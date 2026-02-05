from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import normflows as nf
import numpy as np

from mppi_plus_proto.nn_models.utils import create_mlp, get_activation_cls


class SimpleConditionalMLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 context_dim: int,
                 hidden_dims: list[int],
                 activation: str = "relu",
                 init_method: str | None = None):
        super(SimpleConditionalMLP, self).__init__()  
        self._mlp = create_mlp([input_dim + context_dim] + hidden_dims, activation, 
                                init_method=init_method)

    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
        return self._mlp(torch.cat([x, context], dim=-1))


class ContextOnlyConditionalMLP(nn.Module):

    def __init__(self,
                 context_dim: int,
                 hidden_dims: list[int],
                 activation: str = "relu",
                 init_method: str | None = None) :
        super(ContextOnlyConditionalMLP, self).__init__()
        self._mlp = create_mlp([context_dim] + hidden_dims, activation, 
                                init_method=init_method)

    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
        return self._mlp(context)


class MixtureConditionalLayer(nn.Module):

    def __init__(self, 
                 input_dim: int,
                 condition_dim: int,
                 output_dim: int,
                 n_experts: int,
                 router_hidden_dims: int,
                 router_activation: str = "relu",
                 init_method: str | None = None) -> None:
        super(MixtureConditionalLayer, self).__init__()
        self._router = create_mlp([condition_dim] + router_hidden_dims + [n_experts],
                                  "relu",
                                  init_method=init_method)
        experts = []
        for _ in range(n_experts):
            layer = nn.Linear(input_dim, output_dim, bias=False)
            if init_method is not None:
                if init_method == "zeros":
                    nn.init.zeros_(layer.weight)
                else:
                    raise ValueError(f"Init method {init_method} not supported")
            experts.append(layer)
        self._experts = nn.ModuleList(experts)
    
    def forward(self, 
            x: torch.Tensor, 
            context: torch.Tensor) -> torch.Tensor:
        weights = self._router(context)
        result = None
        for i, expert in range(self._experts):
            expert_output = weights[:, i] * expert(x)
            result = expert_output if result is None else result + expert_output
        return result


class MixtureConditionalMLP(nn.Module):

    def __init__(self,
                 input_dim: int,
                 context_dim: int,
                 output_dim: int,
                 n_layers: int,
                 n_experts: int,
                 router_hidden_dims: int,
                 activation: str = "relu",
                 init_method: str | None = None):
        super(MixtureConditionalMLP, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(MixtureConditionalLayer(
                input_dim=input_dim if i == 0 else output_dim,
                condition_dim=context_dim,
                output_dim=output_dim,
                n_experts=n_experts,
                router_hidden_dims=router_hidden_dims,
                router_activation=activation,
                init_method=init_method,
            ))
            if i != n_layers - 1:
                layers.append(
                    get_activation_cls(activation)()
                )
        self._layers = nn.ModuleList(layers)      

    def forward(self, 
                x: torch.Tensor, 
                context: torch.Tensor) -> torch.Tensor:
        result = x
        for layer in self._layers:
            result = layer(result, context)
        return result


class AffineConstFlowConditional1d(nf.flows.base.Flow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, param_map):
        """Constructor

        Args:
          shape: Shape of the coupling layer
          scale: Flag whether to apply scaling
          shift: Flag whether to apply shift
          logscale_factor: Optional factor which can be used to control the scale of the log scale factor
        """
        super().__init__()
        self.add_module("param_map", param_map)

    def forward(self, z, context: torch.Tensor):
        st_params = self.param_map(z, context)
        s = st_params[:, 0::2, ...]
        t = st_params[:, 1::2, ...]

        z = z * torch.exp(s) + t
        log_det = torch.sum(s, dim=-1)

        return z, log_det

    def inverse(self, z, context: torch.Tensor):
        st_params = self.param_map(z, context)
        s = st_params[:, 0::2, ...]
        t = st_params[:, 1::2, ...]

        z = (z - t) * torch.exp(-s)
        log_det = -torch.sum(s, dim=-1)

        return z, log_det


class AffineCouplingConditional(nf.flows.base.Flow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map="exp"):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid scale when sampling from the model
        """
        super().__init__()
        self.add_module("param_map", param_map)
        self.scale = scale
        self.scale_map = scale_map

    def forward(self, z, context):
        """
        z is a list of z1 and z2; ```z = [z1, z2]```
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1

        Args:
          z
        """
        z1, z2 = z
        param = self.param_map(z1, context)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 + param
            log_det = nf.flows.base.zero_log_det_like_z(z2)
        return [z1, z2], log_det

    def inverse(self, z, context):
        z1, z2 = z
        param = self.param_map(z1, context)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == "exp":
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            elif self.scale_map == "sigmoid_inv":
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale), dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError("This scale map is not implemented.")
        else:
            z2 = z2 - param
            log_det = nf.flows.base.zero_log_det_like_z(z2)
        return [z1, z2], log_det


class AffineCouplingBlockConditional(nf.flows.base.Flow):
    """
    Affine Coupling layer including split and merge operation
    """

    def __init__(self, param_map, scale=True, scale_map="exp", split_mode="channel"):
        """Constructor

        Args:
          param_map: Maps features to shift and scale parameter (if applicable)
          scale: Flag whether scale shall be applied
          scale_map: Map to be applied to the scale parameter, can be 'exp' as in RealNVP or 'sigmoid' as in Glow
          split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [nf.flows.Split(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCouplingConditional(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [nf.flows.Merge(split_mode)]

    def forward(self, z, context):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            if isinstance(flow, AffineCouplingConditional):
                z, log_det = flow(z, context)
            else:
                z, log_det = flow(z)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z, context):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            if isinstance(self.flows[i], AffineCouplingConditional):
                z, log_det = self.flows[i].inverse(z, context)
            else:
                z, log_det = self.flows[i].inverse(z)
            log_det_tot += log_det
        return z, log_det_tot


class TanhFlow(nf.flows.base.Flow):

    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor, context: torch.Tensor = None):
        y = F.tanh(z)
        log_deriv = self._log_deriv_base(z)
        log_det = torch.sum(log_deriv, dim=list(range(1, z.dim())))
        return y, log_det

    def inverse(self, y: torch.Tensor, context: torch.Tensor = None):
        y = torch.clamp(y, -1. + 1e-6, 1 - 1e-6)
        z = torch.atanh(y)
        log_deriv = -self._log_deriv_base(z)
        log_det = torch.sum(log_deriv, dim=list(range(1, y.dim())))
        return z, log_det

    def _log_deriv_base(self, z: torch.Tensor) -> torch.Tensor:
        two = torch.tensor(2.).to(z.device)
        c = two * torch.log(two)
        result = c + two * z - two * F.softplus(two * z)
        return result


class TanhBoundFlow(nf.flows.base.Flow):

    def __init__(self,
                 low: tuple[float],
                 high: tuple[float]):
        super().__init__()
        low = torch.tensor(low, dtype=torch.float32)
        high = torch.tensor(high, dtype=torch.float32)
        assert torch.all(high > low)

        scale = (high - low) / 2.
        shift = (high + low) / 2.

        self.register_buffer("low", low)
        self.register_buffer("high", high)
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, z: torch.Tensor, context: torch.Tensor = None):
        tanh_z = F.tanh(z)
        y = self.scale * tanh_z + self.shift
        log_deriv = self._log_deriv_base(z)
        log_det = torch.sum(log_deriv, dim=list(range(1, z.dim())))
        return y, log_det
    
    def inverse(self, y: torch.Tensor, context: torch.Tensor = None):
        tanh_z = (y - self.shift) / self.scale
        tanh_z = torch.clamp(tanh_z, -1 + 1e-6, 1 - 1e-6)
        z = torch.atanh(tanh_z)
        log_deriv = -self._log_deriv_base(z)
        log_det = torch.sum(log_deriv, dim=list(range(1, y.dim())))
        return z, log_det
    
    def _log_deriv_base(self, z: torch.Tensor) -> torch.Tensor:
        two = torch.tensor(2., dtype=torch.float32, device=z.device)
        c = two * torch.log(two) + torch.log(self.scale)
        result = c + two * z - two * F.softplus(two * z)
        return result
