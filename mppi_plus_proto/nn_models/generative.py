from sympy.polys.polyoptions import Gaussian
import torch
import torch.nn as nn
import normflows as nf

from mppi_plus_proto.nn_models.utils import create_mlp
from mppi_plus_proto.nn_models.layers import TanhBoundFlow


class GaussianModel(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 mlp_dims: list[int],
                 activation: str,
                 log_std_min: float = -10.,
                 log_std_max: float = 2.) -> None:
        super(GaussianModel, self).__init__()
        self._feature_net = create_mlp([input_dim] + mlp_dims, 
                                        activation=activation)
        self._mean_net = nn.Linear(mlp_dims[-1], output_dim)
        self._log_std_net = nn.Linear(mlp_dims[-1], output_dim)
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._feature_net(x)
        mean = self._mean_net(features)
        log_std = self._log_std_net(features)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        return mean, log_std

    def log_prob(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(x)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(u).sum(dim=-1)
        return log_prob


class RecurrentGaussianModel(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 feature_dim: int = 512,
                 num_layers: int = 1,
                 log_std_min: float = -10.,
                 log_std_max: float = 2.) -> None:
        super(RecurrentGaussianModel, self).__init__()
        self._input_dim = input_dim
        self._feature_net = nn.GRU(input_size=input_dim,
                                   hidden_size=feature_dim,
                                   num_layers=num_layers,
                                   batch_first=True)
        self._mean_net = nn.Linear(feature_dim, output_dim)
        self._log_std_net = nn.Linear(feature_dim, output_dim)
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, features = self._feature_net(x)
        features = features.unsqueeze(0)
        mean = self._mean_net(features)
        log_std = self._log_std_net(features)
        log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)
        return mean, log_std

    def log_prob(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        if len(x.shape) == 2:
            x = x.reshape((bs, -1, self._input_dim))
        mean, log_std = self(x)
        std = torch.exp(log_std)
        mean = mean.reshape((bs, -1))
        std = std.reshape((bs, -1))
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(u).sum(dim=-1)
        return log_prob


def generic_gaussian(state_dim: int, action_dim: int, horizon: int, mlp_dims: tuple[int,...] = (512, 512),
                     activation: str = "relu") -> GaussianModel:
    return GaussianModel(input_dim=state_dim * horizon,
                         output_dim=action_dim * horizon,
                         mlp_dims=list(mlp_dims),
                         activation=activation)


def generic_recurrent_gaussian(state_dim: int,
                               action_dim: int,
                               feature_dim: int = 512,
                               num_layers: int = 1) -> RecurrentGaussianModel:
    return RecurrentGaussianModel(input_dim=state_dim,
                                  output_dim=action_dim,
                                  feature_dim=feature_dim,
                                  num_layers=num_layers)


def generic_realnvp(action_dim: int, 
                    horizon: int,
                    num_layers: int = 32,
                    condition_hidden_dims: tuple[int, ...] = (64, 64),
                    low: tuple[float, ...] | None = None,
                    high: tuple[float, ...] | None = None) -> nf.NormalizingFlow:
    action_dim = action_dim * horizon
    base = nf.distributions.base.DiagGaussian(action_dim)

    # Define list of flows
    flows = []
    for i in range(num_layers):
        # Add flow layer
        param_map = nf.nets.MLP([action_dim // 2] +  list(condition_hidden_dims) + [action_dim], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(action_dim, mode='shuffle'))

    if low is not None:
        assert high is not None and len(low) == len(high)
        flows.append(TanhBoundFlow(low=low, high=high))
        
    # Construct flow model
    model = nf.NormalizingFlow(base, flows)

    return model


def generic_autoregressive(action_dim: int, 
                        horizon: int,
                        num_layers: int = 16,
                        hidden_units: int = 256,
                        hidden_layers: int = 2,
                        low: tuple[float, ...] | None = None,
                        high: tuple[float, ...] | None = None) -> nf.NormalizingFlow:
    latent_size = action_dim * horizon

    flows = []
    for i in range(num_layers):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    if low is not None:
        assert high is not None and len(low) == len(high)
        flows.append(TanhBoundFlow(low=low, high=high))

    # Set base distribuiton
    q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
        
    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)

    return nfm