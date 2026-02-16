import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import normflows as nf
    from mppi_plus_proto.nn_models.layers import (
        AffineCouplingBlockConditional,
        SimpleConditionalMLP,
        TanhBoundFlow,
    )
    _NF_IMPORT_ERROR = None
except Exception as _err:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.append(str(root))
    try:
        import normflows as nf
        from mppi_plus_proto.nn_models.layers import (
            AffineCouplingBlockConditional,
            SimpleConditionalMLP,
            TanhBoundFlow,
        )
        _NF_IMPORT_ERROR = None
    except Exception as _err2:
        nf = None
        _NF_IMPORT_ERROR = _err2


class ContextNet(nn.Module):
    def __init__(self, z_dim: int, extra_ctx_dim: int, ctx_dim: int, hidden_dim: int = 128):
        super().__init__()
        in_dim = z_dim + extra_ctx_dim
        self._net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, ctx_dim), nn.ELU()
        )

    def forward(self, z0: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ctx is None:
            x = z0
        else:
            x = torch.cat([z0, ctx], dim=-1)
        return self._net(x)


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, base, flows):
        super().__init__()
        self.base = base
        self.flows = nn.ModuleList(flows)

    def _is_conditional(self, flow: nn.Module) -> bool:
        return isinstance(flow, AffineCouplingBlockConditional)

    def forward(self, z: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for flow in self.flows:
            if self._is_conditional(flow):
                z, ld = flow(z, context)
            else:
                z, ld = flow(z)
            log_det += ld
        return z, log_det

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for flow in reversed(self.flows):
            if self._is_conditional(flow):
                z, ld = flow.inverse(z, context)
            else:
                z, ld = flow.inverse(z)
            log_det += ld
        return z, log_det

    def sample(self, num_samples: int, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.base.sample(num_samples)
        x, log_det = self.forward(z, context)
        logp = self.base.log_prob(z) - log_det
        return x, logp

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        z, log_det = self.inverse(x, context)
        logp = self.base.log_prob(z) + log_det
        return logp


class ConditionalFlowSampler(nn.Module):
    def __init__(
        self,
        z_dim: int,
        action_dim: int,
        horizon: int,
        ctx_dim: int = 128,
        extra_ctx_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 8,
        zero_init_coupling: bool = False,
        bound_actions: bool = True,
        action_low: float = -1.0,
        action_high: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__()
        if nf is None:
            raise ImportError(
                "normflows or mppi_plus_proto is not available. "
                "Install dependencies or disable MI warm-start. "
                f"Original error: {_NF_IMPORT_ERROR}"
            )
        self.action_dim = action_dim
        self.horizon = horizon
        self.total_dim = action_dim * horizon
        if self.total_dim % 2 != 0:
            raise ValueError(
                f"action_dim*horizon must be even for coupling split, got {self.total_dim} "
                f"(action_dim={action_dim}, horizon={horizon})."
            )
        self.device = torch.device(device)

        self.context_net = ContextNet(z_dim, extra_ctx_dim, ctx_dim, hidden_dim)

        flows = []
        init_method = "zeros" if zero_init_coupling else None
        for _ in range(num_layers):
            param_map = SimpleConditionalMLP(
                input_dim=self.total_dim // 2,
                context_dim=ctx_dim,
                hidden_dims=[hidden_dim, hidden_dim, self.total_dim],
                activation="relu",
                init_method=init_method,
            )
            flows.append(AffineCouplingBlockConditional(param_map))
            flows.append(nf.flows.Permute(self.total_dim, mode="shuffle"))

        if bound_actions:
            low = tuple([action_low] * self.total_dim)
            high = tuple([action_high] * self.total_dim)
            flows.append(TanhBoundFlow(low=low, high=high))

        base = nf.distributions.base.DiagGaussian(self.total_dim)
        self.flow = ConditionalNormalizingFlow(base, flows)
        self.to(self.device)

    def sample(
        self,
        z0: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z0 = z0.to(self.device)
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        if ctx is not None:
            ctx = ctx.to(self.device)
        batch = z0.shape[0]
        context = self.context_net(z0, ctx)
        context = context.repeat_interleave(num_samples, dim=0)
        samples, logp = self.flow.sample(batch * num_samples, context)
        samples = samples.view(batch, num_samples, self.horizon, self.action_dim)
        logp = logp.view(batch, num_samples)
        return samples, logp

    @torch.no_grad()
    def sample_no_grad(
        self,
        z0: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
        num_samples: int = 1,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.sample(z0=z0, ctx=ctx, num_samples=num_samples)

    def log_prob(
        self,
        u: torch.Tensor,
        z0: torch.Tensor,
        ctx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z0 = z0.to(self.device)
        u = u.to(self.device)
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        if u.dim() == 3:
            u = u.unsqueeze(1)
        if ctx is not None:
            ctx = ctx.to(self.device)
        batch, num_samples, horizon, act_dim = u.shape
        assert horizon == self.horizon
        assert act_dim == self.action_dim
        context = self.context_net(z0, ctx)
        context = context.repeat_interleave(num_samples, dim=0)
        u_flat = u.reshape(batch * num_samples, self.total_dim)
        logp = self.flow.log_prob(u_flat, context)
        logp = logp.view(batch, num_samples)
        return logp.squeeze(1) if num_samples == 1 else logp
