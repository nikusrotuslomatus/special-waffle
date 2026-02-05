import torch

from typing import Callable


def sample_uniform(num_samples: int, 
                   lb: torch.Tensor, 
                   ub: torch.Tensor,
                   device: str) -> torch.Tensor:
        return torch.rand(num_samples, 
                          lb.shape[0], device=device) * (ub - lb) + lb


class SimpleActionSampler:

    def __init__(self,
                 x_lb: tuple[float, ...],
                 x_ub: tuple[float, ...],
                 u_lb: tuple[float, ...],
                 u_ub: tuple[float, ...],
                 horizon: int,
                 gt_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 start_in_zero: bool = False,
                 device: str = "cuda"):
        self._x_lb = torch.tensor(x_lb, device=device, dtype=torch.float32)
        self._x_ub = torch.tensor(x_ub, device=device, dtype=torch.float32)
        self._u_lb = torch.tensor(u_lb, device=device, dtype=torch.float32)
        self._u_ub = torch.tensor(u_ub, device=device, dtype=torch.float32)
        self._horizon = horizon
        self._start_in_zero = start_in_zero
        self._gt_model = gt_model
        self._device = device

    @property
    def horizon(self) -> int:
        return self._horizon

    def sample(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        u_seq = [sample_uniform(num_samples, self._u_lb, self._u_ub, self._device) 
                 for _ in range(self._horizon)]
        u_seq = torch.stack(u_seq, dim=1)
        
        if not self._start_in_zero:
            x_init = sample_uniform(num_samples, self._x_lb, self._x_ub, self._device)
        else:
            x_init = torch.zeros((num_samples, 3), device=self._device)
        x_init = torch.stack([x_init[:, 0], x_init[:, 1], 
                              torch.cos(x_init[:, 2]), torch.sin(x_init[:, 2])], dim=-1)
        
        x_seq = [x_init]
        x_prev = x_init

        for i in range(self._horizon):
            x_next = self._gt_model(x_prev, u_seq[:, i, :])
            x_seq.append(x_next)
            x_prev = x_next
        x_seq = torch.stack(x_seq, dim=1)

        return u_seq, x_seq
