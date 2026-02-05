import torch
import numpy as np

from abc import ABC, abstractmethod


class AbstractDynamicsModel(ABC):

    def __init__(self, 
                 state_dim: int, 
                 action_dim: int,
                 backend: str,
                 device: str) -> None:
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._backend = backend
        self._device = device

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def device(self) -> str | None:
        if self._backend == "torch":
            return self._device
        return None

    def __call__(self, 
                 x: torch.Tensor | np.ndarray, 
                 u: torch.Tensor | np.ndarray,
                 unnomralize: bool = False,
                 clip: bool = False) -> torch.Tensor | np.ndarray:
        pass

    def rollout(self, 
                u_seq: torch.Tensor | np.ndarray,
                x_init: torch.Tensor | np.ndarray | None = None):
        pass


class UnicycleModel(AbstractDynamicsModel):

    def __init__(self,
                 dt: float,
                 backend: str = "torch",
                 device: str = "cuda") -> None:
        super(UnicycleModel, self).__init__(state_dim=4, action_dim=2, backend=backend, device=device)
        if backend == "torch":
            self._pi = torch.tensor(torch.pi, device=device, dtype=torch.float32)
            self._dt = torch.tensor(dt, device=device, dtype=torch.float32)
            self._device = device
        elif backend == "numpy":
            self._pi = np.pi
            self._dt = dt
            self._device = None
        else:
            raise ValueError(f"Invalid backend: {backend}")
        self._backend = backend

    def __call__(self, 
                 x: torch.Tensor | np.ndarray, 
                 u: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        x_coord = x[:, 0]
        y_coord = x[:, 1]
        cos_yaw = x[:, 2]
        sin_yaw = x[:, 3]
        if self._backend == "torch":
            yaw = torch.atan2(sin_yaw, cos_yaw)
        else:
            yaw = np.arctan2(sin_yaw, cos_yaw)
        v = u[:, 0]
        w = u[:, 1]

        x_coord_next = x_coord + v * cos_yaw * self._dt
        y_coord_next = y_coord + v * sin_yaw * self._dt
        yaw_next = yaw + w * self._dt
        yaw_next = self._wrap_angle(yaw_next)

        if self._backend == "torch":
            return torch.stack([x_coord_next, y_coord_next, 
                                torch.cos(yaw_next), torch.sin(yaw_next)], dim=-1)
        else:
            return np.stack([x_coord_next, y_coord_next, 
                             np.cos(yaw_next), np.sin(yaw_next)], axis=-1)

    def rollout(self, 
                u_seq: torch.Tensor | np.ndarray,
                x_init: torch.Tensor | np.ndarray | None = None):
        bs = u_seq.shape[0]
        horizon = u_seq.shape[1]
        if x_init is None:
            if self._backend == "torch":
                x_init = torch.tensor([0., 0., 1., 0.], device=self._device)
                x_init = torch.tile(x_init, (bs, 1))
            else:
                x_init = np.array([0., 0., 1., 0.])
                x_init = np.tile(x_init, (bs, 1))
        x_seq = [x_init]
        for i in range(horizon):
            x_next = self(x_seq[-1], u_seq[:, i, :])
            x_seq.append(x_next)
        if self._backend == "torch":
            x_seq = torch.stack(x_seq, dim=1)
        else:
            x_seq = np.stack(x_seq, axis=1)
        return x_seq

    def _wrap_angle(self, angle: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        return (angle + self._pi) % (2 * self._pi) - self._pi


class UnicycleSteeringModel(AbstractDynamicsModel):

    def __init__(self, 
                 dt: float,
                 speed: float, 
                 backend: str = "torch",
                 device: str = "cuda") -> None:
        super(UnicycleSteeringModel, self).__init__(state_dim=4, action_dim=1, 
                                                    backend=backend, device=device)
        self._model = UnicycleModel(dt=dt, backend=backend, device=device)
        if backend == "torch":
            self._speed = torch.tensor(speed, device=device, dtype=torch.float32)
        elif backend == "numpy":
            self._speed = speed
        else:
            raise ValueError(f"Invalid backend: {backend}")
        self._backend = backend

    def __call__(self, 
                 x: torch.Tensor | np.ndarray, 
                 u: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if self._backend == "torch":
            linear_vel = torch.tile(self._speed, (x.shape[0], 1))
            u = torch.cat((linear_vel, u), dim=1)
        else:
            linear_vel = np.tile(self._speed, (x.shape[0], 1))
            u = np.concat((linear_vel, u), axis=1)
        return self._model(x, u)

    def rollout(self, 
                u_seq: torch.Tensor | np.ndarray,
                x_init: torch.Tensor | np.ndarray | None = None):
        bs = u_seq.shape[0]
        horizon = u_seq.shape[1]
        if self._backend == "torch":
            linear_vel = torch.tile(self._speed, (bs, horizon, 1))
            u_seq = torch.cat((linear_vel, u_seq), dim=-1)
        else:
            linear_vel = np.tile(self._speed, (bs, horizon, 1))
            u_seq = np.concat((linear_vel, u_seq), axis=-1)
        return self._model.rollout(u_seq, x_init)


class BicycleModel(AbstractDynamicsModel):

    def __init__(self,
                 dt: float,
                 l: float,
                 backend: str = "torch",
                 device: str = "cuda") -> None:
        super(BicycleModel, self).__init__(state_dim=4, action_dim=2, backend=backend, device=device)
        if backend == "torch":
            self._pi = torch.tensor(torch.pi, device=device, dtype=torch.float32)
            self._dt = torch.tensor(dt, device=device, dtype=torch.float32)
            self._l = torch.tensor(l, device=device, dtype=torch.float32)
            self._device = device
        elif backend == "numpy":
            self._pi = np.pi
            self._dt = dt
            self._l = l
            self._device = None
        else:
            raise ValueError(f"Invalid backend: {backend}")
        self._backend = backend

    def __call__(self, 
                 x: torch.Tensor | np.ndarray, 
                 u: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        x_coord = x[:, 0]
        y_coord = x[:, 1]
        cos_yaw = x[:, 2]
        sin_yaw = x[:, 3]
        if self._backend == "torch":
            yaw = torch.atan2(sin_yaw, cos_yaw)
        else:
            yaw = np.arctan2(sin_yaw, cos_yaw)
        v = u[:, 0]
        delta = u[:, 1]

        x_coord_next = x_coord + v * cos_yaw * self._dt
        y_coord_next = y_coord + v * sin_yaw * self._dt
        if self._backend == "torch":
            yaw_next = yaw + (v / self._l) * torch.tan(delta) * self._dt
        else:
            yaw_next = yaw + (v / self._l) * np.tan(delta) * self._dt
        yaw_next = self._wrap_angle(yaw_next)

        if self._backend == "torch":
            return torch.stack([x_coord_next, y_coord_next, 
                                torch.cos(yaw_next), torch.sin(yaw_next)], dim=-1)
        else:
            return np.stack([x_coord_next, y_coord_next, 
                             np.cos(yaw_next), np.sin(yaw_next)], axis=-1)

    def rollout(self, 
                u_seq: torch.Tensor | np.ndarray,
                x_init: torch.Tensor | np.ndarray | None = None):
        bs = u_seq.shape[0]
        horizon = u_seq.shape[1]
        if x_init is None:
            if self._backend == "torch":
                x_init = torch.tensor([0., 0., 1., 0.], device=self._device)
                x_init = torch.tile(x_init, (bs, 1))
            else:
                x_init = np.array([0., 0., 1., 0.])
                x_init = np.tile(x_init, (bs, 1))
        x_seq = [x_init]
        for i in range(horizon):
            x_next = self(x_seq[-1], u_seq[:, i, :])
            x_seq.append(x_next)
        if self._backend == "torch":
            x_seq = torch.stack(x_seq, dim=1)
        else:
            x_seq = np.stack(x_seq, axis=1)
        return x_seq

    def _wrap_angle(self, angle: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        return (angle + self._pi) % (2 * self._pi) - self._pi


class BicycleSteeringModel(AbstractDynamicsModel):

    def __init__(self, 
                 dt: float,
                 l: float,
                 speed: float, 
                 backend: str = "torch",
                 device: str = "cuda") -> None:
        super(BicycleSteeringModel, self).__init__(state_dim=4, action_dim=1, 
                                                   backend=backend, device=device)
        self._model = BicycleModel(dt=dt, l=l, backend=backend, device=device)
        if backend == "torch":
            self._speed = torch.tensor(speed, device=device, dtype=torch.float32)
        elif backend == "numpy":
            self._speed = speed
        else:
            raise ValueError(f"Invalid backend: {backend}")
        self._backend = backend

    def __call__(self, 
                 x: torch.Tensor | np.ndarray, 
                 u: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if self._backend == "torch":
            linear_vel = torch.tile(self._speed, (x.shape[0], 1))
            u = torch.cat((linear_vel, u), dim=1)
        else:
            linear_vel = np.tile(self._speed, (x.shape[0], 1))
            u = np.concat((linear_vel, u), axis=1)
        return self._model(x, u)

    def rollout(self, 
                u_seq: torch.Tensor | np.ndarray,
                x_init: torch.Tensor | np.ndarray | None = None):
        bs = u_seq.shape[0]
        horizon = u_seq.shape[1]
        if self._backend == "torch":
            linear_vel = torch.tile(self._speed, (bs, horizon, 1))
            u_seq = torch.cat((linear_vel, u_seq), dim=-1)
        else:
            linear_vel = np.tile(self._speed, (bs, horizon, 1))
            u_seq = np.concat((linear_vel, u_seq), axis=-1)
        return self._model.rollout(u_seq, x_init)
