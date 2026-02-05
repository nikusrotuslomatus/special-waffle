import numpy as np
import oo_ctrl as octrl

from typing import Tuple


class BicycleSteerMPPIModel(octrl.np.AbstractNumPyModel):

    def __init__(self,
                 dt: float,
                 wheel_base: float,
                 speed: float,
                 angular_bounds: Tuple[float, float],
                 force_clip: bool = False):
        super(BicycleSteerMPPIModel, self).__init__(control_lb=(angular_bounds[0],),
                                                    control_ub=(angular_bounds[1],))
        self._dt = dt
        self._wheel_base = wheel_base
        self._speed = speed
        self._force_clip = force_clip

    def __call__(self,
                 state: np.ndarray,
                 control: np.ndarray) -> np.ndarray:
        if self._force_clip:
            control = self.clip(control)

        x = state[..., 0]
        y = state[..., 1]
        theta = state[..., 2]
        v = self._speed
        delta = control[..., 0]
        l = self._wheel_base

        x_new = x + v * np.cos(theta) * self._dt
        y_new = y + v * np.sin(theta) * self._dt
        theta_new = octrl.np.wrap_angle(theta + (v / l) * np.tan(delta) * self._dt)

        state_new = np.concatenate((
            x_new[..., np.newaxis],
            y_new[..., np.newaxis],
            theta_new[..., np.newaxis]
        ), axis=-1)

        return state_new


class UnicycleSteerMPPIModel(octrl.np.AbstractNumPyModel):

    def __init__(self,
                 dt: float,
                 speed: float,
                 angular_bounds: Tuple[float, float],
                 force_clip: bool = False):
        super(UnicycleSteerMPPIModel, self).__init__(control_lb=(angular_bounds[0],),
                                                     control_ub=(angular_bounds[1],))
        self._dt = dt
        self._speed = speed
        self._force_clip = force_clip

    def __call__(self,
                 state: np.ndarray,
                 control: np.ndarray) -> np.ndarray:
        if self._force_clip:
            control = self.clip(control)

        x = state[..., 0]
        y = state[..., 1]
        theta = state[..., 2]
        v = self._speed
        omega = control[..., 0]

        x_new = x + v * np.cos(theta) * self._dt
        y_new = y + v * np.sin(theta) * self._dt
        theta_new = octrl.np.wrap_angle(theta + omega * self._dt)

        state_new = np.concatenate((
            x_new[..., np.newaxis],
            y_new[..., np.newaxis],
            theta_new[..., np.newaxis]
        ), axis=-1)

        return state_new
