import numpy as np


def state_to_cos_sin(x: np.ndarray) -> np.ndarray:
    x_coord = x[0]
    y_coord = x[1]
    yaw = x[2]
    return np.array([x_coord, y_coord, np.cos(yaw), np.sin(yaw)])


def cos_sin_to_state(x: np.ndarray) -> np.ndarray:
    x_coord = x[0]
    y_coord = x[1]
    cos_yaw = x[2]
    sin_yaw = x[3]
    return np.array([x_coord, y_coord, np.arctan2(sin_yaw, cos_yaw)])
