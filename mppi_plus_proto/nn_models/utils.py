import torch.nn as nn

from typing import Callable


def get_activation_cls(activation: str) -> Callable[[], nn.Module]:
    if activation == "relu":
        return nn.ReLU
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "silu":
        return nn.SiLU
    else:
        raise ValueError(f"Activation {activation} not supported")


def create_mlp(dims: list[int], 
               activation: str,
               last_activation: bool | str = False,
               init_method: str | None = None) -> nn.Module:
    activation_cls = get_activation_cls(activation)
    if isinstance(last_activation, str):
        last_activation_cls = get_activation_cls(last_activation)
    elif last_activation:
        last_activation_cls = activation_cls
    else:
        last_activation_cls = None

    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        
        if init_method is not None:
            if init_method == "zeros":
                nn.init.zeros_(layers[-1].weight)
                nn.init.zeros_(layers[-1].bias)
            else:
                raise ValueError(f"Init method {init_method} not supported")

        if i == (len(dims) - 2) and last_activation_cls is not None:
            layers.append(last_activation_cls())
        elif i < (len(dims) - 2):
            layers.append(activation_cls())
    return nn.Sequential(*layers)
