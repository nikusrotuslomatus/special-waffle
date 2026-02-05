import numpy as np
import torch
import torch.nn as nn
import oo_ctrl as octrl

from pathlib import Path
from mppi_plus_proto.nn_models.generative import generic_realnvp


class NFPresampler(octrl.np.AbstractPresampler):

    def __init__(self,
                 nf_model: nn.Module,
                 weights_path: str | Path,
                 n_samples: int,
                 action_dim: int,
                 horizon: int,
                 device: str = "cuda") -> None:
        super(NFPresampler, self).__init__()
        nf_model = nf_model.to(device)
        nf_model.load_state_dict(torch.load(str(weights_path), 
                                 map_location=device))
        nf_model.eval()
        self._nf_model = nf_model
        self._n_samples = n_samples
        self._action_dim = action_dim
        self._horizon = horizon
        self._device = device

    def sample(self, state, observation) -> np.ndarray:
        with torch.inference_mode():
            samples, _ = self._nf_model.sample(self._n_samples)
            samples = samples.reshape((self._n_samples, self._horizon, self._action_dim))
            samples = samples.cpu().numpy()
        return samples


def presampler_bicycle_steer(n_samples: int) -> NFPresampler:
    model =  generic_realnvp(action_dim=1,
                             horizon=16)
    return NFPresampler(
        nf_model=model,
        weights_path="deploy_checkpoints/bicycle_steer__angle_30__horizon_16__dt_02.pth",
        n_samples=n_samples,
        action_dim=1,
        horizon=16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def presampler_unicycle_steer(n_samples: int) -> NFPresampler:
    model =  generic_realnvp(action_dim=1,
                             horizon=16)
    return NFPresampler(
        nf_model=model,
        weights_path="deploy_checkpoints/unicycle_steer__angle_45__horizon_16__dt_02.pth",
        n_samples=n_samples,
        action_dim=1,
        horizon=16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


def presampler_unicycle_full(n_samples: int) -> NFPresampler:
    model =  generic_realnvp(action_dim=2,
                             horizon=16)
    return NFPresampler(
        nf_model=model,
        weights_path="deploy_checkpoints/unicycle_full__angle_45__horizon_16__dt_02.pth",
        n_samples=n_samples,
        action_dim=2,
        horizon=16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
