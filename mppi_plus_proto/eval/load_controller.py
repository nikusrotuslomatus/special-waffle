import yaml
import numpy as np
import oo_ctrl as octrl

from pathlib import Path
from mppi_plus_proto.mppi.mppi_dynamics_models import BicycleSteerMPPIModel, UnicycleSteerMPPIModel
from mppi_plus_proto.mppi.mppi_presamplers import (
    NFPresampler,
    presampler_bicycle_steer,
    presampler_unicycle_steer
)


def get_available_controllers(config_path: str | Path) -> list[str]:
    with open(config_path, "r") as f:
        return list(yaml.safe_load(f)["controllers"].keys())


def load_controller(config_path: str | Path,
                    controller_name: str,
                    cost,
                    debug: bool = False):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["controllers"][controller_name]

    state_transform = None
    presampler = None

    if config["type"] == "mppi":
        sampler = octrl.np.GaussianActionSampler(
            stds=(np.sqrt(config["variance"]),)
        )
    elif config["type"] == "logmppi":
        sampler = octrl.np.NLNActionSampler(
            stds=(np.sqrt(config["variance"]),)
        )
    else:
        raise ValueError(f"Unknown controller type {config['type']}")


    if config["model"] == "bicycle_steer":
        angle_limit = np.deg2rad(config["angle_limit"])
        model = BicycleSteerMPPIModel(dt=config["dt"],
                                      wheel_base=config["wheel_base"],
                                      speed=config["speed"],
                                      angular_bounds=(-angle_limit, angle_limit))
        state_transform = octrl.np.RearToCenterTransform(config["wheel_base"])
        if "presampler" in config:
            if config["presampler"] == "nf":
                presampler = presampler_bicycle_steer(n_samples=config["n_presamples"])
                assert presampler._horizon == config["horizon"]
            else:
                raise ValueError(f"Unknown config {config['presampler']}")
    
    elif config["model"] == "unicycle_steer":
        angular_vel_limit = np.deg2rad(config["angular_vel_limit"])
        model = UnicycleSteerMPPIModel(dt=config["dt"],
                                       speed=config["speed"],
                                       angular_bounds=(-angular_vel_limit, angular_vel_limit))
        if "presampler" in config:
            if config["presampler"] == "nf":
                presampler = presampler_unicycle_steer(n_samples=config["n_presamples"])
                assert presampler._horizon == config["horizon"]
            else:
                raise ValueError(f"Unknown config {config['presampler']}")
    else:
        raise ValueError(f"Unknown model {config['modle']}") 

    return octrl.np.MPPI(
        horizon=config["horizon"],
        n_samples=config["n_samples"],
        lmbda=config["lmbda"],
        model=model,
        biased=False,
        sampler=sampler,
        cost=cost,
        state_transform=state_transform,
        return_pre_samples=debug,
        return_samples=debug,
        return_state_seq=debug,
        presampler=presampler
    )
