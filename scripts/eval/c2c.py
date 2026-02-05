import autoroot
import autorootcwd
import time
import numpy as np
import oo_ctrl as octrl

from typing import Tuple
from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld
from pyminisim.robot import BicycleRobotModel
from pyminisim.sensors import LidarSensor, LidarSensorConfig, SemanticDetector, SemanticDetectorConfig
from pyminisim.visual import Renderer, CircleDrawing

from mppi_plus_proto.mppi.mppi_dynamics_models import BicycleSteerMPPIModel
from mppi_plus_proto.mppi.mppi_presamplers import (
    presampler_bicycle_steer,
)


OBSTACLES = np.array([[1.5, 0., 0.8]])

WHEEL_BASE = 0.324
SPEED = 1.
DT = 0.2
GOAL_THRESHOLD = 0.25


def create_sim() -> Tuple[Simulation, Renderer]:
    robot_model = BicycleRobotModel(wheel_base=WHEEL_BASE, 
                                    initial_center_pose=np.array([0., 0., 0.]),
                                    initial_control=np.array([0., np.deg2rad(0.)]))
    sensors = []
    sim = Simulation(sim_dt=0.01,
                     # world_map=CirclesWorld(circles=OBSTACLES),
                     world_map=EmptyWorld(),
                     robot_model=robot_model,
                     pedestrians_model=None,
                     sensors=sensors,
                     rt_factor=1.)
    renderer = Renderer(simulation=sim,
                        resolution=90.0,
                        screen_size=(750, 750),
                        camera="robot")
    return sim, renderer


def create_controller() -> octrl.np.MPPI:
    return octrl.np.MPPI(
        horizon=16,
        n_samples=2500,
        lmbda=0.5,
        model=BicycleSteerMPPIModel(dt=DT,
                                wheel_base=WHEEL_BASE,
                                speed=SPEED,
                                angular_bounds=(-np.deg2rad(30.), np.deg2rad(30.))),
        biased=False,
        sampler=octrl.np.GaussianActionSampler(
            stds=(np.sqrt(0.1),)
        ),
        cost=[
            octrl.np.SE2C2CCost(threshold_distance=0.05,
                    threshold_angle=np.deg2rad(20.),
                    weight_distance=2,
                    weight_angle=1.,
                    squared=False,
                    terminal_weight=40.,
                    angle_error="cos_sin")
        ],
        state_transform=octrl.np.RearToCenterTransform(WHEEL_BASE),
        return_pre_samples=True,
        return_samples=True,
        return_state_seq=True,
        presampler=presampler_bicycle_steer(n_samples=1000)
    )


def main():
    sim, renderer = create_sim()
    renderer.initialize()

    goal = np.array([0., 0., np.pi])
    controller = create_controller()

    running = True
    sim.step()  # First step can take some time due to Numba compilation
    n_steps = 0

    u_pred = np.array([0., 0.])
    hold_time = sim.sim_dt

    while running:
        renderer.render()

        if hold_time >= DT:

            x_current = sim.current_state.world.robot.pose
            u_pred, info = controller.step(x_current,
                                           {"goal": goal})
            u_pred = np.array([SPEED, u_pred[0]])
            hold_time = 0.
            if "x_seq_pre_samples" in info:
                renderer.draw("pre_samples", CircleDrawing(info["x_seq_pre_samples"][..., :2].reshape((-1, 2)), 0.03, (247, 200, 245), 0))
                renderer.draw("pre_samples_min", CircleDrawing(info["x_seq_pre_samples_min"][..., :2], 0.03, (222, 16, 57), 0))
            elif "x_seq_samples" in info:
                renderer.draw("samples", CircleDrawing(info["x_seq_samples"][..., :2].reshape((-1, 2)), 0.03, (171, 226, 245), 0))
            renderer.draw("robot_traj", CircleDrawing(info["x_seq"][:, :2], 0.05, (252, 196, 98, 0.5), 0))
            renderer.draw("goal", CircleDrawing(goal[:2], 0.1, (255, 0, 0), 0))
        
        sim.step(u_pred)
        n_steps += 1
        if np.linalg.norm(sim.current_state.world.robot.pose[:2] - goal[:2]) <= GOAL_THRESHOLD and n_steps > 100:
            print("Goal conf reached!")
            break
        # print(f"RT factor: {sim.sim_dt / (finish_time - start_time)}")
        hold_time += sim.sim_dt

    # Done! Time to quit.
    renderer.close()


if __name__ == '__main__':
    main()
