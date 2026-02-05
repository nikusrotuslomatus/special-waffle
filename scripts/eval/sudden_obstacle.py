import autoroot
import autorootcwd
import time
import json
import fire
import numpy as np
import oo_ctrl as octrl

from typing import Tuple
from functools import partial
from tqdm import tqdm
from pyminisim.core import Simulation
from pyminisim.world_map import EmptyWorld
from pyminisim.robot import BicycleRobotModel, UnicycleRobotModel
from pyminisim.sensors import LidarSensor, LidarSensorConfig, SemanticDetector, SemanticDetectorConfig
from pyminisim.visual import Renderer, CircleDrawing

from mppi_plus_proto.mppi.mppi_dynamics_models import BicycleSteerMPPIModel, UnicycleSteerMPPIModel
from mppi_plus_proto.mppi.mppi_presamplers import (
    presampler_bicycle_steer,
)
from mppi_plus_proto.mppi.sudden_circles_world import (
    SuddenCirclesWorld,
    SuddenCirclesWorldSkin
)
from mppi_plus_proto.eval.load_controller import load_controller, get_available_controllers
from mppi_plus_proto.util.parallel_util import do_parallel


LIDAR_DIST = 1.5
APPEAR_DIST = 1000.
ROBOT_RADIUS = 0.25

GOAL_THRESHOLD = 0.3


def create_sim(world_idx: int, 
               controller: octrl.np.MPPI,
               lidar_dist: float,
               appear_dist: float,
               robot_radius: float,
               render: bool) -> Tuple[Simulation, Renderer, SuddenCirclesWorld]:
    circles = np.load(f"worlds/world_{world_idx}.npz")
    inner_circles = np.array(circles["sudden"])
    boundary_circles = np.array(circles["visible"])
    inner_circles = np.stack((inner_circles[:, 1], inner_circles[:, 0], inner_circles[:, 2]), axis=1)
    boundary_circles = np.stack((boundary_circles[:, 1], boundary_circles[:, 0], boundary_circles[:, 2]), axis=1)
    shift = np.abs(boundary_circles[:, 1].min()) / 2.
    inner_circles[:, 1] = inner_circles[:, 1] + shift
    boundary_circles[:, 1] = boundary_circles[:, 1] + shift

    if isinstance(controller._model, BicycleSteerMPPIModel):
        robot_model = BicycleRobotModel(wheel_base=controller._model._wheel_base, 
                                        initial_center_pose=np.array([1, 0., 0.]),
                                        initial_control=np.array([0., 0.]),
                                        robot_radius=robot_radius)
    elif isinstance(controller._model, UnicycleSteerMPPIModel):
        robot_model = UnicycleRobotModel(initial_pose=np.array([1, 0., 0.]),
                                         initial_control=np.array([0., 0.]),
                                         robot_radius=robot_radius)
    else:
        raise ValueError
    sensors = [LidarSensor(config=LidarSensorConfig(max_dist=lidar_dist))]
    world = SuddenCirclesWorld(sudden_circles=inner_circles, 
                               appear_distance=appear_dist,
                               always_visible_circles=boundary_circles)
    sim = Simulation(sim_dt=0.01,
                     world_map=world,
                     robot_model=robot_model,
                     pedestrians_model=None,
                     sensors=sensors,
                     rt_factor=None)
    if render:
        renderer = Renderer(simulation=sim,
                            resolution=60.0,
                            screen_size=(750, 750),
                            camera="robot",
                            map_skin_fn=SuddenCirclesWorldSkin)
    else:
        renderer = None

    return sim, renderer, world


def create_controller(config_path: str,
                      controller_name: str,
                      robot_radius: float,
                      debug: bool) -> octrl.np.MPPI:
    cost=[
            octrl.np.SE2C2CCost(threshold_distance=0.05,
                    threshold_angle=np.deg2rad(20.),
                    weight_distance=2.,
                    weight_angle=1.,
                    squared=False,
                    terminal_weight=20.,
                    angle_error="cos_sin"),
            # octrl.np.CollisionIndicatorCost(Q=100000.,
            #                                 safe_distance=robot_radius + 0.2,
            #                                 name="CA")
            octrl.np.CollisionFieldCost(Q=100.,
                                        safe_distance=robot_radius + 0.3,
                                        name="CA")
        ]
    
    return load_controller(config_path,
                           controller_name,
                           cost,
                           debug)


def run_experiment(world_idx: int,
                   config_path: str,
                   controller_name: str,
                   lidar_dist: float,
                   appear_dist: float,
                   robot_radius: float,
                   render: bool) -> dict:
    controller = create_controller(config_path, controller_name, robot_radius, render)
    dt = controller._model._dt

    sim, renderer, world = create_sim(world_idx=world_idx, 
               controller=controller,
               lidar_dist=lidar_dist,
               appear_dist=appear_dist,
               robot_radius=robot_radius,
               render=render)

    if renderer is not None:
        renderer.initialize()

    goal = np.array([11., 0., 0.])

    running = True
    sim.step() 
    n_steps = 0

    u_pred = np.array([0., 0.])
    hold_time = sim.sim_dt

    trajectory = []
    success = False

    while running:
        if renderer is not None:
            renderer.render()

        if hold_time >= dt:

            lidar_reading = sim.current_state.sensors[LidarSensor.NAME].reading

            x_current = sim.current_state.world.robot.pose
            u_pred, info = controller.step(x_current,
                                           {"goal": goal,
                                            "obstacles": lidar_reading.points})
            if isinstance(controller._model, BicycleSteerMPPIModel) or isinstance(controller._model, UnicycleSteerMPPIModel):
                u_pred = np.array([controller._model._speed, u_pred[0]])
            hold_time = 0.
            if renderer is not None:
                if "x_seq_pre_samples" in info:
                    renderer.draw("pre_samples", CircleDrawing(info["x_seq_pre_samples"][..., :2].reshape((-1, 2)), 0.03, (247, 200, 245), 0))
                    renderer.draw("pre_samples_min", CircleDrawing(info["x_seq_pre_samples_min"][..., :2], 0.03, (222, 16, 57), 0))
                elif "x_seq_samples" in info:
                    renderer.draw("samples", CircleDrawing(info["x_seq_samples"][..., :2].reshape((-1, 2)), 0.03, (171, 226, 245), 0))
                if "x_seq" in info:
                    renderer.draw("robot_traj", CircleDrawing(info["x_seq"][:, :2], 0.05, (252, 196, 98, 0.5), 0))
                renderer.draw("goal", CircleDrawing(goal[:2], 0.1, (255, 0, 0), 0))
        
        world.update(sim.current_state.world.robot.pose, robot_radius)
        trajectory.append(sim.current_state.world.robot.pose[:2])
        sim.step(u_pred)
        n_steps += 1
        if sim.current_state.world.robot_to_world_collision:
            success = False
            break
        if np.linalg.norm(sim.current_state.world.robot.pose[:2] - goal[:2]) <= GOAL_THRESHOLD:
            success = True
            break
        hold_time += sim.sim_dt

    if renderer is not None:
        renderer.close()
    
    trajectory = np.array(trajectory)

    if success:
        path_length = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))
        # print(f"World {world_idx}: success, path length: {path_length}")
        return {
            "success": True,
            "path_length": path_length
        }
    else:
        # print(f"World {world_idx}: failed")
        return {"success": False}


def eval_controller(worlds: list[int],
                   config_path: str,
                   controller_name: str,
                   lidar_dist: float,
                   appear_dist: float,
                   robot_radius: float,
                   render: bool,
                   n_workers: int):
    print(f"\n\n-------- Evaluating {controller_name} --------")
    task_fn = partial(run_experiment, config_path=config_path,
                   controller_name=controller_name,
                   lidar_dist=lidar_dist,
                   appear_dist=appear_dist,
                   robot_radius=robot_radius,
                   render=render)
    results = do_parallel(task_fn, worlds, n_workers, use_tqdm=True, mode="process")

    success_count = 0
    apl = 0.
    for result in results:
        if result["success"]:
            success_count += 1
            apl += result["path_length"]
    
    sr = success_count / len(worlds)
    if success_count != 0:
        apl = apl / success_count
    else:
        apl = None

    print(f"-------- Finished {controller_name} --------")
    print(f"SR: {sr}, APL: {apl}")


def main(config: str = "configs/controllers.yaml",
         controllers: list[str] | None = ["mppi__nf__unicycle_steer__final"],
         worlds: list[int] | None = [299, 293, 294, 295, 59, 296],
         vis: bool = True,
         n_workers: int = 0):
    if controllers is None:
        controllers = get_available_controllers(config)
    if worlds is None:
        worlds = list(range(300))

    for controller in controllers:
        eval_controller(worlds=worlds,
                        config_path=config,
                        controller_name=controller,
                        lidar_dist=LIDAR_DIST,
                        appear_dist=APPEAR_DIST,
                        robot_radius=ROBOT_RADIUS,
                        render=vis,
                        n_workers=n_workers)


if __name__ == '__main__':
    fire.Fire(main)
