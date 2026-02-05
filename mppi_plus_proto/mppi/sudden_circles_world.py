from turtle import circle
from typing import Optional, Union, List

import numpy as np
import pygame
from scipy.spatial.distance import cdist

from pyminisim.core import AbstractWorldMapState, AbstractStaticWorldMap
from pyminisim.visual import AbstractMapSkin, VisualizationParams
from pyminisim.visual.util import PoseConverter
from pyminisim.core import SimulationState


class SuddenCirclesWorldState(AbstractWorldMapState):

    def __init__(self, circles: np.ndarray, appeared_indices: List[int]):
        assert len(circles.shape) == 2 and circles.shape[1] == 3
        super(SuddenCirclesWorldState, self).__init__()
        self._circles = circles
        self._appeared_indices = appeared_indices

    @property
    def all_circles(self) -> np.ndarray:
        return self._circles

    @property
    def appeared_indices(self) -> List[int]:
        return self._appeared_indices.copy()

    @property
    def appeared_circles(self) -> Optional[np.ndarray]:
        if len(self._appeared_indices) == 0:
            return None
        return self._circles[self._appeared_indices]

    @property
    def unappeared_circles(self) -> Optional[np.ndarray]:
        all_indices = set(range(self._circles.shape[0]))
        appeared_indices = set(self._appeared_indices)
        unappeared_indicies = all_indices.difference(appeared_indices)
        unappeared_indicies = list(unappeared_indicies)
        if len(unappeared_indicies) == 0:
            return None
        return self._circles[unappeared_indicies]


class SuddenCirclesWorld(AbstractStaticWorldMap):

    def __init__(self, 
                 sudden_circles: np.ndarray,
                 appear_distance: float,
                 always_visible_circles: Optional[np.ndarray] = None):
        """
        :param circles: List of circles in format [[x, y, radius]] (in metres)
        """
        if always_visible_circles is not None:
            circles = np.concat((always_visible_circles, sudden_circles), axis=0)
            indices = list(range(always_visible_circles.shape[0]))
        else:
            circles = sudden_circles
            indices = []
        super(SuddenCirclesWorld, self).__init__(SuddenCirclesWorldState(circles.copy(), indices))
        self._appear_distance = appear_distance

    def update(self, robot_pose: np.ndarray, robot_radius: float):
        appeared_indicies = set(self._state.appeared_indices)
        all_circles = self._state.all_circles

        dists = np.linalg.norm(all_circles[:, :2] - robot_pose[:2], axis=-1)
        dists = dists - all_circles[:, 2] - robot_radius
        new_appeared_indicies = set(np.where(dists <= self._appear_distance)[0].tolist())
        
        appeared_indicies = appeared_indicies.union(new_appeared_indicies)

        self._state = SuddenCirclesWorldState(all_circles, list(appeared_indicies))

    def closest_distance_to_obstacle(self, point: np.ndarray) -> \
            Union[float, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)
        circles = self._state.appeared_circles
        if circles is None:
            if point.shape == (2,):
                return np.inf
            return np.repeat(np.inf, point.shape[0])

        if len(point.shape) == 1:
            return np.min(np.linalg.norm(point - circles[:, :2], axis=1) - circles[:, 2])

        pairwise_dist = cdist(point, circles[:, :2], metric="euclidean")
        pairwise_dist = pairwise_dist - circles[:, 2]

        return np.min(pairwise_dist, axis=1)

    def is_occupied(self, point: np.ndarray) -> Union[bool, np.ndarray]:
        assert point.shape == (2,) or (len(point.shape) == 2 and point.shape[1] == 2)
        circles = self._state.appeared_circles
        if circles is None:
            if len(point.shape) == 2:
                return np.array([False for _ in range(point.shape[0])])
            return False

        if len(point.shape) == 1:
            return (np.linalg.norm(point - circles[:, :2], axis=1) - circles[:, 2] > 0.).all()

        pairwise_dist = cdist(point, circles[:, :2], metric="euclidean")
        pairwise_dist = pairwise_dist - circles[:, 2]
        result = (pairwise_dist <= 0.).any(axis=1)
        return result


class SuddenCirclesWorldSkin(AbstractMapSkin):

    def __init__(self,
                 world_map: SuddenCirclesWorld,
                 vis_params: VisualizationParams):
        super(SuddenCirclesWorldSkin, self).__init__()
        self._pose_converter = PoseConverter(vis_params)
        self._vis_params = vis_params
        self._resolution = vis_params.resolution

    def render(self, screen, sim_state: SimulationState, global_offset: np.ndarray):
        pixel_offset_x = -int(self._resolution * global_offset[1])
        pixel_offset_y = int(self._resolution * global_offset[0])

        appeared_circles = sim_state.world.world_map.appeared_circles
        if appeared_circles is not None:
            appeared_centers = self._pose_converter.convert(appeared_circles[:, :2])
            appeared_radii = [int(radius * self._vis_params.resolution) for radius in appeared_circles[:, 2]]
            for center, radius in zip(appeared_centers, appeared_radii):
                pygame.draw.circle(screen,
                                (0, 255, 0),
                                (center[0] + pixel_offset_x, center[1] + pixel_offset_y),
                                radius,
                                0)

        unappeared_circles = sim_state.world.world_map.unappeared_circles
        if unappeared_circles is not None:
            unappeared_centers = self._pose_converter.convert(unappeared_circles[:, :2])
            unappeared_radii = [int(radius * self._vis_params.resolution) for radius in unappeared_circles[:, 2]]
            for center, radius in zip(unappeared_centers, unappeared_radii):
                pygame.draw.circle(screen,
                                (179, 177, 171),
                                (center[0] + pixel_offset_x, center[1] + pixel_offset_y),
                                radius,
                                0)
