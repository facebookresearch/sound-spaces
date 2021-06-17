# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Type, Union
import logging

import numpy as np
import torch
import cv2
import librosa
from gym import spaces
from skimage.measure import block_reduce

from habitat.config import Config
from habitat.core.dataset import Episode

from habitat.tasks.nav.nav import DistanceToGoal, Measure, EmbodiedTask, Success
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
)
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING
from soundspaces.utils import convert_semantic_object_to_rgb
from soundspaces.mp3d_utils import HouseReader


@registry.register_sensor
class AudioGoalSensor(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "audiogoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (2, self._sim.config.AUDIO.RIR_SAMPLING_RATE)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_current_audiogoal_observation()


@registry.register_sensor
class SpectrogramSensor(Sensor):
    cls_uuid: str = "spectrogram"
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spectrogram"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        spectrogram = self.compute_spectrogram(np.ones((2, self._sim.config.AUDIO.RIR_SAMPLING_RATE)))

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=spectrogram.shape,
            dtype=np.float32,
        )

    @staticmethod
    def compute_spectrogram(audio_data):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft

        channel1_magnitude = np.log1p(compute_stft(audio_data[0]))
        channel2_magnitude = np.log1p(compute_stft(audio_data[1]))
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)

        return spectrogram

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        spectrogram = self._sim.get_current_spectrogram_observation(self.compute_spectrogram)

        return spectrogram


@registry.register_measure
class NormalizedDistanceToGoal(Measure):
    r""" Distance to goal the episode ends
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._start_end_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "normalized_distance_to_goal"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_goal = task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        self._metric = distance_to_goal / self._start_end_episode_distance


@registry.register_sensor(name="Collision")
class Collision(Sensor):
    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "collision"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        return [self._sim.previous_step_collided]


@registry.register_measure
class SNA(Measure):
    r"""SPL (Success weighted by Path Length)

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._start_end_num_action = None
        self._agent_num_action = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "sna"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._start_end_num_action = episode.info["num_action"]
        self._agent_num_action = 0
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._agent_num_action += 1

        self._metric = ep_success * (
            self._start_end_num_action
            / max(
                self._start_end_num_action, self._agent_num_action
            )
        )


@registry.register_measure
class NA(Measure):
    r""" Number of actions

    ref: On Evaluation of Embodied Agents - Anderson et. al
    https://arxiv.org/pdf/1807.06757.pdf
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._agent_num_action = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "na"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._agent_num_action = 0
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        self._agent_num_action += 1
        self._metric = self._agent_num_action


@registry.register_sensor(name="EgoMap")
class EgoMap(Sensor):
    r"""Estimates the top-down occupancy based on current depth-map.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: contains the MAP_RESOLUTION, MAP_SIZE, HEIGHT_THRESH fields to
                decide grid-size, extents of the projection, and the thresholds
                for determining obstacles and explored space.
    """

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        super().__init__(config=config)

        # Map statistics
        self.map_size = self.config.MAP_SIZE
        self.map_res = self.config.MAP_RESOLUTION

        # Agent height for pointcloud transformation
        self.sensor_height = self.config.POSITION[1]

        # Compute intrinsic matrix
        hfov = float(self._sim.config.DEPTH_SENSOR.HFOV) * np.pi / 180
        self.intrinsic_matrix = np.array([[1 / np.tan(hfov / 2.), 0., 0., 0.],
                                          [0., 1 / np.tan(hfov / 2.), 0., 0.],
                                          [0., 0.,  1, 0],
                                          [0., 0., 0, 1]])
        self.inverse_intrinsic_matrix = np.linalg.inv(self.intrinsic_matrix)

        # Height thresholds for obstacles
        self.height_thresh = self.config.HEIGHT_THRESH

        # Depth processing
        self.min_depth = float(self._sim.config.DEPTH_SENSOR.MIN_DEPTH)
        self.max_depth = float(self._sim.config.DEPTH_SENSOR.MAX_DEPTH)

        # Pre-compute a grid of locations for depth projection
        W = self._sim.config.DEPTH_SENSOR.WIDTH
        H = self._sim.config.DEPTH_SENSOR.HEIGHT
        self.proj_xs, self.proj_ys = np.meshgrid(
                                          np.linspace(-1, 1, W),
                                          np.linspace(1, -1, H)
                                     )

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "ego_map"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self.config.MAP_SIZE, self.config.MAP_SIZE, 2)
        return spaces.Box(
            low=0,
            high=1,
            shape=sensor_shape,
            dtype=np.uint8,
        )

    def convert_to_pointcloud(self, depth):
        """
        Inputs:
            depth = (H, W, 1) numpy array
        Returns:
            xyz_camera = (N, 3) numpy array for (X, Y, Z) in egocentric world coordinates
        """

        depth_float = depth.astype(np.float32)[..., 0]

        # =========== Convert to camera coordinates ============
        W = depth.shape[1]
        xs = self.proj_xs.reshape(-1)
        ys = self.proj_ys.reshape(-1)
        depth_float = depth_float.reshape(-1)

        # Filter out invalid depths
        max_forward_range = self.map_size * self.map_res
        valid_depths = (depth_float != 0.0) & (depth_float <= max_forward_range)
        xs = xs[valid_depths]
        ys = ys[valid_depths]
        depth_float = depth_float[valid_depths]

        # Unproject
        # negate depth as the camera looks along -Z
        xys = np.vstack((xs * depth_float,
                         ys * depth_float,
                         -depth_float, np.ones(depth_float.shape)))
        inv_K = self.inverse_intrinsic_matrix
        xyz_camera = np.matmul(inv_K, xys).T # XYZ in the camera coordinate system
        xyz_camera = xyz_camera[:, :3] / xyz_camera[:, 3][:, np.newaxis]

        return xyz_camera

    def safe_assign(self, im_map, x_idx, y_idx, value):
        try:
            im_map[x_idx, y_idx] = value
        except IndexError:
            valid_idx1 = np.logical_and(x_idx >= 0, x_idx < im_map.shape[0])
            valid_idx2 = np.logical_and(y_idx >= 0, y_idx < im_map.shape[1])
            valid_idx = np.logical_and(valid_idx1, valid_idx2)
            im_map[x_idx[valid_idx], y_idx[valid_idx]] = value

    def _get_depth_projection(self, sim_depth):
        """
        Project pixels visible in depth-map to ground-plane
        """

        if self._sim.config.DEPTH_SENSOR.NORMALIZE_DEPTH:
            depth = sim_depth * (self.max_depth - self.min_depth) + self.min_depth
        else:
            depth = sim_depth

        XYZ_ego = self.convert_to_pointcloud(depth)

        # Adding agent's height to the point cloud
        XYZ_ego[:, 1] += self.sensor_height

        # Convert to grid coordinate system
        V = self.map_size
        Vby2 = V // 2
        points = XYZ_ego

        grid_x = (points[:, 0] / self.map_res) + Vby2
        grid_y = (points[:, 2] / self.map_res) + V

        # Filter out invalid points
        valid_idx = (grid_x >= 0) & (grid_x <= V-1) & (grid_y >= 0) & (grid_y <= V-1)
        points = points[valid_idx, :]
        grid_x = grid_x[valid_idx].astype(int)
        grid_y = grid_y[valid_idx].astype(int)

        # Create empty maps for the two channels
        obstacle_mat = np.zeros((self.map_size, self.map_size), np.uint8)
        explore_mat = np.zeros((self.map_size, self.map_size), np.uint8)

        # Compute obstacle locations
        high_filter_idx = points[:, 1] < self.height_thresh[1]
        low_filter_idx = points[:, 1] > self.height_thresh[0]
        obstacle_idx = np.logical_and(low_filter_idx, high_filter_idx)

        self.safe_assign(obstacle_mat, grid_y[obstacle_idx], grid_x[obstacle_idx], 1)

        # Compute explored locations
        explored_idx = high_filter_idx
        self.safe_assign(explore_mat, grid_y[explored_idx], grid_x[explored_idx], 1)

        # Smoothen the maps
        kernel = np.ones((3, 3), np.uint8)

        obstacle_mat = cv2.morphologyEx(obstacle_mat, cv2.MORPH_CLOSE, kernel)
        explore_mat = cv2.morphologyEx(explore_mat, cv2.MORPH_CLOSE, kernel)

        # Ensure all expanded regions in obstacle_mat are accounted for in explored_mat
        explore_mat = np.logical_or(explore_mat, obstacle_mat)

        return np.stack([obstacle_mat, explore_mat], axis=2)

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        # convert to numpy array
        ego_map_gt = self._sim.get_egomap_observation()
        if ego_map_gt is None:
            sim_depth = asnumpy(observations['depth'])
            ego_map_gt = self._get_depth_projection(sim_depth)
            self._sim.cache_egomap_observation(ego_map_gt)

        return ego_map_gt


def asnumpy(v):
    if torch.is_tensor(v):
        return v.cpu().numpy()
    elif isinstance(v, np.ndarray):
        return v
    else:
        raise ValueError('Invalid input')


@registry.register_sensor(name="Category")
class Category(Sensor):
    cls_uuid: str = "category"

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(len(CATEGORY_INDEX_MAPPING.keys()),),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        index = CATEGORY_INDEX_MAPPING[episode.object_category]
        onehot = np.zeros(len(CATEGORY_INDEX_MAPPING.keys()))
        onehot[index] = 1

        return onehot


@registry.register_sensor(name="CategoryBelief")
class CategoryBelief(Sensor):
    cls_uuid: str = "category_belief"

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(len(CATEGORY_INDEX_MAPPING.keys()),),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        belief = np.zeros(len(CATEGORY_INDEX_MAPPING.keys()))

        return belief


@registry.register_sensor(name="LocationBelief")
class LocationBelief(Sensor):
    cls_uuid: str = "location_belief"

    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(2,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        belief = np.zeros(2)
        return belief


@registry.register_sensor(name="MPCAT40Index")
class MPCAT40Index(Sensor):
    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        self.config = config
        self._category_mapping = {
                'chair': 3,
                'table': 5,
                'picture': 6,
                'cabinet': 7,
                'cushion': 8,
                'sofa': 10,
                'bed': 11,
                'chest_of_drawers': 13,
                'plant': 14,
                'sink': 15,
                'toilet': 18,
                'stool': 19,
                'towel': 20,
                'tv_monitor': 22,
                'shower': 23,
                'bathtub': 25,
                'counter': 26,
                'fireplace': 27,
                'gym_equipment': 33,
                'seating': 34,
                'clothes': 38
            }
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "mpcat40_index"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        index = self._category_mapping[episode.object_category]
        encoding = np.array([index])

        return encoding


@registry.register_sensor(name="SemanticObjectSensor")
class SemanticObjectSensor(Sensor):
    r"""Lists the object categories for each pixel location.

    Args:
        sim: reference to the simulator for calculating task observations.
    """
    cls_uuid: str = "semantic_object"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._current_episode_id = None
        self.mapping = None
        self._initialize_category_mappings()

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _initialize_category_mappings(self):
        self.category_to_task_category_id = {
            'chair': 0,
            'table': 1,
            'picture': 2,
            'cabinet': 3,
            'cushion': 4,
            'sofa': 5,
            'bed': 6,
            'chest_of_drawers': 7,
            'plant': 8,
            'sink': 9,
            'toilet': 10,
            'stool': 11,
            'towel': 12,
            'tv_monitor': 13,
            'shower': 14,
            'bathtub': 15,
            'counter': 16,
            'fireplace': 17,
            'gym_equipment': 18,
            'seating': 19,
            'clothes': 20
        }
        self.category_to_mp3d_category_id = {
            'chair': 3,
            'table': 5,
            'picture': 6,
            'cabinet': 7,
            'cushion': 8,
            'sofa': 10,
            'bed': 11,
            'chest_of_drawers': 13,
            'plant': 14,
            'sink': 15,
            'toilet': 18,
            'stool': 19,
            'towel': 20,
            'tv_monitor': 22,
            'shower': 23,
            'bathtub': 25,
            'counter': 26,
            'fireplace': 27,
            'gym_equipment': 33,
            'seating': 34,
            'clothes': 38
        }
        self.num_task_categories = np.max(
            list(self.category_to_task_category_id.values())
        ) + 1
        self.mp3d_id_to_task_id = np.ones((200, ), dtype=np.int64) * -1
        for k in self.category_to_task_category_id.keys():
            v1 = self.category_to_task_category_id[k]
            v2 = self.category_to_mp3d_category_id[k]
            self.mp3d_id_to_task_id[v2] = v1
        # Map unknown classes to a new category
        self.mp3d_id_to_task_id[
            self.mp3d_id_to_task_id == -1
        ] = self.num_task_categories

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        if self.config.CONVERT_TO_RGB:
            observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.config.HEIGHT, self.config.WIDTH, 3),
                dtype=np.uint8,
            )
        else:
            observation_space = spaces.Box(
                low=np.iinfo(np.uint32).min,
                high=np.iinfo(np.uint32).max,
                shape=(self.config.HEIGHT, self.config.WIDTH),
                dtype=np.uint32,
            )
        return observation_space

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if self._current_episode_id != episode_uniq_id:
            self._current_episode_id = episode_uniq_id
            reader = HouseReader(self._sim._current_scene.replace('.glb', '.house'))
            instance_id_to_mp3d_id = reader.compute_object_to_category_index_mapping()
            self.instance_id_to_mp3d_id = np.array([instance_id_to_mp3d_id[i] for i in range(len(instance_id_to_mp3d_id))])

        # Pre-process semantic observations to remove invalid values
        semantic = np.copy(observations["semantic"])
        semantic[semantic >= self.instance_id_to_mp3d_id.shape[0]] = 0
        # Map from instance id to semantic id
        semantic_object = np.take(self.instance_id_to_mp3d_id, semantic)
        # Map from semantic id to task id
        semantic_object = np.take(self.mp3d_id_to_task_id, semantic_object)
        if self.config.CONVERT_TO_RGB:
            semantic_object = SemanticObjectSensor.convert_semantic_map_to_rgb(
                semantic_object
            )

        return semantic_object

    @staticmethod
    def convert_semantic_map_to_rgb(semantic_map):
        return convert_semantic_object_to_rgb(semantic_map)


@registry.register_sensor(name="PoseSensor")
class PoseSensor(Sensor):
    r"""The agents current location and heading in the coordinate frame defined by the
    episode, i.e. the axis it faces along and the origin is defined by its state at
    t=0. Additionally contains the time-step of the episode.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "pose"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._episode_time = 0
        self._current_episode_id = None
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(4,),
            dtype=np.float32,
        )

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id != self._current_episode_id:
            self._episode_time = 0.0
            self._current_episode_id = episode_uniq_id

        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position_xyz = agent_state.position
        rotation_world_agent = agent_state.rotation

        agent_position_xyz = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position_xyz - origin
        )

        agent_heading = self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )

        ep_time = self._episode_time
        self._episode_time += 1.0

        return np.array(
            [-agent_position_xyz[2], agent_position_xyz[0], agent_heading, ep_time],
            dtype=np.float32
        )


@registry.register_sensor
class ProximitySensor(Sensor):
    r"""Sensor for observing the distance to the closest obstacle

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """
    cls_uuid: str = "proximity"

    def __init__(self, sim, config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._max_detection_radius = getattr(
            config, "MAX_DETECTION_RADIUS", 2.0
        )
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=self._max_detection_radius,
            shape=(1,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position

        return np.array(
            [
                self._sim.distance_to_closest_obstacle(
                    current_position, self._max_detection_radius
                )
            ],
            dtype=np.float32,
        )


@registry.register_sensor
class OracleActionSensor(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.PATH

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode: Episode, **kwargs: Any):
        return self._sim.get_oracle_action()
