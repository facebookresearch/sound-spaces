#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
import logging

import numpy as np
import habitat
import torch
from habitat import Config, Dataset
from habitat.utils.visualizations.utils import observations_to_image
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.av_wan.models.planner import Planner


@baseline_registry.register_env(name="MapNavEnv")
class MapNavEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_target_distance = None
        self._previous_action = None
        self._previous_observation = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)

        self.planner = Planner(model_dir=self._config.MODEL_DIR,
                               use_acoustic_map='ACOUSTIC_MAP' in config.TASK_CONFIG.TASK.SENSORS,
                               masking=self._config.MASKING,
                               task_config=config.TASK_CONFIG
                               )
        torch.set_num_threads(1)

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        self.planner.update_map_and_graph(observations)
        self.planner.add_maps_to_observation(observations)
        self._previous_observation = observations
        logging.debug(super().current_episode)

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, *args, **kwargs):
        intermediate_goal = kwargs["action"]
        self._previous_action = intermediate_goal
        goal = self.planner.get_map_coordinates(intermediate_goal)
        stop = int(self._config.TASK_CONFIG.TASK.ACTION_MAP.MAP_SIZE ** 2 // 2) == intermediate_goal
        observation = self._previous_observation
        cumulative_reward = 0
        done = False
        reaching_waypoint = False
        cant_reach_waypoint = False
        if len(self._config.VIDEO_OPTION) > 0:
            rgb_frames = list()
            audios = list()

        for step_count in range(self._config.PREDICTION_INTERVAL):
            if step_count != 0 and not self.planner.check_navigability(goal):
                cant_reach_waypoint = True
                break
            action = self.planner.plan(observation, goal, stop=stop)
            observation, reward, done, info = super().step({"action": action})
            if len(self._config.VIDEO_OPTION) > 0:
                if "rgb" not in observation:
                    observation["rgb"] = np.zeros((self.config.DISPLAY_RESOLUTION,
                                                   self.config.DISPLAY_RESOLUTION, 3))
                frame = observations_to_image(observation, info)
                rgb_frames.append(frame)
                audios.append(observation['audiogoal'])
            cumulative_reward += reward
            if done:
                self.planner.reset()
                observation = self.reset()
                break
            else:
                self.planner.update_map_and_graph(observation)
                # reaching intermediate goal
                x, y = self.planner.mapper.get_maps_and_agent_pose()[2:4]
                if (x - goal[0]) == (y - goal[1]) == 0:
                    reaching_waypoint = True
                    break

        if not done:
            self.planner.add_maps_to_observation(observation)
        self._previous_observation = observation
        info['reaching_waypoint'] = done or reaching_waypoint
        info['cant_reach_waypoint'] = cant_reach_waypoint
        if len(self._config.VIDEO_OPTION) > 0:
            assert len(rgb_frames) != 0
            info['rgb_frames'] = rgb_frames
            info['audios'] = audios

        return observation, cumulative_reward, done, info

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            # if current_target_distance < self._previous_target_distance:
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = [goal.position for goal in self._env.current_episode.goals]
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
                self._env.task.is_stop_called
                # and self._distance_target() < self._success_distance
                and self._env.sim.reaching_goal
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id

    def global_to_egocentric(self, pg):
        return self.planner.mapper.global_to_egocentric(*pg)

    def egocentric_to_global(self, pg):
        return self.planner.mapper.egocentric_to_global(*pg)
