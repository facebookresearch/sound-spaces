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
from habitat.core.env import Env
from habitat.utils.visualizations.utils import observations_to_image
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.av_wan.models.planner import Planner


@baseline_registry.register_env(name="MapNavEnv")
class MapNavEnv(Env):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._config = config

        self._previous_target_distance = None
        self._previous_action = None
        self._previous_observation = None
        self._episode_distance_covered = None
        super().__init__(config, dataset)

        self.planner = Planner(model_dir='',
                               use_acoustic_map='ACOUSTIC_MAP' in config.TASK.SENSORS,
                               masking=True,
                               task_config=config
                               )
        torch.set_num_threads(1)

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        self.planner.update_map_and_graph(observations)
        self.planner.add_maps_to_observation(observations)
        self._previous_observation = observations
        logging.debug(super().current_episode)

        self._previous_target_distance = 0
        return observations

    def step(self, action):
        intermediate_goal = action
        self._previous_action = intermediate_goal
        goal = self.planner.get_map_coordinates(intermediate_goal)
        stop = int(self._config.TASK.ACTION_MAP.MAP_SIZE ** 2 // 2) == intermediate_goal
        observation = self._previous_observation
        cumulative_reward = 0
        done = False
        reaching_waypoint = False
        cant_reach_waypoint = False

        for step_count in range(10):
            if step_count != 0 and not self.planner.check_navigability(goal):
                cant_reach_waypoint = True
                break
            action = self.planner.plan(observation, goal, stop=stop)
            observation = super().step(action)
            done = self._episode_over
            if done:
                self.planner.reset()
                # observation = self.reset()
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

        return observation

    def global_to_egocentric(self, pg):
        return self.planner.mapper.global_to_egocentric(*pg)

    def egocentric_to_global(self, pg):
        return self.planner.mapper.egocentric_to_global(*pg)