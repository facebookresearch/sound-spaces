# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import os
import argparse
import logging
import pickle
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.utils.common import quat_to_angle_axis, quat_to_coeffs, quat_from_angle_axis, quat_from_coeffs
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, ShortestPathPoint
from soundspaces.task import merge_sim_episode_config
from soundspaces.utils import load_metadata
from baselines.config.default import get_config


class SoundSpaces(HabitatSim):
    def __init__(self, config):
        super().__init__(config)
        self.points = None
        self.graph = None

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> \
            List[ShortestPathPoint]:
        pass

    def reconfigure(self, config) -> None:
        dataset = config.SCENE.split('/')[2]
        scene_name = config.SCENE.split('/')[3]

        is_same_scene = config.SCENE == self._current_scene
        self.config = config
        self.sim_config = self.create_sim_config(self._sensor_suite)
        if not is_same_scene:
            self._current_scene = config.SCENE
            self._sim.close()
            del self._sim
            # HabitatSim is a wrapper class of habitat_sim, which is the backend renderer
            self._sim = habitat_sim.Simulator(self.sim_config)
            logging.info('Loaded scene {}'.format(scene_name))

        if not is_same_scene or self.points is None or self.graph is None:
            # can happen during initialization or reconfiguration
            metadata_dir = os.path.join('data/metadata', dataset, scene_name)
            self.points, self.graph = load_metadata(metadata_dir)

        # after env calls reconfigure to update the config with current episode,
        # simulation needs to update the agent position, rotation in accordance with the new config file
        self._update_agents_state()

        # set agent positions
        self._receiver_position_index = self._position_to_index(self.config.AGENT_0.START_POSITION)
        self._source_position_index = self._position_to_index(self.config.AGENT_0.GOAL_POSITION)
        self._rotation = int(np.around(np.rad2deg(quat_to_angle_axis(quat_from_coeffs(
                             self.config.AGENT_0.START_ROTATION))[0]))) % 360
        self.set_agent_state(list(self.graph.nodes[self._receiver_position_index]['point']),
                             self.config.AGENT_0.START_ROTATION)

        logging.debug("Initial source, agent at: {}, {}, orientation: {}".
                      format(self._source_position_index, self._receiver_position_index,
                             self._rotation))

    def _position_to_index(self, position):
        for node in self.graph:
            if np.allclose(self.graph.nodes()[node]['point'], position):
                return node
        assert True, "Position misalignment."

    def step(self, action):
        sim_obs = self._sim.get_sensor_observations()
        return sim_obs, self._rotation


def main(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default='baselines/config/{}/train_telephone/pointgoal_rgb.yaml'.format(dataset)
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    config = get_config(args.config_path, opts=args.opts)
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.freeze()
    simulator = None
    scene_obs = defaultdict(dict)
    num_obs = 0
    scene_obs_dir = 'data/scene_observations/' + dataset
    os.makedirs(scene_obs_dir, exist_ok=True)
    metadata_dir = 'data/metadata/' + dataset
    for scene in os.listdir(metadata_dir):
        scene_obs = dict()
        scene_metadata_dir = os.path.join(metadata_dir, scene)
        points, graph = load_metadata(scene_metadata_dir)
        if dataset == 'replica':
            scene_mesh_dir = os.path.join('data/scene_datasets', dataset, scene, 'habitat/mesh_semantic.ply')
        else:
            scene_mesh_dir = os.path.join('data/scene_datasets', dataset, scene, scene + '.glb')

        for node in graph.nodes():
            agent_position = graph.nodes()[node]['point']
            for angle in [0, 90, 180, 270]:
                agent_rotation = quat_to_coeffs(quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))).tolist()
                goal_radius = 0.00001
                goal = NavigationGoal(
                    position=agent_position,
                    radius=goal_radius
                )
                episode = NavigationEpisode(
                    goals=[goal],
                    episode_id=str(0),
                    scene_id=scene_mesh_dir,
                    start_position=agent_position,
                    start_rotation=agent_rotation,
                    info={'sound': 'telephone'}
                )

                episode_sim_config = merge_sim_episode_config(config.TASK_CONFIG.SIMULATOR, episode)
                if simulator is None:
                    simulator = SoundSpaces(episode_sim_config)
                simulator.reconfigure(episode_sim_config)

                obs, rotation_index = simulator.step(None)
                scene_obs[(node, rotation_index)] = obs
                num_obs += 1

        print('Total number of observations: {}'.format(num_obs))
        with open(os.path.join(scene_obs_dir, '{}.pkl'.format(scene)), 'wb') as fo:
            pickle.dump(scene_obs, fo)
    simulator.close()
    del simulator


if __name__ == '__main__':
    print('Caching Replica observations ...')
    main('replica')
    print('Caching Matterport3D observations ...')
    main('mp3d')
