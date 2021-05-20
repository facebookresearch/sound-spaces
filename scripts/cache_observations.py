# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional
from abc import ABC
import os
import argparse
import logging
import pickle
from collections import defaultdict

import numpy as np

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Simulator
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from habitat_sim.utils.common import quat_to_angle_axis, quat_to_coeffs, quat_from_angle_axis, quat_from_coeffs
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, ShortestPathPoint
from soundspaces.tasks.audionav_task import merge_sim_episode_config
from soundspaces.utils import load_metadata
from soundspaces.simulator import SoundSpacesSim
from ss_baselines.av_nav.config import get_config


class Sim(SoundSpacesSim):
    def step(self, action):
        sim_obs = self._sim.get_sensor_observations()
        return sim_obs, self._rotation_angle


def main(dataset):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default='ss_baselines/av_nav/config/audionav/{}/train_telephone/pointgoal_rgb.yaml'.format(dataset)
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
    config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
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
                    simulator = Sim(episode_sim_config)
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
