# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pickle

import magnum as mn
import numpy as np

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import SensorSuite
from habitat_sim.utils.common import quat_from_angle_axis
from soundspaces.utils import load_metadata
from ss_baselines.av_nav.config import get_config


def create_sim(scene_id, sensor_suite):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()

    sensor_specifications = []
    for sensor in sensor_suite.sensors.values():
        sim_sensor_cfg = sensor._get_default_spec()
        sim_sensor_cfg.uuid = sensor.uuid
        sim_sensor_cfg.resolution = list(
            sensor.observation_space.shape[:2]
        )
        sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
        sensor_specifications.append(sim_sensor_cfg)

    agent_cfg.sensor_specifications = sensor_specifications

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def main(dataset):
    """
    This functions computes and saves the visual observations for the pre-defined grid points in SoundSpaces 1.0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default='ss_baselines/av_nav/config/audionav/{}/train_telephone/pointgoal_rgb.yaml'.format(dataset)
    )
    args = parser.parse_args()

    config = get_config(args.config_path)

    sim_sensors = []
    for sensor_name in ["RGB_SENSOR", "DEPTH_SENSOR"]:
        sensor_cfg = getattr(config.TASK_CONFIG.SIMULATOR, sensor_name)
        sensor_type = registry.get_sensor(sensor_cfg.TYPE)
        sim_sensors.append(sensor_type(sensor_cfg))
    sensor_suite = SensorSuite(sim_sensors)

    num_obs = 0
    scene_obs_dir = 'data/scene_observations/' + dataset
    os.makedirs(scene_obs_dir, exist_ok=True)
    metadata_dir = 'data/metadata/' + dataset
    for scene in os.listdir(metadata_dir):
        scene_obs = dict()
        scene_metadata_dir = os.path.join(metadata_dir, scene)
        points, graph = load_metadata(scene_metadata_dir)
        if dataset == 'replica':
            scene_id = os.path.join('data/scene_datasets', dataset, scene, 'habitat/mesh_semantic.ply')
        else:
            scene_id = os.path.join('data/scene_datasets', dataset, scene, scene + '.glb')

        sim_config = create_sim(scene_id, sensor_suite)
        sim = habitat_sim.Simulator(sim_config)

        for node in graph.nodes():
            agent_position = graph.nodes()[node]['point']
            for angle in [0, 90, 180, 270]:
                agent = sim.get_agent(0)
                new_state = sim.get_agent(0).get_state()
                new_state.position = agent_position
                new_state.rotation = quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))
                new_state.sensor_states = {}
                agent.set_state(new_state, True)

                sim_obs = sim.get_sensor_observations()
                obs = sensor_suite.get_observations(sim_obs)
                scene_obs[(node, angle)] = obs
                num_obs += 1

        print('Total number of observations: {}'.format(num_obs))
        with open(os.path.join(scene_obs_dir, '{}.pkl'.format(scene)), 'wb') as fo:
            pickle.dump(scene_obs, fo)
        sim.close()
        del sim


if __name__ == '__main__':
    print('Caching Replica observations ...')
    main('replica')
    print('Caching Matterport3D observations ...')
    main('mp3d')
