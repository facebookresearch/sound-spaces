# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import json
import shutil
import glob

import magnum as mn
import numpy as np

from habitat_sim.utils.common import quat_from_angle_axis
import habitat_sim
from scipy.io import wavfile

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 600


def make_configuration(scene_id, resolution=(512, 256), fov=20, visual_sensors=True):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    backend_cfg.scene_dataset_config_file = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    if visual_sensors:
        # agent configuration
        rgb_sensor_cfg = habitat_sim.CameraSensorSpec()
        rgb_sensor_cfg.resolution = resolution
        rgb_sensor_cfg.far = np.iinfo(np.int32).max
        rgb_sensor_cfg.hfov = mn.Deg(fov)
        rgb_sensor_cfg.position = np.array([0, 1.5, 0])

        depth_sensor_cfg = habitat_sim.CameraSensorSpec()
        depth_sensor_cfg.uuid = 'depth_camera'
        depth_sensor_cfg.resolution = resolution
        depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_cfg.hfov = mn.Deg(fov)
        depth_sensor_cfg.position = np.array([0, 1.5, 0])

        # semantic_sensor_cfg = habitat_sim.CameraSensorSpec()
        # semantic_sensor_cfg.uuid = "semantic_camera"
        # semantic_sensor_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
        # semantic_sensor_cfg.resolution = resolution
        # semantic_sensor_cfg.hfov = mn.Deg(fov)
        # semantic_sensor_cfg.position = np.array([0, 1.5, 0])

        agent_cfg.sensor_specifications = [rgb_sensor_cfg, depth_sensor_cfg]
    else:
        agent_cfg.sensor_specifications = []

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def add_acoustic_config(sim, args):
    # create the acoustic configs
    acoustics_config = habitat_sim.sensor.RLRAudioPropagationConfiguration()
    acoustics_config.enableMaterials = (args.dataset in ['mp3d', 'gibson'])
    acoustics_config.sampleRate = 44100

    # create channel layout
    channel_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayout()
    channel_layout.channelType = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
    )
    channel_layout.channelCount = 1

    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.acousticsConfig = acoustics_config
    audio_sensor_spec.channelLayout = channel_layout

    # add the audio sensor
    sim.add_sensor(audio_sensor_spec)
    if args.dataset in ['mp3d', 'gibson']:
        audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
        audio_sensor.setAudioMaterialsJSON('data/mp3d_material_config.json')


def get_res_angles_for(fov):
    if fov == 20:
        resolution = (384, 64)
        angles = [170, 150, 130, 110, 90, 70, 50, 30, 10, 350, 330, 310, 290, 270, 250, 230, 210, 190]
    elif fov == 30:
        resolution = (384, 128)
        angles = [0, 330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30]
    elif fov == 60:
        resolution = (256, 128)
        angles = [0, 300, 240, 180, 120, 60]
    elif fov == 90:
        resolution = (256, 256)
        angles = [0, 270, 180, 90]
    else:
        raise ValueError

    return resolution, angles


def visual_render(sim, receiver, angles):
    rgb_panorama = []
    depth_panorama = []
    for angle in angles:
        agent = sim.get_agent(0)
        new_state = sim.get_agent(0).get_state()
        new_state.position = receiver
        new_state.rotation = quat_from_angle_axis(math.radians(angle), np.array([0, 1, 0]))
        new_state.sensor_states = {}
        agent.set_state(new_state, True)

        observation = sim.get_sensor_observations()
        rgb_panorama.append(observation["rgba_camera"][..., :3])
        depth_panorama.append(normalize_depth(observation['depth_camera']))

    return rgb_panorama, depth_panorama


def acoustic_render(sim, receiver, source):
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(source)

    agent = sim.get_agent(0)
    new_state = sim.get_agent(0).get_state()
    new_state.position = receiver
    new_state.rotation = quat_from_angle_axis(0, np.array([0, 1, 0]))
    new_state.sensor_states = {}
    agent.set_state(new_state, True)

    observation = sim.get_sensor_observations()

    return np.array(observation['audio_sensor'])


def normalize_depth(depth):
    min_depth = 0
    max_depth = 10
    depth = np.clip(depth, min_depth, max_depth)
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)
    return normalized_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mp3d')
    parser.add_argument('--partition', type=str, default='learnlab,learnfair')
    parser.add_argument('--output-dir', type=str, default='data/PanoIR')
    parser.add_argument('--num-per-scene', type=int, default=1000)
    parser.add_argument('--reset', default=False, action='store_true')
    parser.add_argument('--slurm', default=False, action='store_true')
    args = parser.parse_args()

    if args.reset:
        shutil.rmtree(args.output_dir)

    if args.dataset == 'mp3d':
        from soundspaces.mp3d_utils import SCENE_SPLITS
        scenes = SCENE_SPLITS['train'] + SCENE_SPLITS['val'] + SCENE_SPLITS['test']
        scenes.remove('2n8kARJN3HM')
        scene_ids = [f"data/scene_datasets/mp3d/{scene}/{scene}.glb" for scene in scenes]
    elif args.dataset == 'gibson':
        scene_ids = glob.glob(f"data/scene_datasets/gibson/*.glb")
    elif args.dataset == 'hm3d':
        scene_ids = glob.glob("data/scene_datasets/hm3d/**/*.basis.glb", recursive=True)
    else:
        raise ValueError
    print(f'{args.dataset} has {len(scene_ids)} environments')

    if args.slurm:
        import submitit
        executor = submitit.AutoExecutor(folder="data/logs/submitit/%j")
        executor.update_parameters(timeout_min=60 * 3, slurm_partition=args.partition, cpus_per_task=10,
                                   gpus_per_node=1)

        with executor.batch():
            for scene_id in scene_ids[:1]:
                executor.submit(run, args, scene_id)
    else:
        for scene_id in scene_ids:
            run(args, scene_id)


def run(args, scene_id):
    scene = scene_id.split('/')[-1].split('.')[0]
    scene_obs_dir = f'{args.output_dir}/{args.dataset}/{scene}'
    os.makedirs(scene_obs_dir, exist_ok=True)

    receivers = []
    sources = []
    metadata = {}
    rgb_obs = []
    depth_obs = []
    ir_obs = []

    angles = get_res_angles_for(fov=20)[1]
    cfg = make_configuration(scene_id, resolution=get_res_angles_for(fov=20)[0])
    sim = habitat_sim.Simulator(cfg)
    for i in range(args.num_per_scene):
        receiver = sim.pathfinder.get_random_navigable_point()
        while True:
            source = sim.pathfinder.get_random_navigable_point()
            if np.sqrt((source[0] - receiver[0]) ** 2 + (source[2] - receiver[2]) ** 2) < 5 and\
                    abs(source[1] - receiver[1]) < 2:
                break
        receivers.append(receiver)
        sources.append(source)
        distance = np.sqrt((source[0] - receiver[0]) ** 2 + (source[2] - receiver[2]) ** 2)
        direction = (270 - np.rad2deg(np.arctan2(source[2] - receiver[2], source[0] - receiver[0]))) % 360
        metadata[i] = (direction, distance)

    for receiver in receivers:
        rgb, depth = visual_render(sim, receiver, angles)
        rgb_obs.append(np.concatenate(rgb, axis=1))
        depth_obs.append(np.concatenate(depth, axis=1))
    sim.close()

    cfg = make_configuration(scene_id, resolution=get_res_angles_for(fov=20)[0], visual_sensors=False)
    sim = habitat_sim.Simulator(cfg)
    add_acoustic_config(sim, args)
    for receiver, source in zip(receivers, sources):
        ir = acoustic_render(sim, receiver, source)
        ir_obs.append(ir)
    sim.close()

    for i, (rgb, depth, ir) in enumerate(zip(rgb_obs, depth_obs, ir_obs)):
        plt.imsave(os.path.join(scene_obs_dir, f'{i}-rgb.png'), rgb)
        plt.imsave(os.path.join(scene_obs_dir, f'{i}-depth.png'), depth)
        wavfile.write(os.path.join(scene_obs_dir, f'{i}-ir.wav'), 44100, ir[0])

    with open(os.path.join(scene_obs_dir, 'metadata.json'), 'w') as fo:
        json.dump(metadata, fo)

    sim.close()


if __name__ == '__main__':
    main()
