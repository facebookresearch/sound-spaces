# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import logging
import argparse

import numpy as np
import pygame

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
from habitat.datasets import make_dataset
from habitat.utils.visualizations.utils import observations_to_image
import soundspaces
from ss_baselines.common.environments import AudioNavRLEnv
from ss_baselines.common.utils import images_to_video_with_audio
from ss_baselines.av_nav.config import get_config


def draw_top_down_map(info):
    top_down_map = info["top_down_map"]["map"]

    top_down_map = maps.colorize_topdown_map(top_down_map)
    map_agent_pos = info["top_down_map"]["agent_map_coord"]
    top_down_map = maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["top_down_map"]["agent_angle"],
        agent_radius_px=top_down_map.shape[0] // 25,
    )

    return top_down_map


def interactive_demo(config, env):
    # Set the width and height of the screen [width, height]
    pygame.init()
    size = (728, 256)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Interactive Demo")
    # Loop until the user clicks the close button.
    done = False
    # Used to manage how fast the screen updates
    clock = pygame.time.Clock()

    frames = list()
    audios = list()
    observation = env.reset()
    rgb_image = np.swapaxes(observation['rgb'], 0, 1)
    # screen.blit(pygame.surfarray.make_surface(rgb_image), (0, 0))
    pygame.display.flip()
    # -------- Main Program Loop -----------
    keys = []
    while not done:
        # --- Main event loop
        def wait():
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN:
                        action = None
                        if event.key == pygame.K_w:  # w
                            action = HabitatSimActions.MOVE_FORWARD
                            keys.append('w')
                        elif event.key == pygame.K_a:  # a
                            action = HabitatSimActions.TURN_LEFT
                            keys.append('a')
                        elif event.key == pygame.K_d:  # d
                            action = HabitatSimActions.TURN_RIGHT
                            keys.append('d')
                        elif event.key == pygame.K_f:  # f
                            action = HabitatSimActions.STOP
                            keys.append('f')
                        if action is not None:
                            return action

        action = wait()
        # --- Game logic should go here
        observation, reward, done, info = env.step(**{'action': action})
        if env.get_done(None):
            # observation = env.reset()
            break

        if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in observation:
            for obs in observation['intermediate']:
                frame = observations_to_image(obs, info)
                frames.append(frame)
        frame = observations_to_image(observation, info)
        frames.append(frame)
        frame = np.swapaxes(frame, 0, 1)
        audio = observation['audiogoal']
        audios.append(audio)

        # Here, we clear the screen to white. Don't put other drawing commands
        # above this, or they will be erased with this command.
        screen.fill((255, 255, 255))
        screen.blit(pygame.surfarray.make_surface(frame), (0, 0))
        # smaller_frame = block_reduce(frame, block_size=(down_sampling, down_sampling, 1), func=np.mean)
        # screen.blit(pygame.surfarray.make_surface(smaller_frame), (0, 0))

        # play sound
        # temp_file = 'data/temp/temp.wav'
        # sr = config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE
        # audio = np.int16(audio * 32767).T
        # wavfile.write(temp_file, sr, audio)
        # pygame.mixer.music.load(temp_file)
        # pygame.mixer.music.play(-1)

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # --- Limit to 60 frames per second
        clock.tick(1)

    # Close the window and quit.
    pygame.quit()
    env.close()
    print('Keys: {}'.format(','.join(keys)))

    # write frames and audio into videos
    video_dir = 'data/visualizations/demo'
    video_name = 'demo'
    fps = config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS \
        if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE else 1
    images_to_video_with_audio(frames, video_dir, video_name, audios,
                               sr=config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE, fps=fps)


def following(config, env, keys):
    observation = env.reset()
    frames = list()
    audios = list()
    for key in keys:
        if key == 'w':  # w
            action = HabitatSimActions.MOVE_FORWARD
        elif key == 'a':  # a
            action = HabitatSimActions.TURN_LEFT
        elif key == 'd':  # d
            action = HabitatSimActions.TURN_RIGHT
        elif key == 'f':  # f
            action = HabitatSimActions.STOP

        # --- Game logic should go here
        observation, reward, done, info = env.step(**{'action': action})
        if env.get_done(None):
            break

        if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE and 'intermediate' in observation:
            for obs in observation['intermediate']:
                frame = observations_to_image(obs, info)
                frames.append(frame)
        frame = observations_to_image(observation, info)
        frames.append(frame)
        audio = observation['audiogoal']
        audios.append(audio)

    env.close()

    # write frames and audio into videos
    video_dir = 'data/visualizations/demo'
    video_name = 'demo'
    fps = config.TASK_CONFIG.SIMULATOR.VIEW_CHANGE_FPS \
        if config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE else 1
    images_to_video_with_audio(frames, video_dir, video_name, audios,
                               sr=config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE, fps=fps)


def main():
    # import os
    # os.environ["SDL_VIDEODRIVER"] = "dummy"
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sound', default=False, action='store_true')
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default='eval',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=False,
        default='ss_baselines/av_nav/config/audionav/mp3d/interactive_demo.yaml',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action='store_true',
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--keys",
        default='',
        type=str,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    # file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    config = get_config(
        config_paths=args.exp_config,
        opts=args.opts,
        run_type=args.run_type)
    config.defrost()
    config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
    if args.keys == '':
        config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = \
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = 256
        config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False
    else:
        config.TASK_CONFIG.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = False
    config.freeze()
    print(config)
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    env = AudioNavRLEnv(config=config, dataset=dataset)

    if args.keys == '':
        interactive_demo(config, env)
    else:
        keys = args.keys.split(',')
        following(config, env, keys)


if __name__ == '__main__':
    main()
