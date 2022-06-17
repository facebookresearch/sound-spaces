#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union
import os
import logging
import shutil


from habitat import get_config as get_task_config
from habitat.config import Config as CN
from habitat.config.default import SIMULATOR_SENSOR
import habitat

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","
# -----------------------------------------------------------------------------
# EXPERIMENT CONFIG
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 0
_C.BASE_TASK_CONFIG_PATH = "configs/tasks/pointnav.yaml"
_C.TASK_CONFIG = CN()  # task_config will be stored as a config node
_C.CMD_TRAILING_OPTS = []  # store command line options as list of strings
_C.TRAINER_NAME = "AVWanTrainer"
_C.ENV_NAME = "MapNavEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.MODEL_DIR = 'data/models/output'
_C.VIDEO_OPTION = ["disk", "tensorboard"]
_C.VISUALIZATION_OPTION = ["top_down_map"]
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"  # path to ckpt or path to ckpts dir
_C.NUM_PROCESSES = 16
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 10000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.USE_VECENV = True
_C.USE_SYNC_VECENV = False
_C.ENCODE_RGB = True
_C.ENCODE_DEPTH = True
_C.DEBUG = False
_C.USE_LAST_CKPT = False
_C.PREDICTION_INTERVAL = 10
_C.DATASET_FILTER = []
_C.VISUALIZE_FAILURE_ONLY = False
_C.MASKING = True
_C.DISPLAY_RESOLUTION = 128
_C.CONTINUOUS = False
# -----------------------------------------------------------------------------
# EVAL CONFIG
# -----------------------------------------------------------------------------
_C.EVAL = CN()
# The split to evaluate on
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL) ENVIRONMENT CONFIG
# -----------------------------------------------------------------------------
_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.WITH_TIME_PENALTY = True
_C.RL.WITH_DISTANCE_REWARD = True
_C.RL.DISTANCE_REWARD_SCALE = 1.0
_C.RL.WITH_PREDICTION_REWARD = False
_C.RL.GOAL_PREDICTION_SCALE = 1.0
_C.RL.TIME_DIFF = False
# -----------------------------------------------------------------------------
# PROXIMAL POLICY OPTIMIZATION (PPO)
# -----------------------------------------------------------------------------
_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 16
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 7e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 5
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = False
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.use_exponential_lr_decay = False
_C.RL.PPO.exp_decay_lambda = 1.0
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
# -----------------------------------------------------------------------------
# TASK CONFIG
# -----------------------------------------------------------------------------
_TC = habitat.get_config()
_TC.defrost()
_TC.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = int(1e4)
# -----------------------------------------------------------------------------
# AUDIOGOAL_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.AUDIOGOAL_SENSOR = CN()
_TC.TASK.AUDIOGOAL_SENSOR.TYPE = "AudioGoalSensor"
# -----------------------------------------------------------------------------
# SPECTROGRAM_SENSOR
# -----------------------------------------------------------------------------
_TC.TASK.SPECTROGRAM_SENSOR = CN()
_TC.TASK.SPECTROGRAM_SENSOR.TYPE = "SpectrogramSensor"
# -----------------------------------------------------------------------------
# habitat_audio
# -----------------------------------------------------------------------------
_TC.SIMULATOR.GRID_SIZE = 0.5
_TC.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False
_TC.SIMULATOR.VIEW_CHANGE_FPS = 10
_TC.SIMULATOR.SCENE_DATASET = 'replica'
_TC.SIMULATOR.USE_RENDERED_OBSERVATIONS = True
_TC.SIMULATOR.SCENE_OBSERVATION_DIR = 'data/scene_observations'
_TC.SIMULATOR.AUDIO = CN()
_TC.SIMULATOR.AUDIO.SCENE = ""
_TC.SIMULATOR.AUDIO.BINAURAL_RIR_DIR = "data/binaural_rirs"
_TC.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = 44100
_TC.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = "data/sounds/1s_all"
_TC.SIMULATOR.AUDIO.METADATA_DIR = "data/metadata"
_TC.SIMULATOR.AUDIO.POINTS_FILE = 'points.txt'
_TC.SIMULATOR.AUDIO.GRAPH_FILE = 'graph.pkl'
_TC.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND = False
_TC.SIMULATOR.AUDIO.EVERLASTING = True
# -----------------------------------------------------------------------------
# DistanceToGoal Measure
# -----------------------------------------------------------------------------
_TC.TASK.DISTANCE_TO_GOAL = CN()
_TC.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
_TC.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# NormalizedDistanceToGoal Measure
# -----------------------------------------------------------------------------
_TC.TASK.NORMALIZED_DISTANCE_TO_GOAL = CN()
_TC.TASK.NORMALIZED_DISTANCE_TO_GOAL.TYPE = "NormalizedDistanceToGoal"
# -----------------------------------------------------------------------------
# Dataset extension
# -----------------------------------------------------------------------------
_TC.DATASET.VERSION = 'v1'
# -----------------------------------------------------------------------------
# Egocentric occupancy map projected from depth image
# -----------------------------------------------------------------------------
_TC.TASK.EGOMAP_SENSOR = SIMULATOR_SENSOR.clone()
_TC.TASK.EGOMAP_SENSOR.TYPE = "EgoMap"
_TC.TASK.EGOMAP_SENSOR.MAP_SIZE = 31
_TC.TASK.EGOMAP_SENSOR.MAP_RESOLUTION = 0.1
_TC.TASK.EGOMAP_SENSOR.HEIGHT_THRESH = (0.5, 2.0)
# -----------------------------------------------------------------------------
# Global map placeholder
# -----------------------------------------------------------------------------
_TC.TASK.GEOMETRIC_MAP = SIMULATOR_SENSOR.clone()
_TC.TASK.GEOMETRIC_MAP.TYPE = "GeometricMap"
_TC.TASK.GEOMETRIC_MAP.MAP_SIZE = 200
_TC.TASK.GEOMETRIC_MAP.INTERNAL_MAP_SIZE = 500
_TC.TASK.GEOMETRIC_MAP.MAP_RESOLUTION = 0.1
_TC.TASK.GEOMETRIC_MAP.NUM_CHANNEL = 2
# -----------------------------------------------------------------------------
# Acoustic map placeholder
# -----------------------------------------------------------------------------
_TC.TASK.ACOUSTIC_MAP = SIMULATOR_SENSOR.clone()
_TC.TASK.ACOUSTIC_MAP.TYPE = "AcousticMap"
_TC.TASK.ACOUSTIC_MAP.MAP_SIZE = 20
_TC.TASK.ACOUSTIC_MAP.MAP_RESOLUTION = 0.5
_TC.TASK.ACOUSTIC_MAP.NUM_CHANNEL = 1
_TC.TASK.ACOUSTIC_MAP.ENCODING = "average_intensity"
# -----------------------------------------------------------------------------
# Local occupancy map placeholder
# -----------------------------------------------------------------------------
_TC.TASK.ACTION_MAP = SIMULATOR_SENSOR.clone()
_TC.TASK.ACTION_MAP.TYPE = "ActionMap"
_TC.TASK.ACTION_MAP.MAP_SIZE = 9
_TC.TASK.ACTION_MAP.MAP_RESOLUTION = 0.5
_TC.TASK.ACTION_MAP.NUM_CHANNEL = 1
# -----------------------------------------------------------------------------
# Collision Sensor in habitat-audio
# -----------------------------------------------------------------------------
_TC.TASK.COLLISION = SIMULATOR_SENSOR.clone()
_TC.TASK.COLLISION.TYPE = "Collision"
# -----------------------------------------------------------------------------
# Intensity value placeholder
# -----------------------------------------------------------------------------
_TC.TASK.INTENSITY = SIMULATOR_SENSOR.clone()
_TC.TASK.INTENSITY.TYPE = "Intensity"
# -----------------------------------------------------------------------------
# Number of action metric
# -----------------------------------------------------------------------------
_TC.TASK.NUM_ACTION = CN()
_TC.TASK.NUM_ACTION.TYPE = "NA"
# -----------------------------------------------------------------------------
# Success normalized by number of action metric
# -----------------------------------------------------------------------------
_TC.TASK.SUCCESS_WEIGHTED_BY_NUM_ACTION = CN()
_TC.TASK.SUCCESS_WEIGHTED_BY_NUM_ACTION.TYPE = "SNA"


def merge_from_path(config, config_paths):
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)
    return config


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    model_dir: Optional[str] = None,
    run_type: Optional[str] = None,
    overwrite: bool = False
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    `config_paths` and overwritten by options from `opts`.
    Args:
        config_paths: List of config paths or string that contains comma
        separated list of config paths.
        opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example, `opts = ['FOO.BAR',
        0.5]`. Argument can be used for parameter sweeping or quick tests.
        model_dir: suffix for output dirs
        run_type: either train or eval
        overwrite: overwrite model directory
    """
    config = merge_from_path(_C.clone(), config_paths)
    config.TASK_CONFIG = get_task_config(config_paths=config.BASE_TASK_CONFIG_PATH)

    # config_name = os.path.basename(config_paths).split('.')[0]
    if model_dir is not None:
        config.MODEL_DIR = model_dir
    config.TENSORBOARD_DIR = os.path.join(config.MODEL_DIR, 'tb')
    config.CHECKPOINT_FOLDER = os.path.join(config.MODEL_DIR, 'data')
    config.VIDEO_DIR = os.path.join(config.MODEL_DIR, 'video_dir')
    config.LOG_FILE = os.path.join(config.MODEL_DIR, 'train.log')
    config.EVAL_CKPT_PATH_DIR = os.path.join(config.MODEL_DIR, 'data')

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    dirs = [config.VIDEO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
    if run_type == 'train':
        # check dirs
        if any([os.path.exists(d) for d in dirs]):
            for d in dirs:
                if os.path.exists(d):
                    logging.warning('{} exists'.format(d))
                    # if overwrite or input('Output directory already exists! Overwrite the folder? (y/n)') == 'y':
            if overwrite:
                for d in dirs:
                    if os.path.exists(d):
                        shutil.rmtree(d)

    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.SIMULATOR.USE_SYNC_VECENV = config.USE_SYNC_VECENV
    config.TASK_CONFIG.freeze()

    config.freeze()
    return config


def get_task_config(
        config_paths: Optional[Union[List[str], str]] = None,
        opts: Optional[list] = None
) -> habitat.Config:
    config = _TC.clone()
    config.set_new_allowed(False)
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
