#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import shutil
from pprint import pprint

import git
import torch
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
import tensorflow as tf

from av_wan.common.baseline_registry import baseline_registry
from av_wan.config.default import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        # required=True,
        default='audionav_blind,audionav_rgb,audionav_depth,pointnav_blind,pointnav_rgb,pointnav_depth,'
                'pointaudionav_blind,pointaudionav_rgb,pointaudionav_depth',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        # required=True,
        default='av_wan/config/test_telephone',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        required=True,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=0,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=100000,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--event-dir",
        type=str,
        # required=True,
        default='eval_spl_val_telephone_average_spl',
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--prev-ckpt-ind",
        type=int,
        default=-1,
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--collect-ckpts",
        default=False,
        action='store_true',
        help="Evaluation interval of checkpoints",
    )
    parser.add_argument(
        "--no-search",
        default=False,
        action='store_true',
        help="Evaluation interval of checkpoints",
    )
    args = parser.parse_args()

    models = args.models.split(',')
    results = {}
    if args.collect_ckpts:
        shutil.rmtree('data/ckpts', ignore_errors=True)
    for model in models:
        model_config_path = os.path.join(args.config_dir, model + '.yaml')
        model_dir = os.path.join(args.models_dir, model)
        model_ckpt_dir = os.path.join(model_dir, 'data')
        model_log_dir = os.path.join(model_dir, 'tb')

        if args.no_search:
            ckpt_file = os.listdir(model_ckpt_dir)[0]
            best_ckpt_file = os.path.join(model_ckpt_dir, ckpt_file)
        else:
            best_ckpt_idx = find_best_ckpt_idx(model_log_dir, args.event_dir, args.min_step, args.max_step)
            best_ckpt_file = os.path.join(model_ckpt_dir, 'ckpt.{}.pth'.format(best_ckpt_idx))
        if args.collect_ckpts:
            target_dir = os.path.join('data/ckpts', model, 'data', model)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(best_ckpt_file, target_dir)
        else:
            result = test(model_config_path, model_dir, best_ckpt_file, args.opts)
            results[model] = result

    pprint(results)


def find_best_ckpt_idx(model_log_dir, event_dir, min_step, max_step):
    event_dir_path = os.path.join(model_log_dir, event_dir)
    events = os.listdir(event_dir_path)

    max_value = 0
    max_index = -1
    for event in events:
        iterator = tf.compat.v1.train.summary_iterator(os.path.join(event_dir_path, event))
        for e in iterator:
            if not min_step <= e.step <= max_step:
                continue
            if len(e.summary.value) > 0 and e.summary.value[0].simple_value > max_value:
                max_value = e.summary.value[0].simple_value
                max_index = e.step

    if max_index == -1:
        logging.warning('No max index is found in {}'.format(event_dir_path))
    else:
        logging.info('The best index in {} is {}'.format(event_dir_path, max_index))

    return max_index


def test(model_config_path, model_dir, best_ckpt_file, opts):
    config = get_config(model_config_path, opts, model_dir, 'eval')
    config.defrost()
    config.EVAL_CKPT_PATH_DIR = best_ckpt_file
    # config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = 256
    # config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = 256
    config.freeze()

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    repo = git.Repo(search_parent_directories=True)
    logging.info('Current git head hash code: {}'.format(repo.head.object.hexsha))

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    torch.set_num_threads(1)

    result = trainer.eval()
    return result


if __name__ == "__main__":
    main()
