#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Implements evaluation of ``habitat.Agent`` inside ``habitat.Env``.
``habitat.Benchmark`` creates a ``habitat.Env`` which is specified through
the ``config_env`` parameter in constructor. The evaluation is task agnostic
and is implemented through metrics defined for ``habitat.EmbodiedTask``.
"""

from collections import defaultdict
from typing import Dict, Optional
import logging

from tqdm import tqdm

from habitat import Config
from habitat.core.agent import Agent
# from habitat.core.env import Env
from ss_baselines.common.environments import NavRLEnv
from habitat.datasets import make_dataset


class Benchmark:
    r"""Benchmark for evaluating agents in environments.
    """

    def __init__(self, task_config: Optional[Config] = None) -> None:
        r"""..

        :param task_config: config to be used for creating the environment
        """
        dummy_config = Config()
        dummy_config.RL = Config()
        dummy_config.RL.SLACK_REWARD = -0.01
        dummy_config.RL.SUCCESS_REWARD = 10
        dummy_config.RL.WITH_TIME_PENALTY = True
        dummy_config.RL.DISTANCE_REWARD_SCALE = 1
        dummy_config.RL.WITH_DISTANCE_REWARD = True
        dummy_config.RL.defrost()
        dummy_config.TASK_CONFIG = task_config
        dummy_config.freeze()

        dataset = make_dataset(id_dataset=task_config.DATASET.TYPE, config=task_config.DATASET)
        self._env = NavRLEnv(config=dummy_config, dataset=dataset)

    def evaluate(
        self, agent: Agent, num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        r"""..

        :param agent: agent to be evaluated in environment.
        :param num_episodes: count of number of episodes for which the
            evaluation should be run.
        :return: dict containing metrics tracked by environment.
        """

        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0
        reward_episodes = 0
        step_episodes = 0
        success_count = 0
        for count_episodes in tqdm(range(num_episodes)):
            agent.reset()
            observations = self._env.reset()
            episode_reward = 0

            while not self._env.habitat_env.episode_over:
                action = agent.act(observations)
                observations, reward, done, info = self._env.step(**action)
                logging.debug("Reward: {}".format(reward))
                if done:
                    logging.debug('Episode reward: {}'.format(episode_reward))
                episode_reward += reward
                step_episodes += 1

            metrics = self._env.habitat_env.get_metrics()
            for m, v in metrics.items():
                agg_metrics[m] += v
            reward_episodes += episode_reward
            success_count += metrics['spl'] > 0

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}
        logging.info("Average reward: {} in {} episodes".format(reward_episodes / count_episodes, count_episodes))
        logging.info("Average episode steps: {}".format(step_episodes / count_episodes))
        logging.info('Success rate: {}'.format(success_count / num_episodes))

        return avg_metrics
