#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from math import pi
import logging

import numpy as np

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# from habitat.config.default import get_config
from av_nav.common.benchmark import Benchmark
from av_nav.config.default import get_task_config as get_config


class RandomAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        # because the frame is in with polar coordinates
        dist = observations[self.goal_sensor_uuid][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.STOP
        else:
            action = np.random.choice(
                [
                    HabitatSimActions.MOVE_FORWARD,
                    HabitatSimActions.TURN_LEFT,
                    HabitatSimActions.TURN_RIGHT,
                ]
            )
        return {"action": action}


class ForwardOnlyAgent(RandomAgent):
    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.STOP
        else:
            action = HabitatSimActions.MOVE_FORWARD
        return {"action": action}


class ForwardTurnAgent(RandomAgent):
    def __init__(self, success_distance, goal_sensor_uuid):
        self._last_obs = None
        super(ForwardTurnAgent, self).__init__(success_distance, goal_sensor_uuid)

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.STOP
            self._last_obs = None
        else:
            if self._last_obs is None:
                action = HabitatSimActions.MOVE_FORWARD
                self._last_obs = observations
            else:
                if (self._last_obs[self.goal_sensor_uuid] == observations[self.goal_sensor_uuid]).all():
                    action = HabitatSimActions.TURN_RIGHT
                    self._last_obs = None
                else:
                    action = HabitatSimActions.MOVE_FORWARD
                    self._last_obs = observations
        return {"action": action}


class RandomForwardAgent(RandomAgent):
    def __init__(self, success_distance, goal_sensor_uuid):
        super().__init__(success_distance, goal_sensor_uuid)
        self.FORWARD_PROBABILITY = 0.8

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.STOP
        else:
            if np.random.uniform(0, 1, 1) < self.FORWARD_PROBABILITY:
                action = HabitatSimActions.MOVE_FORWARD
            else:
                action = np.random.choice(
                    [HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]
                )

        return {"action": action}


class GoalFollower(RandomAgent):
    def __init__(self, success_distance, goal_sensor_uuid):
        super().__init__(success_distance, goal_sensor_uuid)
        self.pos_th = self.dist_threshold_to_stop
        self.angle_th = float(np.deg2rad(60))
        self.random_prob = 0

    def normalize_angle(self, angle):
        if angle < -pi:
            angle = 2.0 * pi + angle
        if angle > pi:
            angle = -2.0 * pi + angle
        return angle

    def turn_towards_goal(self, angle_to_goal):
        if angle_to_goal > pi or (
            (angle_to_goal < 0) and (angle_to_goal > -pi)
        ):
            action = HabitatSimActions.TURN_RIGHT
        else:
            action = HabitatSimActions.TURN_LEFT
        return action

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.STOP
        else:
            angle_to_goal = self.normalize_angle(
                np.array(observations[self.goal_sensor_uuid][1])
            )
            if abs(angle_to_goal) < self.angle_th:
                action = HabitatSimActions.MOVE_FORWARD
            else:
                action = self.turn_towards_goal(angle_to_goal)

        return {"action": action}


def get_all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )


def get_agent_cls(agent_class_name):
    sub_classes = [
        sub_class
        for sub_class in get_all_subclasses(habitat.Agent)
        if sub_class.__name__ == agent_class_name
    ]
    return sub_classes[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-distance", type=float, default=0.2)
    parser.add_argument(
        "--task-config", type=str, default="configs/replica/pointgoal.yaml"
    )
    parser.add_argument("--agent-class", type=str, default="RandomAgent")
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    task_config = get_config(args.task_config)
    task_config.defrost()
    task_config.DATASET.SPLIT = 'test_telephone'
    task_config.freeze()

    agent = get_agent_cls(args.agent_class)(
        success_distance=args.success_distance,
        goal_sensor_uuid=task_config.TASK.GOAL_SENSOR_UUID,
    )
    benchmark = Benchmark(task_config)
    metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
