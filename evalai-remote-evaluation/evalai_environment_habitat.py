import argparse
import grpc
import gym
import pickle 
import sys
import os
import requests
import json
import random

from environment_utils import EvalAI_Interface

from concurrent import futures
from collections import defaultdict
import time
from timeout import Timeout

import evaluation_pb2
import evaluation_pb2_grpc

import numpy as np

import habitat
import soundspaces
from ss_baselines.av_nav.config.default import get_config
# from habitat.config.default import get_config
from habitat import make_dataset
from habitat.core.env import Env


def make_eval_env_fn(config, env_class, rank):
    dataset = make_dataset(
        config.DATASET.TYPE, config=config.DATASET
    )
    env = env_class(config=config, dataset=dataset)
    env.seed(rank)
    return env


def construct_eval_envs(env_class, config, num_processes):
    configs = []
    env_classes = [env_class for _ in range(num_processes)]
    dataset = make_dataset(config.DATASET.TYPE)
    scenes = dataset.get_scenes_to_load(config.DATASET)

    if len(scenes) > 0:
        random.shuffle(scenes)

        assert len(scenes) >= num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

    scene_splits = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):

        task_config = config.clone()
        task_config.defrost()
        if len(scenes) > 0:
            task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.freeze()

        configs.append(task_config.clone())

    envs = habitat.VectorEnv(
        make_env_fn=make_eval_env_fn,
        env_fn_args=tuple(
            tuple(zip(configs, env_classes, range(num_processes)))
        ),
        auto_reset_done=False,
    )
    return envs


class EvaluatorEnvironment:
    def __init__(self, config_path, environment='CartPole-v0'):
        self.feedback = None

        config = get_config(config_path)

        self.env = construct_eval_envs(Env, config, num_processes=1)
        self.current_episode_steps = 0

    def count_episodes(self):
        return self.env.count_episodes()[0]


class Environment(evaluation_pb2_grpc.EnvironmentServicer):
    def __init__(self, challenge_pk, phase_pk, submission_pk, config_path):
        self.challenge_pk = challenge_pk
        self.phase_pk = phase_pk
        self.submission_pk = submission_pk
        self._env = EvaluatorEnvironment(config_path)
        self._agg_metrics = defaultdict(list)

    def num_episodes(self, request, context):
        res = {"num_episodes": self._env.count_episodes()}
        return evaluation_pb2.Package(SerializedEntity=pack_for_grpc(res))

    def reset(self, request, context):
        obs = self._env.env.reset()[0]
        return evaluation_pb2.Package(SerializedEntity=pack_for_grpc(
            {
                "observations": obs
            }
        ))

    def episode_over(self, request, context):
        episode_over = self._env.env.episode_over()[0]
        return evaluation_pb2.Package(SerializedEntity=pack_for_grpc(
            {
                "episode_over": episode_over
            }
        ))

    def get_metrics(self, request, context):
        metrics = {"spl": -1.0, "softspl": -1.0, "distance_to_goal": -1.0, "success": -1.0}
        return evaluation_pb2.Package(SerializedEntity=pack_for_grpc(
            {
                "metrics": metrics
           }
        ))

    def evalai_update_submission(self, request, context):
        res = self._agg_metrics

        # TODO(akadian): check number of episodes executed
        print("TESTING EPISODES: {} {}".format(
                len(res["spl"]),
                self._env.env.count_episodes()[0],
            )
        )

        final_res = {}
        if len(res["spl"]) != self._env.env.count_episodes()[0]:
            logging.info("Missing episodes")
            final_res["spl"] = -1.0
            final_res["softspl"] = -1.0
            final_res["distance_to_goal"] = -1.0
            final_res["success"] = -1.0
            submission_success = False
        else:
            final_res["spl"] = np.mean(res["spl"])
            final_res["softspl"] = np.mean(res["softspl"])
            final_res["distance_to_goal"] = np.mean(res["distance_to_goal"])
            final_res["success"] = np.mean(res["success"])
            submission_success = True

        try: 
            update_submission_result(self.challenge_pk, self.phase_pk, self.submission_pk, final_res, "FINISHED")
            return evaluation_pb2.Package(SerializedEntity=pack_for_grpc(
                {
                    "submission_success": submission_success
               }
            ))
        except:
            return evaluation_pb2.Package(SerializedEntity=pack_for_grpc(
                {
                    "submission_success": False
               }
            ))

    def act_on_environment(self, request, context):
        # TODO(akadian): check if you have to handle any corner cases here

        action = unpack_for_grpc(request.SerializedEntity)
        res = self._env.env.step([action])
        obs = res[0]

        self._env.current_episode_steps += 1

        if self._env.env.episode_over()[0]:
            mtr = self._env.env.get_metrics()[0]
            for m, v in mtr.items():
                self._agg_metrics[m].append(v)

        return evaluation_pb2.Package(SerializedEntity=pack_for_grpc(
            {
                "observations": obs
            }
        ))


def pack_for_grpc(entity):
    return pickle.dumps(entity)


def unpack_for_grpc(entity):
    return pickle.loads(entity)


def get_action_space(env):
    return list(range(env.action_space.n))


def metric_show_participant(phase):
    if phase == "1233" or phase == "1236":
        return False
    return True


def update_submission_result(challenge_pk, phase_pk, submission_pk, metrics, submission_status):
    print("Final Score: spl: {:.6f}, softspl: {:.6f}, distance_to_goal: {:.6f}, success: {:.6f}".format(
        metrics["spl"], metrics["softspl"], metrics["distance_to_goal"], metrics["success"])
    )
    print("Metrics: {}".format(metrics))
    print("Stopping Evaluator")
    submission_data = {
        "submission_status": "finished",
        "submission": submission_pk,
    }
    submission_data = {
        "challenge_phase": phase_pk,
        "submission":submission_pk,
        "stdout": "standard_ouput",
        "stderr": "standard_error",
        "submission_status": submission_status,
        "result": json.dumps(
            [
                {
                    'split': 'split',
                    'show_to_participant': metric_show_participant(phase_pk),
                    'accuracies': {
                        'SPL': metrics["spl"], 
                        "SOFT_SPL": metrics["softspl"],
                        "DISTANCE_TO_GOAL": metrics["distance_to_goal"],
                        "SUCCESS": metrics["success"],
                    }
                }
            ]
        )
    }
    print("SUBMISSION DATA: {0}".format(submission_data))
    api.update_submission_data(submission_data, challenge_pk)
    print("Data updated successfully")
    exit(0)

    
api = EvalAI_Interface(
    AUTH_TOKEN = os.environ.get("AUTH_TOKEN", ""),
    EVALAI_API_SERVER = os.environ.get("EVALAI_API_SERVER", "http://localhost:8000"),
)


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--agent-timelimit", type=int, required=True)
        args = parser.parse_args()

        BODY = os.environ.get("BODY")
        BODY = BODY.replace("'", '"')
        BODY = json.loads(BODY)
        print("BODY", BODY)
        challenge_pk = BODY["challenge_pk"]
        phase_pk = BODY["phase_pk"]
        submission_pk = BODY["submission_pk"]

        server = grpc.server(futures.ThreadPoolExecutor(max_workers = 1))
        evaluation_pb2_grpc.add_EnvironmentServicer_to_server(Environment(challenge_pk, phase_pk, submission_pk, args.config), server)
        print('Starting server. Listening on port 8085.')
        server.add_insecure_port('[::]:8085')
        server.start()
        try:
            with Timeout(seconds=args.agent_timelimit):
                while True:
                    time.sleep(4)
        except KeyboardInterrupt:
            server.stop(0)
    except Exception as e:
        print("Exception:", e)
        update_submission_result(challenge_pk, phase_pk, submission_pk, {
            "spl": -1.0,
            "softspl": -1.0,
            "distance_to_goal": -1.0,
            "success": -1.0,
        }, "FAILED")
        import traceback
        print(traceback.print_exc())
    

if __name__ == "__main__":
    main()