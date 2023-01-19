''#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, List, Optional
import logging
import gzip

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint
from habitat.core.dataset import Dataset
from habitat.core.utils import DatasetFloatJSONEncoder
from soundspaces.tasks.semantic_audionav_task import SemanticAudioGoalNavEpisode, SemanticAudioGoal, ObjectViewLocation

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"


@registry.register_dataset(name="SemanticAudioNav")
class SemanticAudioNavDataset(Dataset):
    category_to_task_category_id: Dict[str, int]
    category_to_scene_annotation_category_id: Dict[str, int]
    episodes: List[SemanticAudioGoalNavEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"
    goals_by_category: Dict[str, List[SemanticAudioGoal]]

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def get_scenes_to_load(config: Config, **kwargs) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        assert SemanticAudioNavDataset.check_config_paths_exist(config), \
            (config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT), config.SCENES_DIR)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = SemanticAudioNavDataset(cfg)
        return SemanticAudioNavDataset._get_scenes_from_folder(
            content_scenes_path=dataset.content_scenes_path,
            dataset_dir=dataset_dir,
        )

    @staticmethod
    def _get_scenes_from_folder(content_scenes_path, dataset_dir):
        scenes = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    @staticmethod
    def dedup_goals(dataset: Dict[str, Any]) -> Dict[str, Any]:
        if len(dataset["episodes"]) == 0:
            return dataset

        goals_by_category = dict()
        for i, ep in enumerate(dataset["episodes"]):
            dataset["episodes"][i]["object_category"] = ep["goals"][0][
                "object_category"
            ]
            ep = SemanticAudioGoalNavEpisode(**ep)

            goals_key = ep.goals_key
            if goals_key not in goals_by_category:
                goals_by_category[goals_key] = ep.goals

            dataset["episodes"][i]["goals"] = []

        dataset["goals_by_category"] = goals_by_category

        return dataset

    def to_json(self) -> str:
        for i in range(len(self.episodes)):
            self.episodes[i].goals = []

        result = DatasetFloatJSONEncoder().encode(self)

        for i in range(len(self.episodes)):
            self.episodes[i].goals = self.goals_by_category[
                self.episodes[i].goals_key
            ]

        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self._config = config

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=datasetfile_path)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        scenes = config.CONTENT_SCENES
        if ALL_SCENES_MASK in scenes:
            scenes = SemanticAudioNavDataset._get_scenes_from_folder(
                content_scenes_path=self.content_scenes_path,
                dataset_dir=dataset_dir,
            )

        last_episode_cnt = 0
        for scene in scenes:
            scene_filename = self.content_scenes_path.format(
                data_path=dataset_dir, scene=scene
            )
            with gzip.open(scene_filename, "rt") as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=scene_filename)

            num_episode = len(self.episodes) - last_episode_cnt
            last_episode_cnt = len(self.episodes)
            logging.debug('Sampled {} from {}'.format(num_episode, scene))
        logging.info(f"Sampled {len(self.episodes)} episodes from {len(scenes)} scenes.")

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> SemanticAudioGoal:
        g = SemanticAudioGoal(**serialized_goal)

        for vidx, view in enumerate(g.view_points):
            view_location = ObjectViewLocation(view, iou=0)
            view_location.agent_state = AgentState(view_location.agent_state)
            g.view_points[vidx] = view_location

        return g

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, scene_filename: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        # if "category_to_task_category_id" in deserialized:
        #     self.category_to_task_category_id = deserialized[
        #         "category_to_task_category_id"
        #     ]
        #
        # if "category_to_scene_annotation_category_id" in deserialized:
        #     self.category_to_scene_annotation_category_id = deserialized[
        #         "category_to_scene_annotation_category_id"
        #     ]
        #
        # if "category_to_mp3d_category_id" in deserialized:
        #     self.category_to_scene_annotation_category_id = deserialized[
        #         "category_to_mp3d_category_id"
        #     ]
        #
        # assert len(self.category_to_task_category_id) == len(
        #     self.category_to_scene_annotation_category_id
        # )

        # assert set(self.category_to_task_category_id.keys()) == set(
        #     self.category_to_scene_annotation_category_id.keys()
        # ), "category_to_task and category_to_mp3d must have the same keys"

        if len(deserialized["episodes"]) == 0:
            return

        # if "goals_by_category" not in deserialized:
        #     deserialized = self.dedup_goals(deserialized)
        #
        # for k, v in deserialized["goals_by_category"].items():
        #     self.goals_by_category[k] = [self.__deserialize_goal(g) for g in v]

        for i, episode in enumerate(deserialized["episodes"]):
            episode = SemanticAudioGoalNavEpisode(**episode)
            # a temporal workaround to set scene_dataset_config attribute
            episode.scene_dataset_config = self._config.SCENES_DIR.split('/')[-1]

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = self.__deserialize_goal(goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)

            if hasattr(self._config, 'CONTINUOUS') and self._config.CONTINUOUS:
                # TODO: simplify
                episode.goals[0].position[1] += 0.1
                for view_point in episode.goals[0].view_points:
                    view_point.agent_state.position[1] += 0.1
            self.episodes.append(episode)

            # the agent can navigate to any instance of the target object category
            # episode.goals = self.goals_by_category[episode.goals_key]

            # if episode.shortest_paths is not None:
            #     for path in episode.shortest_paths:
            #         for p_index, point in enumerate(path):
            #             if point is None or isinstance(point, (int, str)):
            #                 point = {
            #                     "action": point,
            #                     "rotation": None,
            #                     "position": None,
            #                 }
            #
            #             path[p_index] = ShortestPathPoint(**point)

            # self.episodes.append(episode)
