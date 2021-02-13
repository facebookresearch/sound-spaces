# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Type, Union

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.tasks.nav.nav import NavigationTask, Measure, EmbodiedTask
from habitat.core.registry import registry


@registry.register_task(name="AudioNav")
class AudioNavigationTask(NavigationTask):
    def overwrite_sim_config(
        self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)


def merge_sim_episode_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    # here's where the scene update happens, extract the scene name out of the path
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
        episode.start_position is not None
        and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        agent_cfg.GOAL_POSITION = episode.goals[0].position
        agent_cfg.SOUND_ID = episode.info['sound'] + '.wav'
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config
