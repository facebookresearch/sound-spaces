#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Type, Union
import logging

import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    Sensor,
    SensorTypes,
    Simulator,
)


class MapPlaceHolder(Sensor):
    def __init__(
            self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.config.MAP_SIZE, self.config.MAP_SIZE, self.config.NUM_CHANNEL),
            dtype=np.uint8,
        )

    def get_observation(
            self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        return np.zeros((self.config.MAP_SIZE, self.config.MAP_SIZE, self.config.NUM_CHANNEL))


@registry.register_sensor(name="GeometricMap")
class GeometricMap(MapPlaceHolder):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "gm"


@registry.register_sensor(name="ActionMap")
class ActionMap(MapPlaceHolder):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "action_map"


@registry.register_sensor(name="AcousticMap")
class AcousticMap(MapPlaceHolder):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "am"


@registry.register_sensor(name="Intensity")
class Intensity(Sensor):
    def __init__(
        self, sim: Union[Simulator, Config], config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(config=config)
        self._sim = sim

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "intensity"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1,
            shape=(1,),
            dtype=bool
        )

    def get_observation(
        self, *args: Any, observations, episode: Episode, **kwargs: Any
    ) -> object:
        num_frame = 150
        audiogoal = self._sim.get_current_audiogoal_observation()
        nonzero_idx = np.min((audiogoal > 0.1 * audiogoal.max()).argmax(axis=1))
        impulse = audiogoal[:, nonzero_idx: nonzero_idx + num_frame]
        rms = np.mean(impulse ** 2)

        return [rms]
