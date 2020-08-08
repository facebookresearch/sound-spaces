# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import HabitatSimActions

HabitatSimActions.extend_action_space("MOVE_BACKWARD")
HabitatSimActions.extend_action_space("MOVE_LEFT")
HabitatSimActions.extend_action_space("MOVE_RIGHT")


@registry.register_action_space_configuration(name="move-all")
class MoveOnlySpaceConfiguration(ActionSpaceConfiguration):
    def get(self):
        return {
            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.MOVE_BACKWARD: habitat_sim.ActionSpec(
                "move_backward",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.MOVE_RIGHT: habitat_sim.ActionSpec(
                "move_right",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            ),
            HabitatSimActions.MOVE_LEFT: habitat_sim.ActionSpec(
                "move_left",
                habitat_sim.ActuationSpec(
                    amount=self.config.FORWARD_STEP_SIZE
                ),
            )
        }
