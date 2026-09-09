"""MolmoSpaces policy wrapper for whole-body tracking and Dex1-1 control."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from molmo_spaces.policy.base_policy import BasePolicy
from molmo_spaces.policy.motion_tracking import MotionTrackingRuntime, MujocoStateBridge
from molmo_spaces.robots.g1_dex1_constants import DEX1_FINGER_LOWER, DEX1_FINGER_UPPER

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.tasks.task import BaseMujocoTask


class MotionTrackingPolicy(BasePolicy):
    """Return body and Dex1 actions from the bundled tracking policy."""

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask | None = None) -> None:
        super().__init__(config, task)
        if task is None:
            raise ValueError("MotionTrackingPolicy requires a live MolmoSpaces task")
        robot_view = task.env.current_robot.robot_view
        required_groups = {"base", "body", "left_dex1", "right_dex1"}
        if not required_groups.issubset(robot_view.move_group_ids()):
            raise ValueError(
                f"MotionTrackingPolicy requires G1Dex1RobotView groups {required_groups}, "
                f"got {robot_view.move_group_ids()}"
            )

        policy_config = config.policy_config
        self.bridge = MujocoStateBridge(robot_view)
        self.runtime = MotionTrackingRuntime(
            self.bridge,
            tracking_config_path=policy_config.tracking_config_path,
            policy_path=policy_config.policy_path,
            motion_source=policy_config.motion_source,
            enable_transport=policy_config.enable_transport,
        )

    def set_dex1_open_amount(self, side: Literal["left", "right"], open_amount: float) -> None:
        """Set one Dex1-1 jaw: 0 is closed, 1 is fully open."""

        self.bridge.set_dex1_open_amount(side, open_amount)

    def reset(self) -> None:
        self.runtime.reset()
        self.set_dex1_open_amount("left", 1.0)
        self.set_dex1_open_amount("right", 1.0)

    def get_action(self, observation) -> dict[str, np.ndarray]:
        del observation
        dex1_targets = {}
        for side in ("left", "right"):
            amount = self.bridge.dex1_open_amount[side]
            target = DEX1_FINGER_LOWER + amount * (DEX1_FINGER_UPPER - DEX1_FINGER_LOWER)
            dex1_targets[f"{side}_dex1"] = np.full(2, target, dtype=np.float64)
        return {
            "body": self.runtime.step(),
            **dex1_targets,
        }

    def get_info(self) -> dict:
        return {
            "policy": "motion_tracking",
            "observation_dim": self.runtime.observation_dim,
            "motion_source": self.runtime.policy.motion_source,
        }

    def close(self) -> None:
        self.runtime.close()
