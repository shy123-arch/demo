"""MolmoSpaces view of the free-base G1 with two Dex1-1 grippers."""

from __future__ import annotations

from typing import Literal

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.g1_dex1_constants import (
    BODY_JOINT_NAMES,
    DEX1_FINGER_LOWER,
    DEX1_FINGER_UPPER,
    dex1_joint_names,
)
from molmo_spaces.robots.robot_views.abstract import (
    FreeJointRobotBaseGroup,
    GripperGroup,
    MJCFFrameMixin,
    RobotView,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose


def _actuator_for_joint(model, joint_id: int) -> int:
    matches = [aid for aid in range(model.nu) if int(model.actuator_trnid[aid, 0]) == joint_id]
    if len(matches) != 1:
        joint_name = model.joint(joint_id).name
        raise ValueError(f"Expected one actuator for {joint_name}, found {len(matches)}")
    return matches[0]


class G1Dex1BaseGroup(FreeJointRobotBaseGroup):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        model = mj_data.model
        base_joint_id = model.joint(f"{namespace}floating_base_joint").id
        super().__init__(mj_data, base_joint_id, [], [], floating=False)

    @property
    def noop_ctrl(self) -> np.ndarray:
        # The pelvis is deliberately unactuated; locomotion emerges from the
        # 29 body torques and ground contact.
        return np.zeros(0, dtype=np.float64)


class G1Dex1BodyGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    def __init__(self, mj_data: MjData, base: G1Dex1BaseGroup, namespace: str = "") -> None:
        model = mj_data.model
        joint_ids = [model.joint(f"{namespace}{name}").id for name in BODY_JOINT_NAMES]
        actuator_ids = [_actuator_for_joint(model, jid) for jid in joint_ids]
        self._root_id = model.body(f"{namespace}pelvis").id
        self._leaf_id = model.body(f"{namespace}torso_link").id
        super().__init__(mj_data, joint_ids, actuator_ids, self._root_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._leaf_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_id)


class Dex1GripperGroup(MJCFFrameMixin, GripperGroup):
    def __init__(
        self,
        mj_data: MjData,
        side: Literal["left", "right"],
        base: G1Dex1BaseGroup,
        namespace: str = "",
    ) -> None:
        model = mj_data.model
        self.side = side
        joint_ids = [model.joint(f"{namespace}{name}").id for name in dex1_joint_names(side)]
        actuator_ids = [_actuator_for_joint(model, jid) for jid in joint_ids]
        self._base_id = model.body(f"{namespace}{side}_dex1_base_link").id
        super().__init__(mj_data, joint_ids, actuator_ids, self._base_id, base)

    @property
    def leaf_frame_id(self) -> int:
        return self._base_id

    @property
    def leaf_frame_type(self):
        return "body"

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._base_id)

    def set_gripper_ctrl_open(self, open: bool) -> None:
        target = DEX1_FINGER_UPPER if open else DEX1_FINGER_LOWER
        self.ctrl = np.full(2, target, dtype=np.float64)

    @property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        return 0.0, 2.0 * (DEX1_FINGER_UPPER - DEX1_FINGER_LOWER)

    @property
    def inter_finger_dist(self) -> float:
        opening = self.joint_pos - DEX1_FINGER_LOWER
        return float(np.clip(opening.sum(), *self.inter_finger_dist_range))


class G1Dex1RobotView(RobotView):
    def __init__(self, mj_data: MjData, namespace: str = "") -> None:
        self._namespace = namespace
        base = G1Dex1BaseGroup(mj_data, namespace)
        super().__init__(
            mj_data,
            {
                "base": base,
                "body": G1Dex1BodyGroup(mj_data, base, namespace),
                "left_dex1": Dex1GripperGroup(mj_data, "left", base, namespace),
                "right_dex1": Dex1GripperGroup(mj_data, "right", base, namespace),
            },
        )

    @property
    def name(self) -> str:
        return f"{self._namespace}g1_dex1"

    @property
    def base(self) -> G1Dex1BaseGroup:
        return self._move_groups["base"]
