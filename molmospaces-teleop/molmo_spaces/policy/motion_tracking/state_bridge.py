"""Expose MolmoSpaces MuJoCo state using tracking-policy semantics."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np

from molmo_spaces.robots.g1_dex1_constants import (
    BODY_JOINT_NAMES,
    BODY_TO_POLICY_INDICES,
    DEFAULT_BODY_QPOS,
    POLICY_JOINT_NAMES,
)
from molmo_spaces.robots.robot_views.g1_dex1_view import G1Dex1RobotView


class MujocoStateBridge:
    """Controller-shaped state object consumed by tracking observation modules.

    It replaces the Unitree SDK/DDS controller with direct MuJoCo reads.
    """

    def __init__(self, robot_view: G1Dex1RobotView) -> None:
        self.robot_view = robot_view
        self.mj_model = robot_view.mj_model
        self.mj_data = robot_view.mj_data
        self.namespace = getattr(robot_view, "_namespace", "")
        self.config = SimpleNamespace(
            real_joint_names=list(BODY_JOINT_NAMES),
            isaac_joint_names_state=list(POLICY_JOINT_NAMES),
            default_qpos_real=DEFAULT_BODY_QPOS.astype(np.float32).tolist(),
        )
        self.dof_size_real = len(BODY_JOINT_NAMES)
        self.default_qpos_real = DEFAULT_BODY_QPOS.astype(np.float32).copy()

        self.qj_real = np.zeros(self.dof_size_real, dtype=np.float32)
        self.dqj_real = np.zeros(self.dof_size_real, dtype=np.float32)
        self.tau_real = np.zeros(self.dof_size_real, dtype=np.float32)
        self.qj_isaac = np.zeros(self.dof_size_real, dtype=np.float32)
        self.dqj_isaac = np.zeros(self.dof_size_real, dtype=np.float32)
        self.tau_isaac = np.zeros(self.dof_size_real, dtype=np.float32)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.gyro = np.zeros(3, dtype=np.float32)
        self.linacc = np.zeros(3, dtype=np.float32)
        self.root_pos_w = np.zeros(3, dtype=np.float32)
        self.root_quat_w = self.quat.copy()
        self.root_vel_w = np.zeros(3, dtype=np.float32)
        self.root_yaw_speed = 0.0
        self._odom_origin_xy = None
        self.dex1_open_amount = {"left": 1.0, "right": 1.0}

        self._gyro_sensor = self._sensor_slice("imu_ang_vel")
        self._linacc_sensor = self._sensor_slice("imu_lin_acc")
        self.update()

    def _sensor_slice(self, unprefixed_name: str) -> slice | None:
        sensor_id = mujoco.mj_name2id(
            self.mj_model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            f"{self.namespace}{unprefixed_name}",
        )
        if sensor_id < 0:
            return None
        start = int(self.mj_model.sensor_adr[sensor_id])
        size = int(self.mj_model.sensor_dim[sensor_id])
        return slice(start, start + size)

    def update(self) -> None:
        body = self.robot_view.get_move_group("body")
        base = self.robot_view.base

        self.qj_real[:] = np.asarray(body.joint_pos, dtype=np.float32)
        self.dqj_real[:] = np.asarray(body.joint_vel, dtype=np.float32)
        self.tau_real[:] = np.asarray(body.ctrl, dtype=np.float32)
        self.qj_isaac[:] = self.qj_real[BODY_TO_POLICY_INDICES]
        self.dqj_isaac[:] = self.dqj_real[BODY_TO_POLICY_INDICES]
        self.tau_isaac[:] = self.tau_real[BODY_TO_POLICY_INDICES]

        base_qpos = np.asarray(base.joint_pos, dtype=np.float32)
        base_qvel = np.asarray(base.joint_vel, dtype=np.float32)
        self.root_pos_w[:] = base_qpos[:3]
        quat = base_qpos[3:7]
        self.quat[:] = quat / max(float(np.linalg.norm(quat)), 1e-8)
        self.root_quat_w[:] = self.quat
        self.root_vel_w[:] = base_qvel[:3]
        self.root_yaw_speed = float(base_qvel[5])

        if self._gyro_sensor is not None:
            self.gyro[:] = self.mj_data.sensordata[self._gyro_sensor][:3]
        else:
            self.gyro[:] = base_qvel[3:6]
        if self._linacc_sensor is not None:
            self.linacc[:] = self.mj_data.sensordata[self._linacc_sensor][:3]
        else:
            self.linacc[:] = 0.0

    def current_reference_anchor(self) -> dict[str, np.ndarray]:
        """State accepted by ``TrackingPolicyRaw.reset_reference_to_state``."""

        return {
            "joint_pos": self.qj_isaac.copy(),
            "root_pos": self.root_pos_w.copy(),
            "root_quat": self.root_quat_w.copy(),
        }

    def set_dex1_open_amount(self, side: str, open_amount: float) -> None:
        if side not in self.dex1_open_amount:
            raise ValueError(f"Unknown Dex1-1 side: {side!r}")
        self.dex1_open_amount[side] = float(np.clip(open_amount, 0.0, 1.0))

    def set_hand_from_controller_buttons(self, buttons: dict) -> None:
        """Map continuous PICO trigger/grip values onto the Dex1-1 jaws."""

        def close_amount(side: str) -> float:
            trigger = float(np.clip(buttons.get(f"{side}_trigger_value", 0.0), 0.0, 1.0))
            grip = float(np.clip(buttons.get(f"{side}_grip_value", 0.0), 0.0, 1.0))
            return 0.5 + 0.5 * trigger if grip > 1e-4 else trigger

        for side in ("left", "right"):
            self.set_dex1_open_amount(side, 1.0 - close_amount(side))
