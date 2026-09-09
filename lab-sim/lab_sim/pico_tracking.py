"""Adapt the proven MolmoSpaces PICO tracking runtime to the lab-sim G1."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import mujoco
import numpy as np
import yaml

from .env import ROOT


LEGACY_MOLMO_ROOT = (
    ROOT.parent / "molospace" / "molmospaces-teleop" / "molmospaces-teleop"
)
DEFAULT_MOLMO_ROOT = ROOT.parent / "molmospaces-teleop"


@dataclass(frozen=True)
class MolmoTrackingBindings:
    root: Path
    config_path: Path
    policy_path: Path
    robot_model_path: Path
    runtime_cls: type
    body_joint_names: tuple[str, ...]
    policy_joint_names: tuple[str, ...]
    default_body_qpos: np.ndarray
    body_kp: np.ndarray
    body_kd: np.ndarray
    body_to_policy_indices: np.ndarray


def resolve_molmo_root(path: str | Path | None = None) -> Path:
    """Resolve and validate the MolmoSpaces checkout used by the adapter."""

    candidate = Path(
        path or os.environ.get("MOLMO_TELEOP_ROOT") or (
            DEFAULT_MOLMO_ROOT if DEFAULT_MOLMO_ROOT.is_dir() else LEGACY_MOLMO_ROOT
        )
    ).expanduser().resolve()
    required = (
        candidate / "configs" / "motion_tracking_live.yaml",
        candidate / "assets" / "policies" / "motion_tracking" / "policy.onnx",
        candidate / "assets" / "policies" / "motion_tracking" / "policy.onnx.data",
        candidate / "assets" / "robots" / "g1_dex1" / "model.xml",
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            "MolmoSpaces tracking checkout is incomplete; missing: "
            + ", ".join(missing)
        )
    return candidate


def load_molmo_tracking(path: str | Path | None = None) -> MolmoTrackingBindings:
    """Import only the portable tracking runtime and its G1 constants."""

    root = resolve_molmo_root(path)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    runtime = importlib.import_module(
        "molmo_spaces.policy.motion_tracking.runtime"
    ).MotionTrackingRuntime
    constants = importlib.import_module("molmo_spaces.robots.g1_dex1_constants")
    config = yaml.safe_load((root / "configs" / "motion_tracking_live.yaml").read_text())
    expected = set(constants.BODY_JOINT_NAMES)
    for label, names in (
        ("body", constants.BODY_JOINT_NAMES),
        ("policy", constants.POLICY_JOINT_NAMES),
        ("dataset_joint_names", config.get("dataset_joint_names", [])),
        ("action_joint_names", config.get("action_joint_names", [])),
    ):
        if len(names) != 29 or len(set(names)) != 29 or set(names) != expected:
            raise ValueError(f"Invalid 29D joint-name contract: {label}")
    if sorted(constants.BODY_TO_POLICY_INDICES) != list(range(29)):
        raise ValueError("Body-to-policy mapping must be a 29D permutation")
    return MolmoTrackingBindings(
        root=root,
        config_path=root / "configs" / "motion_tracking_live.yaml",
        policy_path=root
        / "assets"
        / "policies"
        / "motion_tracking"
        / "policy.onnx",
        robot_model_path=root / "assets" / "robots" / "g1_dex1" / "model.xml",
        runtime_cls=runtime,
        body_joint_names=tuple(constants.BODY_JOINT_NAMES),
        policy_joint_names=tuple(constants.POLICY_JOINT_NAMES),
        default_body_qpos=np.asarray(
            constants.DEFAULT_BODY_QPOS, dtype=np.float64
        ).copy(),
        body_kp=np.asarray(constants.BODY_KP, dtype=np.float64).copy(),
        body_kd=np.asarray(constants.BODY_KD, dtype=np.float64).copy(),
        body_to_policy_indices=np.asarray(
            constants.BODY_TO_POLICY_INDICES, dtype=np.int64
        ).copy(),
    )


def _actuators_for_joints(model: mujoco.MjModel, joint_ids: np.ndarray) -> np.ndarray:
    actuator_ids = []
    for joint_id in joint_ids:
        matches = np.flatnonzero(model.actuator_trnid[:, 0] == int(joint_id))
        if len(matches) != 1:
            name = model.joint(int(joint_id)).name
            raise ValueError(
                f"Expected one actuator for body joint {name!r}, found {len(matches)}"
            )
        actuator_ids.append(int(matches[0]))
    return np.asarray(actuator_ids, dtype=np.int64)


class Dex3HandMapper:
    """Map continuous PICO trigger/grip values to mirrored Dex3 poses."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data
        self.names = tuple(
            f"{side}_hand_{name}_joint"
            for side in ("left", "right")
            for name in (
                "thumb_0",
                "thumb_1",
                "thumb_2",
                "middle_0",
                "middle_1",
                "index_0",
                "index_1",
            )
        )
        self.joint_ids = np.asarray(
            [model.joint(name).id for name in self.names], dtype=np.int64
        )
        self.qadr = model.jnt_qposadr[self.joint_ids]
        self.vadr = model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = _actuators_for_joints(model, self.joint_ids)
        self.ranges = model.jnt_range[self.joint_ids].copy()
        self.targets = np.zeros(len(self.names), dtype=np.float64)
        self.closed_targets = np.asarray(
            [
                (lo if abs(lo) > abs(hi) else hi) * 0.55
                for lo, hi in self.ranges
            ],
            dtype=np.float64,
        )
        self.kp = np.full(len(self.names), 15.0, dtype=np.float64)
        self.kd = np.full(len(self.names), 0.4, dtype=np.float64)
        self.torque_low = np.full(len(self.names), -np.inf, dtype=np.float64)
        self.torque_high = np.full(len(self.names), np.inf, dtype=np.float64)
        limited = model.jnt_actfrclimited[self.joint_ids].astype(bool)
        self.torque_low[limited] = model.jnt_actfrcrange[self.joint_ids][limited, 0]
        self.torque_high[limited] = model.jnt_actfrcrange[self.joint_ids][limited, 1]

    @staticmethod
    def _analog(buttons: dict[str, Any], side: str, kind: str) -> float:
        keys = (
            f"{side}_{kind}_value",
            f"{side}_{'index_trig' if kind == 'trigger' else kind}_value",
            f"{side}_{'index_trig' if kind == 'trigger' else kind}",
        )
        for key in keys:
            if key in buttons:
                try:
                    value = float(buttons[key])
                    if not np.isfinite(value):
                        return float("nan")
                    return float(np.clip(value, 0.0, 1.0))
                except (TypeError, ValueError):
                    return float("nan")
        return 0.0

    def set_from_controller_buttons(self, buttons: dict[str, Any]) -> None:
        if not isinstance(buttons, dict):
            return
        for side_index, side in enumerate(("left", "right")):
            trigger = self._analog(buttons, side, "trigger")
            grip = self._analog(buttons, side, "grip")
            if not np.isfinite([trigger, grip]).all():
                continue
            close = 0.5 + 0.5 * trigger if grip > 1e-4 else trigger
            sl = slice(side_index * 7, (side_index + 1) * 7)
            self.targets[sl] = np.clip(
                close * self.closed_targets[sl],
                self.ranges[sl, 0],
                self.ranges[sl, 1],
            )

    def compute_torque(self) -> np.ndarray:
        torque = self.kp * (self.targets - self.data.qpos[self.qadr])
        torque -= self.kd * self.data.qvel[self.vadr]
        return np.clip(torque, self.torque_low, self.torque_high)


class LabSimStateBridge:
    """Expose lab-sim MuJoCo state with MotionTrackingRuntime semantics."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        bindings: MolmoTrackingBindings,
        hands: Dex3HandMapper,
    ) -> None:
        self.model = model
        self.data = data
        self.bindings = bindings
        self.hands = hands
        self.config = SimpleNamespace(
            real_joint_names=list(bindings.body_joint_names),
            isaac_joint_names_state=list(bindings.policy_joint_names),
            default_qpos_real=bindings.default_body_qpos.astype(
                np.float32
            ).tolist(),
        )
        self.dof_size_real = len(bindings.body_joint_names)
        if self.dof_size_real != 29:
            raise ValueError(f"Expected 29 G1 body joints, got {self.dof_size_real}")
        self.default_qpos_real = bindings.default_body_qpos.astype(
            np.float32
        ).copy()
        self.joint_ids = np.asarray(
            [model.joint(name).id for name in bindings.body_joint_names],
            dtype=np.int64,
        )
        self.qadr = model.jnt_qposadr[self.joint_ids]
        self.vadr = model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = _actuators_for_joints(model, self.joint_ids)
        base = model.joint("floating_base_joint")
        self.base_qadr = int(base.qposadr[0])
        self.base_vadr = int(base.dofadr[0])

        self.qj_real = np.zeros(29, dtype=np.float32)
        self.dqj_real = np.zeros(29, dtype=np.float32)
        self.tau_real = np.zeros(29, dtype=np.float32)
        self.qj_isaac = np.zeros(29, dtype=np.float32)
        self.dqj_isaac = np.zeros(29, dtype=np.float32)
        self.tau_isaac = np.zeros(29, dtype=np.float32)
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.gyro = np.zeros(3, dtype=np.float32)
        self.linacc = np.zeros(3, dtype=np.float32)
        self.root_pos_w = np.zeros(3, dtype=np.float32)
        self.root_quat_w = self.quat.copy()
        self.root_vel_w = np.zeros(3, dtype=np.float32)
        self.root_yaw_speed = 0.0
        self._odom_origin_xy: np.ndarray | None = None
        self.latest_buttons: dict[str, Any] = {}
        self.finish_requested = False
        self.pause_reason: str | None = None
        self._previous_finish_button = False
        self._gyro_slice = self._sensor_slice("imu-pelvis-angular-velocity")
        self._linacc_slice = self._sensor_slice("imu-pelvis-linear-acceleration")
        if self._linacc_slice is None:
            raise ValueError("Scene requires imu-pelvis-linear-acceleration sensor")
        self.update()

    def _sensor_slice(self, name: str) -> slice | None:
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        if sensor_id < 0:
            return None
        start = int(self.model.sensor_adr[sensor_id])
        return slice(start, start + int(self.model.sensor_dim[sensor_id]))

    def update(self) -> None:
        self.qj_real[:] = self.data.qpos[self.qadr]
        self.dqj_real[:] = self.data.qvel[self.vadr]
        self.tau_real[:] = self.data.ctrl[self.actuator_ids]
        order = self.bindings.body_to_policy_indices
        self.qj_isaac[:] = self.qj_real[order]
        self.dqj_isaac[:] = self.dqj_real[order]
        self.tau_isaac[:] = self.tau_real[order]

        base_qpos = self.data.qpos[self.base_qadr : self.base_qadr + 7]
        base_qvel = self.data.qvel[self.base_vadr : self.base_vadr + 6]
        self.root_pos_w[:] = base_qpos[:3]
        norm = max(float(np.linalg.norm(base_qpos[3:7])), 1e-8)
        self.quat[:] = base_qpos[3:7] / norm
        self.root_quat_w[:] = self.quat
        self.root_vel_w[:] = base_qvel[:3]
        self.root_yaw_speed = float(base_qvel[5])
        self.gyro[:] = (
            self.data.sensordata[self._gyro_slice][:3]
            if self._gyro_slice is not None
            else base_qvel[3:6]
        )
        self.linacc[:] = (
            self.data.sensordata[self._linacc_slice][:3]
            if self._linacc_slice is not None
            else 0.0
        )

    def current_reference_anchor(self) -> dict[str, np.ndarray]:
        self.update()
        return {
            "joint_pos": self.qj_isaac.copy(),
            "root_pos": self.root_pos_w.copy(),
            "root_quat": self.root_quat_w.copy(),
        }

    def reset_root_xy_origin(self, reason: str = "") -> None:
        del reason
        self.update()
        self._odom_origin_xy = self.root_pos_w[:2].copy()

    def set_hand_from_controller_buttons(self, buttons: dict[str, Any]) -> None:
        self.latest_buttons = dict(buttons)
        self.hands.set_from_controller_buttons(buttons)

    def handle_vr_record_buttons(self, buttons: dict[str, Any]) -> None:
        self.latest_buttons = dict(buttons)
        pressed = bool(buttons.get("right_key_two", False))
        if pressed and not self._previous_finish_button:
            self.finish_requested = True
        self._previous_finish_button = pressed

    def notify_vr_fail_safe_pause(
        self, reason: str, action: str = "hold"
    ) -> None:
        self.pause_reason = f"{action}: {reason}"


class BodyPDController:
    """Apply MolmoSpaces deployment dynamics and 29D PD gains in lab-sim."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        bindings: MolmoTrackingBindings,
    ) -> None:
        self.model = model
        self.data = data
        self.names = bindings.body_joint_names
        self.joint_ids = np.asarray(
            [model.joint(name).id for name in self.names], dtype=np.int64
        )
        self.qadr = model.jnt_qposadr[self.joint_ids]
        self.vadr = model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = _actuators_for_joints(model, self.joint_ids)
        self.kp = bindings.body_kp.copy()
        self.kd = bindings.body_kd.copy()
        self.joint_ranges = model.jnt_range[self.joint_ids].copy()
        self.torque_low = np.full(29, -np.inf, dtype=np.float64)
        self.torque_high = np.full(29, np.inf, dtype=np.float64)
        limited = model.jnt_actfrclimited[self.joint_ids].astype(bool)
        self.torque_low[limited] = model.jnt_actfrcrange[self.joint_ids][
            limited, 0
        ]
        self.torque_high[limited] = model.jnt_actfrcrange[self.joint_ids][
            limited, 1
        ]
        self._transfer_dynamics(bindings.robot_model_path)

    def _transfer_dynamics(self, reference_xml: Path) -> None:
        reference = mujoco.MjModel.from_xml_path(str(reference_xml))
        for joint_id, name in zip(self.joint_ids, self.names):
            reference_joint = reference.joint(name)
            target_vadr = int(self.model.jnt_dofadr[joint_id])
            source_vadr = int(reference_joint.dofadr[0])
            self.model.dof_armature[target_vadr] = reference.dof_armature[
                source_vadr
            ]
            self.model.dof_damping[target_vadr] = reference.dof_damping[
                source_vadr
            ]
            self.model.dof_frictionloss[target_vadr] = reference.dof_frictionloss[
                source_vadr
            ]

    def compute(self, target: np.ndarray) -> np.ndarray:
        desired = np.asarray(target, dtype=np.float64)
        if desired.shape != (29,) or not np.isfinite(desired).all():
            raise ValueError("Body target must contain 29 finite joint positions")
        desired = np.clip(
            desired, self.joint_ranges[:, 0], self.joint_ranges[:, 1]
        )
        torque = self.kp * (desired - self.data.qpos[self.qadr])
        torque -= self.kd * self.data.qvel[self.vadr]
        return np.clip(torque, self.torque_low, self.torque_high)
