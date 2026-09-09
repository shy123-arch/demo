"""Named-joint adapter for the NVIDIA GR00T G1 walking policies.

The two ONNX policies shipped with ``decoupled_wbc`` are small lower-body
controllers.  They do *not* consume the Dex3 hand state and they do not output
arm or hand commands.  This module keeps that contract explicit while making
the controller usable with any MuJoCo model that has the original Unitree G1
joint names and one free base joint.

The observation layout and gains follow the ``g1_gear_wbc`` runner from the
SONIC/decoupled-WBC project.  The ONNX files themselves are distributed under
the NVIDIA Open Model License; this adapter does not change their weights.

Typical use at the simulation control rate (50 Hz)::

    policy = G1WalkingPolicy()
    q_target = policy.apply(model, data, velocity=(0.25, 0.0, 0.0))
    # q_target is 43 entries in ``TARGET_JOINTS`` order.

For environments whose target array follows MuJoCo actuator order, use
``policy.targets_for_model(model, q_target)`` or call ``apply(...,
output_order="model")``.  Torque mode is available with
``apply(..., mode="torque")``; it writes ``data.ctrl`` and returns the same
named 43-vector of torques.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


# This is the policy's fixed observation order.  Keep it independent of the
# MuJoCo model's qpos/actuator order: Dex3 joints are interleaved in the tree.
BODY_JOINTS: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Canonical output order for the complete G1 body + Dex3 target.  It follows
# the order used by SIMPLE's G1 wrapper (left thumb/index/middle, then right)
# rather than relying on the order in which a particular XML happens to list
# child bodies or actuators.
HAND_JOINTS: tuple[str, ...] = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

TARGET_JOINTS: tuple[str, ...] = BODY_JOINTS + HAND_JOINTS
LOWER_JOINTS: tuple[str, ...] = BODY_JOINTS[:15]
ARM_JOINTS: tuple[str, ...] = BODY_JOINTS[15:]

LOWER_NEUTRAL = np.asarray(
    (
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        -0.1,
        0.0,
        0.0,
        0.3,
        -0.2,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    dtype=np.float32,
)

# The values are copied from g1_gear_wbc.yaml.  Arm/hand gains are the same
# simple position PD used by the reference MuJoCo runner / lab environment.
LOWER_KP = np.asarray(
    (150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250),
    dtype=np.float32,
)
LOWER_KD = np.asarray(
    (2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5), dtype=np.float32
)
ARM_KP = np.full(14, 100.0, dtype=np.float32)
ARM_KD = np.full(14, 0.5, dtype=np.float32)
HAND_KP = np.full(14, 15.0, dtype=np.float32)
HAND_KD = np.full(14, 0.4, dtype=np.float32)

SINGLE_OBS_DIM = 86
OBS_HISTORY_LEN = 6
OBS_DIM = SINGLE_OBS_DIM * OBS_HISTORY_LEN
ACTION_DIM = 15
ACTION_SCALE = 0.25
CMD_SCALE = np.asarray((2.0, 2.0, 0.5), dtype=np.float32)
HEIGHT_CMD = 0.74
RPY_CMD = np.zeros(3, dtype=np.float32)


def _policy_search_dirs() -> tuple[Path, ...]:
    """Return likely locations without requiring a package install.

    The first locations are inside lab-sim, so a final checkout can vendor the
    model there.  The two workspace locations keep the adapter usable against
    the existing checkout while the asset copy is being prepared.
    """

    root = Path(__file__).resolve().parents[1]
    return (
        root / "assets" / "g1" / "policy",
        root / "assets" / "policy",
        root / "third_party" / "decoupled_wbc" / "sim2mujoco" / "resources" / "robots" / "g1" / "policy",
        Path("/mnt/workspace/Wilson/HIW-500-controoler-main/third_party/decoupled_wbc/sim2mujoco/resources/robots/g1/policy"),
        Path("/mnt/workspace/Wilson/SIMPLE/third_party/decoupled_wbc/sim2mujoco/resources/robots/g1/policy"),
    )


def _default_policy_path(name: str) -> Path:
    filename = f"GR00T-WholeBodyControl-{name}.onnx"
    for directory in _policy_search_dirs():
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    searched = "\n  ".join(str(p / filename) for p in _policy_search_dirs())
    raise FileNotFoundError(
        f"Could not find {filename}. Vendor the ONNX file under "
        "assets/g1/policy or pass an explicit policy path. Searched:\n  "
        + searched
    )


def _as_float_vector(value: Iterable[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float32)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _quat_rotate_inverse(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a world vector into the free-base frame (wxyz quaternion)."""

    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("free-base quaternion is not finite or has near-zero norm")
    w, x, y, z = q / norm
    q_conj = np.asarray((w, -x, -y, -z), dtype=np.float32)
    v = np.asarray(vector, dtype=np.float32)
    return np.asarray(
        (
            v[0] * (q_conj[0] ** 2 + q_conj[1] ** 2 - q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3])
            + v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
            v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3])
            + v[1] * (q_conj[0] ** 2 - q_conj[1] ** 2 + q_conj[2] ** 2 - q_conj[3] ** 2)
            + v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
            v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2])
            + v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1])
            + v[2] * (q_conj[0] ** 2 - q_conj[1] ** 2 - q_conj[2] ** 2 + q_conj[3] ** 2),
        ),
        dtype=np.float32,
    )


class G1WalkingPolicy:
    """GR00T lower-body walking policy with a named-joint MuJoCo adapter.

    ``apply`` is intended to run once every 20 ms (50 Hz), while MuJoCo may
    integrate at 200--500 Hz between calls.  Upper-body joints are held at
    their current positions by default because this controller has no hand or
    arm policy.  Pass ``upper_body="neutral"`` at construction, or explicit
    ``upper_target``/``hand_target`` to ``apply``, when a neutral upper body is
    desired.
    """

    body_joints = BODY_JOINTS
    hand_joints = HAND_JOINTS
    target_joints = TARGET_JOINTS

    def __init__(
        self,
        policy_path: str | Path | None = None,
        walk_policy_path: str | Path | None = None,
        *,
        balance_policy: str | Path | None = None,
        walk_policy: str | Path | None = None,
        balance_session: Any | None = None,
        walk_session: Any | None = None,
        output_order: str = "canonical",
        mode: str = "targets",
        upper_body: str = "hold",
        upper_target: Sequence[float] | None = None,
        hand_target: Sequence[float] | None = None,
        height_cmd: float = HEIGHT_CMD,
        rpy_cmd: Sequence[float] = RPY_CMD,
        cmd_scale: Sequence[float] = CMD_SCALE,
        action_scale: float = ACTION_SCALE,
        control_hz: float = 50.0,
        providers: Sequence[str] | None = None,
    ) -> None:
        if balance_policy is not None:
            policy_path = balance_policy
        if walk_policy is not None:
            walk_policy_path = walk_policy
        if policy_path is None and balance_session is None:
            policy_path = _default_policy_path("Balance")
        if walk_policy_path is None and walk_session is None:
            walk_policy_path = _default_policy_path("Walk")

        self.output_order = str(output_order).lower()
        if self.output_order not in {"canonical", "model"}:
            raise ValueError("output_order must be 'canonical' or 'model'")
        self.mode = str(mode).lower()
        if self.mode not in {"targets", "torque", "torques"}:
            raise ValueError("mode must be 'targets' or 'torque'")
        self.upper_body = str(upper_body).lower()
        if self.upper_body not in {"hold", "neutral"}:
            raise ValueError("upper_body must be 'hold' or 'neutral'")
        self.height_cmd = float(height_cmd)
        if not np.isfinite(self.height_cmd):
            raise ValueError("height_cmd must be finite")
        self.rpy_cmd = _as_float_vector(rpy_cmd, 3, "rpy_cmd")
        self.cmd_scale = _as_float_vector(cmd_scale, 3, "cmd_scale")
        self.action_scale = float(action_scale)
        if not np.isfinite(self.action_scale):
            raise ValueError("action_scale must be finite")
        self.control_hz = float(control_hz)
        if not np.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be positive and finite")

        self.arm_neutral = (
            np.zeros(14, dtype=np.float32)
            if upper_target is None
            else _as_float_vector(upper_target, 14, "upper_target")
        )
        self.hand_neutral = (
            np.zeros(14, dtype=np.float32)
            if hand_target is None
            else _as_float_vector(hand_target, 14, "hand_target")
        )

        self._providers = tuple(providers) if providers is not None else None
        self._balance = balance_session or self._load_session(policy_path, "Balance", self._providers)
        self._walk = walk_session or self._load_session(walk_policy_path, "Walk", self._providers)
        self._balance_input = self._input_name_and_validate(self._balance, "Balance")
        self._walk_input = self._input_name_and_validate(self._walk, "Walk")

        self._history: deque[np.ndarray] = deque(maxlen=OBS_HISTORY_LEN)
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._model_key: int | None = None
        self._model_info: dict[str, Any] | None = None
        self.last_observation = np.zeros(OBS_DIM, dtype=np.float32)
        self.last_target = np.zeros(len(TARGET_JOINTS), dtype=np.float32)
        self.last_torque = np.zeros(len(TARGET_JOINTS), dtype=np.float32)
        self.reset()

    @staticmethod
    def _load_session(
        path: str | Path | None,
        label: str,
        providers: Sequence[str] | None = None,
    ) -> Any:
        if path is None:
            raise ValueError(f"{label} policy path/session was not supplied")
        model_path = Path(path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(f"{label} policy does not exist: {model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:  # lazy so target-only users can import module
            raise ImportError(
                "onnxruntime is required to run G1WalkingPolicy; install it in "
                "the lab-sim environment or provide an ONNX session"
            ) from exc
        # Default ORT provider selection is most portable for this CPU policy.
        kwargs = {} if providers is None else {"providers": list(providers)}
        return ort.InferenceSession(str(model_path), **kwargs)

    @staticmethod
    def _input_name_and_validate(session: Any, label: str) -> str:
        inputs = session.get_inputs()
        if not inputs:
            raise ValueError(f"{label} policy has no ONNX inputs")
        inp = inputs[0]
        shape = getattr(inp, "shape", None)
        if shape is not None and len(shape) == 2 and shape[-1] not in (None, "None", "batch_size", "dynamic"):
            try:
                width = int(shape[-1])
            except (TypeError, ValueError):
                width = OBS_DIM
            if width != OBS_DIM:
                raise ValueError(f"{label} policy expects width {width}, adapter builds {OBS_DIM}")
        return str(inp.name)

    def reset(self) -> None:
        """Reset temporal action/history state without touching MuJoCo data."""

        self._history.clear()
        self._last_action.fill(0.0)
        self.last_observation.fill(0.0)
        self.last_target.fill(0.0)
        self.last_torque.fill(0.0)

    @property
    def action(self) -> np.ndarray:
        return self._last_action.copy()

    @property
    def neutral_pose(self) -> np.ndarray:
        """43-vector: lower-body neutral, zero arms, and zero Dex3 joints."""

        return np.concatenate((LOWER_NEUTRAL, self.arm_neutral, self.hand_neutral)).astype(
            np.float32, copy=False
        )

    def _load_model_info(self, model: Any) -> dict[str, Any]:
        if self._model_key == id(model) and self._model_info is not None:
            return self._model_info
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - MuJoCo is lab-sim's base dep
            raise ImportError("mujoco is required for G1WalkingPolicy.apply") from exc

        joint_ids: dict[str, int] = {}
        qadr: dict[str, int] = {}
        vadr: dict[str, int] = {}
        actuator_ids: dict[str, int | None] = {}
        for name in TARGET_JOINTS:
            jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
            if jid < 0:
                raise ValueError(f"model is missing required G1 joint {name!r}")
            if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError(f"required G1 joint {name!r} unexpectedly has free type")
            # The target is a scalar hinge in the G1/Dex3 model.  Rejecting
            # multi-DoF joints here prevents silent address corruption.
            if int(model.jnt_type[jid]) not in {
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            }:
                raise ValueError(f"joint {name!r} is not a scalar hinge/slide joint")
            joint_ids[name] = jid
            qadr[name] = int(model.jnt_qposadr[jid])
            vadr[name] = int(model.jnt_dofadr[jid])
            aid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
            if aid < 0:
                # A model may expose a valid target-only model without motors.
                # Torque mode will report this clearly; target mode still works.
                matches = np.flatnonzero(np.asarray(model.actuator_trnid[:, 0]) == jid)
                aid = int(matches[0]) if matches.size else -1
            actuator_ids[name] = None if aid < 0 else aid

        free_ids = np.flatnonzero(
            np.asarray(model.jnt_type, dtype=np.int32)
            == int(mujoco.mjtJoint.mjJNT_FREE)
        )
        # Scenes commonly contain free props (bottles, boxes, etc.).  Prefer
        # the canonical G1 name and then a free joint on the pelvis body; only
        # fall back to the sole free joint when the model is robot-only.
        named_free = int(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        )
        if named_free in set(int(i) for i in free_ids):
            free_id = named_free
        else:
            pelvis_free = [
                int(i)
                for i in free_ids
                if str(model.body(int(model.jnt_bodyid[i])).name).lower() in {"pelvis", "g1", "robot"}
            ]
            if len(pelvis_free) == 1:
                free_id = pelvis_free[0]
            elif free_ids.size == 1:
                free_id = int(free_ids[0])
            else:
                raise ValueError(
                    "G1WalkingPolicy could not identify the free G1 base joint; "
                    f"model has free joint ids {free_ids.tolist()}"
                )
        if free_id < 0:
            raise ValueError(
                "G1WalkingPolicy requires one free G1 base joint"
            )
        target_joint_by_id = {jid: name for name, jid in joint_ids.items()}
        model_actuator_names = tuple(
            target_joint_by_id[int(model.actuator_trnid[i, 0])]
            for i in range(int(model.nu))
            if int(model.actuator_trnid[i, 0]) in target_joint_by_id
        )
        info = {
            "joint_ids": joint_ids,
            "qadr": qadr,
            "vadr": vadr,
            "actuator_ids": actuator_ids,
            "free_qadr": int(model.jnt_qposadr[free_id]),
            "free_vadr": int(model.jnt_dofadr[free_id]),
            "model_actuator_names": model_actuator_names,
        }
        self._model_key = id(model)
        self._model_info = info
        return info

    @staticmethod
    def _check_data(data: Any, model: Any) -> None:
        qpos = np.asarray(data.qpos)
        qvel = np.asarray(data.qvel)
        if qpos.ndim != 1 or qpos.shape[0] != int(model.nq):
            raise ValueError(f"data.qpos has shape {qpos.shape}, model.nq={model.nq}")
        if qvel.ndim != 1 or qvel.shape[0] != int(model.nv):
            raise ValueError(f"data.qvel has shape {qvel.shape}, model.nv={model.nv}")
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("MuJoCo state contains non-finite qpos/qvel")

    def observation(self, model: Any, data: Any, velocity: Sequence[float]) -> np.ndarray:
        """Build one 86-D policy observation without changing history."""

        info = self._load_model_info(model)
        self._check_data(data, model)
        command = _as_float_vector(velocity, 3, "velocity")
        qpos = np.asarray(data.qpos)
        qvel = np.asarray(data.qvel)
        qj = np.asarray([qpos[info["qadr"][n]] for n in BODY_JOINTS], dtype=np.float32)
        dqj = np.asarray([qvel[info["vadr"][n]] for n in BODY_JOINTS], dtype=np.float32)
        base_qadr = info["free_qadr"]
        base_vadr = info["free_vadr"]
        quat = qpos[base_qadr + 3 : base_qadr + 7]
        omega = qvel[base_vadr + 3 : base_vadr + 6]

        single = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        single[0:3] = command * self.cmd_scale
        single[3] = self.height_cmd
        single[4:7] = self.rpy_cmd
        single[7:10] = np.asarray(omega, dtype=np.float32) * 0.5
        single[10:13] = _quat_rotate_inverse(quat, np.asarray((0.0, 0.0, -1.0), dtype=np.float32))
        defaults = np.zeros(29, dtype=np.float32)
        defaults[:15] = LOWER_NEUTRAL
        single[13:42] = (qj - defaults)
        single[42:71] = dqj * 0.05
        single[71:86] = self._last_action
        if not np.isfinite(single).all():
            raise ValueError("constructed policy observation is non-finite")
        return single

    def _infer(self, obs: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        history_obs = self._history
        history_obs.append(obs)
        batch = np.concatenate(tuple(history_obs), axis=0).astype(np.float32, copy=False)
        # A deque starts empty after reset, so pad on the left exactly as the
        # reference runner does for the first five calls.
        if batch.shape != (OBS_DIM,):
            padded = np.zeros(OBS_DIM, dtype=np.float32)
            n = len(history_obs)
            padded[-n * SINGLE_OBS_DIM :] = batch
            batch = padded
        self.last_observation = batch.copy()
        walking = float(np.linalg.norm(velocity)) > 0.05
        session = self._walk if walking else self._balance
        input_name = self._walk_input if walking else self._balance_input
        result = session.run(None, {input_name: batch[None, :]})
        if not result:
            raise ValueError("walking policy returned no outputs")
        action = np.asarray(result[0], dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(
                f"walking policy must return 15 finite values, got shape {action.shape}"
            )
        self._last_action = action.copy()
        return action

    def _targets_by_name(
        self,
        model: Any,
        data: Any,
        lower_target: np.ndarray,
        upper_target: Sequence[float] | None,
        hand_target: Sequence[float] | None,
    ) -> dict[str, float]:
        info = self._load_model_info(model)
        qpos = np.asarray(data.qpos)
        target = {}
        if self.upper_body == "hold":
            target.update({name: float(qpos[info["qadr"][name]]) for name in ARM_JOINTS + HAND_JOINTS})
        else:
            target.update({name: float(v) for name, v in zip(ARM_JOINTS, self.arm_neutral)})
            target.update({name: float(v) for name, v in zip(HAND_JOINTS, self.hand_neutral)})
        if upper_target is not None:
            arm = _as_float_vector(upper_target, 14, "upper_target")
            target.update({name: float(v) for name, v in zip(ARM_JOINTS, arm)})
        if hand_target is not None:
            hand = _as_float_vector(hand_target, 14, "hand_target")
            target.update({name: float(v) for name, v in zip(HAND_JOINTS, hand)})
        target.update({name: float(v) for name, v in zip(LOWER_JOINTS, lower_target)})
        return target

    def _ordered_vector(self, model: Any, values: Mapping[str, float], order: str) -> np.ndarray:
        if order == "canonical":
            names = TARGET_JOINTS
        elif order == "model":
            names = self._load_model_info(model)["model_actuator_names"]
            if len(names) != len(TARGET_JOINTS):
                raise ValueError(
                    "model output order requested, but model has "
                    f"{len(names)} named G1 actuators; expected {len(TARGET_JOINTS)}"
                )
        else:
            raise ValueError("output_order must be 'canonical' or 'model'")
        return np.asarray([values[name] for name in names], dtype=np.float32)

    def targets_for_model(self, model: Any, target: Sequence[float]) -> np.ndarray:
        """Convert a canonical 43-vector to model actuator order by name."""

        vector = _as_float_vector(target, len(TARGET_JOINTS), "target")
        values = dict(zip(TARGET_JOINTS, vector))
        return self._ordered_vector(model, values, "model")

    def _torques_by_name(self, model: Any, data: Any, values: Mapping[str, float]) -> dict[str, float]:
        info = self._load_model_info(model)
        qpos = np.asarray(data.qpos)
        qvel = np.asarray(data.qvel)
        torques: dict[str, float] = {}
        for i, name in enumerate(LOWER_JOINTS):
            qa, va = info["qadr"][name], info["vadr"][name]
            torques[name] = float(LOWER_KP[i] * (values[name] - qpos[qa]) - LOWER_KD[i] * qvel[va])
        for i, name in enumerate(ARM_JOINTS):
            qa, va = info["qadr"][name], info["vadr"][name]
            torques[name] = float(ARM_KP[i] * (values[name] - qpos[qa]) - ARM_KD[i] * qvel[va])
        for i, name in enumerate(HAND_JOINTS):
            qa, va = info["qadr"][name], info["vadr"][name]
            torques[name] = float(HAND_KP[i] * (values[name] - qpos[qa]) - HAND_KD[i] * qvel[va])

        # Clip only when the named actuator exposes a finite force limit.  A
        # motor without an explicit limit may still inherit the Unitree joint
        # ``actuatorfrcrange``; use that as the fallback.
        for name, aid in info["actuator_ids"].items():
            jid = info["joint_ids"][name]
            limited = False
            limit = None
            try:
                joint_limited = bool(np.asarray(model.jnt_actfrclimited)[jid])
                joint_limit = np.asarray(model.jnt_actfrcrange)[jid]
                if joint_limited and np.isfinite(joint_limit).all():
                    limited, limit = True, joint_limit
            except (AttributeError, IndexError):
                pass
            if not limited and aid is not None:
                try:
                    actuator_limited = bool(np.asarray(model.actuator_forcelimited)[aid])
                    actuator_limit = np.asarray(model.actuator_forcerange)[aid]
                    if actuator_limited and np.isfinite(actuator_limit).all():
                        limited, limit = True, actuator_limit
                except (AttributeError, IndexError):
                    pass
            if limited and limit is not None and limit[0] <= limit[1]:
                torques[name] = float(np.clip(torques[name], limit[0], limit[1]))
        return torques

    def apply(
        self,
        model: Any,
        data: Any,
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        mode: str | None = None,
        torque: bool | None = None,
        output_order: str | None = None,
        upper_target: Sequence[float] | None = None,
        hand_target: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Run one policy update and return targets or torques.

        Args:
            model, data: Matching MuJoCo model/state objects.
            velocity: Body-frame ``(vx, vy, yaw_rate)`` command in SI units.
            mode: ``"targets"`` (default) or ``"torque"``.  If omitted, the
                constructor's mode is used.
            torque: Boolean shorthand for ``mode="torque"``.
            output_order: ``"canonical"`` (fixed 43-name order) or ``"model"``
                (the model's named actuator order).
            upper_target: Optional 14-arm override.
            hand_target: Optional 14-hand override.

        In torque mode, ``data.ctrl`` is updated for every named actuator and
        the returned vector contains the same torques in the selected order.
        """

        command = _as_float_vector(velocity, 3, "velocity")
        info = self._load_model_info(model)
        self._check_data(data, model)
        obs = self.observation(model, data, command)
        action = self._infer(obs, command)
        lower_target = action * self.action_scale + LOWER_NEUTRAL
        target_values = self._targets_by_name(model, data, lower_target, upper_target, hand_target)

        selected_mode = self.mode if mode is None else str(mode).lower()
        if torque is not None:
            selected_mode = "torque" if torque else "targets"
        if selected_mode not in {"targets", "torque", "torques"}:
            raise ValueError("mode must be 'targets' or 'torque'")
        order = self.output_order if output_order is None else str(output_order).lower()
        target = self._ordered_vector(model, target_values, "canonical")
        self.last_target = target.copy()
        if selected_mode in {"targets"}:
            return target if order == "canonical" else self._ordered_vector(model, target_values, "model")

        torque_values = self._torques_by_name(model, data, target_values)
        self.last_torque = self._ordered_vector(model, torque_values, "canonical")
        missing = [name for name, aid in info["actuator_ids"].items() if aid is None]
        if missing:
            raise ValueError("torque mode requires actuators for: " + ", ".join(missing))
        for name, aid in info["actuator_ids"].items():
            data.ctrl[aid] = torque_values[name]
        return self.last_torque.copy() if order == "canonical" else self._ordered_vector(model, torque_values, "model")

    def apply_torque(self, model: Any, data: Any, velocity: Sequence[float] = (0.0, 0.0, 0.0), **kwargs: Any) -> np.ndarray:
        """Convenience alias for ``apply(..., mode='torque')``."""

        kwargs["mode"] = "torque"
        return self.apply(model, data, velocity, **kwargs)


# Small aliases keep integration code readable and preserve likely import
# spellings without duplicating implementation.
G1LocomotionPolicy = G1WalkingPolicy
WalkingPolicy = G1WalkingPolicy
LocomotionPolicy = G1WalkingPolicy

__all__ = [
    "ARM_JOINTS",
    "BODY_JOINTS",
    "G1LocomotionPolicy",
    "G1WalkingPolicy",
    "HAND_JOINTS",
    "LOWER_JOINTS",
    "LocomotionPolicy",
    "TARGET_JOINTS",
    "WalkingPolicy",
]
