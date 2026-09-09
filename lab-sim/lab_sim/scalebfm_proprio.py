"""ScaleBFM controlled by joint encoders and a pelvis IMU only.

Environment state is deliberately absent from this module. Camera-derived
commands use an estimated frame whose origin is the pelvis at reset, whose
vertical is the IMU inertial vertical, and whose initial heading is zero. The
pelvis translation estimate anchors both ankle frames using robot kinematics;
it assumes that both feet remain planted. It cannot recover foot slip or walk.

Sensor joint arrays follow ``metadata['joint_names']``. IMU orientation is wxyz
in its inertial frame; gyro is angular velocity in the pelvis/body frame.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import time

import numpy as np
import torch

from .vendor.scalebfm_network import HumanoidTransformer, TaskEmbedder
from .vendor.scalebfm_export import (
    HumanoidTransformerPolicyWrapperWithMode, build_mode_mappings, parse_xml,
    quat_apply_inverse, quat_mul_inverse_left,
)

ASSETS = Path(__file__).resolve().parents[1] / "assets/g1/policy/scalebfm_m"
FIVE_BODIES = ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link",
               "left_ankle_roll_link", "right_ankle_roll_link")


def _quat_mul(a, b):
    """Hamilton product of broadcastable wxyz arrays."""
    a, b = np.broadcast_arrays(a, b)
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate((aw * bw - np.sum(av * bv, axis=-1, keepdims=True),
                           aw * bv + bw * av + np.cross(av, bv)), axis=-1)


def _quat_apply(quat, vector):
    t = 2 * np.cross(quat[..., 1:], vector)
    return vector + quat[..., :1] * t + np.cross(quat[..., 1:], t)


class ProprioScaleBFM:
    """Official 50 Hz five-point policy with a stationary-feet state estimate.

    Call ``reset`` once, then alternate ``infer`` and ``update``. ``infer``
    returns desired joint positions and the policy's raw action, both in
    metadata ``action_names`` order. It retains the action for the next sensor
    update so observation/action history matches the original exporter.
    Neither method accepts a simulator object or global translation.
    """

    def __init__(self, device="cpu", assets=ASSETS):
        self.assets = Path(assets)
        self.metadata = json.loads((self.assets / "metadata.json").read_text())
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        meta = self.metadata
        arch = meta["policy_architecture"]
        actor = HumanoidTransformer(
            arch["prop_obs_dim"], arch["action_dim"], arch["output_dim"],
            arch["embedding_dim"], arch["num_heads"], arch["ff_dim"], arch["num_layers"],
        )
        embedder = TaskEmbedder(arch["task_obs_dim"], arch["embedding_dim"],
                                arch["reduced_task_dim"], arch["task_embedder_hidden_dims"])
        state = torch.load(self.assets / "model_22200.pt", map_location="cpu",
                           weights_only=True)["model_state_dict"]
        actor.load_state_dict({k[len("actor."):]: v for k, v in state.items()
                               if k.startswith("actor.")}, strict=True)
        embedder.load_state_dict({k[len("actor_task_embedder."):]: v for k, v in state.items()
                                  if k.startswith("actor_task_embedder.")}, strict=True)
        mode_table = torch.load(self.assets / "mode_table.pt", map_location=self.device,
                                weights_only=True)
        bodies, joints, parents, axes, translations, rotations = parse_xml(
            self.assets / "kinematics.xml", self.device)
        rotations = rotations / rotations.norm(dim=-1, keepdim=True)
        self.names = list(meta["joint_names"])
        self.action_names = list(meta["action_names"])
        self.selected_names = list(meta["selected_body_names"])
        self.five_indices = [self.selected_names.index(n) for n in FIVE_BODIES]
        if set(torch.nonzero(mode_table[4]).flatten().tolist()) != set(self.five_indices):
            raise ValueError("ScaleBFM mode 4 does not match the five controlled bodies")
        self.default = np.asarray(meta["default_dof_pos"], dtype=float)
        self.kp = np.asarray(meta["stiffness"], dtype=float)
        self.kd = np.asarray(meta["damping"], dtype=float)
        self.limits = np.asarray(meta["torque_limit"], dtype=float)
        joint_indices = [self.names.index(n) for n in joints]
        selected_indices = [bodies.index(n) for n in self.selected_names]
        self.policy = HumanoidTransformerPolicyWrapperWithMode(
            SimpleNamespace(actor=actor, actor_task_embedder=embedder),
            build_mode_mappings(mode_table, meta["mode_feature_dims"], True), mode_table,
            self.tensor(self.default)[None], self.tensor(meta["action_scale"])[None],
            meta["history_buffer_size"], len(meta["future_idx"]),
            translations, rotations, parents, axes,
            torch.tensor(selected_indices, device=self.device),
            torch.tensor(joint_indices, device=self.device),
        ).to(self.device).eval()
        self.mode = torch.tensor([4], dtype=torch.long, device=self.device)
        self.offsets = np.asarray(meta["future_idx"], dtype=int)
        self.time_offsets = torch.as_tensor(self.offsets, dtype=torch.long,
                                            device=self.device)[None, :, None]
        self._body_names = bodies
        self._parents = parents.cpu().numpy()
        self._axes = axes.cpu().numpy().astype(float)
        self._translations = translations.cpu().numpy().astype(float)
        self._rotations = rotations.cpu().numpy().astype(float)
        self._rotations /= np.linalg.norm(self._rotations, axis=-1, keepdims=True)
        self._joint_indices = np.asarray(joint_indices)
        self._selected_indices = np.asarray(selected_indices)
        self._five_body_indices = np.asarray([bodies.index(n) for n in FIVE_BODIES])
        self._ankle_indices = self._five_body_indices[3:]
        self._torso_index = bodies.index("torso_link")
        self.history = None
        self.inference_ms = []

    def tensor(self, value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)

    @staticmethod
    def _sensor_array(value, shape, name):
        result = np.asarray(value, dtype=float)
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite array with shape {shape}")
        return result.copy()

    def _read_sensors(self, joint_pos, joint_vel, imu_wxyz, gyro):
        n = len(self.names)
        pos = self._sensor_array(joint_pos, (n,), "joint_pos")
        vel = self._sensor_array(joint_vel, (n,), "joint_vel")
        imu = self._sensor_array(imu_wxyz, (4,), "imu_wxyz")
        angular_vel = self._sensor_array(gyro, (3,), "gyro")
        norm = np.linalg.norm(imu)
        if not np.isclose(norm, 1, atol=1e-3):
            raise ValueError("imu_wxyz must be normalized")
        return pos, vel, imu / norm, angular_vel

    def _forward_kinematics(self, joint_pos):
        """Robot-model FK relative to pelvis, with no root/world pose input."""
        positions = np.zeros((len(self._body_names), 3))
        rotations = np.zeros((len(self._body_names), 4))
        rotations[:, 0] = 1
        half = joint_pos[self._joint_indices] / 2
        joint_rot = np.concatenate((np.cos(half)[:, None],
                                    self._axes * np.sin(half)[:, None]), axis=-1)
        for body in range(1, len(self._body_names)):
            parent = self._parents[body]
            positions[body] = positions[parent] + _quat_apply(
                rotations[parent], self._translations[body - 1])
            rotations[body] = _quat_mul(rotations[parent], _quat_mul(
                self._rotations[body - 1], joint_rot[body - 1]))
        return positions, rotations

    def _set_sensor_state(self, sensors, *, initial=False):
        self.joint_pos, self.joint_vel, imu, self.gyro = sensors
        orientation = _quat_mul(self._inertial_to_initial, imu)
        if not initial and np.dot(orientation, self.root_quaternion) < 0:
            orientation = -orientation
        self.root_quaternion = orientation / np.linalg.norm(orientation)
        local_pos, local_quat = self._forward_kinematics(self.joint_pos)
        rotated_pos = _quat_apply(self.root_quaternion, local_pos)
        if initial:
            self.root_position = np.zeros(3)
            self._foot_anchors = rotated_pos[self._ankle_indices].copy()
        else:
            # Least-squares translation satisfying both stationary ankle anchors.
            self.root_position = np.mean(
                self._foot_anchors - rotated_pos[self._ankle_indices], axis=0)
        self._body_positions = rotated_pos + self.root_position
        self._body_quaternions = _quat_mul(self.root_quaternion, local_quat)
        self.foot_anchor_residual = np.linalg.norm(
            self._body_positions[self._ankle_indices] - self._foot_anchors, axis=1)

    def reset(self, joint_pos, joint_vel, imu_wxyz, gyro):
        """Anchor both feet and initialize a translation-free local frame."""
        sensors = self._read_sensors(joint_pos, joint_vel, imu_wxyz, gyro)
        w, x, y, z = sensors[2]
        initial_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self._inertial_to_initial = np.array(
            [np.cos(initial_yaw / 2), 0, 0, -np.sin(initial_yaw / 2)])
        self._set_sensor_state(sensors, initial=True)
        self.initial_pos = self._body_positions[self._selected_indices].copy()
        self.initial_quat = self._body_quaternions[self._selected_indices].copy()
        self.previous_action = np.zeros(len(self.action_names))
        h = self.metadata["history_buffer_size"]
        self.history = [self.tensor(a)[None, None].repeat(1, h, 1) for a in (
            self.root_quaternion, self.gyro, self.joint_pos, self.joint_vel,
            self.previous_action,
        )]
        self.inference_ms = []
        return self.poses()

    def update(self, joint_pos, joint_vel, imu_wxyz, gyro):
        """Consume one 50 Hz encoder/IMU sample and the preceding raw action."""
        self._require_reset()
        self._set_sensor_state(self._read_sensors(joint_pos, joint_vel, imu_wxyz, gyro))
        for i, value in enumerate((self.root_quaternion, self.gyro, self.joint_pos,
                                   self.joint_vel, self.previous_action)):
            self.history[i] = torch.cat(
                (self.history[i][:, 1:], self.tensor(value)[None, None]), dim=1)

    def _require_reset(self):
        if self.history is None:
            raise RuntimeError("Call reset with encoder and IMU measurements first")

    def poses(self):
        """Return pelvis, left/right wrists and left/right ankles, in that order."""
        self._require_reset()
        return (self._body_positions[self._five_body_indices].copy(),
                self._body_quaternions[self._five_body_indices].copy())

    def torso_pose(self):
        """Estimated torso pose for a fixed robot-mounted camera calibration."""
        self._require_reset()
        return (self._body_positions[self._torso_index].copy(),
                self._body_quaternions[self._torso_index].copy())

    def infer(self, positions, quaternions):
        """Infer from six future frames of five poses in the estimated frame."""
        self._require_reset()
        pos, quat = np.asarray(positions), np.asarray(quaternions)
        if pos.shape != (6, 5, 3) or quat.shape != (6, 5, 4):
            raise ValueError("Expected positions (6,5,3) and wxyz quaternions (6,5,4)")
        if not np.isfinite(pos).all() or not np.isfinite(quat).all():
            raise ValueError("Target poses must be finite")
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-3):
            raise ValueError("Target quaternions must be normalized")
        n_future, n_selected = len(self.offsets), len(self.selected_names)
        target_pos = np.broadcast_to(self.initial_pos, (n_future, n_selected, 3)).copy()
        target_quat = np.broadcast_to(self.initial_quat, (n_future, n_selected, 4)).copy()
        target_pos[:, self.five_indices] = pos
        target_quat[:, self.five_indices] = quat
        root_quat = self.history[0][:, -1, None, None].expand(1, n_future, n_selected, 4)
        target_local_pos = quat_apply_inverse(root_quat,
            self.tensor(target_pos - self.root_position)[None])
        target_local_quat = quat_mul_inverse_left(root_quat, self.tensor(target_quat)[None])
        started = time.perf_counter()
        with torch.inference_mode():
            target, raw_action = self.policy(*self.history, target_local_pos,
                                             target_local_quat, self.mode, self.time_offsets)
        self.inference_ms.append(1000 * (time.perf_counter() - started))
        desired = target[0].cpu().numpy().copy()
        raw = raw_action[0].cpu().numpy().copy()
        if not np.isfinite(desired).all() or not np.isfinite(raw).all():
            raise RuntimeError("ScaleBFM produced a non-finite action")
        self.previous_action = raw.copy()
        return desired, raw
