#!/usr/bin/env python3
"""
Low-latency PICO/XRobot teleop bridge for MolmoSpaces.

Architecture:
1. XR callback thread stores the latest VR snapshot with a monotonic timestamp.
2. A retarget thread runs at a fixed rate and only retargets the latest snapshot.
3. A request thread serves the newest ZMQ request using time-based interpolation over a
   short retarget history buffer.
4. A control thread publishes controller buttons at a fixed rate.
5. Optionally, a UDP latest-state stream publishes the same pose/control data without
   TCP head-of-line blocking.
"""

import argparse
import json
import multiprocessing as mp
import os
import signal
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from teleop.default_pose import DEFAULT_MIMIC_OBS

GMR = None
RobotMotionViewer = None
quat_mul_np = None
xrt = None


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _enable_debug_log_file(path: str) -> None:
    if not str(path).strip():
        return
    log_path = Path(path).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"[DebugLog] tee stdout/stderr -> {log_path}")

XR_BODY_JOINT_NAMES = [
    "Pelvis",
    "Left_Hip",
    "Right_Hip",
    "Spine1",
    "Left_Knee",
    "Right_Knee",
    "Spine2",
    "Left_Ankle",
    "Right_Ankle",
    "Spine3",
    "Left_Foot",
    "Right_Foot",
    "Neck",
    "Left_Collar",
    "Right_Collar",
    "Head",
    "Left_Shoulder",
    "Right_Shoulder",
    "Left_Elbow",
    "Right_Elbow",
    "Left_Wrist",
    "Right_Wrist",
    "Left_Hand",
    "Right_Hand",
]

G1_DOF_JOINT_NAMES = [
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
]

RECORDER_BUTTON_NAMES = [
    "left_key_one",
    "left_key_two",
    "left_axis_click",
    "left_index_trig",
    "left_grip",
    "right_key_one",
    "right_key_two",
    "right_axis_click",
    "right_index_trig",
    "right_grip",
]

RECORDER_VALUE_NAMES = [
    "left_trigger_value",
    "left_grip_value",
    "left_axis_x",
    "left_axis_y",
    "right_trigger_value",
    "right_grip_value",
    "right_axis_x",
    "right_axis_y",
]


def _load_runtime_dependencies() -> None:
    global GMR, RobotMotionViewer, quat_mul_np, xrt

    try:
        from general_motion_retargeting import GeneralMotionRetargeting as _GMR
        from general_motion_retargeting import RobotMotionViewer as _RobotMotionViewer
        from general_motion_retargeting.rot_utils import quat_mul_np as _quat_mul_np
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'general_motion_retargeting'. Install GMR in the active Python environment."
        ) from exc

    try:
        import xrobotoolkit_sdk as _xrt
    except ImportError as exc:
        raise ImportError(
            "Failed to import 'xrobotoolkit_sdk'. Install the patched SDK in the active Python environment."
        ) from exc

    for name in (
        "register_frame_callback",
        "clear_frame_callback",
        "has_frame_callback",
    ):
        if not hasattr(_xrt, name):
            raise ImportError(
                "Installed xrobotoolkit_sdk does not expose callback APIs. "
                "Reinstall the patched XRoboToolkit-PC-Service-Pybind build."
            )

    GMR = _GMR
    RobotMotionViewer = _RobotMotionViewer
    quat_mul_np = _quat_mul_np
    xrt = _xrt


@dataclass
class RetargetedFrame:
    recv_ns: int
    qpos: np.ndarray


def _pose7_to_rotation_xyzw(pose: Any) -> R:
    if not isinstance(pose, (list, tuple)) or len(pose) < 7:
        raise ValueError(f"invalid pose payload: {pose!r}")
    quat_xyzw = np.asarray([float(v) for v in pose[3:7]], dtype=np.float64)
    return R.from_quat(quat_xyzw)


def _forward_axis_vector(name: str) -> np.ndarray:
    axis_map = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "-y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "-z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    }
    return axis_map[name]


def _compute_head_pose_relative_to_pelvis(snapshot: dict[str, Any], head_forward_axis: str) -> tuple[float, float]:
    body = snapshot.get("body", {})
    poses = body.get("poses") if isinstance(body, dict) else None
    if not isinstance(poses, (list, tuple)) or not poses:
        raise ValueError("body poses are not available")

    headset_pose = snapshot.get("headset_pose")
    if not isinstance(headset_pose, (list, tuple)) or len(headset_pose) < 7:
        raise ValueError("headset pose is not available")

    pelvis_rot = _pose7_to_rotation_xyzw(poses[0])
    head_rot = _pose7_to_rotation_xyzw(headset_pose)
    rel_rot = pelvis_rot.inv() * head_rot
    forward_local = rel_rot.apply(_forward_axis_vector(head_forward_axis))

    yaw_deg = float(np.degrees(np.arctan2(forward_local[0], forward_local[2])))
    pitch_deg = float(
        np.degrees(
            np.arctan2(
                -forward_local[1],
                np.linalg.norm(forward_local[[0, 2]]) + 1e-8,
            )
        )
    )
    return yaw_deg, pitch_deg


class PicoRawRecorder:
    def __init__(
        self,
        output_dir: str,
        prefix: str,
        gmr_output_dir: str,
        gmr_prefix: str,
        auto_start: bool,
        button_control: bool,
        include_snapshot_json: bool,
        start_button: str,
        stop_button: str,
        record_config: Dict[str, Any],
    ):
        self.output_dir = Path(output_dir).expanduser()
        self.prefix = str(prefix).strip() or "pico_raw"
        self.gmr_output_dir = Path(gmr_output_dir).expanduser() if str(gmr_output_dir).strip() else None
        self.gmr_prefix = str(gmr_prefix).strip() or "gmr_mt"
        self.session_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.button_control = bool(button_control)
        self.include_snapshot_json = bool(include_snapshot_json)
        self.start_button = str(start_button)
        self.stop_button = str(stop_button)
        self.record_config = dict(record_config)
        self.lock = threading.Lock()
        self.recording = False
        self.prev_start_button = False
        self.prev_stop_button = False
        self.prev_quit_combo = False
        self.saved_count = 0
        self.segment_start_monotonic: Optional[float] = None

        self.raw_seq: list[int] = []
        self.raw_recv_ns: list[int] = []
        self.raw_motion_timestamp_ns: list[int] = []
        self.raw_body_timestamp_ns: list[int] = []
        self.raw_top_timestamp_ns: list[int] = []
        self.raw_body_poses: list[np.ndarray] = []
        self.raw_headset_pose: list[np.ndarray] = []
        self.raw_left_controller_pose: list[np.ndarray] = []
        self.raw_right_controller_pose: list[np.ndarray] = []
        self.raw_controller_buttons: list[np.ndarray] = []
        self.raw_controller_values: list[np.ndarray] = []
        self.raw_snapshot_json: list[str] = []
        self.gmr_seq: list[int] = []
        self.gmr_recv_ns: list[int] = []
        self.gmr_qpos: list[np.ndarray] = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.gmr_output_dir is not None:
            self.gmr_output_dir.mkdir(parents=True, exist_ok=True)
        if auto_start:
            self.start()

    @staticmethod
    def _none_to_i64(value: Optional[int]) -> int:
        return -1 if value is None else int(value)

    @staticmethod
    def _pose7_from_any(value: Any) -> np.ndarray:
        out = np.full((7,), np.nan, dtype=np.float32)
        if isinstance(value, dict):
            for key in ("pose", "controller_pose", "tracking_pose"):
                if key in value:
                    value = value[key]
                    break
        if isinstance(value, (list, tuple)) and len(value) >= 7:
            try:
                out[:] = np.asarray(value[:7], dtype=np.float32)
            except Exception:
                pass
        return out

    @staticmethod
    def _body_poses_from_any(value: Any) -> np.ndarray:
        out = np.full((len(XR_BODY_JOINT_NAMES), 7), np.nan, dtype=np.float32)
        if not isinstance(value, (list, tuple)):
            return out
        count = min(len(value), len(XR_BODY_JOINT_NAMES))
        for idx in range(count):
            pose = value[idx]
            if not isinstance(pose, (list, tuple)) or len(pose) < 7:
                continue
            try:
                out[idx] = np.asarray(pose[:7], dtype=np.float32)
            except Exception:
                continue
        return out

    @staticmethod
    def _controller_pose(snapshot: Any, side: str) -> np.ndarray:
        if not isinstance(snapshot, dict):
            return np.full((7,), np.nan, dtype=np.float32)

        controllers = snapshot.get("controllers", {})
        if isinstance(controllers, dict):
            controller = controllers.get(side, {})
            pose = PicoRawRecorder._pose7_from_any(controller)
            if np.isfinite(pose).any():
                return pose

        for key in (
            f"{side}_controller_pose",
            f"{side}_controller",
            f"{side}_hand_pose",
        ):
            if key in snapshot:
                pose = PicoRawRecorder._pose7_from_any(snapshot[key])
                if np.isfinite(pose).any():
                    return pose

        return np.full((7,), np.nan, dtype=np.float32)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): PicoRawRecorder._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PicoRawRecorder._json_safe(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _controller_buttons_array(controller_buttons: Dict[str, Any]) -> np.ndarray:
        return np.asarray([bool(controller_buttons.get(name, False)) for name in RECORDER_BUTTON_NAMES], dtype=bool)

    @staticmethod
    def _controller_values_array(controller_buttons: Dict[str, Any]) -> np.ndarray:
        left_axis = controller_buttons.get("left_axis", [0.0, 0.0])
        right_axis = controller_buttons.get("right_axis", [0.0, 0.0])

        def _axis_value(axis: Any, idx: int) -> float:
            if isinstance(axis, (list, tuple)) and len(axis) > idx:
                try:
                    return float(axis[idx])
                except Exception:
                    return 0.0
            return 0.0

        values = [
            float(controller_buttons.get("left_trigger_value", 0.0)),
            float(controller_buttons.get("left_grip_value", 0.0)),
            _axis_value(left_axis, 0),
            _axis_value(left_axis, 1),
            float(controller_buttons.get("right_trigger_value", 0.0)),
            float(controller_buttons.get("right_grip_value", 0.0)),
            _axis_value(right_axis, 0),
            _axis_value(right_axis, 1),
        ]
        return np.asarray(values, dtype=np.float32)

    def _next_output_path(self, output_dir: Path, prefix: str) -> Path:
        name_parts = [prefix, self.session_id]
        name_parts.append(f"seg{self.saved_count:06d}")
        return output_dir / ("_".join(name_parts) + ".npz")

    def _clear_locked(self) -> None:
        self.raw_seq.clear()
        self.raw_recv_ns.clear()
        self.raw_motion_timestamp_ns.clear()
        self.raw_body_timestamp_ns.clear()
        self.raw_top_timestamp_ns.clear()
        self.raw_body_poses.clear()
        self.raw_headset_pose.clear()
        self.raw_left_controller_pose.clear()
        self.raw_right_controller_pose.clear()
        self.raw_controller_buttons.clear()
        self.raw_controller_values.clear()
        self.raw_snapshot_json.clear()
        self.gmr_seq.clear()
        self.gmr_recv_ns.clear()
        self.gmr_qpos.clear()

    @staticmethod
    def _button_label(button: str) -> str:
        labels = {
            "left_key_one": "left X",
            "left_key_two": "left Y",
            "left_axis_click": "left stick click",
            "left_index_trig": "left trigger",
            "left_grip": "left grip",
            "right_key_one": "right A",
            "right_key_two": "right B",
            "right_axis_click": "right stick click",
            "right_index_trig": "right trigger",
            "right_grip": "right grip",
        }
        return labels.get(button, button)

    def print_ready(self, keyboard_control: bool, status_interval_s: float) -> None:
        status = self.status_snapshot()
        state = "recording" if status["recording"] else "idle"
        button_msg = "disabled"
        if self.button_control:
            button_msg = (
                f"{self._button_label(self.start_button)}=start, "
                f"{self._button_label(self.stop_button)}=stop/save, "
                "left Y+right B=save+quit"
            )
        keyboard_msg = "r=start, p=stop/save, q=save+quit" if keyboard_control else "disabled"
        status_msg = "off" if status_interval_s <= 0.0 else f"every {status_interval_s:g}s while recording"
        print(
            "[Recorder] ready | "
            f"state={state} saved={int(status['saved'])} dir={self.output_dir} "
            f"buttons=({button_msg}) keyboard=({keyboard_msg}) status={status_msg}"
        )

    def status_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            elapsed_s = 0.0
            if self.recording and self.segment_start_monotonic is not None:
                elapsed_s = max(0.0, time.monotonic() - self.segment_start_monotonic)
            return {
                "recording": self.recording,
                "segment": self.saved_count,
                "elapsed_s": elapsed_s,
                "raw": len(self.raw_seq),
                "gmr": len(self.gmr_seq),
                "saved": self.saved_count,
            }

    def start(self) -> None:
        with self.lock:
            if self.recording:
                print("[Recorder] already recording, ignore start")
                return
            self._clear_locked()
            self.recording = True
            self.segment_start_monotonic = time.monotonic()
            segment_id = self.saved_count
        print(f"[Recorder] start pico raw capture | state=recording segment={segment_id:08d} dir={self.output_dir}")

    def stop_and_save(self) -> Optional[Path]:
        with self.lock:
            if not self.recording:
                return None
            self.recording = False
            self.segment_start_monotonic = None
            if len(self.raw_seq) == 0:
                self._clear_locked()
                print("[Recorder] stop pico raw capture | state=idle no frames captured, skip save")
                return None
            raw_payload = self._build_raw_payload_locked()
            gmr_payload = self._build_gmr_payload_locked()
            self._clear_locked()

        raw_output_path = self._next_output_path(self.output_dir, self.prefix)
        gmr_output_path = (
            self._next_output_path(self.gmr_output_dir, self.gmr_prefix)
            if self.gmr_output_dir is not None and gmr_payload is not None
            else None
        )
        raw_payload["gmr_record_filename"] = np.asarray("" if gmr_output_path is None else gmr_output_path.name)
        if gmr_payload is not None:
            gmr_payload["raw_record_filename"] = np.asarray(raw_output_path.name)

        old_sigint_handler = None
        signal_guarded = threading.current_thread() is threading.main_thread()
        if signal_guarded:
            old_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            self._atomic_save_npz(raw_output_path, raw_payload)
            if gmr_output_path is not None and gmr_payload is not None:
                self._atomic_save_npz(gmr_output_path, gmr_payload)
            self.saved_count += 1
            gmr_count = 0 if gmr_payload is None else int(gmr_payload["gmr_seq"].shape[0])
            print(
                f"[Recorder] saved capture | state=idle raw={raw_payload['raw_seq'].shape[0]} "
                f"gmr={gmr_count} raw_path={raw_output_path} "
                f"gmr_path={gmr_output_path if gmr_output_path is not None else 'none'} next=start"
            )
        finally:
            if signal_guarded:
                signal.signal(signal.SIGINT, old_sigint_handler)
        return raw_output_path

    @staticmethod
    def _atomic_save_npz(output_path: Path, payload: Dict[str, Any]) -> None:
        tmp_path = output_path.with_name(f".{output_path.stem}.tmp.npz")
        try:
            np.savez_compressed(tmp_path, **payload)
            os.replace(tmp_path, output_path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def handle_buttons(self, controller_buttons: Dict[str, Any]) -> bool:
        if not self.button_control:
            return False

        start_pressed = bool(controller_buttons.get(self.start_button, False))
        stop_pressed = bool(controller_buttons.get(self.stop_button, False))
        quit_combo_pressed = bool(controller_buttons.get("left_key_two", False)) and bool(
            controller_buttons.get("right_key_two", False)
        )
        should_quit = quit_combo_pressed and not self.prev_quit_combo
        should_start = start_pressed and not self.prev_start_button
        should_stop = stop_pressed and not self.prev_stop_button
        self.prev_quit_combo = quit_combo_pressed
        self.prev_start_button = start_pressed
        self.prev_stop_button = stop_pressed

        if should_quit:
            self.stop_and_save()
            print("[Recorder] PICO left Y + right B pressed | save+quit")
            return True
        if should_start:
            with self.lock:
                already_recording = self.recording
            if not already_recording:
                self.start()
        if should_stop:
            self.stop_and_save()
        return False

    def append_raw(
        self,
        *,
        seq: int,
        recv_ns: int,
        motion_timestamp_ns: Optional[int],
        body_timestamp_ns: Optional[int],
        top_timestamp_ns: Optional[int],
        snapshot: dict,
        controller_buttons: Dict[str, Any],
    ) -> None:
        with self.lock:
            if not self.recording:
                return
            body = snapshot.get("body", {}) if isinstance(snapshot, dict) else {}
            body_poses = body.get("poses", None) if isinstance(body, dict) else None
            self.raw_seq.append(int(seq))
            self.raw_recv_ns.append(int(recv_ns))
            self.raw_motion_timestamp_ns.append(self._none_to_i64(motion_timestamp_ns))
            self.raw_body_timestamp_ns.append(self._none_to_i64(body_timestamp_ns))
            self.raw_top_timestamp_ns.append(self._none_to_i64(top_timestamp_ns))
            self.raw_body_poses.append(self._body_poses_from_any(body_poses))
            self.raw_headset_pose.append(self._pose7_from_any(snapshot.get("headset_pose", None)))
            self.raw_left_controller_pose.append(self._controller_pose(snapshot, "left"))
            self.raw_right_controller_pose.append(self._controller_pose(snapshot, "right"))
            self.raw_controller_buttons.append(self._controller_buttons_array(controller_buttons))
            self.raw_controller_values.append(self._controller_values_array(controller_buttons))
            if self.include_snapshot_json:
                try:
                    self.raw_snapshot_json.append(
                        json.dumps(self._json_safe(snapshot), separators=(",", ":"), ensure_ascii=True)
                    )
                except Exception:
                    self.raw_snapshot_json.append("")

    def append_gmr(self, *, seq: int, recv_ns: int, qpos: np.ndarray) -> None:
        with self.lock:
            if not self.recording:
                return
            self.gmr_seq.append(int(seq))
            self.gmr_recv_ns.append(int(recv_ns))
            self.gmr_qpos.append(np.asarray(qpos, dtype=np.float32).reshape(-1).copy())

    def _build_raw_payload_locked(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": np.asarray(1, dtype=np.int32),
            "format": np.asarray("pico_raw"),
            "created_unix_ms": np.asarray(int(time.time() * 1000), dtype=np.int64),
            "body_joint_names": np.asarray(XR_BODY_JOINT_NAMES),
            "controller_button_names": np.asarray(RECORDER_BUTTON_NAMES),
            "controller_value_names": np.asarray(RECORDER_VALUE_NAMES),
            "raw_seq": np.asarray(self.raw_seq, dtype=np.int64),
            "raw_recv_ns": np.asarray(self.raw_recv_ns, dtype=np.int64),
            "raw_motion_timestamp_ns": np.asarray(self.raw_motion_timestamp_ns, dtype=np.int64),
            "raw_body_timestamp_ns": np.asarray(self.raw_body_timestamp_ns, dtype=np.int64),
            "raw_top_timestamp_ns": np.asarray(self.raw_top_timestamp_ns, dtype=np.int64),
            "raw_body_poses": np.stack(self.raw_body_poses, axis=0).astype(np.float32)
            if self.raw_body_poses
            else np.zeros((0, len(XR_BODY_JOINT_NAMES), 7), dtype=np.float32),
            "raw_headset_pose": np.stack(self.raw_headset_pose, axis=0).astype(np.float32)
            if self.raw_headset_pose
            else np.zeros((0, 7), dtype=np.float32),
            "raw_left_controller_pose": np.stack(self.raw_left_controller_pose, axis=0).astype(np.float32)
            if self.raw_left_controller_pose
            else np.zeros((0, 7), dtype=np.float32),
            "raw_right_controller_pose": np.stack(self.raw_right_controller_pose, axis=0).astype(np.float32)
            if self.raw_right_controller_pose
            else np.zeros((0, 7), dtype=np.float32),
            "raw_controller_buttons": np.stack(self.raw_controller_buttons, axis=0).astype(bool)
            if self.raw_controller_buttons
            else np.zeros((0, len(RECORDER_BUTTON_NAMES)), dtype=bool),
            "raw_controller_values": np.stack(self.raw_controller_values, axis=0).astype(np.float32)
            if self.raw_controller_values
            else np.zeros((0, len(RECORDER_VALUE_NAMES)), dtype=np.float32),
            "actual_human_height": np.asarray(float(self.record_config.get("actual_human_height", 1.6)), dtype=np.float32),
            "coordinate_version": np.asarray("xrobot_raw_pose_xyzw_v1"),
            "session_id": np.asarray(self.session_id),
            "segment_id": np.asarray(self.saved_count, dtype=np.int32),
            "record_config_json": np.asarray(
                json.dumps(self._json_safe(self.record_config), separators=(",", ":"), ensure_ascii=True)
            ),
        }
        if self.include_snapshot_json:
            payload["raw_snapshot_json"] = np.asarray(self.raw_snapshot_json)
        return payload

    def _build_gmr_payload_locked(self) -> Optional[Dict[str, Any]]:
        if not self.gmr_qpos:
            return None

        raw_seq = np.asarray(self.raw_seq, dtype=np.int64)
        gmr_seq = np.asarray(self.gmr_seq, dtype=np.int64)
        gmr_qpos = np.stack(self.gmr_qpos, axis=0).astype(np.float32)
        common_seq, raw_indices, gmr_indices = np.intersect1d(
            raw_seq,
            gmr_seq,
            assume_unique=False,
            return_indices=True,
        )

        return {
            "schema_version": np.asarray(1, dtype=np.int32),
            "format": np.asarray("gmr_mt"),
            "created_unix_ms": np.asarray(int(time.time() * 1000), dtype=np.int64),
            "session_id": np.asarray(self.session_id),
            "segment_id": np.asarray(self.saved_count, dtype=np.int32),
            "source_format": np.asarray("pico_raw"),
            "dof_joint_names": np.asarray(G1_DOF_JOINT_NAMES),
            "joint_names": np.asarray(G1_DOF_JOINT_NAMES),
            "gmr_seq": gmr_seq,
            "gmr_recv_ns": np.asarray(self.gmr_recv_ns, dtype=np.int64),
            "gmr_qpos": gmr_qpos,
            "gmr_root_pos": gmr_qpos[:, 0:3],
            "gmr_root_quat_wxyz": gmr_qpos[:, 3:7],
            "gmr_root_rot_xyzw": gmr_qpos[:, [4, 5, 6, 3]],
            "gmr_dof_pos": gmr_qpos[:, 7:36],
            "fps": np.asarray(50, dtype=np.int64),
            "root_pos": gmr_qpos[:, 0:3],
            "root_rot": gmr_qpos[:, [4, 5, 6, 3]],
            "dof_pos": gmr_qpos[:, 7:36],
            "raw_seq": raw_seq,
            "aligned_seq": common_seq.astype(np.int64),
            "aligned_raw_indices": raw_indices.astype(np.int64),
            "aligned_gmr_indices": gmr_indices.astype(np.int64),
            "record_config_json": np.asarray(
                json.dumps(self._json_safe(self.record_config), separators=(",", ":"), ensure_ascii=True)
            ),
        }


class _RetargetWorkerRuntime:
    ROBOT_GROUND_REFERENCE_BODY_NAMES = ("left_toe_link", "right_toe_link")

    def __init__(self, worker_config: Dict[str, Any]):
        from general_motion_retargeting import GeneralMotionRetargeting
        from general_motion_retargeting.rot_utils import quat_mul_np as worker_quat_mul_np

        self._quat_mul_np = worker_quat_mul_np
        self.retarget = GeneralMotionRetargeting(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=float(worker_config["actual_human_height"]),
        )
        self.retarget.max_iter = int(worker_config["gmr_max_iter"])
        self.send_human_motion = bool(worker_config["send_human_motion"])
        self.min_link_height = float(worker_config["min_link_height"])
        self.min_link_height_align_strategy = str(worker_config["min_link_height_align_strategy"])
        self.min_link_height_bootstrap_frames = max(1, int(worker_config["min_link_height_bootstrap_frames"]))
        self.fixed_min_link_height_offset: Optional[float] = None
        self.min_link_height_offset_samples: list[float] = []
        self.rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        self.rotation_quat = R.from_matrix(self.rotation_matrix).as_quat(scalar_first=True)

    def _body_poses_to_pose_dict(self, poses: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(poses, (list, tuple)) or len(poses) < len(XR_BODY_JOINT_NAMES):
            return None

        body_pose_dict: Dict[str, Any] = {}
        for i, joint_name in enumerate(XR_BODY_JOINT_NAMES):
            pose = poses[i]
            if not isinstance(pose, (list, tuple)) or len(pose) < 7:
                return None
            x, y, z, qx, qy, qz, qw = [float(v) for v in pose[:7]]
            pos = np.array([x, y, z], dtype=np.float64) @ self.rotation_matrix.T
            rot = self._quat_mul_np(
                self.rotation_quat.reshape(1, 4),
                np.array([[qw, qx, qy, qz]], dtype=np.float64),
                scalar_first=True,
            )[0]
            body_pose_dict[joint_name] = [pos.tolist(), rot.tolist()]
        return body_pose_dict

    def _get_current_min_body_z(self) -> Optional[float]:
        body_z = self.retarget.configuration.data.xpos[1:, 2]
        if body_z.size == 0:
            return None
        min_body_z = float(np.min(body_z))
        if not np.isfinite(min_body_z):
            return None
        return min_body_z

    def _get_current_ground_reference_z(self) -> Optional[float]:
        toe_z_values: list[float] = []
        body_name_map = getattr(self.retarget, "robot_body_names", {})
        data = self.retarget.configuration.data

        for body_name in self.ROBOT_GROUND_REFERENCE_BODY_NAMES:
            body_id = body_name_map.get(body_name)
            if body_id is None:
                continue
            if body_id < 0 or body_id >= data.xpos.shape[0]:
                continue
            z = float(data.xpos[body_id, 2])
            if np.isfinite(z):
                toe_z_values.append(z)

        if toe_z_values:
            return float(min(toe_z_values))
        return self._get_current_min_body_z()

    def _apply_min_link_height_offset(self, qpos: np.ndarray) -> np.ndarray:
        qpos_adj = np.asarray(qpos, dtype=np.float32).copy()
        ground_ref_z = self._get_current_ground_reference_z()
        if ground_ref_z is None:
            return qpos_adj

        if self.min_link_height_align_strategy == "per_frame":
            qpos_adj[2] += self.min_link_height - ground_ref_z
            return qpos_adj

        if self.fixed_min_link_height_offset is None:
            offset = self.min_link_height - ground_ref_z
            self.min_link_height_offset_samples.append(offset)
            if len(self.min_link_height_offset_samples) >= self.min_link_height_bootstrap_frames:
                self.fixed_min_link_height_offset = float(np.mean(self.min_link_height_offset_samples))
                print(
                    "[Info] worker startup_fixed ground calibration: "
                    f"{self.fixed_min_link_height_offset:.6f} m from "
                    f"{len(self.min_link_height_offset_samples)} frames"
                )
                self.min_link_height_offset_samples.clear()

        applied_offset = (
            self.fixed_min_link_height_offset
            if self.fixed_min_link_height_offset is not None
            else float(np.mean(self.min_link_height_offset_samples))
            if self.min_link_height_offset_samples
            else 0.0
        )
        qpos_adj[2] += applied_offset
        return qpos_adj

    @staticmethod
    def _copy_human_motion_data(human_motion_data: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(human_motion_data, dict):
            return None
        copied: Dict[str, Any] = {}
        for key, value in human_motion_data.items():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            pos = np.asarray(value[0], dtype=np.float32).copy()
            rot = np.asarray(value[1], dtype=np.float32).copy()
            copied[key] = (pos, rot)
        return copied

    def process_packet(self, packet: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body_pose_dict = self._body_poses_to_pose_dict(packet.get("poses"))
        if body_pose_dict is None:
            return None

        qpos_curr = self.retarget.retarget(body_pose_dict, offset_to_ground=False)
        if qpos_curr is None:
            return None

        qpos_curr = np.asarray(qpos_curr, dtype=np.float32).reshape(-1)
        if qpos_curr.shape[0] < 36:
            raise ValueError(f"retarget qpos too short: {qpos_curr.shape[0]}")
        qpos_curr = self._apply_min_link_height_offset(qpos_curr[:36])

        return {
            "type": "retarget_result",
            "seq": int(packet["seq"]),
            "recv_ns": int(packet["recv_ns"]),
            "qpos": qpos_curr.astype(np.float32, copy=True),
            "human_motion_data": self._copy_human_motion_data(self.retarget.scaled_human_data)
            if self.send_human_motion
            else None,
        }


def _retarget_worker_main(
    raw_recv_conn: Any,
    result_send_conn: Any,
    worker_config: Dict[str, Any],
) -> None:
    try:
        runtime = _RetargetWorkerRuntime(worker_config)
    except Exception as exc:
        try:
            result_send_conn.send({"type": "worker_init_error", "error": str(exc)})
        except Exception:
            pass
        return

    try:
        result_send_conn.send({"type": "worker_ready"})
    except Exception:
        return

    last_processed_seq = 0
    while True:
        try:
            if not raw_recv_conn.poll(0.1):
                continue
            packet = raw_recv_conn.recv()
        except EOFError:
            break
        except Exception as exc:
            try:
                result_send_conn.send({"type": "worker_runtime_error", "error": str(exc)})
            except Exception:
                pass
            continue

        if isinstance(packet, dict) and packet.get("type") == "shutdown":
            break

        dropped_before_process = 0
        while raw_recv_conn.poll():
            try:
                newer_packet = raw_recv_conn.recv()
            except EOFError:
                newer_packet = None
            if newer_packet is None:
                break
            if isinstance(newer_packet, dict) and newer_packet.get("type") == "shutdown":
                return
            dropped_before_process += 1
            packet = newer_packet

        prev_processed_seq = last_processed_seq
        try:
            result = runtime.process_packet(packet)
        except Exception as exc:
            try:
                result_send_conn.send({"type": "worker_runtime_error", "error": str(exc)})
            except Exception:
                pass
            continue

        if result is None:
            continue

        result["dropped_before_process"] = int(dropped_before_process)
        result["prev_processed_seq"] = int(prev_processed_seq)
        last_processed_seq = int(result["seq"])

        try:
            result_send_conn.send(result)
        except (BrokenPipeError, EOFError, OSError):
            break


class LowLatencyTeleopPoseZMQServer:
    BODY_JOINT_NAMES = XR_BODY_JOINT_NAMES

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.robot = args.robot
        self.vis_fps = int(args.vis_fps)
        self.ctrl_fps = int(args.ctrl_fps)
        self.retarget_fps = float(args.retarget_fps)
        self.lookback_ns = int(float(args.lookback_ms) * 1e6)
        self.retarget_buffer_window_ns = int(float(args.retarget_buffer_window_s) * 1e9)
        self.log_interval_s = float(args.log_interval_s)
        self.record_status_interval_s = float(args.record_status_interval_s)
        self.debug_warning_interval_s = float(args.debug_warning_interval_s)
        self.debug_retarget_stall_warn_ms = float(args.debug_retarget_stall_warn_ms)
        self.debug_raw_fresh_ms = float(args.debug_raw_fresh_ms)
        self.debug_reply_process_warn_ms = float(args.debug_reply_process_warn_ms)
        self.debug_pico_callback_warn_ms = float(args.debug_pico_callback_warn_ms)
        self.debug_pico_health_interval_s = float(args.debug_pico_health_interval_s)
        self.udp_stream_host = str(args.udp_stream_host).strip()
        self.udp_stream_port = int(args.udp_stream_port)
        self.udp_stream_fps = float(args.udp_stream_fps)
        self.udp_stream_qos = bool(args.udp_stream_qos)

        if self.vis_fps <= 0:
            raise ValueError("vis_fps must be > 0")
        if self.ctrl_fps <= 0:
            raise ValueError("ctrl_fps must be > 0")
        if self.retarget_fps <= 0:
            raise ValueError("retarget_fps must be > 0")
        if self.lookback_ns < 0:
            raise ValueError("lookback_ms must be >= 0")
        if self.retarget_buffer_window_ns <= 0:
            raise ValueError("retarget_buffer_window_s must be > 0")
        if self.log_interval_s < 0:
            raise ValueError("log_interval_s must be >= 0")
        if self.record_status_interval_s < 0:
            raise ValueError("record_status_interval_s must be >= 0")
        if self.debug_pico_callback_warn_ms < 0:
            raise ValueError("debug_pico_callback_warn_ms must be >= 0")
        if self.debug_pico_health_interval_s < 0:
            raise ValueError("debug_pico_health_interval_s must be >= 0")
        if self.udp_stream_fps <= 0:
            raise ValueError("udp_stream_fps must be > 0")
        if self.udp_stream_port <= 0 or self.udp_stream_port > 65535:
            raise ValueError("udp_stream_port must be in 1..65535")

        self.retarget = None
        self.viewer = None
        self.gmr_max_iter = 5
        self.retarget_worker_ready_timeout_s = max(1.0, float(args.retarget_worker_ready_timeout_s))

        self.zmq_context = None
        self.req_sock = None
        self.rep_sock = None
        self.ctrl_sock = None
        self.udp_stream_socket: Optional[socket.socket] = None
        self.udp_stream_addr: Optional[tuple[str, int]] = None
        self.recorder: Optional[PicoRawRecorder] = None
        if str(args.pico_record_output_dir).strip():
            self.recorder = PicoRawRecorder(
                output_dir=str(args.pico_record_output_dir),
                prefix=str(args.pico_record_prefix),
                gmr_output_dir=str(args.gmr_record_output_dir),
                gmr_prefix=str(args.gmr_record_prefix),
                auto_start=bool(args.pico_record_auto_start),
                button_control=bool(args.pico_record_enable_buttons)
                and not bool(args.pico_record_disable_buttons),
                include_snapshot_json=bool(args.pico_record_include_snapshot_json),
                start_button=str(args.pico_record_start_button),
                stop_button=str(args.pico_record_stop_button),
                record_config={
                    "actual_human_height": float(args.actual_human_height),
                    "robot": str(args.robot),
                    "ctrl_fps": int(args.ctrl_fps),
                    "retarget_fps": float(args.retarget_fps),
                    "lookback_ms": float(args.lookback_ms),
                    "record_source": "xrobot_frame_callback",
                },
            )

        self.default_qpos = self._build_default_qpos()
        print(
            "[PoseBridge] default_qpos "
            f"root_z={float(self.default_qpos[2]):.3f} "
            f"dof={np.array2string(self.default_qpos[7:36], precision=3, suppress_small=True)}"
        )
        self.last_controller_buttons: Dict[str, Any] = self._default_controller_buttons()
        self.controller_pressed_edge_until: Dict[str, float] = {}
        self.controller_pressed_edge_hold_s = 1.0

        self.min_link_height = float(args.min_link_height)
        self.min_link_height_align_strategy = str(args.min_link_height_align_strategy)
        self.min_link_height_bootstrap_frames = max(1, int(args.min_link_height_bootstrap_frames))
        self.fixed_min_link_height_offset: Optional[float] = None
        self.min_link_height_offset_samples: list[float] = []

        self.rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        self.rotation_quat = R.from_matrix(self.rotation_matrix).as_quat(scalar_first=True)

        self.latest_vr_lock = threading.Lock()
        self.latest_vr_poses: Optional[Any] = None
        self.latest_vr_recv_ns: int = 0
        self.latest_vr_seq: int = 0
        self.latest_vr_motion_timestamp_ns: Optional[int] = None
        self.latest_controller_recv_ns: int = 0
        self.latest_controller_seq: int = 0
        self.latest_callback_gap_ms: Optional[float] = None
        self.latest_body_available = False
        self.latest_motion_timestamp_duplicate = False
        self.duplicate_motion_timestamp_count = 0
        self.body_unavailable_count = 0
        self.motion_timestamp_missing_count = 0

        self.retarget_buffer_lock = threading.Lock()
        self.retarget_buffer: deque[RetargetedFrame] = deque()
        self.stream_payload_lock = threading.Lock()
        self.latest_udp_stream_payload: Optional[Dict[str, Any]] = None
        self.vis_lock = threading.Lock()
        self.latest_vis_qpos: Optional[np.ndarray] = None
        self.latest_vis_human_motion: Optional[Dict[str, Any]] = None

        self.vr_frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.stats_lock = threading.Lock()

        self.frame_seq = 0
        self.last_vis_monotonic = 0.0
        self.last_req_monotonic: Optional[float] = None
        self.req_count = 0
        self.reply_count = 0
        self.reply_drop_count = 0
        self.req_merged_total = 0
        self.fallback_count = 0
        self.raw_motion_drop_count = 0
        self.retarget_input_skip_count = 0
        self.ctrl_send_count = 0
        self.udp_stream_count = 0
        self.udp_stream_drop_count = 0
        self.last_udp_stream_send_monotonic: Optional[float] = None
        self.last_ctrl_send_monotonic: Optional[float] = None
        self.latest_req_dt_ms: Optional[float] = None
        self.latest_merged_reqs = 0
        self.latest_reply_process_ms: Optional[float] = None
        self._last_debug_warning: Dict[str, float] = {}
        self._last_pico_health_log_monotonic: Optional[float] = None
        self._last_pico_health_callback_count = 0
        self._last_pico_health_raw_seq = 0
        self._last_pico_health_controller_seq = 0
        self._last_pico_health_body_unavailable_count = 0
        self._last_pico_health_duplicate_motion_timestamp_count = 0
        self._last_pico_health_motion_timestamp_missing_count = 0

        self.callback_count = 0
        self.retarget_count = 0
        self.latest_debug_info: Dict[str, Any] = {
            "mode": "no_data",
            "target_age_ms": None,
            "older_age_ms": None,
            "newer_age_ms": None,
            "span_ms": None,
            "buffer_len": 0,
            "retarget_age_ms": None,
            "raw_motion_age_ms": None,
        }

        self.retarget_thread = None
        self.raw_sender_thread = None
        self.worker_result_thread = None
        self.request_thread = None
        self.control_thread = None
        self.udp_stream_thread = None
        self.stats_thread = None
        self.visualization_thread = None
        self.recorder_command_thread = None
        self.recorder_status_thread = None

        self.mp_ctx = mp.get_context("spawn")
        self.raw_send_conn = None
        self.raw_recv_conn = None
        self.result_send_conn = None
        self.result_recv_conn = None
        self.retarget_process = None

        self.tiny2_head_follow_enabled = bool(str(args.tiny2_head_follow_host).strip())
        self.tiny2_head_follow_addr = (str(args.tiny2_head_follow_host).strip(), int(args.tiny2_head_follow_port))
        self.tiny2_head_follow_socket: Optional[socket.socket] = None
        self.tiny2_head_follow_last_send_ns = 0
        self.tiny2_head_follow_last_report_ns = 0
        self.tiny2_head_follow_sent = 0
        self.tiny2_head_follow_period_ns = int(1e9 / max(float(args.tiny2_head_follow_rate_hz), 1e-3))

    def _debug_warn(self, key: str, msg: str, interval_s: Optional[float] = None) -> None:
        interval = self.debug_warning_interval_s if interval_s is None else float(interval_s)
        now = time.monotonic()
        last = self._last_debug_warning.get(key)
        if last is None or (now - last) >= interval:
            print(msg)
            self._last_debug_warning[key] = now

    @staticmethod
    def _configure_latest_zmq_socket(zmq_mod: Any, sock: Any, *, send: bool = False, recv: bool = False) -> None:
        hwm = 64
        sock.setsockopt(zmq_mod.LINGER, 0)
        if send:
            sock.setsockopt(zmq_mod.SNDHWM, hwm)
            if hasattr(zmq_mod, "IMMEDIATE"):
                sock.setsockopt(zmq_mod.IMMEDIATE, 1)
        if recv:
            sock.setsockopt(zmq_mod.RCVHWM, hwm)
        # Explicit drain keeps the newest message; a small queue absorbs brief
        # transport jitter without dropping reply packets.

    @staticmethod
    def _nonfinite_summary(name: str, values: np.ndarray, limit: int = 8) -> Optional[str]:
        arr = np.asarray(values).reshape(-1)
        bad = np.flatnonzero(~np.isfinite(arr))
        if bad.size == 0:
            return None
        shown = bad[:limit]
        pairs = ", ".join(f"{int(i)}={arr[i]}" for i in shown)
        suffix = "" if bad.size <= limit else f", ... total={bad.size}"
        return f"{name}[{pairs}{suffix}]"

    def _build_default_qpos(self) -> np.ndarray:
        mimic = np.asarray(DEFAULT_MIMIC_OBS[self.robot], dtype=np.float32).reshape(-1)
        if mimic.shape[0] < 35:
            raise ValueError(f"DEFAULT_MIMIC_OBS[{self.robot}] must be at least 35 dims")
        dof_pos = mimic[6:35]
        root_z = float(mimic[2])
        root_pos = np.array([0.0, 0.0, root_z], dtype=np.float32)
        root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([root_pos, root_quat, dof_pos], axis=0).astype(np.float32)

    @staticmethod
    def _default_controller_buttons() -> Dict[str, Any]:
        return {
            "left_key_one": False,
            "left_key_two": False,
            "left_axis_click": False,
            "left_index_trig": False,
            "left_trigger_value": 0.0,
            "left_grip": False,
            "left_grip_value": 0.0,
            "left_axis": [0.0, 0.0],
            "right_key_one": False,
            "right_key_two": False,
            "right_axis_click": False,
            "right_index_trig": False,
            "right_trigger_value": 0.0,
            "right_grip": False,
            "right_grip_value": 0.0,
            "right_axis": [0.0, 0.0],
        }

    @staticmethod
    def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
        q = np.asarray(quat, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(q))
        if not np.isfinite(norm) or norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return (q / norm).astype(np.float32)

    def _slerp_quat_wxyz(self, quat0: np.ndarray, quat1: np.ndarray, alpha: float) -> np.ndarray:
        q0 = self._normalize_quat_wxyz(quat0).astype(np.float64)
        q1 = self._normalize_quat_wxyz(quat1).astype(np.float64)
        t = float(np.clip(alpha, 0.0, 1.0))

        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        if dot > 0.9995:
            out = q0 + t * (q1 - q0)
            return self._normalize_quat_wxyz(out)

        theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        sin_theta_0 = float(np.sin(theta_0))
        if abs(sin_theta_0) < 1e-8:
            return self._normalize_quat_wxyz(q0)

        theta = theta_0 * t
        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = np.sin(theta) / sin_theta_0
        out = s0 * q0 + s1 * q1
        return self._normalize_quat_wxyz(out)

    def _interpolate_qpos(self, prev_qpos: np.ndarray, next_qpos: np.ndarray, alpha: float) -> np.ndarray:
        t = float(np.clip(alpha, 0.0, 1.0))
        frame = prev_qpos * (1.0 - t) + next_qpos * t
        frame[3:7] = self._slerp_quat_wxyz(prev_qpos[3:7], next_qpos[3:7], t)
        return frame.astype(np.float32)

    def _extract_controller_buttons_from_snapshot(self, snapshot: Optional[dict]) -> Dict[str, Any]:
        if snapshot is None:
            return self.last_controller_buttons

        controllers = snapshot.get("controllers", {}) if isinstance(snapshot, dict) else {}
        left = controllers.get("left", {}) if isinstance(controllers, dict) else {}
        right = controllers.get("right", {}) if isinstance(controllers, dict) else {}

        def _first(mapping: dict, *keys: str, default: Any = None) -> Any:
            for key in keys:
                if key in mapping:
                    return mapping.get(key)
            return default

        def _as_bool(value: Any) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        def _axis(controller: dict) -> list[float]:
            values = _first(controller, "axis", "primary2DAxis", "primary_2d_axis")
            if isinstance(values, (list, tuple)) and len(values) >= 2:
                return [float(values[0]), float(values[1])]
            return [
                float(_first(controller, "axis_x", "axisX", default=0.0)),
                float(_first(controller, "axis_y", "axisY", default=0.0)),
            ]

        left_trigger = float(_first(left, "trigger", "trigger_value", "triggerValue", default=0.0))
        left_grip = float(_first(left, "grip", "grip_value", "gripValue", default=0.0))
        right_trigger = float(_first(right, "trigger", "trigger_value", "triggerValue", default=0.0))
        right_grip = float(_first(right, "grip", "grip_value", "gripValue", default=0.0))

        return {
            "left_key_one": _as_bool(_first(left, "primary_button", "primaryButton", default=False)),
            "left_key_two": _as_bool(_first(left, "secondary_button", "secondaryButton", default=False)),
            "left_axis_click": _as_bool(_first(left, "axis_click", "axisClick", "primary2DAxisClick", default=False)),
            "left_index_trig": left_trigger > 1e-4,
            "left_trigger_value": float(np.clip(left_trigger, 0.0, 1.0)),
            "left_grip": left_grip > 1e-4,
            "left_grip_value": float(np.clip(left_grip, 0.0, 1.0)),
            "left_axis": _axis(left),
            "right_key_one": _as_bool(_first(right, "primary_button", "primaryButton", default=False)),
            "right_key_two": _as_bool(_first(right, "secondary_button", "secondaryButton", default=False)),
            "right_axis_click": _as_bool(_first(right, "axis_click", "axisClick", "primary2DAxisClick", default=False)),
            "right_index_trig": right_trigger > 1e-4,
            "right_trigger_value": float(np.clip(right_trigger, 0.0, 1.0)),
            "right_grip": right_grip > 1e-4,
            "right_grip_value": float(np.clip(right_grip, 0.0, 1.0)),
            "right_axis": _axis(right),
        }

    def _active_controller_pressed_edges_locked(self) -> list[str]:
        now = time.monotonic()
        expired = [
            name for name, edge_until in self.controller_pressed_edge_until.items() if edge_until <= now
        ]
        for name in expired:
            self.controller_pressed_edge_until.pop(name, None)
        return sorted(self.controller_pressed_edge_until.keys())

    @staticmethod
    def _serialize_qpos_frame(qpos: np.ndarray) -> Dict[str, Any]:
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        return {
            "root_pos": q[0:3].tolist(),
            "root_quat": q[3:7].tolist(),
            "dof_pos": q[7:36].tolist(),
        }

    def _publish_tiny2_head_pose(self, snapshot: dict, recv_ns: int) -> None:
        if self.tiny2_head_follow_socket is None:
            return
        if recv_ns - self.tiny2_head_follow_last_send_ns < self.tiny2_head_follow_period_ns:
            return
        self.tiny2_head_follow_last_send_ns = recv_ns

        try:
            head_yaw, head_pitch = _compute_head_pose_relative_to_pelvis(
                snapshot, str(self.args.tiny2_head_forward_axis)
            )
            yaw = head_yaw * float(self.args.tiny2_head_follow_yaw_gain) + float(
                self.args.tiny2_head_follow_yaw_offset
            )
            pitch = head_pitch * float(self.args.tiny2_head_follow_pitch_gain) + float(
                self.args.tiny2_head_follow_pitch_offset
            )
            payload = {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "head_yaw": float(head_yaw),
                "head_pitch": float(head_pitch),
                "timestamp_ns": int(recv_ns),
                "source": "molmospaces_xrobot",
            }
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.tiny2_head_follow_socket.sendto(data, self.tiny2_head_follow_addr)
            self.tiny2_head_follow_sent += 1

            if recv_ns - self.tiny2_head_follow_last_report_ns >= int(1e9):
                print(
                    "[Tiny2] head pose udp | "
                    f"sent={self.tiny2_head_follow_sent} "
                    f"yaw={yaw:.2f} pitch={pitch:.2f} "
                    f"raw_yaw={head_yaw:.2f} raw_pitch={head_pitch:.2f} "
                    f"target={self.tiny2_head_follow_addr[0]}:{self.tiny2_head_follow_addr[1]}",
                    flush=True,
                )
                self.tiny2_head_follow_last_report_ns = recv_ns
        except Exception as exc:
            if recv_ns - self.tiny2_head_follow_last_report_ns >= int(1e9):
                print(f"[Tiny2] waiting for XR head pose: {exc}", flush=True)
                self.tiny2_head_follow_last_report_ns = recv_ns

    def _on_vr_frame(self, snapshot: dict) -> None:
        recv_ns = time.monotonic_ns()
        controller_buttons = self._extract_controller_buttons_from_snapshot(snapshot)

        previous_buttons = dict(self.last_controller_buttons)
        pressed_edges = [
            name
            for name, value in controller_buttons.items()
            if isinstance(value, bool) and value and not bool(previous_buttons.get(name, False))
        ]
        if pressed_edges:
            print(f"[PICO][Buttons] pressed_edges={pressed_edges}", flush=True)

        controllers = snapshot.get("controllers", {}) if isinstance(snapshot, dict) else {}
        if isinstance(controllers, dict):
            left = controllers.get("left", {})
            right = controllers.get("right", {})
            left_keys = tuple(sorted(left.keys())) if isinstance(left, dict) else ()
            right_keys = tuple(sorted(right.keys())) if isinstance(right, dict) else ()
            key_signature = (left_keys, right_keys)
            if key_signature != getattr(self, "_last_controller_key_debug", None):
                self._last_controller_key_debug = key_signature
                print(
                    f"[PICO][ControllerKeys] left={list(left_keys)} right={list(right_keys)}",
                    flush=True,
                )

        if self.recorder is not None:
            if self.recorder.handle_buttons(controller_buttons):
                self.stop_event.set()
                self.vr_frame_event.set()
                return

        top_timestamp_ns = None
        try:
            top_timestamp_ns = int(snapshot.get("timestamp_ns", 0)) if isinstance(snapshot, dict) else None
        except Exception:
            top_timestamp_ns = None
        body = snapshot.get("body", {}) if isinstance(snapshot, dict) else {}
        body_available = bool(body.get("available", False)) if isinstance(body, dict) else False
        body_timestamp_ns = None
        if body_available:
            try:
                body_timestamp_ns = int(body.get("timestamp_ns", 0))
            except Exception:
                body_timestamp_ns = None
        motion_timestamp_ns = body_timestamp_ns if body_timestamp_ns not in (None, 0) else top_timestamp_ns

        if self.tiny2_head_follow_enabled and body_available:
            self._publish_tiny2_head_pose(snapshot, recv_ns)

        should_wake_retarget = False
        raw_record_seq = None
        callback_gap_ms = None
        motion_timestamp_duplicate = False
        motion_timestamp_missing = bool(body_available and motion_timestamp_ns is None)
        callback_count = 0
        raw_seq = 0
        controller_seq = 0
        body_unavailable_count = 0
        duplicate_motion_timestamp_count = 0
        motion_timestamp_missing_count = 0
        with self.latest_vr_lock:
            if self.latest_controller_recv_ns > 0:
                callback_gap_ms = round((recv_ns - int(self.latest_controller_recv_ns)) / 1e6, 3)
            self.latest_controller_recv_ns = recv_ns
            self.latest_controller_seq += 1
            self.latest_callback_gap_ms = callback_gap_ms
            self.latest_body_available = bool(body_available)
            self.last_controller_buttons = controller_buttons
            if pressed_edges:
                edge_until = time.monotonic() + self.controller_pressed_edge_hold_s
                for name in pressed_edges:
                    self.controller_pressed_edge_until[str(name)] = edge_until
            self._active_controller_pressed_edges_locked()
            self.callback_count += 1
            if not body_available:
                self.body_unavailable_count += 1
            if motion_timestamp_missing:
                self.motion_timestamp_missing_count += 1
            if body_available and motion_timestamp_ns is not None:
                motion_timestamp_duplicate = self.latest_vr_motion_timestamp_ns == motion_timestamp_ns
                self.latest_motion_timestamp_duplicate = bool(motion_timestamp_duplicate)
                if not motion_timestamp_duplicate:
                    self.latest_vr_poses = body.get("poses", None)
                    self.latest_vr_recv_ns = recv_ns
                    self.latest_vr_seq += 1
                    self.latest_vr_motion_timestamp_ns = motion_timestamp_ns
                    should_wake_retarget = True
                    raw_record_seq = int(self.latest_vr_seq)
                else:
                    self.duplicate_motion_timestamp_count += 1
            else:
                self.latest_motion_timestamp_duplicate = False
            callback_count = int(self.callback_count)
            raw_seq = int(self.latest_vr_seq)
            controller_seq = int(self.latest_controller_seq)
            body_unavailable_count = int(self.body_unavailable_count)
            duplicate_motion_timestamp_count = int(self.duplicate_motion_timestamp_count)
            motion_timestamp_missing_count = int(self.motion_timestamp_missing_count)
        if (
            self.debug_pico_callback_warn_ms > 0.0
            and callback_gap_ms is not None
            and callback_gap_ms >= self.debug_pico_callback_warn_ms
        ):
            self._debug_warn(
                "xrobot_pico_callback_gap",
                "[PICO][Gap] callback gap | "
                f"gap_ms={callback_gap_ms}, threshold_ms={self.debug_pico_callback_warn_ms:.1f}, "
                f"cb={callback_count}, raw_seq={raw_seq}, controller_seq={controller_seq}, "
                f"body_available={body_available}, motion_timestamp_ns={motion_timestamp_ns}, "
                f"motion_ts_duplicate={motion_timestamp_duplicate}, "
                f"body_unavailable={body_unavailable_count}, "
                f"dup_motion_ts={duplicate_motion_timestamp_count}, "
                f"missing_motion_ts={motion_timestamp_missing_count}",
                interval_s=0.0,
            )
        if callback_gap_ms is not None and callback_gap_ms >= self.debug_retarget_stall_warn_ms:
            self._debug_warn(
                "xrobot_callback_gap",
                "[Error] PICO/XRobot callback gap | "
                f"gap_ms={callback_gap_ms}, body_available={body_available}, "
                f"motion_timestamp_ns={motion_timestamp_ns}",
                interval_s=0.0,
            )
        if not body_available:
            self._debug_warn(
                "xrobot_body_unavailable",
                "[PICO][BodyUnavailable] body unavailable | "
                f"cb={callback_count}, controller_seq={controller_seq}, "
                f"body_unavailable={body_unavailable_count}",
            )
        if motion_timestamp_missing:
            self._debug_warn(
                "xrobot_motion_timestamp_missing",
                "[PICO][TimestampMissing] body frame missing motion timestamp | "
                f"cb={callback_count}, raw_seq={raw_seq}, missing_motion_ts={motion_timestamp_missing_count}",
            )
        if body_available and motion_timestamp_duplicate:
            self._debug_warn(
                "xrobot_duplicate_motion_timestamp",
                "[PICO][DuplicateTimestamp] duplicate motion timestamp | "
                f"motion_timestamp_ns={motion_timestamp_ns}, cb={callback_count}, raw_seq={raw_seq}, "
                f"dup_motion_ts={duplicate_motion_timestamp_count}",
            )
        if self.recorder is not None and raw_record_seq is not None:
            self.recorder.append_raw(
                seq=raw_record_seq,
                recv_ns=recv_ns,
                motion_timestamp_ns=motion_timestamp_ns,
                body_timestamp_ns=body_timestamp_ns,
                top_timestamp_ns=top_timestamp_ns,
                snapshot=snapshot,
                controller_buttons=controller_buttons,
            )
        if should_wake_retarget:
            self.vr_frame_event.set()

    def _append_retarget_frame(self, recv_ns: int, qpos: np.ndarray) -> None:
        cutoff_ns = recv_ns - self.retarget_buffer_window_ns
        with self.retarget_buffer_lock:
            self.retarget_buffer.append(RetargetedFrame(recv_ns=recv_ns, qpos=qpos.astype(np.float32, copy=True)))
            while self.retarget_buffer and self.retarget_buffer[0].recv_ns < cutoff_ns:
                self.retarget_buffer.popleft()

    @staticmethod
    def _copy_human_motion_data(human_motion_data: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(human_motion_data, dict):
            return None

        copied: Dict[str, Any] = {}
        for key, value in human_motion_data.items():
            if not isinstance(value, (list, tuple)) or len(value) < 2:
                continue
            pos = np.asarray(value[0], dtype=np.float32).copy()
            rot = np.asarray(value[1], dtype=np.float32).copy()
            copied[key] = (pos, rot)
        return copied

    def _get_retarget_frames_snapshot(self) -> list[RetargetedFrame]:
        with self.retarget_buffer_lock:
            return list(self.retarget_buffer)

    def _sample_target_qpos(self, frames: list[RetargetedFrame], target_ns: int) -> tuple[np.ndarray, bool, Dict[str, Any]]:
        if not frames:
            return self.default_qpos.copy(), True, {
                "mode": "default",
                "target_ns": target_ns,
                "older_ns": None,
                "newer_ns": None,
                "alpha": None,
                "buffer_len": 0,
        }
        if len(frames) == 1:
            only_ns = frames[0].recv_ns
            return frames[0].qpos.astype(np.float32, copy=True), True, {
                "mode": "single_frame",
                "target_ns": target_ns,
                "older_ns": only_ns,
                "newer_ns": only_ns,
                "alpha": None,
                "buffer_len": 1,
            }
        if target_ns <= frames[0].recv_ns:
            oldest_ns = frames[0].recv_ns
            return frames[0].qpos.astype(np.float32, copy=True), True, {
                "mode": "fallback_oldest",
                "target_ns": target_ns,
                "older_ns": oldest_ns,
                "newer_ns": oldest_ns,
                "alpha": None,
                "buffer_len": len(frames),
            }
        if target_ns >= frames[-1].recv_ns:
            latest_ns = frames[-1].recv_ns
            return frames[-1].qpos.astype(np.float32, copy=True), True, {
                "mode": "fallback_latest",
                "target_ns": target_ns,
                "older_ns": latest_ns,
                "newer_ns": latest_ns,
                "alpha": None,
                "buffer_len": len(frames),
            }

        for idx in range(1, len(frames)):
            prev_frame = frames[idx - 1]
            next_frame = frames[idx]
            if target_ns <= next_frame.recv_ns:
                dt = next_frame.recv_ns - prev_frame.recv_ns
                if dt <= 0:
                    same_ns = next_frame.recv_ns
                    return next_frame.qpos.astype(np.float32, copy=True), True, {
                        "mode": "degenerate_dt",
                        "target_ns": target_ns,
                        "older_ns": same_ns,
                        "newer_ns": same_ns,
                        "alpha": None,
                        "buffer_len": len(frames),
                    }
                alpha = float(target_ns - prev_frame.recv_ns) / float(dt)
                return self._interpolate_qpos(prev_frame.qpos, next_frame.qpos, alpha), False, {
                    "mode": "interpolate",
                    "target_ns": target_ns,
                    "older_ns": prev_frame.recv_ns,
                    "newer_ns": next_frame.recv_ns,
                    "alpha": alpha,
                    "buffer_len": len(frames),
                }

        latest_ns = frames[-1].recv_ns
        return frames[-1].qpos.astype(np.float32, copy=True), True, {
            "mode": "fallback_latest",
            "target_ns": target_ns,
            "older_ns": latest_ns,
            "newer_ns": latest_ns,
            "alpha": None,
            "buffer_len": len(frames),
        }

    def _build_reply_frames(self, req_recv_ns: int) -> tuple[list[np.ndarray], bool, Dict[str, Any]]:
        frames = self._get_retarget_frames_snapshot()
        target_base_ns = req_recv_ns - self.lookback_ns
        qpos, used_fallback, sample_info = self._sample_target_qpos(frames, target_base_ns)
        return [qpos], used_fallback, sample_info

    def _get_latest_frame_ages_ms(self, now_ns: Optional[int] = None) -> tuple[Optional[float], Optional[float]]:
        health = self._get_input_health_snapshot(now_ns=now_ns)
        return health["retarget_age_ms"], health["raw_motion_age_ms"]

    def _get_input_health_snapshot(self, now_ns: Optional[int] = None) -> Dict[str, Any]:
        if now_ns is None:
            now_ns = time.monotonic_ns()
        with self.latest_vr_lock:
            latest_raw_recv_ns = int(self.latest_vr_recv_ns) if self.latest_vr_recv_ns > 0 else None
            latest_controller_recv_ns = (
                int(self.latest_controller_recv_ns) if self.latest_controller_recv_ns > 0 else None
            )
            callback_count = int(self.callback_count)
            raw_seq = int(self.latest_vr_seq)
            controller_seq = int(self.latest_controller_seq)
            callback_gap_ms = self.latest_callback_gap_ms
            body_available = bool(self.latest_body_available)
            motion_timestamp_duplicate = bool(self.latest_motion_timestamp_duplicate)
            duplicate_motion_timestamp_count = int(self.duplicate_motion_timestamp_count)
            body_unavailable_count = int(self.body_unavailable_count)
            motion_timestamp_missing_count = int(self.motion_timestamp_missing_count)

        with self.retarget_buffer_lock:
            latest_retarget_recv_ns = self.retarget_buffer[-1].recv_ns if self.retarget_buffer else None
            retarget_buffer_len = len(self.retarget_buffer)

        raw_motion_age_ms = None
        if latest_raw_recv_ns is not None:
            raw_motion_age_ms = round((now_ns - latest_raw_recv_ns) / 1e6, 3)

        controller_age_ms = None
        if latest_controller_recv_ns is not None:
            controller_age_ms = round((now_ns - latest_controller_recv_ns) / 1e6, 3)

        retarget_age_ms = None
        if latest_retarget_recv_ns is not None:
            retarget_age_ms = round((now_ns - latest_retarget_recv_ns) / 1e6, 3)

        return {
            "retarget_age_ms": retarget_age_ms,
            "raw_motion_age_ms": raw_motion_age_ms,
            "controller_age_ms": controller_age_ms,
            "callback_gap_ms": callback_gap_ms,
            "callback_count": callback_count,
            "controller_seq": controller_seq,
            "raw_seq": raw_seq,
            "retarget_count": int(self.retarget_count),
            "retarget_buffer_len": int(retarget_buffer_len),
            "body_available": body_available,
            "motion_timestamp_duplicate": motion_timestamp_duplicate,
            "duplicate_motion_timestamp_count": duplicate_motion_timestamp_count,
            "body_unavailable_count": body_unavailable_count,
            "motion_timestamp_missing_count": motion_timestamp_missing_count,
        }

    def _update_debug_info(self, sample_info: Dict[str, Any], req_recv_ns: int) -> None:
        older_ns = sample_info.get("older_ns")
        newer_ns = sample_info.get("newer_ns")
        health = self._get_input_health_snapshot()

        info = {
            "mode": sample_info.get("mode"),
            "target_age_ms": round((req_recv_ns - int(sample_info["target_ns"])) / 1e6, 3),
            "older_age_ms": None if older_ns is None else round((req_recv_ns - int(older_ns)) / 1e6, 3),
            "newer_age_ms": None if newer_ns is None else round((req_recv_ns - int(newer_ns)) / 1e6, 3),
            "span_ms": None
            if older_ns is None or newer_ns is None
            else round((int(newer_ns) - int(older_ns)) / 1e6, 3),
            "alpha": sample_info.get("alpha"),
            "buffer_len": int(sample_info.get("buffer_len", 0)),
            "reply_process_ms": self.latest_reply_process_ms,
            **health,
        }
        with self.stats_lock:
            self.latest_debug_info = info

    def _warn_on_fallback(self, sample_info: Dict[str, Any]) -> None:
        now_ns = time.monotonic_ns()
        retarget_age_ms, raw_motion_age_ms = self._get_latest_frame_ages_ms(now_ns=now_ns)
        target_age_ms = round((now_ns - int(sample_info["target_ns"])) / 1e6, 3)
        older_ns = sample_info.get("older_ns")
        newer_ns = sample_info.get("newer_ns")
        older_age_ms = None if older_ns is None else round((now_ns - int(older_ns)) / 1e6, 3)
        newer_age_ms = None if newer_ns is None else round((now_ns - int(newer_ns)) / 1e6, 3)
        self._debug_warn(
            "interpolation_fallback",
            "[Warning] interpolation fallback "
            f"mode={sample_info.get('mode')}, "
            f"target_age_ms={target_age_ms}, "
            f"older_age_ms={older_age_ms}, "
            f"newer_age_ms={newer_age_ms}, "
            f"buffer={int(sample_info.get('buffer_len', 0))}, "
            f"latest_retarget_age_ms={retarget_age_ms}, "
            f"latest_raw_motion_age_ms={raw_motion_age_ms}",
            interval_s=max(1.0, float(self.log_interval_s)),
        )

    def _warn_on_raw_motion_drop(self, dropped_count: int, latest_seq: int, last_processed_seq: int) -> None:
        now_ns = time.monotonic_ns()
        retarget_age_ms, raw_motion_age_ms = self._get_latest_frame_ages_ms(now_ns=now_ns)
        print(
            "[Warning] retarget lag dropped raw motion frames "
            f"dropped={int(dropped_count)}, "
            f"last_processed_seq={int(last_processed_seq)}, "
            f"latest_seq={int(latest_seq)}, "
            f"latest_retarget_age_ms={retarget_age_ms}, "
            f"latest_raw_motion_age_ms={raw_motion_age_ms}"
        )

    def _raw_sender_loop(self) -> None:
        last_sent_seq = 0
        period_ns = max(1, int(1e9 / float(self.retarget_fps)))
        next_send_ns = 0

        while not self.stop_event.is_set():
            now_ns = time.monotonic_ns()
            if next_send_ns > now_ns:
                wait_s = min(0.1, (next_send_ns - now_ns) / 1e9)
                self.stop_event.wait(timeout=wait_s)
                continue

            if not self.vr_frame_event.is_set() and not self.vr_frame_event.wait(timeout=0.1):
                continue

            while not self.stop_event.is_set():
                now_ns = time.monotonic_ns()
                if next_send_ns > now_ns:
                    break

                with self.latest_vr_lock:
                    poses = self.latest_vr_poses
                    recv_ns = self.latest_vr_recv_ns
                    seq = self.latest_vr_seq

                if poses is None or seq == last_sent_seq:
                    with self.latest_vr_lock:
                        if self.latest_vr_seq == last_sent_seq:
                            self.vr_frame_event.clear()
                            break
                    continue

                if last_sent_seq != 0 and seq > last_sent_seq + 1:
                    dropped_count = seq - last_sent_seq - 1
                    self.retarget_input_skip_count += int(dropped_count)

                try:
                    self.raw_send_conn.send(
                        {
                            "seq": int(seq),
                            "recv_ns": int(recv_ns),
                            "poses": poses,
                        }
                    )
                except (BrokenPipeError, EOFError, OSError) as exc:
                    print(f"[Warning] raw->worker pipe failed: {exc}")
                    self.stop_event.set()
                    self.vr_frame_event.set()
                    break

                last_sent_seq = seq
                next_send_ns = time.monotonic_ns() + period_ns

    def _worker_result_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                if not self.result_recv_conn.poll(0.1):
                    continue
                payload = self.result_recv_conn.recv()
            except EOFError:
                print("[Warning] worker->main pipe closed")
                self.stop_event.set()
                break
            except Exception as exc:
                print(f"[Warning] worker result recv failed: {exc}")
                self.stop_event.set()
                break

            if not isinstance(payload, dict):
                continue

            payload_type = payload.get("type")
            if payload_type == "worker_ready":
                continue
            if payload_type in ("worker_init_error", "worker_runtime_error"):
                print(f"[Warning] retarget worker error: {payload.get('error')}")
                if payload_type == "worker_init_error":
                    self.stop_event.set()
                continue
            if payload_type != "retarget_result":
                continue

            dropped_before_process = int(payload.get("dropped_before_process", 0))
            if dropped_before_process > 0:
                self.raw_motion_drop_count += dropped_before_process
                self._warn_on_raw_motion_drop(
                    dropped_count=dropped_before_process,
                    latest_seq=int(payload.get("seq", 0)),
                    last_processed_seq=int(payload.get("prev_processed_seq", 0)),
                )

            qpos_curr = np.asarray(payload.get("qpos"), dtype=np.float32).reshape(-1)
            summary = self._nonfinite_summary("gmr_qpos", qpos_curr)
            if summary is not None:
                self._debug_warn(
                    "retarget_qpos_nonfinite",
                    f"[Error] retarget output contains nan/inf | seq={payload.get('seq', 0)} | {summary}",
                )
                continue
            recv_ns = int(payload["recv_ns"])
            self._append_retarget_frame(recv_ns=recv_ns, qpos=qpos_curr)
            if self.recorder is not None:
                self.recorder.append_gmr(
                    seq=int(payload.get("seq", 0)),
                    recv_ns=recv_ns,
                    qpos=qpos_curr,
                )
            self.retarget_count += 1

            if self.viewer is not None:
                with self.vis_lock:
                    self.latest_vis_qpos = qpos_curr.astype(np.float32, copy=True)
                    self.latest_vis_human_motion = payload.get("human_motion_data")

    def _drain_requests_blocking(self) -> tuple[Optional[Dict[str, Any]], Optional[int], int]:
        import zmq

        poller = zmq.Poller()
        poller.register(self.req_sock, zmq.POLLIN)

        while not self.stop_event.is_set():
            events = dict(poller.poll(timeout=100))
            if self.req_sock not in events:
                continue

            latest_req: Optional[Dict[str, Any]] = None
            req_recv_ns: Optional[int] = None
            merged_reqs = 0
            any_start = False

            while True:
                try:
                    raw = self.req_sock.recv_string(flags=zmq.NOBLOCK)
                    req_recv_ns = time.monotonic_ns()
                except zmq.Again:
                    break
                except Exception as exc:
                    print(f"[Warning] request recv failed: {exc}")
                    break

                try:
                    req = json.loads(raw)
                except Exception:
                    print("[Warning] bad request JSON")
                    continue
                if not isinstance(req, dict):
                    continue

                merged_reqs += 1
                any_start = any_start or bool(req.get("start", False))
                latest_req = req

            if latest_req is None:
                continue

            latest_req["start"] = any_start
            return latest_req, req_recv_ns, merged_reqs

        return None, None, 0

    def _build_pose_payload(
        self,
        *,
        req_recv_ns: int,
        req: Dict[str, Any],
        merged_reqs: int,
        transport: str,
        stream_seq: int = 0,
        stream_send_gap_ms: Optional[float] = None,
        count_fallback: bool = True,
    ) -> tuple[
        Dict[str, Any],
        float,
        Optional[float],
        Optional[float],
        Optional[float],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        out_frames, used_fallback, sample_info = self._build_reply_frames(req_recv_ns=req_recv_ns)
        reply_process_ms = round((time.monotonic_ns() - int(req_recv_ns)) / 1e6, 3)
        self.latest_reply_process_ms = reply_process_ms
        self._update_debug_info(sample_info=sample_info, req_recv_ns=req_recv_ns)
        if used_fallback and count_fallback:
            self.fallback_count += 1
            self._warn_on_fallback(sample_info=sample_info)

        seq_start = int(self.frame_seq)
        self.frame_seq += len(out_frames)

        health = self._get_input_health_snapshot()
        retarget_age_ms = health.get("retarget_age_ms")
        raw_motion_age_ms = health.get("raw_motion_age_ms")
        controller_age_ms = health.get("controller_age_ms")

        payload: Dict[str, Any] = {
            "transport": transport,
            "start": bool(req.get("start", False)),
            "no_interp_applied": bool(used_fallback),
            "chunk_size": len(out_frames),
            "frame_seq_start": seq_start,
            "req_seq": int(req.get("req_seq", 0)),
            "retarget_age_ms": retarget_age_ms,
            "raw_motion_age_ms": raw_motion_age_ms,
            "controller_age_ms": controller_age_ms,
            "callback_gap_ms": health.get("callback_gap_ms"),
            "reply_process_ms": reply_process_ms,
            "sample_mode": sample_info.get("mode"),
            "buffer_len": int(sample_info.get("buffer_len", 0)),
            "req_count": int(self.req_count),
            "req_merged": int(merged_reqs),
            "callback_count": int(health.get("callback_count", 0)),
            "retarget_count": int(health.get("retarget_count", 0)),
            "raw_seq": int(health.get("raw_seq", 0)),
            "controller_seq": int(health.get("controller_seq", 0)),
            "body_available": bool(health.get("body_available", False)),
            "motion_timestamp_duplicate": bool(health.get("motion_timestamp_duplicate", False)),
            "duplicate_motion_timestamp_count": int(health.get("duplicate_motion_timestamp_count", 0)),
            "body_unavailable_count": int(health.get("body_unavailable_count", 0)),
            "motion_timestamp_missing_count": int(health.get("motion_timestamp_missing_count", 0)),
            "robot_req_send_gap_ms": req.get("robot_req_send_gap_ms"),
            "robot_post_step_gap_ms": req.get("robot_post_step_gap_ms"),
            "robot_future_horizon": req.get("robot_future_horizon"),
            "robot_vr_active": req.get("robot_vr_active"),
            "robot_vr_pending": req.get("robot_vr_pending"),
            "t_rep_ms": int(time.time() * 1000),
            "frames": [self._serialize_qpos_frame(x) for x in out_frames],
        }

        if transport == "udp_stream":
            with self.latest_vr_lock:
                buttons = dict(self.last_controller_buttons)
                pressed_edges = self._active_controller_pressed_edges_locked()
            payload.update(
                {
                    "stream": True,
                    "stream_seq": int(stream_seq),
                    "ctrl_seq": int(stream_seq),
                    "t_ms": int(time.time() * 1000),
                    "pc_udp_stream_send_gap_ms": None
                    if stream_send_gap_ms is None
                    else round(float(stream_send_gap_ms), 3),
                    "pc_ctrl_send_gap_ms": None
                    if stream_send_gap_ms is None
                    else round(float(stream_send_gap_ms), 3),
                    "controller_buttons": buttons,
                    "pressed_edges": pressed_edges,
                }
            )

        return payload, reply_process_ms, retarget_age_ms, raw_motion_age_ms, controller_age_ms, sample_info, health

    def _request_loop(self) -> None:
        import zmq

        while not self.stop_event.is_set():
            req, req_recv_ns, merged_reqs = self._drain_requests_blocking()
            if req is None or req_recv_ns is None:
                continue

            now = time.monotonic()
            self.req_count += 1
            self.req_merged_total += int(merged_reqs)
            self.latest_merged_reqs = int(merged_reqs)
            if self.last_req_monotonic is None:
                self.latest_req_dt_ms = None
            else:
                self.latest_req_dt_ms = (now - self.last_req_monotonic) * 1000.0
            self.last_req_monotonic = now
            if self.latest_req_dt_ms is not None and self.latest_req_dt_ms >= 100.0:
                self._debug_warn(
                    "robot_request_gap",
                    "[Warning] robot request receive gap | "
                    f"pc_req_dt_ms={self.latest_req_dt_ms:.2f}, merged={int(merged_reqs)}, "
                    f"req_seq={req.get('req_seq')}, "
                    f"robot_req_send_gap_ms={req.get('robot_req_send_gap_ms')}, "
                    f"robot_post_step_gap_ms={req.get('robot_post_step_gap_ms')}, "
                    f"robot_h={req.get('robot_future_horizon')}, "
                    f"robot_active={req.get('robot_vr_active')}",
                    interval_s=0.0,
                )

            (
                payload,
                reply_process_ms,
                retarget_age_ms,
                raw_motion_age_ms,
                controller_age_ms,
                sample_info,
                health,
            ) = self._build_pose_payload(
                req_recv_ns=req_recv_ns,
                req=req,
                merged_reqs=merged_reqs,
                transport="zmq",
            )
            reply_should_warn = (
                reply_process_ms >= self.debug_reply_process_warn_ms
                or (retarget_age_ms is not None and float(retarget_age_ms) >= self.debug_retarget_stall_warn_ms)
                or (raw_motion_age_ms is not None and float(raw_motion_age_ms) >= self.debug_retarget_stall_warn_ms)
            )
            if reply_should_warn:
                self._debug_warn(
                    "slow_or_stale_reply",
                    "[Warning] ZMQ reply health | "
                    f"reply_process_ms={reply_process_ms}, req={int(self.req_count)}, "
                    f"merged={int(merged_reqs)}, mode={sample_info.get('mode')}, "
                    f"buffer={int(sample_info.get('buffer_len', 0))}, "
                    f"retarget_age_ms={retarget_age_ms}, raw_motion_age_ms={raw_motion_age_ms}, "
                    f"controller_age_ms={controller_age_ms}, callback_gap_ms={health.get('callback_gap_ms')}, "
                    f"cb={int(health.get('callback_count', 0))}, retarget={int(health.get('retarget_count', 0))}, "
                    f"raw_seq={int(health.get('raw_seq', 0))}, body_available={bool(health.get('body_available', False))}, "
                    f"motion_ts_duplicate={bool(health.get('motion_timestamp_duplicate', False))}",
                )

            try:
                self.rep_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
                self.reply_count += 1
            except zmq.Again:
                self.reply_drop_count += 1
                print("[Warning] reply queue full, drop one reply")
            except Exception as exc:
                print(f"[Warning] reply send failed: {exc}")

    def _stats_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.stop_event.wait(timeout=self.log_interval_s):
                break

            with self.stats_lock:
                info = dict(self.latest_debug_info)
            with self.latest_vr_lock:
                callback_count = int(self.callback_count)
                raw_seq_snapshot = int(self.latest_vr_seq)
                controller_seq_snapshot = int(self.latest_controller_seq)
                body_unavailable_count_snapshot = int(self.body_unavailable_count)
                duplicate_motion_timestamp_count_snapshot = int(self.duplicate_motion_timestamp_count)
                motion_timestamp_missing_count_snapshot = int(self.motion_timestamp_missing_count)
            retarget_count = int(self.retarget_count)
            req_count = int(self.req_count)
            reply_count = int(self.reply_count)
            reply_drop_count = int(self.reply_drop_count)
            udp_stream_count = int(self.udp_stream_count)
            udp_stream_drop_count = int(self.udp_stream_drop_count)
            req_merged_total = int(self.req_merged_total)
            fallback_count = int(self.fallback_count)
            raw_motion_drop_count = int(self.raw_motion_drop_count)
            retarget_input_skip_count = int(self.retarget_input_skip_count)
            latest_merged_reqs = int(self.latest_merged_reqs)
            latest_req_dt_ms = self.latest_req_dt_ms
            retarget_age_ms = info.get("retarget_age_ms")
            raw_motion_age_ms = info.get("raw_motion_age_ms")
            controller_age_ms = info.get("controller_age_ms")
            callback_gap_ms = info.get("callback_gap_ms")
            reply_process_ms = info.get("reply_process_ms")
            now_mono = time.monotonic()

            if self.debug_pico_health_interval_s > 0.0:
                should_log_pico_health = (
                    self._last_pico_health_log_monotonic is None
                    or (now_mono - self._last_pico_health_log_monotonic) >= self.debug_pico_health_interval_s
                )
                if should_log_pico_health:
                    if self._last_pico_health_log_monotonic is None:
                        interval_s = None
                        cb_rate_hz = None
                        raw_rate_hz = None
                        controller_rate_hz = None
                    else:
                        interval_s = max(1e-6, now_mono - self._last_pico_health_log_monotonic)
                        cb_rate_hz = (callback_count - self._last_pico_health_callback_count) / interval_s
                        raw_rate_hz = (raw_seq_snapshot - self._last_pico_health_raw_seq) / interval_s
                        controller_rate_hz = (
                            controller_seq_snapshot - self._last_pico_health_controller_seq
                        ) / interval_s
                    print(
                        "[PICO][Health] "
                        f"cb={callback_count}, raw_seq={raw_seq_snapshot}, "
                        f"controller_seq={controller_seq_snapshot}, "
                        f"cb_rate_hz={None if cb_rate_hz is None else round(cb_rate_hz, 1)}, "
                        f"raw_rate_hz={None if raw_rate_hz is None else round(raw_rate_hz, 1)}, "
                        f"controller_rate_hz={None if controller_rate_hz is None else round(controller_rate_hz, 1)}, "
                        f"callback_gap_ms={callback_gap_ms}, "
                        f"raw_motion_age_ms={raw_motion_age_ms}, "
                        f"controller_age_ms={controller_age_ms}, "
                        f"body_available={info.get('body_available')}, "
                        f"body_unavailable={body_unavailable_count_snapshot} "
                        f"(+{body_unavailable_count_snapshot - self._last_pico_health_body_unavailable_count}), "
                        f"dup_motion_ts={duplicate_motion_timestamp_count_snapshot} "
                        f"(+{duplicate_motion_timestamp_count_snapshot - self._last_pico_health_duplicate_motion_timestamp_count}), "
                        f"missing_motion_ts={motion_timestamp_missing_count_snapshot} "
                        f"(+{motion_timestamp_missing_count_snapshot - self._last_pico_health_motion_timestamp_missing_count})"
                    )
                    self._last_pico_health_log_monotonic = now_mono
                    self._last_pico_health_callback_count = callback_count
                    self._last_pico_health_raw_seq = raw_seq_snapshot
                    self._last_pico_health_controller_seq = controller_seq_snapshot
                    self._last_pico_health_body_unavailable_count = body_unavailable_count_snapshot
                    self._last_pico_health_duplicate_motion_timestamp_count = duplicate_motion_timestamp_count_snapshot
                    self._last_pico_health_motion_timestamp_missing_count = motion_timestamp_missing_count_snapshot

            raw_is_fresh = (
                raw_motion_age_ms is not None
                and float(raw_motion_age_ms) <= self.debug_raw_fresh_ms
            )
            retarget_is_stale = (
                retarget_age_ms is None
                or float(retarget_age_ms) >= self.debug_retarget_stall_warn_ms
            )
            if raw_is_fresh and retarget_is_stale:
                self._debug_warn(
                    "retarget_stall",
                    "[Error] retarget stalled while raw input is fresh | "
                    f"raw_motion_age_ms={raw_motion_age_ms}, "
                    f"retarget_age_ms={retarget_age_ms}, "
                    f"cb={callback_count}, retarget={retarget_count}, "
                    f"raw_drop={raw_motion_drop_count}, raw_skip={retarget_input_skip_count}, "
                    f"fallback={fallback_count}, "
                    f"mode={info.get('mode')}, buffer={info.get('buffer_len')}, "
                    f"controller_age_ms={controller_age_ms}, callback_gap_ms={callback_gap_ms}, "
                    f"raw_seq={info.get('raw_seq')}, body_available={info.get('body_available')}, "
                    f"motion_ts_duplicate={info.get('motion_timestamp_duplicate')}",
                )
            elif raw_motion_age_ms is not None and float(raw_motion_age_ms) >= self.debug_retarget_stall_warn_ms:
                self._debug_warn(
                    "raw_stall",
                    "[Error] raw PICO/XRobot input stalled | "
                    f"raw_motion_age_ms={raw_motion_age_ms}, "
                    f"retarget_age_ms={retarget_age_ms}, controller_age_ms={controller_age_ms}, "
                    f"callback_gap_ms={callback_gap_ms}, cb={callback_count}, retarget={retarget_count}, "
                    f"raw_seq={info.get('raw_seq')}, controller_seq={info.get('controller_seq')}, "
                    f"body_available={info.get('body_available')}, "
                    f"motion_ts_duplicate={info.get('motion_timestamp_duplicate')}, "
                    f"dup_motion_ts={info.get('duplicate_motion_timestamp_count')}, "
                    f"body_unavailable={info.get('body_unavailable_count')}, "
                    f"missing_motion_ts={info.get('motion_timestamp_missing_count')}",
                )

            alpha = info.get("alpha")
            alpha_str = "None" if alpha is None else f"{float(alpha):.3f}"
            req_dt_str = "None" if latest_req_dt_ms is None else f"{float(latest_req_dt_ms):.2f}"
            print(
                "[Stats] "
                f"req={req_count}, rep={reply_count}, rep_drop={reply_drop_count}, "
                f"udp_stream={udp_stream_count}, udp_drop={udp_stream_drop_count}, "
                f"req_merged_total={req_merged_total}, latest_merged={latest_merged_reqs}, "
                f"fallback={fallback_count}, raw_drop={raw_motion_drop_count}, "
                f"raw_skip={retarget_input_skip_count}, "
                f"cb={callback_count}, retarget={retarget_count}, "
                f"mode={info.get('mode')}, buffer={info.get('buffer_len')}, "
                f"latest_req_dt_ms={req_dt_str}, "
                f"reply_process_ms={reply_process_ms}, "
                f"target_age_ms={info.get('target_age_ms')}, "
                f"older_age_ms={info.get('older_age_ms')}, "
                f"newer_age_ms={info.get('newer_age_ms')}, "
                f"span_ms={info.get('span_ms')}, alpha={alpha_str}, "
                f"retarget_age_ms={info.get('retarget_age_ms')}, "
                f"raw_motion_age_ms={info.get('raw_motion_age_ms')}, "
                f"controller_age_ms={controller_age_ms}, "
                f"callback_gap_ms={callback_gap_ms}, "
                f"raw_seq={info.get('raw_seq')}, "
                f"controller_seq={info.get('controller_seq')}, "
                f"body_available={info.get('body_available')}, "
                f"motion_ts_duplicate={info.get('motion_timestamp_duplicate')}, "
                f"dup_motion_ts={info.get('duplicate_motion_timestamp_count')}, "
                f"body_unavailable={info.get('body_unavailable_count')}, "
                f"missing_motion_ts={info.get('motion_timestamp_missing_count')}"
            )

    def _recorder_command_loop(self) -> None:
        if self.recorder is None:
            return
        print("[Recorder] keyboard controls: r=start/resume, p=stop+save, q=save+quit")
        while not self.stop_event.is_set():
            try:
                cmd = input().strip().lower()
            except EOFError:
                return
            except Exception:
                continue
            if cmd in ("r", "record", "start"):
                self.recorder.start()
            elif cmd in ("p", "pause", "stop", "save"):
                self.recorder.stop_and_save()
            elif cmd in ("q", "quit", "exit"):
                self.recorder.stop_and_save()
                self.stop_event.set()
                self.vr_frame_event.set()
                return
            elif cmd in ("h", "help", "?"):
                print("[Recorder] keyboard controls: r=start/resume, p=stop+save, q=save+quit")
            elif cmd:
                print(f"[Recorder] unknown command '{cmd}'. Use r, p, q, or h.")

    def _recorder_status_loop(self) -> None:
        if self.recorder is None or self.record_status_interval_s <= 0.0:
            return
        while not self.stop_event.wait(timeout=self.record_status_interval_s):
            status = self.recorder.status_snapshot()
            if not status["recording"]:
                continue
            print(
                "[Recorder] status | "
                f"state=recording segment={int(status['segment']):08d} "
                f"elapsed_s={float(status['elapsed_s']):.1f} raw={int(status['raw'])} "
                f"gmr={int(status['gmr'])}"
            )

    def _control_loop(self) -> None:
        import zmq

        period_s = 1.0 / float(self.ctrl_fps)
        while not self.stop_event.is_set():
            now = time.monotonic()
            if self.last_ctrl_send_monotonic is None:
                ctrl_send_gap_ms = None
            else:
                ctrl_send_gap_ms = (now - self.last_ctrl_send_monotonic) * 1000.0
                if ctrl_send_gap_ms >= 100.0:
                    self._debug_warn(
                        "pc_control_send_gap",
                        "[Warning] PC control send loop gap | "
                        f"gap_ms={ctrl_send_gap_ms:.2f}, ctrl_seq={self.ctrl_send_count + 1}",
                        interval_s=0.0,
                    )
            self.last_ctrl_send_monotonic = now
            self.ctrl_send_count += 1

            with self.latest_vr_lock:
                buttons = dict(self.last_controller_buttons)
                pressed_edges = self._active_controller_pressed_edges_locked()
            with self.stream_payload_lock:
                pose_payload = self.latest_udp_stream_payload

            payload = {
                "t_ms": int(time.time() * 1000),
                "ctrl_seq": int(self.ctrl_send_count),
                "pc_ctrl_send_gap_ms": None
                if ctrl_send_gap_ms is None
                else round(float(ctrl_send_gap_ms), 3),
                "controller_buttons": buttons,
                "pressed_edges": pressed_edges,
            }
            if isinstance(pose_payload, dict):
                payload["pose_payload"] = pose_payload
            try:
                self.ctrl_sock.send_string(json.dumps(payload), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
            except Exception as exc:
                print(f"[Warning] control send failed: {exc}")

            self.stop_event.wait(timeout=period_s)

    def _udp_stream_loop(self) -> None:
        if self.udp_stream_socket is None or self.udp_stream_addr is None:
            return

        period_s = 1.0 / float(self.udp_stream_fps)
        while not self.stop_event.is_set():
            now = time.monotonic()
            if self.last_udp_stream_send_monotonic is None:
                stream_send_gap_ms = None
            else:
                stream_send_gap_ms = (now - self.last_udp_stream_send_monotonic) * 1000.0
                if stream_send_gap_ms >= 100.0:
                    self._debug_warn(
                        "pc_udp_stream_send_gap",
                        "[Warning] PC UDP stream send loop gap | "
                        f"gap_ms={stream_send_gap_ms:.2f}, stream_seq={self.udp_stream_count + 1}",
                        interval_s=0.0,
                    )
            self.last_udp_stream_send_monotonic = now
            self.udp_stream_count += 1
            stream_seq = int(self.udp_stream_count)
            req_recv_ns = time.monotonic_ns()
            req = {
                "req_seq": 0,
                "start": False,
                "robot_req_send_gap_ms": None,
                "robot_post_step_gap_ms": None,
                "robot_future_horizon": None,
                "robot_vr_active": None,
                "robot_vr_pending": None,
            }
            (
                payload,
                reply_process_ms,
                retarget_age_ms,
                raw_motion_age_ms,
                controller_age_ms,
                sample_info,
                health,
            ) = self._build_pose_payload(
                req_recv_ns=req_recv_ns,
                req=req,
                merged_reqs=0,
                transport="udp_stream",
                stream_seq=stream_seq,
                stream_send_gap_ms=stream_send_gap_ms,
                count_fallback=False,
            )

            stream_should_warn = (
                reply_process_ms >= self.debug_reply_process_warn_ms
                or (retarget_age_ms is not None and float(retarget_age_ms) >= self.debug_retarget_stall_warn_ms)
                or (raw_motion_age_ms is not None and float(raw_motion_age_ms) >= self.debug_retarget_stall_warn_ms)
            )
            if stream_should_warn:
                self._debug_warn(
                    "slow_or_stale_udp_stream",
                    "[Warning] UDP stream health | "
                    f"reply_process_ms={reply_process_ms}, stream_seq={stream_seq}, "
                    f"mode={sample_info.get('mode')}, buffer={int(sample_info.get('buffer_len', 0))}, "
                    f"retarget_age_ms={retarget_age_ms}, raw_motion_age_ms={raw_motion_age_ms}, "
                    f"controller_age_ms={controller_age_ms}, callback_gap_ms={health.get('callback_gap_ms')}, "
                    f"cb={int(health.get('callback_count', 0))}, retarget={int(health.get('retarget_count', 0))}, "
                    f"raw_seq={int(health.get('raw_seq', 0))}, body_available={bool(health.get('body_available', False))}, "
                    f"motion_ts_duplicate={bool(health.get('motion_timestamp_duplicate', False))}",
                )

            with self.stream_payload_lock:
                self.latest_udp_stream_payload = payload
            try:
                raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.udp_stream_socket.sendto(raw, self.udp_stream_addr)
            except (BlockingIOError, InterruptedError):
                self.udp_stream_drop_count += 1
            except Exception as exc:
                self.udp_stream_drop_count += 1
                self._debug_warn(
                    "udp_stream_send_failed",
                    f"[Warning] UDP stream send failed: {exc}",
                )

            self.stop_event.wait(timeout=period_s)

    def _visualization_loop(self) -> None:
        if self.viewer is None:
            return

        period_s = 1.0 / float(self.vis_fps)
        while not self.stop_event.is_set():
            with self.vis_lock:
                qpos = None if self.latest_vis_qpos is None else self.latest_vis_qpos.copy()
                human_motion_data = self.latest_vis_human_motion

            if qpos is not None:
                try:
                    self.viewer.step(
                        root_pos=qpos[:3],
                        root_rot=qpos[3:7],
                        dof_pos=qpos[7:36],
                        human_motion_data=human_motion_data,
                        rate_limit=False,
                        follow_camera=True,
                    )
                except Exception as exc:
                    print(f"[Warning] visualization failed, disabling viewer: {exc}")
                    self.viewer.close()
                    self.viewer = None
                    return

            self.stop_event.wait(timeout=period_s)

    def setup(self) -> None:
        try:
            import zmq
        except ImportError as exc:
            raise ImportError("pyzmq is required for the teleop ZMQ server.") from exc

        if self.args.visualize:
            self.viewer = RobotMotionViewer(
                robot_type=self.robot,
                motion_fps=self.vis_fps,
                transparent_robot=1,
            )

        self.raw_recv_conn, self.raw_send_conn = self.mp_ctx.Pipe(duplex=False)
        self.result_recv_conn, self.result_send_conn = self.mp_ctx.Pipe(duplex=False)
        worker_config = {
            "actual_human_height": float(self.args.actual_human_height),
            "gmr_max_iter": int(self.gmr_max_iter),
            "send_human_motion": bool(self.args.visualize),
            "min_link_height": self.min_link_height,
            "min_link_height_align_strategy": self.min_link_height_align_strategy,
            "min_link_height_bootstrap_frames": self.min_link_height_bootstrap_frames,
        }
        self.retarget_process = self.mp_ctx.Process(
            target=_retarget_worker_main,
            args=(self.raw_recv_conn, self.result_send_conn, worker_config),
            name="teleop-retarget-worker",
            daemon=True,
        )
        self.retarget_process.start()
        self.raw_recv_conn.close()
        self.raw_recv_conn = None
        self.result_send_conn.close()
        self.result_send_conn = None

        ready_deadline = time.monotonic() + self.retarget_worker_ready_timeout_s
        while not self.result_recv_conn.poll(0.25):
            if self.retarget_process is not None and not self.retarget_process.is_alive():
                raise RuntimeError(
                    "Retarget worker exited before becoming ready "
                    f"(exitcode={self.retarget_process.exitcode})."
                )
            if time.monotonic() >= ready_deadline:
                pid = self.retarget_process.pid if self.retarget_process is not None else None
                alive = self.retarget_process.is_alive() if self.retarget_process is not None else False
                raise RuntimeError(
                    "Retarget worker did not become ready within "
                    f"{self.retarget_worker_ready_timeout_s:.1f} seconds "
                    f"(pid={pid}, alive={alive})."
                )
        worker_msg = self.result_recv_conn.recv()
        if not isinstance(worker_msg, dict) or worker_msg.get("type") != "worker_ready":
            raise RuntimeError(f"Retarget worker failed to start: {worker_msg}")

        xrt.init()
        xrt.register_frame_callback(self._on_vr_frame)

        if self.tiny2_head_follow_enabled:
            self.tiny2_head_follow_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.tiny2_head_follow_socket.setblocking(False)

        self.zmq_context = zmq.Context.instance()

        self.req_sock = self.zmq_context.socket(zmq.PULL)
        self._configure_latest_zmq_socket(zmq, self.req_sock, recv=True)
        self.req_sock.bind(self.args.req_bind_addr)

        self.rep_sock = self.zmq_context.socket(zmq.PUSH)
        self._configure_latest_zmq_socket(zmq, self.rep_sock, send=True)
        self.rep_sock.bind(self.args.rep_bind_addr)

        self.ctrl_sock = self.zmq_context.socket(zmq.PUSH)
        self._configure_latest_zmq_socket(zmq, self.ctrl_sock, send=True)
        self.ctrl_sock.bind(self.args.ctrl_bind_addr)

        if self.udp_stream_host:
            self.udp_stream_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            if self.udp_stream_qos:
                try:
                    self.udp_stream_socket.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0xB8)
                except Exception as exc:
                    print(f"[Warning] failed to set UDP stream IP_TOS: {exc}")
                try:
                    self.udp_stream_socket.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)
                except Exception as exc:
                    print(f"[Warning] failed to set UDP stream SO_PRIORITY: {exc}")
            self.udp_stream_socket.setblocking(False)
            self.udp_stream_addr = (self.udp_stream_host, self.udp_stream_port)

        print("Low-latency teleop ZMQ pose server initialized")
        print(f"  req_bind_addr: {self.args.req_bind_addr}")
        print(f"  rep_bind_addr: {self.args.rep_bind_addr}")
        print(f"  ctrl_bind_addr: {self.args.ctrl_bind_addr}")
        if self.udp_stream_addr is not None:
            print(
                "  udp_stream_target: "
                f"udp://{self.udp_stream_addr[0]}:{self.udp_stream_addr[1]} "
                f"fps={self.udp_stream_fps:.1f} "
                f"qos={'on_tos_0xb8_priority_6' if self.udp_stream_qos else 'off'}"
            )
        print(f"  ctrl_fps: {self.ctrl_fps}")
        print(f"  retarget_fps: {self.retarget_fps:.3f}")
        print("  zmq_hwm: 64")
        print("  zmq_conflate: False")
        print(f"  gmr_max_iter: {self.gmr_max_iter}")
        print(f"  retarget_worker_ready_timeout_s: {self.retarget_worker_ready_timeout_s:.1f}")
        print("  chunk_size: fixed to 1 frame per reply")
        print(f"  lookback_ms: {self.lookback_ns / 1e6:.3f}")
        print(f"  retarget_buffer_window_s: {self.retarget_buffer_window_ns / 1e9:.3f}")
        print(f"  log_interval_s: {self.log_interval_s:.3f}")
        print(f"  visualize: {self.args.visualize}")
        if self.tiny2_head_follow_enabled:
            print(
                "  tiny2_head_follow: "
                f"udp://{self.tiny2_head_follow_addr[0]}:{self.tiny2_head_follow_addr[1]} "
                f"rate={float(self.args.tiny2_head_follow_rate_hz):.1f}Hz "
                f"axis={self.args.tiny2_head_forward_axis}"
            )
        if self.recorder is not None:
            self.recorder.print_ready(
                keyboard_control=not self.args.pico_record_disable_keyboard,
                status_interval_s=self.record_status_interval_s,
            )
            print(f"  pico_record_output_dir: {self.recorder.output_dir}")
            print(f"  gmr_record_output_dir: {self.recorder.gmr_output_dir}")
            print(f"  pico_record_auto_start: {self.args.pico_record_auto_start}")
            print(f"  pico_record_include_snapshot_json: {self.recorder.include_snapshot_json}")
        print(f"  retarget_worker_pid: {self.retarget_process.pid if self.retarget_process else None}")

    def run(self) -> None:
        self.setup()

        self.raw_sender_thread = threading.Thread(
            target=self._raw_sender_loop,
            name="teleop-raw-sender",
            daemon=True,
        )
        self.worker_result_thread = threading.Thread(
            target=self._worker_result_loop,
            name="teleop-worker-result",
            daemon=True,
        )
        self.request_thread = threading.Thread(
            target=self._request_loop,
            name="teleop-request",
            daemon=True,
        )
        self.control_thread = threading.Thread(
            target=self._control_loop,
            name="teleop-control",
            daemon=True,
        )
        if self.udp_stream_socket is not None:
            self.udp_stream_thread = threading.Thread(
                target=self._udp_stream_loop,
                name="teleop-udp-stream",
                daemon=True,
            )
        if self.viewer is not None:
            self.visualization_thread = threading.Thread(
                target=self._visualization_loop,
                name="teleop-visualization",
                daemon=True,
            )
        if self.log_interval_s > 0.0:
            self.stats_thread = threading.Thread(
                target=self._stats_loop,
                name="teleop-stats",
                daemon=True,
            )
        if self.recorder is not None and not self.args.pico_record_disable_keyboard:
            self.recorder_command_thread = threading.Thread(
                target=self._recorder_command_loop,
                name="teleop-recorder-command",
                daemon=True,
            )
        if self.recorder is not None and self.record_status_interval_s > 0.0:
            self.recorder_status_thread = threading.Thread(
                target=self._recorder_status_loop,
                name="teleop-recorder-status",
                daemon=True,
            )

        self.raw_sender_thread.start()
        self.worker_result_thread.start()
        self.request_thread.start()
        self.control_thread.start()
        if self.udp_stream_thread is not None:
            self.udp_stream_thread.start()
        if self.visualization_thread is not None:
            self.visualization_thread.start()
        if self.stats_thread is not None:
            self.stats_thread.start()
        if self.recorder_command_thread is not None:
            self.recorder_command_thread.start()
        if self.recorder_status_thread is not None:
            self.recorder_status_thread.start()

        try:
            while not self.stop_event.is_set():
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("KeyboardInterrupt, exiting low-latency teleop ZMQ pose server.")
        finally:
            self.stop_event.set()
            self.vr_frame_event.set()
            if self.recorder is not None:
                self.recorder.stop_and_save()
            try:
                xrt.clear_frame_callback()
            except Exception:
                pass
            if self.tiny2_head_follow_socket is not None:
                self.tiny2_head_follow_socket.close()
                self.tiny2_head_follow_socket = None
            if self.udp_stream_socket is not None:
                self.udp_stream_socket.close()
                self.udp_stream_socket = None

            for thread in (
                self.raw_sender_thread,
                self.worker_result_thread,
                self.request_thread,
                self.control_thread,
                self.udp_stream_thread,
                self.visualization_thread,
                self.stats_thread,
                self.recorder_command_thread,
                self.recorder_status_thread,
            ):
                if thread is not None:
                    thread.join(timeout=1.0)

            if self.raw_send_conn is not None:
                try:
                    self.raw_send_conn.send({"type": "shutdown"})
                except Exception:
                    pass
            if self.raw_send_conn is not None:
                self.raw_send_conn.close()
            if self.raw_recv_conn is not None:
                self.raw_recv_conn.close()
            if self.result_send_conn is not None:
                self.result_send_conn.close()
            if self.result_recv_conn is not None:
                self.result_recv_conn.close()
            if self.retarget_process is not None:
                self.retarget_process.join(timeout=2.0)
                if self.retarget_process.is_alive():
                    self.retarget_process.terminate()
                    self.retarget_process.join(timeout=1.0)

            if self.viewer is not None:
                self.viewer.close()
            if self.req_sock is not None:
                self.req_sock.close(0)
            if self.rep_sock is not None:
                self.rep_sock.close(0)
            if self.ctrl_sock is not None:
                self.ctrl_sock.close(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency ZMQ teleop pose server")
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands"],
        default="unitree_g1",
        help="Robot key for defaults",
    )
    parser.add_argument("--actual_human_height", type=float, default=1.6)
    parser.add_argument("--vis_fps", type=int, default=10, help="Viewer update frequency")
    parser.add_argument("--ctrl_fps", type=int, default=50, help="Controller button publish frequency")
    parser.add_argument(
        "--retarget_fps",
        type=float,
        default=60.0,
        help="Maximum retarget rate. The latest PICO/XRobot frame is sampled at this fixed rate.",
    )
    parser.add_argument(
        "--retarget_worker_ready_timeout_s",
        type=float,
        default=60.0,
        help="How long to wait for the GMR retarget worker to finish initialization.",
    )
    parser.add_argument(
        "--lookback_ms",
        type=float,
        default=15.0,
        help="Sample reply frames at request_time - lookback_ms",
    )
    parser.add_argument(
        "--retarget_buffer_window_s",
        type=float,
        default=0.5,
        help="How much retarget history to keep for timestamp interpolation",
    )
    parser.add_argument(
        "--log_interval_s",
        type=float,
        default=1.0,
        help="Periodic debug log interval. Set to 0 to disable.",
    )
    parser.add_argument(
        "--debug_log_file",
        type=str,
        default="",
        help="Optional file path. When set, tee stdout/stderr to this log file.",
    )
    parser.add_argument(
        "--debug_warning_interval_s",
        type=float,
        default=1.0,
        help="Rate limit repeated disconnect/stall warnings.",
    )
    parser.add_argument(
        "--debug_retarget_stall_warn_ms",
        type=float,
        default=500.0,
        help="Warn when retarget output is older than this while raw input is fresh.",
    )
    parser.add_argument(
        "--debug_raw_fresh_ms",
        type=float,
        default=200.0,
        help="Raw input age below this threshold is considered fresh.",
    )
    parser.add_argument(
        "--debug_reply_process_warn_ms",
        type=float,
        default=50.0,
        help="Warn when request handling plus reply construction exceeds this duration.",
    )
    parser.add_argument(
        "--debug_pico_callback_warn_ms",
        type=float,
        default=100.0,
        help="Warn when the PICO/XRobot frame callback gap exceeds this duration. Set 0 to disable.",
    )
    parser.add_argument(
        "--debug_pico_health_interval_s",
        type=float,
        default=1.0,
        help="Periodic PICO/XRobot health log interval. Set 0 to disable.",
    )
    parser.add_argument("--req_bind_addr", type=str, default="tcp://*:28701")
    parser.add_argument("--rep_bind_addr", type=str, default="tcp://*:28702")
    parser.add_argument("--ctrl_bind_addr", type=str, default="tcp://*:28703")
    parser.add_argument(
        "--udp_stream_host",
        type=str,
        default="127.0.0.1",
        help="MolmoSpaces host/IP for UDP latest-state pose/control stream. Empty disables UDP stream.",
    )
    parser.add_argument("--udp_stream_port", type=int, default=28704)
    parser.add_argument("--udp_stream_fps", type=float, default=50.0)
    parser.add_argument(
        "--udp_stream_qos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set DSCP/WMM-like low-latency marks on UDP stream packets.",
    )
    parser.add_argument("--min_link_height", type=float, default=0.0)
    parser.add_argument(
        "--min_link_height_align_strategy",
        type=str,
        choices=["startup_fixed", "per_frame"],
        default="startup_fixed",
    )
    parser.add_argument("--min_link_height_bootstrap_frames", type=int, default=10)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--pico_record_output_dir",
        type=str,
        default="",
        help="Enable raw PICO/XRobot npz capture and write sessions to this directory.",
    )
    parser.add_argument(
        "--pico_record_prefix",
        type=str,
        default="pico_raw",
        help="Filename prefix for raw PICO capture files.",
    )
    parser.add_argument(
        "--gmr_record_output_dir",
        type=str,
        default="",
        help="Optional output directory for synchronized online GMR/MT files.",
    )
    parser.add_argument(
        "--gmr_record_prefix",
        type=str,
        default="gmr_mt",
        help="Filename prefix for synchronized online GMR/MT files.",
    )
    parser.add_argument(
        "--pico_record_auto_start",
        action="store_true",
        help="Start raw PICO recording immediately.",
    )
    parser.add_argument(
        "--pico_record_enable_buttons",
        action="store_true",
        help="Enable controller recorder controls.",
    )
    parser.add_argument(
        "--pico_record_disable_buttons",
        action="store_true",
        help="Disable controller recorder controls.",
    )
    parser.add_argument(
        "--pico_record_start_button",
        type=str,
        default="right_key_two",
        help="Controller button key used to start recording when button control is enabled.",
    )
    parser.add_argument(
        "--pico_record_stop_button",
        type=str,
        default="left_key_two",
        help="Controller button key used to stop and save recording when button control is enabled.",
    )
    parser.add_argument(
        "--pico_record_disable_keyboard",
        action="store_true",
        help="Disable stdin recorder controls (r=start, p=stop+save, q=save+quit).",
    )
    parser.add_argument(
        "--record_status_interval_s",
        type=float,
        default=5.0,
        help="Recorder-only status interval while recording. Set to 0 to disable.",
    )
    parser.add_argument(
        "--pico_record_include_snapshot_json",
        action="store_true",
        help="Also store each full raw callback snapshot as JSON strings in raw_snapshot_json.",
    )
    parser.add_argument(
        "--tiny2_head_follow_host",
        type=str,
        default="",
        help="Optional robot host/IP for OBSBOT Tiny2 UDP head-follow bridge. Empty disables publishing.",
    )
    parser.add_argument(
        "--tiny2_head_follow_port",
        type=int,
        default=29900,
        help="UDP port consumed by scripts/tiny2_udp_head_bridge.py on the robot.",
    )
    parser.add_argument(
        "--tiny2_head_follow_rate_hz",
        type=float,
        default=30.0,
        help="Maximum Tiny2 head-pose UDP publish rate.",
    )
    parser.add_argument("--tiny2_head_follow_yaw_gain", type=float, default=1.0)
    parser.add_argument("--tiny2_head_follow_pitch_gain", type=float, default=1.0)
    parser.add_argument("--tiny2_head_follow_yaw_offset", type=float, default=0.0)
    parser.add_argument("--tiny2_head_follow_pitch_offset", type=float, default=0.0)
    parser.add_argument(
        "--tiny2_head_forward_axis",
        choices=["x", "-x", "y", "-y", "z", "-z"],
        default="z",
        help="Which local headset axis should be treated as forward for Tiny2 head-follow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _enable_debug_log_file(args.debug_log_file)
    _load_runtime_dependencies()
    server = LowLatencyTeleopPoseZMQServer(args)
    server.run()


if __name__ == "__main__":
    main()

