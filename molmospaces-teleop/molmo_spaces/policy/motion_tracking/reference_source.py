import errno
import json
import os
import socket
import time
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .math_utils import _linspace_rows, _slerp, _yaw_component_wxyz
from .paths import REAL_G1_ROOT
from .utils import DictToClass, MotionUDPServer

try:
    import zmq
except Exception:
    zmq = None

if TYPE_CHECKING:
    from .core import TrackingPolicyRaw


_VR_CONTROL_EDGE_BUTTONS = {"right_key_one", "left_key_one", "right_key_two", "left_key_two"}


def remap_joint_array_by_names(
    data: np.ndarray,
    source_joint_names,
    target_joint_names,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D joint array [T, J], got shape={data.shape}")
    if data.shape[1] != len(source_joint_names):
        raise ValueError(
            f"Joint dim mismatch: data has {data.shape[1]} dims, "
            f"but source_joint_names has {len(source_joint_names)} names."
        )

    name_to_idx = {name: i for i, name in enumerate(source_joint_names)}
    remap = np.zeros((data.shape[0], len(target_joint_names)), dtype=np.float32)
    for i, name in enumerate(target_joint_names):
        j = name_to_idx.get(name, None)
        if j is not None:
            remap[:, i] = data[:, j]
    return remap


class MotionSourceBase(ABC):
    def __init__(self, policy: "TrackingPolicyRaw", policy_cfg: DictToClass):
        self.policy = policy
        self.config = policy_cfg
        self.motions: Dict[str, Dict[str, np.ndarray]] = self._load_motions()

    def _load_motion_file(self, path: str, motion_name: str, start: int = 0, end: int = -1) -> Dict[str, np.ndarray]:
        mp = Path(path)
        resolved = str(mp if mp.is_absolute() else (REAL_G1_ROOT / mp))
        data = np.load(resolved, allow_pickle=True)
        if not isinstance(data, np.lib.npyio.NpzFile):
            raise ValueError(f"[{self.__class__.__name__}] Only .npz is supported: {resolved}")

        joint_pos = data["dof_pos"][start:end].astype(np.float32)
        root_pos = data["root_pos"][start:end].astype(np.float32)
        root_rot_xyzw = data["root_rot"][start:end].astype(np.float32)
        root_quat = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)

        joint_names = data.get("joint_names", None)
        if joint_names is None:
            raise ValueError(
                f"[{self.__class__.__name__}] Motion '{motion_name}' is missing 'joint_names' in npz. "
                "Please export joint_names with the dataset."
            )
        source_joint_names = []
        for n in joint_names.tolist():
            if isinstance(n, (bytes, np.bytes_)):
                source_joint_names.append(n.decode("utf-8"))
            else:
                source_joint_names.append(str(n))
        joint_pos = remap_joint_array_by_names(joint_pos, source_joint_names, self.policy.obs_joint_names)

        return {
            "joint_pos": joint_pos,
            "root_quat": root_quat,
            "root_pos": root_pos,
        }

    def _load_motions(self) -> Dict[str, Dict[str, np.ndarray]]:
        motions: Dict[str, Dict[str, np.ndarray]] = {}

        for m in getattr(self.config, "motions", []):
            mc = DictToClass(m)
            motion_name = mc.name
            t0, t1 = int(mc.start), int(mc.end)
            motions[motion_name] = self._load_motion_file(mc.path, motion_name, t0, t1)

        for m in getattr(self.config, "motion_clips", []):
            mc = DictToClass(m)
            motion_name = mc.name
            joint_pos_1 = np.asarray(mc.joint_pos, dtype=np.float32).reshape(1, -1)
            if joint_pos_1.shape[1] != len(self.policy.dataset_joint_names):
                raise ValueError(
                    f"[{self.__class__.__name__}] Motion clip '{motion_name}' dim={joint_pos_1.shape[1]} "
                    f"does not match dataset_joint_names size={len(self.policy.dataset_joint_names)}."
                )
            source_joint_names = self.policy.dataset_joint_names
            joint_pos_1 = remap_joint_array_by_names(joint_pos_1, source_joint_names, self.policy.obs_joint_names)
            root_quat_1 = np.asarray(mc.root_quat, dtype=np.float32).reshape(1, 4)
            root_pos_1 = np.asarray(mc.root_pos, dtype=np.float32).reshape(1, 3)

            motions[motion_name] = {
                "joint_pos": joint_pos_1,
                "root_quat": root_quat_1,
                "root_pos": root_pos_1,
            }

        if "default" not in motions:
            raise ValueError(f"[{self.__class__.__name__}] motions must include a 'default' clip (length==1).")

        return motions

    @staticmethod
    def _empty_frames(n_joints: int) -> Dict[str, np.ndarray]:
        return {
            "joint_pos": np.zeros((0, n_joints), dtype=np.float32),
            "root_quat": np.zeros((0, 4), dtype=np.float32),
            "root_pos": np.zeros((0, 3), dtype=np.float32),
        }

    def _align_motion_to_anchor(
        self,
        motion: Dict[str, np.ndarray],
        anchor: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        p0 = motion["root_pos"][0]
        q0_yaw = _yaw_component_wxyz(motion["root_quat"][0])
        pa = anchor["root_pos"]
        qa_yaw = _yaw_component_wxyz(anchor["root_quat"])

        r0 = R.from_quat(q0_yaw, scalar_first=True)
        ra = R.from_quat(qa_yaw, scalar_first=True)
        r_delta = ra * r0.inv()

        root_pos_aligned = r_delta.apply(motion["root_pos"] - p0) + pa
        root_pos_aligned[:, 2] = motion["root_pos"][:, 2]

        root_quat_all = R.from_quat(motion["root_quat"], scalar_first=True)
        root_quat_aligned = (r_delta * root_quat_all).as_quat(scalar_first=True)

        return {
            "joint_pos": motion["joint_pos"].astype(np.float32, copy=True),
            "root_quat": root_quat_aligned.astype(np.float32),
            "root_pos": root_pos_aligned.astype(np.float32),
        }

    def _build_transition_prefix(
        self,
        anchor: Dict[str, np.ndarray],
        tgt_first: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        t_steps = int(self.policy.transition_steps)
        if t_steps <= 0:
            return self._empty_frames(self.policy.n_joints)

        joints_tr = _linspace_rows(anchor["joint_pos"], tgt_first["joint_pos"], t_steps)
        root_pos_tr = _linspace_rows(anchor["root_pos"], tgt_first["root_pos"], t_steps)
        root_quat_tr = _slerp(anchor["root_quat"], tgt_first["root_quat"], t_steps)

        return {
            "joint_pos": joints_tr,
            "root_quat": root_quat_tr,
            "root_pos": root_pos_tr,
        }

    def append_motion_from_tail(self, name: str) -> bool:
        if name not in self.motions:
            print(f"[{self.__class__.__name__}] Unknown motion '{name}'")
            return False

        anchor = self.policy.read_ref_tail_state()
        aligned_motion = self._align_motion_to_anchor(self.motions[name], anchor)

        tgt_first = {
            "joint_pos": aligned_motion["joint_pos"][0],
            "root_quat": aligned_motion["root_quat"][0],
            "root_pos": aligned_motion["root_pos"][0],
        }
        trans_motion = self._build_transition_prefix(anchor, tgt_first)

        segment = {
            "joint_pos": np.concatenate([trans_motion["joint_pos"], aligned_motion["joint_pos"]], axis=0),
            "root_quat": np.concatenate([trans_motion["root_quat"], aligned_motion["root_quat"]], axis=0),
            "root_pos": np.concatenate([trans_motion["root_pos"], aligned_motion["root_pos"]], axis=0),
        }
        self.policy.append_ref_frames(segment)

        self.policy.current_name = name
        self.policy.current_done = (self.policy.ref_idx >= self.policy.ref_len - 1)

        print(
            f"[{self.__class__.__name__}] Append motion '{name}' | appended={segment['joint_pos'].shape[0]}, "
            f"ref_len={self.policy.ref_len}, transition={self.policy.transition_steps}"
        )
        return True

    def append_hold_from_tail(self, steps: int, name: str = "hold") -> bool:
        n = max(1, int(steps))
        anchor = self.policy.read_ref_tail_state()
        segment = {
            "joint_pos": np.repeat(anchor["joint_pos"].reshape(1, -1), n, axis=0).astype(np.float32),
            "root_quat": np.repeat(anchor["root_quat"].reshape(1, -1), n, axis=0).astype(np.float32),
            "root_pos": np.repeat(anchor["root_pos"].reshape(1, -1), n, axis=0).astype(np.float32),
        }
        self.policy.append_ref_frames(segment)
        self.policy.current_name = name
        self.policy.current_done = False
        print(f"[{self.__class__.__name__}] Append hold '{name}' | appended={n}, ref_len={self.policy.ref_len}")
        return True

    def on_fade_in(self):
        self.append_motion_from_tail("default")

    def on_fade_out(self):
        self.append_motion_from_tail("default")

    def deactivate(self):
        return

    def post_step(self):
        return


class UDPMotionSource(MotionSourceBase):
    def __init__(self, policy: "TrackingPolicyRaw", policy_cfg: DictToClass):
        self.udp_enable = bool(getattr(policy_cfg, "udp_enable", True))
        self.udp_host = str(getattr(policy_cfg, "udp_host", "127.0.0.1"))
        self.udp_port = int(getattr(policy_cfg, "udp_port", 28562))
        self.auto_return_default = bool(getattr(policy_cfg, "udp_auto_return_default", True))
        self._udp_server: Optional[MotionUDPServer] = None

        super().__init__(policy, policy_cfg)

        if self.udp_enable:
            try:
                self._udp_server = MotionUDPServer(self.udp_host, self.udp_port)
                self._udp_server.start()
            except Exception as e:
                self._udp_server = None
                print(f"[UDPMotionSource] Failed to start UDP server: {e}")

    def request_motion(self, name: str) -> bool:
        if name not in self.motions:
            print(f"[UDPMotionSource] Unknown motion '{name}'")
            return False

        if (self.policy.current_name == "default" or name == "default") and self.policy.current_done:
            return self.append_motion_from_tail(name)

        print(
            f"[UDPMotionSource] Reject '{name}': "
            f"current='{self.policy.current_name}', done={self.policy.current_done}"
        )
        return False

    def request_runtime_motion(self, payload: Dict) -> bool:
        path = str(payload.get("path", "")).strip()
        if not path:
            name = str(payload.get("name", "")).strip()
            if not name:
                return False
            return self.request_motion("default" if name == "default" else name)

        name = str(payload.get("name", "")).strip() or Path(path).stem
        start = int(payload.get("start", 0))
        end = int(payload.get("end", -1))
        try:
            self.motions[name] = self._load_motion_file(path, name, start, end)
            print(f"[UDPMotionSource] Loaded runtime motion '{name}' from {path}")
        except Exception as e:
            print(f"[UDPMotionSource] Failed to load runtime motion '{name}' from {path}: {e}")
            return False
        return self.request_motion(name)

    def post_step(self):
        if self._udp_server is None:
            return

        handled_command = False
        for cmd in self._udp_server.pop_all():
            handled_command = True
            if cmd.startswith("{"):
                try:
                    payload = json.loads(cmd)
                except Exception as e:
                    print(f"[UDPMotionSource] Invalid JSON command: {e}")
                    continue
                if isinstance(payload, dict):
                    self.request_runtime_motion(payload)
                continue
            self.request_motion("default" if cmd == "default" else cmd)

        if (
            self.auto_return_default
            and not handled_command
            and self.policy.current_done
            and self.policy.current_name != "default"
        ):
            print(f"[UDPMotionSource] Motion '{self.policy.current_name}' done; auto returning to default")
            self.append_motion_from_tail("default")

    def deactivate(self):
        if self._udp_server is not None:
            self._udp_server.stop()


class VRMotionSource(MotionSourceBase):
    _next_episode_button_armed = True

    def __init__(self, policy: "TrackingPolicyRaw", policy_cfg: DictToClass):
        self.vr_transport = str(getattr(policy_cfg, "vr_transport", "zmq")).strip().lower()
        self.vr_req_addr = str(getattr(policy_cfg, "vr_req_addr", "tcp://127.0.0.1:28701"))
        self.vr_rep_addr = str(getattr(policy_cfg, "vr_rep_addr", "tcp://127.0.0.1:28702"))
        self.vr_ctrl_addr = str(getattr(policy_cfg, "vr_ctrl_addr", "tcp://127.0.0.1:28703"))
        self.vr_udp_stream_bind_addr = str(getattr(policy_cfg, "vr_udp_stream_bind_addr", "0.0.0.0"))
        self.vr_udp_stream_port = int(getattr(policy_cfg, "vr_udp_stream_port", 28704))
        self.vr_udp_stream_reuse_addr = bool(getattr(policy_cfg, "vr_udp_stream_reuse_addr", False))
        self.vr_udp_stream_rcvbuf = int(getattr(policy_cfg, "vr_udp_stream_rcvbuf", 4 * 1024 * 1024))
        self.vr_low_watermark = int(getattr(policy_cfg, "vr_low_watermark", 10))
        self.vr_high_watermark = int(getattr(policy_cfg, "vr_high_watermark", 0))
        self.vr_chunk_frames = int(getattr(policy_cfg, "vr_chunk_frames", 5))
        self.vr_inflight_lifetime_steps = int(getattr(policy_cfg, "vr_inflight_lifetime_steps", 3))
        self.vr_verbose_io = bool(getattr(policy_cfg, "vr_verbose_io", False))
        self.vr_debug_log_interval_s = float(getattr(policy_cfg, "vr_debug_log_interval_s", 1.0))
        self.vr_debug_rep_dt_warn_ms = float(getattr(policy_cfg, "vr_debug_rep_dt_warn_ms", 30.0))
        self.vr_debug_no_reply_warn_ms = float(getattr(policy_cfg, "vr_debug_no_reply_warn_ms", 100.0))
        self.vr_gap_recovery_enable = bool(getattr(policy_cfg, "vr_gap_recovery_enable", False))
        self.vr_fail_safe_no_reply_ms = float(getattr(policy_cfg, "vr_fail_safe_no_reply_ms", 500.0))
        self.vr_fail_safe_retarget_age_ms = float(getattr(policy_cfg, "vr_fail_safe_retarget_age_ms", 500.0))
        default_fail_safe_action = "default" if bool(getattr(policy_cfg, "vr_fail_safe_return_default", False)) else "hold"
        self.vr_fail_safe_action = str(getattr(policy_cfg, "vr_fail_safe_action", default_fail_safe_action)).strip().lower()
        self.vr_fail_safe_hold_steps = int(getattr(policy_cfg, "vr_fail_safe_hold_steps", 50))
        self.vr_fail_safe_auto_restart = bool(getattr(policy_cfg, "vr_fail_safe_auto_restart", True))
        self.vr_next_episode_button = os.getenv("MOLMO_TELEOP_NEXT_BUTTON", "right_key_two").strip()
        self.vr_require_pico_video_ready = os.getenv(
            "MOLMO_REQUIRE_PICO_VIDEO_READY",
            "1" if os.getenv("MOLMO_PICO_VIRTUAL_CAMERA") else "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if self.vr_transport not in ("zmq", "udp_stream"):
            raise ValueError("vr_transport must be 'zmq' or 'udp_stream'")
        if self.vr_udp_stream_port <= 0 or self.vr_udp_stream_port > 65535:
            raise ValueError("vr_udp_stream_port must be in 1..65535")
        if self.vr_udp_stream_rcvbuf < 0:
            raise ValueError("vr_udp_stream_rcvbuf must be >= 0")
        if self.vr_inflight_lifetime_steps < 0:
            raise ValueError("vr_inflight_lifetime_steps must be >= 0")
        if self.vr_high_watermark > 0 and self.vr_high_watermark < self.vr_low_watermark:
            raise ValueError("vr_high_watermark must be >= vr_low_watermark when enabled")

        self._vr_active = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        # Start-time anchor of deploy reference stream, used as transition start pose.
        self._vr_anchor_joint_pos: Optional[np.ndarray] = None
        self._vr_anchor_root_pos: Optional[np.ndarray] = None
        self._vr_anchor_root_quat: Optional[np.ndarray] = None
        self._vr_align_ready = False
        # Yaw-only alignment rotation: source(VR at start) -> target(deploy anchor at start).
        self._vr_r_delta: Optional[R] = None
        # Source VR root position at start; later VR root translation is measured relative to this origin.
        self._vr_source_root_pos0: Optional[np.ndarray] = None
        self._vr_target_anchor_pos: Optional[np.ndarray] = None

        positive_steps = [int(s) for s in np.asarray(policy.future_steps).reshape(-1).tolist() if int(s) > 0]
        self._target_future_horizon = int(max(positive_steps)) if len(positive_steps) > 0 else 0

        self._zmq_ctx = None
        self._req_sock = None
        self._rep_sock = None
        self._ctrl_sock = None
        self._udp_stream_sock: Optional[socket.socket] = None
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._pending_start_request = False
        self._vr_user_enabled = False
        self._prev_start_btn = False
        self._prev_stop_btn = False
        self._prev_next_episode_btn = False
        self._teleop_next_episode_requested = False
        self._req_log_count = 0
        self._rep_log_count = 0
        self._last_req_log_monotonic: Optional[float] = None
        self._last_rep_log_monotonic: Optional[float] = None
        self._oldest_unanswered_req_monotonic: Optional[float] = None
        self._last_event_log: Dict[str, float] = {}
        self._vr_fail_safe_triggered = False
        self._auto_restart_after_fail_safe = False
        self._last_req_send_monotonic: Optional[float] = None
        self._last_req_horizon: Optional[int] = None
        self._last_req_start = False
        self._last_ctrl_recv_monotonic: Optional[float] = None
        self._last_ctrl_buttons: Dict[str, bool] = {}
        self._last_ctrl_seq: Optional[int] = None
        self._last_udp_stream_seq: Optional[int] = None
        self._latest_ctrl_pose_payload: Optional[dict] = None
        self._last_ctrl_pose_stream_seq: Optional[int] = None
        self._udp_stream_debug_count = 0
        self._ctrl_log_count = 0
        self._last_rep_recv_monotonic: Optional[float] = None
        self._last_post_step_monotonic: Optional[float] = None
        self._latest_post_step_gap_ms: Optional[float] = None
        self._last_req_send_gap_ms: Optional[float] = None

        super().__init__(policy, policy_cfg)

        if self.vr_transport == "zmq":
            if zmq is None:
                raise ImportError("[VRMotionSource] pyzmq is required for vr_transport='zmq'.")
            try:
                self._zmq_ctx = zmq.Context.instance()
                self._req_sock = self._zmq_ctx.socket(zmq.PUSH)
                self._configure_latest_zmq_socket(self._req_sock, send=True)
                self._req_sock.connect(self.vr_req_addr)

                self._rep_sock = self._zmq_ctx.socket(zmq.PULL)
                self._configure_latest_zmq_socket(self._rep_sock, recv=True)
                self._rep_sock.connect(self.vr_rep_addr)

                self._ctrl_sock = self._zmq_ctx.socket(zmq.PULL)
                self._configure_latest_zmq_socket(self._ctrl_sock, recv=True)
                self._ctrl_sock.connect(self.vr_ctrl_addr)

                print(
                    "[VRMotionSource] Connected "
                    f"transport=zmq, req->{self.vr_req_addr}, rep<-{self.vr_rep_addr}, "
                    f"ctrl<-{self.vr_ctrl_addr}, low_watermark={self.vr_low_watermark}, "
                    f"high_watermark={self.vr_high_watermark}, chunk_frames={self.vr_chunk_frames}, "
                    f"inflight_lifetime_steps={self.vr_inflight_lifetime_steps}, "
                    f"no_reply_warn_ms={self.vr_debug_no_reply_warn_ms:.1f}, "
                    f"gap_recovery={self.vr_gap_recovery_enable}, "
                    f"fail_safe_no_reply_ms={self.vr_fail_safe_no_reply_ms:.1f}, "
                    f"fail_safe_retarget_age_ms={self.vr_fail_safe_retarget_age_ms:.1f}, "
                    f"fail_safe_action={self.vr_fail_safe_action}, "
                    f"fail_safe_auto_restart={self.vr_fail_safe_auto_restart}, "
                    "zmq_hwm=64, zmq_conflate=False, "
                    f"verbose_io={self.vr_verbose_io}"
                )
            except Exception as e:
                self._req_sock = None
                self._rep_sock = None
                self._ctrl_sock = None
                print(f"[VRMotionSource] Failed to create ZMQ sockets: {e}")
        else:
            ctrl_status = ""
            if zmq is not None and self.vr_ctrl_addr:
                try:
                    self._zmq_ctx = zmq.Context.instance()
                    self._ctrl_sock = self._zmq_ctx.socket(zmq.PULL)
                    self._configure_latest_zmq_socket(self._ctrl_sock, recv=True)
                    self._ctrl_sock.connect(self.vr_ctrl_addr)
                    ctrl_status = f", ctrl<-{self.vr_ctrl_addr}"
                except Exception as e:
                    self._ctrl_sock = None
                    ctrl_status = f", ctrl_unavailable={e}"
                    print(f"[VRMotionSource][Warning] Failed to create ZMQ control socket in UDP mode: {e}")
            try:
                self._udp_stream_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                if self.vr_udp_stream_reuse_addr:
                    self._udp_stream_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if self.vr_udp_stream_rcvbuf > 0:
                    try:
                        self._udp_stream_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.vr_udp_stream_rcvbuf)
                    except OSError as e:
                        print(f"[VRMotionSource][Warning] failed to set UDP SO_RCVBUF={self.vr_udp_stream_rcvbuf}: {e}")
                self._udp_stream_sock.bind((self.vr_udp_stream_bind_addr, self.vr_udp_stream_port))
                self._udp_stream_sock.setblocking(False)
                actual_rcvbuf = None
                try:
                    actual_rcvbuf = self._udp_stream_sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                except OSError:
                    pass
                print(
                    "[VRMotionSource] Connected "
                    f"transport=udp_stream, stream<-udp://{self.vr_udp_stream_bind_addr}:{self.vr_udp_stream_port}{ctrl_status}, "
                    f"low_watermark={self.vr_low_watermark}, high_watermark={self.vr_high_watermark}, "
                    f"reuse_addr={self.vr_udp_stream_reuse_addr}, rcvbuf={actual_rcvbuf}, "
                    f"no_reply_warn_ms={self.vr_debug_no_reply_warn_ms:.1f}, "
                    f"gap_recovery={self.vr_gap_recovery_enable}, "
                    f"fail_safe_no_reply_ms={self.vr_fail_safe_no_reply_ms:.1f}, "
                    f"fail_safe_retarget_age_ms={self.vr_fail_safe_retarget_age_ms:.1f}, "
                    f"fail_safe_action={self.vr_fail_safe_action}, "
                    f"fail_safe_auto_restart={self.vr_fail_safe_auto_restart}, "
                    f"verbose_io={self.vr_verbose_io}"
                )
            except Exception as e:
                if self._udp_stream_sock is not None:
                    try:
                        self._udp_stream_sock.close()
                    except Exception:
                        pass
                self._udp_stream_sock = None
                hint = ""
                if isinstance(e, OSError) and getattr(e, "errno", None) in (errno.EADDRINUSE, 10048):
                    hint = " | port already in use; stop old MolmoSpaces processes or run: ss -lunp | grep 28704"
                print(f"[VRMotionSource] Failed to create UDP stream socket: {e}{hint}")

    def _log_event(self, key: str, msg: str, interval_s: Optional[float] = None) -> None:
        interval = self.vr_debug_log_interval_s if interval_s is None else float(interval_s)
        now = time.monotonic()
        last = self._last_event_log.get(key)
        if last is None or (now - last) >= interval:
            print(msg)
            self._last_event_log[key] = now

    @staticmethod
    def _extract_buttons(payload: dict) -> Optional[dict]:
        if not isinstance(payload, dict):
            return None
        buttons = payload.get("controller_buttons", None)
        if not isinstance(buttons, dict):
            return None
        buttons = dict(buttons)
        for edge_key in ("pressed_edges", "controller_pressed_edges", "latched_control_buttons"):
            edges = payload.get(edge_key)
            if not isinstance(edges, (list, tuple)):
                continue
            for key in edges:
                if isinstance(key, str) and key in _VR_CONTROL_EDGE_BUTTONS:
                    buttons[key] = True
        return buttons

    def _has_ctrl_pose_fallback(self) -> bool:
        payload = self._latest_ctrl_pose_payload
        if not isinstance(payload, dict):
            return False
        frames = payload.get("frames")
        return isinstance(frames, list) and len(frames) > 0

    @staticmethod
    def _configure_latest_zmq_socket(sock, *, send: bool = False, recv: bool = False) -> None:
        hwm = 64
        sock.setsockopt(zmq.LINGER, 0)
        if send:
            sock.setsockopt(zmq.SNDHWM, hwm)
            if hasattr(zmq, "IMMEDIATE"):
                sock.setsockopt(zmq.IMMEDIATE, 1)
        if recv:
            sock.setsockopt(zmq.RCVHWM, hwm)
        # Explicit drain keeps the newest message; a small queue absorbs brief
        # transport jitter without dropping reply packets.

    @staticmethod
    def _ms_since(t: Optional[float]) -> Optional[float]:
        if t is None:
            return None
        return round((time.monotonic() - float(t)) * 1000.0, 1)

    @staticmethod
    def _pressed_buttons(buttons: Optional[dict]) -> list[str]:
        if not isinstance(buttons, dict):
            return []
        pressed = []
        for key, value in buttons.items():
            if isinstance(value, bool) and value:
                pressed.append(str(key))
        return sorted(pressed)

    @staticmethod
    def _reply_debug_suffix(payload: dict) -> str:
        fields = [
            ("transport", "transport"),
            ("stream_seq", "stream_seq"),
            ("udp_drained", "udp_drained"),
            ("req_seq", "req_seq"),
            ("retarget_age_ms", "retarget_age_ms"),
            ("raw_motion_age_ms", "raw_age_ms"),
            ("controller_age_ms", "controller_age_ms"),
            ("callback_gap_ms", "callback_gap_ms"),
            ("reply_process_ms", "reply_process_ms"),
            ("sample_mode", "mode"),
            ("buffer_len", "buffer"),
            ("req_count", "pc_req"),
            ("req_merged", "merged"),
            ("callback_count", "cb"),
            ("retarget_count", "retarget"),
            ("raw_seq", "raw_seq"),
            ("controller_seq", "ctrl_seq"),
            ("body_available", "body"),
            ("motion_timestamp_duplicate", "dup_ts"),
            ("robot_req_send_gap_ms", "robot_req_gap_ms"),
            ("pc_udp_stream_send_gap_ms", "pc_stream_gap_ms"),
            ("robot_post_step_gap_ms", "robot_step_gap_ms"),
            ("robot_future_horizon", "robot_h"),
            ("robot_vr_active", "robot_active"),
        ]
        parts = []
        for key, label in fields:
            if key in payload:
                parts.append(f"{label}={payload.get(key)}")
        return "" if not parts else " | " + ", ".join(parts)

    def _pico_video_ready(self) -> bool:
        if not self.vr_require_pico_video_ready:
            return True
        try:
            from molmo_spaces.utils.teleop_live_outputs import pico_virtual_camera_ready

            return bool(pico_virtual_camera_ready())
        except Exception:
            return False

    def _drain_control(self) -> None:
        if self._ctrl_sock is None:
            return

        latest_buttons: Optional[dict] = None
        latest_payload: Optional[dict] = None
        latched_control_buttons: dict[str, bool] = {}
        drained = 0
        while True:
            try:
                raw = self._ctrl_sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                print(f"[VRMotionSource] control recv failed: {e}")
                break

            try:
                payload = json.loads(raw)
            except Exception:
                continue
            buttons = self._extract_buttons(payload)
            if buttons is not None:
                for key in _VR_CONTROL_EDGE_BUTTONS:
                    if bool(buttons.get(key, False)):
                        latched_control_buttons[key] = True
                latest_buttons = buttons
                pose_payload = payload.get("pose_payload")
                if isinstance(pose_payload, dict):
                    self._latest_ctrl_pose_payload = pose_payload
                latest_payload = payload
                drained += 1

        if latest_buttons is None:
            return

        if latched_control_buttons:
            latest_buttons = dict(latest_buttons)
            for key, value in latched_control_buttons.items():
                latest_buttons[key] = bool(latest_buttons.get(key, False)) or bool(value)
            if isinstance(latest_payload, dict):
                latest_payload = dict(latest_payload)
                latest_payload["controller_buttons"] = latest_buttons
                latest_payload["latched_control_buttons"] = sorted(latched_control_buttons.keys())
            latest_ctrl_seq = None
            if isinstance(latest_payload, dict):
                try:
                    latest_ctrl_seq = int(latest_payload.get("ctrl_seq"))
                except Exception:
                    latest_ctrl_seq = None
            print(
                "[VRMotionSource][ControlLatch] "
                f"latched={sorted(latched_control_buttons.keys())}, drained_ctrl={drained}, "
                f"ctrl_seq={latest_ctrl_seq}, pressed={self._pressed_buttons(latest_buttons)}",
                flush=True,
            )
        if isinstance(latest_payload, dict):
            pose_payload = latest_payload.get("pose_payload")
            if isinstance(pose_payload, dict):
                if latched_control_buttons:
                    pose_payload = dict(pose_payload)
                    pose_buttons = self._extract_buttons(pose_payload)
                    if not isinstance(pose_buttons, dict):
                        pose_buttons = {}
                    pose_buttons = dict(pose_buttons)
                    for key, value in latched_control_buttons.items():
                        pose_buttons[key] = bool(pose_buttons.get(key, False)) or bool(value)
                    pose_payload["controller_buttons"] = pose_buttons
                    pose_payload["latched_control_buttons"] = sorted(latched_control_buttons.keys())
                    latest_payload = dict(latest_payload)
                    latest_payload["pose_payload"] = pose_payload
                self._latest_ctrl_pose_payload = pose_payload

        self._apply_control_payload(latest_payload, latest_buttons, drained)

    def _apply_control_payload(self, latest_payload: Optional[dict], latest_buttons: dict, drained: int) -> None:
        recv_mono = time.monotonic()
        latest_ctrl_seq: Optional[int] = None
        pc_ctrl_send_gap_ms = None
        ctrl_seq_gap = None
        if isinstance(latest_payload, dict):
            try:
                latest_ctrl_seq = int(latest_payload.get("ctrl_seq"))
            except Exception:
                latest_ctrl_seq = None
            pc_ctrl_send_gap_ms = latest_payload.get("pc_ctrl_send_gap_ms")
            if latest_ctrl_seq is not None and self._last_ctrl_seq is not None:
                ctrl_seq_gap = latest_ctrl_seq - self._last_ctrl_seq
        if self._last_ctrl_recv_monotonic is not None:
            ctrl_gap_ms = (recv_mono - self._last_ctrl_recv_monotonic) * 1000.0
            if ctrl_gap_ms >= self.vr_debug_no_reply_warn_ms:
                self._log_event(
                    "ctrl_gap_warn",
                    "[VRMotionSource][Warning] control packet gap "
                    f"{ctrl_gap_ms:.1f} ms | pc_send_gap_ms={pc_ctrl_send_gap_ms}, "
                    f"ctrl_seq={latest_ctrl_seq}, seq_gap={ctrl_seq_gap}, drained={drained}",
                    interval_s=0.0,
                )
        if pc_ctrl_send_gap_ms is not None:
            try:
                if float(pc_ctrl_send_gap_ms) >= self.vr_debug_no_reply_warn_ms:
                    self._log_event(
                        "pc_ctrl_send_gap_warn",
                        "[VRMotionSource][Warning] PC control sender gap reported "
                        f"{float(pc_ctrl_send_gap_ms):.1f} ms | ctrl_seq={latest_ctrl_seq}, "
                        f"seq_gap={ctrl_seq_gap}, drained={drained}",
                        interval_s=0.0,
                    )
            except Exception:
                pass
        self._last_ctrl_recv_monotonic = recv_mono
        if latest_ctrl_seq is not None:
            self._last_ctrl_seq = latest_ctrl_seq
        self._last_ctrl_buttons = {k: bool(v) for k, v in latest_buttons.items() if isinstance(v, bool)}
        self._ctrl_log_count += int(drained)

        ctrl_wall_age_ms: Optional[int] = None
        if isinstance(latest_payload, dict):
            try:
                ctrl_wall_age_ms = int(time.time() * 1000) - int(latest_payload.get("t_ms"))
            except Exception:
                ctrl_wall_age_ms = None

        record_fn = getattr(self.policy.controller, "handle_vr_record_buttons", None)
        if callable(record_fn):
            record_fn(latest_buttons)

        start_btn = bool(latest_buttons.get("right_key_one", False))
        stop_btn = bool(latest_buttons.get("left_key_one", False))
        next_btn = bool(
            self.vr_next_episode_button
            and latest_buttons.get(self.vr_next_episode_button, False)
        )
        if not next_btn:
            type(self)._next_episode_button_armed = True
        start_rise = start_btn and (not self._prev_start_btn)
        stop_rise = stop_btn and (not self._prev_stop_btn)
        next_rise = (
            next_btn
            and (not self._prev_next_episode_btn)
            and type(self)._next_episode_button_armed
        )
        self._prev_start_btn = start_btn
        self._prev_stop_btn = stop_btn
        self._prev_next_episode_btn = next_btn
        stop_and_start_rise = stop_rise and start_rise

        if next_rise:
            type(self)._next_episode_button_armed = False
            self._teleop_next_episode_requested = True
            notify_pause = getattr(self.policy.controller, "notify_vr_fail_safe_pause", None)
            self._auto_restart_after_fail_safe = False
            self._vr_user_enabled = False
            self._pending_start_request = False
            self._req_inflight = False
            self._req_inflight_steps_left = 0
            self._oldest_unanswered_req_monotonic = None
            self._vr_active = False
            self._vr_align_ready = False
            self._vr_in_transition = False
            self._vr_transition_count = 0
            self._vr_fail_safe_triggered = False
            print(
                "[VRMotionSource] Next random scene/task requested from control button | "
                f"button={self.vr_next_episode_button}, pressed={self._pressed_buttons(latest_buttons)}, "
                f"drained_ctrl={drained}, ctrl_wall_age_ms={ctrl_wall_age_ms}, "
                f"ctrl_count={self._ctrl_log_count}, ctrl_seq={latest_ctrl_seq}, "
                f"pc_send_gap_ms={pc_ctrl_send_gap_ms}",
                flush=True,
            )
            if callable(notify_pause):
                notify_pause(
                    "manual next random scene/task from control button",
                    action="hold",
                )

        if stop_rise:
            notify_pause = getattr(self.policy.controller, "notify_vr_fail_safe_pause", None)
            was_controlling = bool(
                self._vr_user_enabled or self._vr_active or self._pending_start_request
            )
            self._auto_restart_after_fail_safe = False
            self._vr_user_enabled = False
            self._pending_start_request = False
            self._req_inflight = False
            self._req_inflight_steps_left = 0
            self._oldest_unanswered_req_monotonic = None
            self._vr_active = False
            self._vr_align_ready = False
            self._vr_in_transition = False
            self._vr_transition_count = 0
            self._vr_fail_safe_triggered = False
            print(
                "[VRMotionSource] VR stop from control button | "
                f"pressed={self._pressed_buttons(latest_buttons)}, drained_ctrl={drained}, "
                f"ctrl_wall_age_ms={ctrl_wall_age_ms}, ctrl_count={self._ctrl_log_count}, "
                f"ctrl_seq={latest_ctrl_seq}, pc_send_gap_ms={pc_ctrl_send_gap_ms}"
            )
            if callable(notify_pause):
                notify_pause(
                    "manual VR stop from control button",
                    action="hold",
                )

        if stop_and_start_rise:
            self._log_event(
                "ignore_start_with_stop",
                "[VRMotionSource] Ignore VR start because LEFT X stop was latched in the same control batch | "
                f"pressed={self._pressed_buttons(latest_buttons)}, drained_ctrl={drained}, "
                f"ctrl_seq={latest_ctrl_seq}",
                interval_s=0.0,
            )

        start_when_ready = (
            start_btn
            and not stop_btn
            and not self._vr_active
            and not self._pending_start_request
            and self._pico_video_ready()
        )

        if (start_rise or start_when_ready) and not stop_rise and not next_rise:
            if self._vr_active and not self._pending_start_request:
                self._log_event(
                    "ignore_start_active",
                    "[VRMotionSource] Ignore VR start button because stream is already active | "
                    f"pressed={self._pressed_buttons(latest_buttons)}, drained_ctrl={drained}, "
                    f"ctrl_seq={latest_ctrl_seq}",
                )
            elif not self._pico_video_ready():
                self._log_event(
                    "ignore_start_pico_video_not_ready",
                    "[VRMotionSource][WAITING] Ignored RIGHT A because PICO Remote Vision video is not ready yet. "
                    "Open PICO: WEBCAM Remote Vision -> Listen, wait for OPEN_CAMERA/H264 encoder, then keep holding or press RIGHT A again.",
                    interval_s=0.0,
                )
            else:
                self._auto_restart_after_fail_safe = False
                self._request_vr_start("PICO VR start")
                print(
                    "[VRMotionSource] VR start requested from control button | "
                    f"pressed={self._pressed_buttons(latest_buttons)}, drained_ctrl={drained}, "
                    f"ctrl_wall_age_ms={ctrl_wall_age_ms}, ctrl_count={self._ctrl_log_count}, "
                    f"ctrl_seq={latest_ctrl_seq}, pc_send_gap_ms={pc_ctrl_send_gap_ms}"
                )

        set_hand_fn = getattr(self.policy.controller, "set_hand_from_controller_buttons", None)
        hand_control_enabled = bool(
            callable(set_hand_fn)
            and (self._vr_active or self._pending_start_request or self._vr_user_enabled)
            and not stop_btn
            and not next_btn
        )
        if hand_control_enabled:
            set_hand_fn(latest_buttons)
        elif callable(set_hand_fn) and any(
            bool(latest_buttons.get(name, False))
            for name in ("left_index_trig", "left_grip", "right_index_trig", "right_grip")
        ):
            self._log_event(
                "ignore_inactive_hand",
                "[VRMotionSource] Ignore Dex1 hand input while VR control is stopped; press RIGHT A to resume.",
                interval_s=2.0,
            )

    def consume_next_episode_request(self) -> bool:
        requested = bool(self._teleop_next_episode_requested)
        self._teleop_next_episode_requested = False
        return requested

    def _request_vr_start(self, reset_reason: str) -> None:
        if getattr(self.policy, "uses_root_xy_obs", False):
            reset_root_xy_origin = getattr(self.policy.controller, "reset_root_xy_origin", None)
            if callable(reset_root_xy_origin):
                reset_root_xy_origin(reset_reason)
        read_robot_state = getattr(self.policy, "read_robot_state", None)
        reset_reference_to_state = getattr(self.policy, "reset_reference_to_state", None)
        if callable(read_robot_state) and callable(reset_reference_to_state):
            reset_reference_to_state(read_robot_state(), name="vr_anchor")
            self._log_event(
                "vr_start_ref_reset",
                "[VRMotionSource] Reset reference queue for VR start",
            )
        self._vr_user_enabled = True
        self._pending_start_request = True
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._oldest_unanswered_req_monotonic = None
        self._vr_active = False
        self._vr_align_ready = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        self._vr_fail_safe_triggered = False

    def _future_horizon(self) -> int:
        if self.policy.ref_len <= 0:
            return 0
        return max(0, int(self.policy.ref_len - 1 - self.policy.ref_idx))

    @staticmethod
    def _repeat_frame(frame: Dict[str, np.ndarray], count: int) -> Dict[str, np.ndarray]:
        c = int(count)
        return {
            "joint_pos": np.repeat(frame["joint_pos"].reshape(1, -1), c, axis=0).astype(np.float32),
            "root_pos": np.repeat(frame["root_pos"].reshape(1, -1), c, axis=0).astype(np.float32),
            "root_quat": np.repeat(frame["root_quat"].reshape(1, -1), c, axis=0).astype(np.float32),
        }

    def _pad_future_once_on_start(self, frame: Dict[str, np.ndarray]) -> None:
        if self._target_future_horizon <= 0:
            return
        deficit = int(self._target_future_horizon - self._future_horizon())
        if deficit > 0:
            self.policy.append_ref_frames(self._repeat_frame(frame, deficit))

    def _pad_future_to_low_watermark(self, frame: Dict[str, np.ndarray]) -> None:
        deficit = int(self.vr_low_watermark - self._future_horizon())
        if deficit <= 0:
            return
        self.policy.append_ref_frames(self._repeat_frame(frame, deficit))
        self._log_event(
            "pad_future",
            "[VRMotionSource] Padded repeated frame "
            f"to low_watermark={self.vr_low_watermark} (added={deficit}, h={self._future_horizon()})"
        )

    def _appendable_reply_frames(self, frames: list[Dict[str, np.ndarray]]) -> list[Dict[str, np.ndarray]]:
        if self.vr_high_watermark <= 0:
            return frames
        h_now = self._future_horizon()
        if h_now >= self.vr_high_watermark:
            self._log_event(
                "drop_high_watermark",
                "[VRMotionSource][Warning] Drop reply frames because future buffer "
                f"already reached high_watermark={self.vr_high_watermark} (h={h_now})"
            )
            return []
        capacity = int(self.vr_high_watermark - h_now)
        kept = frames[:capacity]
        dropped = max(0, len(frames) - len(kept))
        if dropped > 0:
            self._log_event(
                "drop_excess_reply",
                "[VRMotionSource][Warning] Drop excess reply frames to respect "
                f"high_watermark={self.vr_high_watermark} (kept={len(kept)}, dropped={dropped}, h={h_now})"
            )
        return kept

    def _warn_horizon_if_needed(self, tag: str) -> None:
        if self._target_future_horizon <= 0:
            return
        if (not self._vr_user_enabled) and (not self._pending_start_request) and (not self._vr_active):
            return
        h = self._future_horizon()
        if h < self._target_future_horizon:
            self._log_event(
                "horizon_below_required",
                f"[VRMotionSource][Warning] horizon={h} below required={self._target_future_horizon} ({tag})"
            )

    def _fail_safe_stop(self, reason: str) -> None:
        if self._vr_fail_safe_triggered:
            return
        should_auto_restart = bool(
            self.vr_fail_safe_auto_restart
            and (self._vr_user_enabled or self._pending_start_request or self._vr_active)
        )
        notify_pause = getattr(self.policy.controller, "notify_vr_fail_safe_pause", None)
        if self.vr_fail_safe_action in ("hold_active", "recover", "continue"):
            self._req_inflight = False
            self._req_inflight_steps_left = 0
            self._oldest_unanswered_req_monotonic = None
            self._pending_start_request = False
            print(f"[VRMotionSource][FailSafe] hold active VR stream: {reason}")
            self.append_hold_from_tail(self._recovery_hold_steps(), name="vr_fail_safe_hold_active")
            print("[VRMotionSource][FailSafe] appended hold reference; VR stream remains active")
            if callable(notify_pause):
                notify_pause(reason, action=self.vr_fail_safe_action)
            return
        self._vr_fail_safe_triggered = True
        self._vr_user_enabled = False
        self._pending_start_request = False
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._oldest_unanswered_req_monotonic = None
        self._vr_active = False
        self._vr_align_ready = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        self._auto_restart_after_fail_safe = should_auto_restart
        print(f"[VRMotionSource][FailSafe] stop VR stream: {reason}")
        if callable(notify_pause):
            notify_pause(reason, action=self.vr_fail_safe_action)
        if self._auto_restart_after_fail_safe:
            print("[VRMotionSource][FailSafe] auto restart armed; fresh VR data will resume stream")
        if self.vr_fail_safe_action == "default":
            self.append_motion_from_tail("default")
            print("[VRMotionSource][FailSafe] appended default motion after VR stream failure")
        elif self.vr_fail_safe_action == "hold":
            self.append_hold_from_tail(self.vr_fail_safe_hold_steps, name="vr_fail_safe_hold")
            print("[VRMotionSource][FailSafe] appended hold reference after VR stream failure")
        elif self.vr_fail_safe_action in ("none", "stop", "off"):
            print("[VRMotionSource][FailSafe] no reference appended after VR stream failure")
        else:
            print(
                f"[VRMotionSource][FailSafe] unknown action '{self.vr_fail_safe_action}', "
                "using hold"
            )
            self.append_hold_from_tail(self.vr_fail_safe_hold_steps, name="vr_fail_safe_hold")

    def _recovery_hold_steps(self) -> int:
        return max(1, int(max(self.vr_low_watermark, self._target_future_horizon)))

    def _recover_no_reply_if_needed(self, reason: str) -> bool:
        if not self.vr_gap_recovery_enable:
            return False
        if self._future_horizon() > self._target_future_horizon:
            return False
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        steps = self._recovery_hold_steps()
        print(f"[VRMotionSource][Recovery] append hold frames while waiting for VR reply: {reason} | steps={steps}")
        self.append_hold_from_tail(steps, name="vr_reply_gap_hold")
        return True

    @staticmethod
    def _slerp_single_shortest(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        a = float(np.clip(alpha, 0.0, 1.0))
        qq0 = np.asarray(q0, dtype=np.float64).reshape(4)
        qq1 = np.asarray(q1, dtype=np.float64).reshape(4)
        qq0 /= max(np.linalg.norm(qq0), 1e-9)
        qq1 /= max(np.linalg.norm(qq1), 1e-9)
        if float(np.dot(qq0, qq1)) < 0.0:
            qq1 = -qq1
        key = R.from_quat(np.stack([qq0, qq1], axis=0), scalar_first=True)
        interp = Slerp([0.0, 1.0], key)([a]).as_quat(scalar_first=True)[0]
        interp = interp / max(np.linalg.norm(interp), 1e-9)
        return interp.astype(np.float32)

    def _parse_frame(self, payload: dict) -> Optional[Dict[str, np.ndarray]]:
        if not isinstance(payload, dict):
            return None
        required = ["root_pos", "root_quat", "dof_pos"]
        if not all(k in payload for k in required):
            return None
        try:
            root_pos = np.asarray(payload["root_pos"], dtype=np.float32).reshape(3)
            root_quat = np.asarray(payload["root_quat"], dtype=np.float32).reshape(4)
            joint_pos = np.asarray(payload["dof_pos"], dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if joint_pos.shape[0] != self.policy.n_joints:
            print(
                f"[VRMotionSource] dof dim mismatch: "
                f"got={joint_pos.shape[0]}, expected={self.policy.n_joints}"
            )
            return None
        if len(self.policy.dataset_joint_names) != joint_pos.shape[0]:
            print(
                f"[VRMotionSource] dataset_joint_names mismatch: "
                f"got={len(self.policy.dataset_joint_names)}, expected={joint_pos.shape[0]}"
            )
            return None
        joint_pos = remap_joint_array_by_names(
            joint_pos.reshape(1, -1),
            self.policy.dataset_joint_names,
            self.policy.obs_joint_names,
        )[0]
        qn = float(np.linalg.norm(root_quat))
        if not np.isfinite(qn) or qn < 1e-6:
            return None
        root_quat = (root_quat / qn).astype(np.float32)
        return {
            "joint_pos": joint_pos.astype(np.float32),
            "root_pos": root_pos.astype(np.float32),
            "root_quat": root_quat.astype(np.float32),
        }

    def _start_vr_session(self, first_frame: Dict[str, np.ndarray]) -> None:
        if getattr(self.policy, "uses_root_xy_obs", False):
            read_robot_state = getattr(self.policy, "read_robot_state", None)
            anchor = read_robot_state() if callable(read_robot_state) else self.policy.read_current_state()
        else:
            anchor = self.policy.read_ref_tail_state()
        self._vr_anchor_joint_pos = anchor["joint_pos"].astype(np.float32, copy=True)
        self._vr_anchor_root_pos = anchor["root_pos"].astype(np.float32, copy=True)
        self._vr_anchor_root_quat = anchor["root_quat"].astype(np.float32, copy=True)
        src_yaw = _yaw_component_wxyz(first_frame["root_quat"])
        tgt_yaw = _yaw_component_wxyz(anchor["root_quat"])
        r0 = R.from_quat(src_yaw, scalar_first=True)
        rc = R.from_quat(tgt_yaw, scalar_first=True)
        self._vr_r_delta = rc * r0.inv()
        self._vr_source_root_pos0 = first_frame["root_pos"].astype(np.float32, copy=True)
        self._vr_target_anchor_pos = anchor["root_pos"].astype(np.float32, copy=True)
        self._vr_align_ready = True
        self._vr_active = True
        self._vr_transition_count = 0
        self._vr_in_transition = int(self.policy.transition_steps) > 0
        self._pending_start_request = False
        # Bootstrap future horizon once at start using current ref-buffer tail.
        self._pad_future_once_on_start(
            {
                "joint_pos": anchor["joint_pos"].astype(np.float32, copy=True),
                "root_pos": anchor["root_pos"].astype(np.float32, copy=True),
                "root_quat": anchor["root_quat"].astype(np.float32, copy=True),
            }
        )
        print(
            "\n"
            "================ TELEOP_VR_STARTED ================\n"
            f"transition_steps={int(self.policy.transition_steps)}\n"
            "state=active\n"
            "Do not press LEFT X unless you want to stop VR control.\n"
            "===================================================\n",
            flush=True,
        )

    def _apply_start_transition(self, aligned: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if (
            not self._vr_in_transition
            or self._vr_anchor_joint_pos is None
            or self._vr_anchor_root_pos is None
            or self._vr_anchor_root_quat is None
        ):
            return aligned

        self._vr_transition_count += 1
        t_steps = max(1, int(self.policy.transition_steps))
        alpha = min(1.0, float(self._vr_transition_count) / float(t_steps))

        out_joint = (self._vr_anchor_joint_pos * (1.0 - alpha) + aligned["joint_pos"] * alpha).astype(np.float32)
        out_pos = (self._vr_anchor_root_pos * (1.0 - alpha) + aligned["root_pos"] * alpha).astype(np.float32)
        out_quat = self._slerp_single_shortest(self._vr_anchor_root_quat, aligned["root_quat"], alpha)

        if alpha >= 1.0:
            self._vr_in_transition = False
        return {
            "joint_pos": out_joint,
            "root_pos": out_pos,
            "root_quat": out_quat,
        }

    def _align_vr_frame(self, frame: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        if (
            not self._vr_align_ready
            or self._vr_r_delta is None
            or self._vr_source_root_pos0 is None
            or self._vr_target_anchor_pos is None
        ):
            return None
        root_pos = frame["root_pos"].astype(np.float32)
        root_quat = frame["root_quat"].astype(np.float32)

        aligned_pos = self._vr_r_delta.apply(root_pos - self._vr_source_root_pos0) + self._vr_target_anchor_pos
        aligned_pos = aligned_pos.astype(np.float32)
        aligned_pos[2] = root_pos[2]

        aligned_quat = (self._vr_r_delta * R.from_quat(root_quat, scalar_first=True)).as_quat(scalar_first=True)
        aligned_quat = aligned_quat.astype(np.float32)
        aligned_quat /= max(np.linalg.norm(aligned_quat), 1e-6)
        return {
            "joint_pos": frame["joint_pos"].astype(np.float32, copy=True),
            "root_pos": aligned_pos,
            "root_quat": aligned_quat,
        }

    def _drain_udp_stream_payload(self) -> tuple[Optional[dict], int]:
        if self._udp_stream_sock is None:
            return None, 0

        latest_payload: Optional[dict] = None
        latched_control_buttons: dict[str, bool] = {}
        drained = 0
        while True:
            try:
                raw, _addr = self._udp_stream_sock.recvfrom(65535)
            except (BlockingIOError, InterruptedError):
                break
            except Exception as e:
                print(f"[VRMotionSource] UDP stream recv failed: {e}")
                break

            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                print("[VRMotionSource] bad JSON UDP stream packet")
                continue
            if not isinstance(payload, dict):
                continue
            latest_payload = payload
            drained += 1

            # The simulator can step at ~1 Hz on this laptop while the bridge streams
            # at 50 Hz. Preserve short start/stop pulses seen anywhere in this drain.
            buttons = self._extract_buttons(payload)
            if isinstance(buttons, dict):
                for key in _VR_CONTROL_EDGE_BUTTONS:
                    if bool(buttons.get(key, False)):
                        latched_control_buttons[key] = True

        if latest_payload is None:
            return None, 0

        latest_stream_seq: Optional[int] = None
        try:
            latest_stream_seq = int(latest_payload.get("stream_seq"))
        except Exception:
            latest_stream_seq = None
        if (
            not isinstance(latest_payload.get("frames"), list)
            or len(latest_payload.get("frames", [])) == 0
        ):
            self._log_event(
                "udp_stream_empty",
                "[VRMotionSource][Warning] ignore UDP stream packet without frames | "
                f"stream_seq={latest_stream_seq}, drained={drained}",
            )
            return None, drained

        if (
            latest_stream_seq is not None
            and self._last_udp_stream_seq is not None
            and latest_stream_seq <= self._last_udp_stream_seq
        ):
            self._log_event(
                "drop_old_udp_stream",
                "[VRMotionSource][Warning] drop old UDP stream packet | "
                f"stream_seq={latest_stream_seq}, last_stream_seq={self._last_udp_stream_seq}, drained={drained}",
            )
            return None, drained
        if latest_stream_seq is not None:
            self._last_udp_stream_seq = latest_stream_seq
        latest_payload["udp_drained"] = int(drained)

        frames = latest_payload.get("frames", [])
        frame_count = len(frames) if isinstance(frames, list) else -1
        buttons = self._extract_buttons(latest_payload)
        self._udp_stream_debug_count += 1
        self._log_event(
            "udp_stream_rx",
            "[VRMotionSource][StreamRx] "
            f"#{self._udp_stream_debug_count} drained={drained}, stream_seq={latest_stream_seq}, "
            f"frames={frame_count}, has_buttons={buttons is not None}, "
            f"pressed={self._pressed_buttons(buttons)}, h={self._future_horizon()}, "
            f"active={self._vr_active}, pending={self._pending_start_request}",
            interval_s=2.0,
        )
        if buttons is not None:
            if latched_control_buttons:
                buttons = dict(buttons)
                for key, value in latched_control_buttons.items():
                    buttons[key] = bool(buttons.get(key, False)) or bool(value)
                latest_payload = dict(latest_payload)
                latest_payload["controller_buttons"] = buttons
                latest_payload["latched_control_buttons"] = sorted(latched_control_buttons.keys())
                print(
                    "[VRMotionSource][ControlLatch] "
                    f"latched={sorted(latched_control_buttons.keys())}, drained={drained}, "
                    f"stream_seq={latest_stream_seq}, pressed={self._pressed_buttons(buttons)}",
                    flush=True,
                )
            self._apply_control_payload(latest_payload, buttons, drained)
        return latest_payload, drained
    def _drain_replies(self) -> Optional[Dict[str, np.ndarray]]:
        last_aligned_frame: Optional[Dict[str, np.ndarray]] = None
        source_label = "Rep"
        payloads: list[dict] = []

        if self.vr_transport == "udp_stream":
            payload, _drained = self._drain_udp_stream_payload()
            if payload is not None:
                payloads.append(payload)
                source_label = "Stream"
            else:
                fallback_payload = self._latest_ctrl_pose_payload
                if isinstance(fallback_payload, dict):
                    fallback_stream_seq: Optional[int] = None
                    try:
                        fallback_stream_seq = int(fallback_payload.get("stream_seq"))
                    except Exception:
                        fallback_stream_seq = None
                    if (
                        fallback_stream_seq is None
                        or self._last_ctrl_pose_stream_seq is None
                        or fallback_stream_seq > self._last_ctrl_pose_stream_seq
                    ):
                        payload = dict(fallback_payload)
                        payload["transport"] = "zmq_control_fallback"
                        payload["ctrl_pose_fallback"] = True
                        if fallback_stream_seq is not None:
                            self._last_ctrl_pose_stream_seq = fallback_stream_seq
                        payloads.append(payload)
                        source_label = "CtrlFallback"
                        frames = payload.get("frames", [])
                        frame_count = len(frames) if isinstance(frames, list) else -1
                        buttons = self._extract_buttons(payload)
                        self._log_event(
                            "ctrl_pose_fallback_rx",
                            "[VRMotionSource][CtrlFallbackRx] "
                            f"stream_seq={fallback_stream_seq}, frames={frame_count}, "
                            f"has_buttons={buttons is not None}, pressed={self._pressed_buttons(buttons)}, "
                            f"h={self._future_horizon()}, active={self._vr_active}, pending={self._pending_start_request}",
                            interval_s=2.0,
                        )
        else:
            if self._rep_sock is None:
                return last_aligned_frame
            while True:
                try:
                    raw = self._rep_sock.recv_string(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                except Exception as e:
                    print(f"[VRMotionSource] recv failed: {e}")
                    break

                try:
                    payload = json.loads(raw)
                except Exception:
                    print("[VRMotionSource] bad JSON reply")
                    continue
                if isinstance(payload, dict):
                    payloads.append(payload)

        for payload in payloads:
            raw_frames = payload.get("frames", [])
            if not isinstance(raw_frames, list):
                raw_frames = []
            parsed_frames = [f for f in (self._parse_frame(x) for x in raw_frames) if f is not None]
            if len(parsed_frames) == 0:
                continue

            recv_mono = time.monotonic()
            h_recv = self._future_horizon()
            self._last_rep_recv_monotonic = recv_mono
            reply_debug = self._reply_debug_suffix(payload)
            retarget_age_ms = payload.get("retarget_age_ms")
            if (
                retarget_age_ms is not None
                and self.vr_fail_safe_retarget_age_ms > 0.0
                and float(retarget_age_ms) >= self.vr_fail_safe_retarget_age_ms
            ):
                self._fail_safe_stop(
                    f"stale VR retarget reply age={float(retarget_age_ms):.1f} ms "
                    f">= {self.vr_fail_safe_retarget_age_ms:.1f} ms{reply_debug}"
                )
                continue

            if self._auto_restart_after_fail_safe:
                self._auto_restart_after_fail_safe = False
                self._request_vr_start("PICO VR auto restart after fail-safe")
                print(
                    "[VRMotionSource][FailSafe] auto restart requested from fresh VR data"
                    f"{reply_debug}"
                )

            self._rep_log_count += 1
            rep_dt_ms: Optional[float] = None
            if self._last_rep_log_monotonic is None:
                rep_dt_msg = "first"
            else:
                rep_dt_ms = (recv_mono - self._last_rep_log_monotonic) * 1000.0
                rep_dt_msg = f"dt={rep_dt_ms:7.2f} ms"
            self._last_rep_log_monotonic = recv_mono

            start_flag = bool(payload.get("start", False))
            if self.vr_transport == "udp_stream" and self._pending_start_request:
                start_flag = True
            if self._pending_start_request and not start_flag:
                self._log_event(
                    "ignore_non_start_pending",
                    f"[VRMotionSource][{source_label}] #{self._rep_log_count:06d} {rep_dt_msg} | "
                    f"h_recv={h_recv}, frames={len(parsed_frames)}, start={start_flag}, "
                    f"action=ignore_non_start_pending{reply_debug}"
                )
                self._log_event(
                    "ignore_non_start_pending_warn",
                    "[VRMotionSource][Warning] ignore non-start reply while pending start alignment",
                )
                continue
            if start_flag:
                if self._pending_start_request:
                    self._start_vr_session(parsed_frames[0])
                else:
                    self._log_event(
                        "drop_delayed_start",
                        f"[VRMotionSource][{source_label}] #{self._rep_log_count:06d} {rep_dt_msg} | "
                        f"h_recv={h_recv}, frames={len(parsed_frames)}, start={start_flag}, "
                        f"action=drop_delayed_start{reply_debug}"
                    )
                    self._log_event(
                        "drop_delayed_start_warn",
                        "[VRMotionSource][Warning] drop delayed start reply after alignment is ready",
                    )
                    continue

            if not self._vr_active:
                self._log_event(
                    "ignore_inactive",
                    f"[VRMotionSource][{source_label}] #{self._rep_log_count:06d} {rep_dt_msg} | "
                    f"h_recv={h_recv}, frames={len(parsed_frames)}, start={start_flag}, "
                    f"action=ignore_inactive{reply_debug}"
                )
                continue

            out_frames = []
            for f in parsed_frames:
                aligned = self._align_vr_frame(f)
                if aligned is not None:
                    out_frames.append(self._apply_start_transition(aligned))
            out_frames = self._appendable_reply_frames(out_frames)
            if len(out_frames) == 0:
                if self.vr_verbose_io:
                    self._log_event(
                        "ignore_no_aligned",
                        f"[VRMotionSource][{source_label}] #{self._rep_log_count:06d} {rep_dt_msg} | "
                        f"h_recv={h_recv}, frames={len(parsed_frames)}, "
                        f"start={start_flag}, action=ignore_no_aligned{reply_debug}",
                    )
                continue

            seg = {
                "joint_pos": np.stack([f["joint_pos"] for f in out_frames], axis=0).astype(np.float32),
                "root_pos": np.stack([f["root_pos"] for f in out_frames], axis=0).astype(np.float32),
                "root_quat": np.stack([f["root_quat"] for f in out_frames], axis=0).astype(np.float32),
            }
            self.policy.append_ref_frames(seg)
            last_aligned_frame = out_frames[-1]
            self._oldest_unanswered_req_monotonic = None
            h_after = self._future_horizon()
            slow_reply = rep_dt_ms is not None and rep_dt_ms >= self.vr_debug_rep_dt_warn_ms
            horizon_low = h_recv < self._target_future_horizon or h_after < self._target_future_horizon
            if self.vr_verbose_io or start_flag or slow_reply or horizon_low:
                level = "[Warning] " if slow_reply or horizon_low else ""
                self._log_event(
                    "rep_append_warn" if slow_reply or horizon_low else "rep_append",
                    f"[VRMotionSource]{level}[{source_label}] #{self._rep_log_count:06d} {rep_dt_msg} | "
                    f"h_recv={h_recv}, h_after={h_after}, frames={len(out_frames)}, "
                    f"start={start_flag}, action=append{reply_debug}",
                    interval_s=0.0 if self.vr_verbose_io or start_flag else self.vr_debug_log_interval_s,
                )
        return last_aligned_frame

    def _send_request_if_needed(self) -> None:
        if self.vr_transport != "zmq":
            return
        if self._req_sock is None:
            return
        if not self._vr_user_enabled:
            return
        if self._req_inflight:
            return
        h = self._future_horizon()
        should_request = (h <= self.vr_low_watermark) or self._pending_start_request
        if not should_request:
            return
        now = time.monotonic()
        start_flag = bool(self._pending_start_request)
        req_seq = int(self._req_log_count + 1)
        if self._last_req_send_monotonic is None:
            req_send_gap_ms = None
        else:
            req_send_gap_ms = (now - self._last_req_send_monotonic) * 1000.0
        req = {
            "req_seq": req_seq,
            "start": start_flag,
            "need_frames": int(max(1, self.vr_chunk_frames)),
            "future_horizon": int(h),
            "robot_req_send_gap_ms": None
            if req_send_gap_ms is None
            else round(float(req_send_gap_ms), 3),
            "robot_post_step_gap_ms": None
            if self._latest_post_step_gap_ms is None
            else round(float(self._latest_post_step_gap_ms), 3),
            "robot_future_horizon": int(h),
            "robot_vr_active": bool(self._vr_active),
            "robot_vr_pending": bool(self._pending_start_request),
            "t_req_ms": int(time.time() * 1000),
        }
        try:
            self._req_sock.send_string(json.dumps(req), flags=zmq.NOBLOCK)
            self._req_inflight = True
            self._req_inflight_steps_left = int(self.vr_inflight_lifetime_steps)
            if self._oldest_unanswered_req_monotonic is None:
                self._oldest_unanswered_req_monotonic = now
            self._last_req_send_monotonic = now
            self._last_req_send_gap_ms = req_send_gap_ms
            self._last_req_horizon = int(h)
            self._last_req_start = bool(start_flag)
            self._req_log_count = req_seq
            if self._last_req_log_monotonic is None:
                req_dt_msg = "first"
            else:
                req_dt_ms = (now - self._last_req_log_monotonic) * 1000.0
                req_dt_msg = f"dt={req_dt_ms:7.2f} ms"
            self._last_req_log_monotonic = now
            horizon_low = h < self._target_future_horizon
            if self.vr_verbose_io or start_flag or horizon_low:
                level = "[Warning] " if horizon_low else ""
                self._log_event(
                    "req_send_warn" if horizon_low else "req_send",
                    f"[VRMotionSource]{level}[Req] #{self._req_log_count:06d} {req_dt_msg} | "
                    f"h_req={h}, start={start_flag}, need_frames={int(req['need_frames'])}, "
                    f"post_step_gap_ms={req.get('robot_post_step_gap_ms')}",
                    interval_s=0.0 if self.vr_verbose_io or start_flag else self.vr_debug_log_interval_s,
                )
        except zmq.Again:
            return
        except Exception as e:
            print(f"[VRMotionSource] send request failed: {e}")

    def _warn_no_reply_if_needed(self) -> None:
        if self.vr_transport != "zmq":
            return
        if self._oldest_unanswered_req_monotonic is None:
            return
        if not self._vr_user_enabled:
            return
        elapsed_ms = (time.monotonic() - self._oldest_unanswered_req_monotonic) * 1000.0
        if elapsed_ms < self.vr_debug_no_reply_warn_ms:
            return
        last_req_age_ms = self._ms_since(self._last_req_send_monotonic)
        last_rep_age_ms = self._ms_since(self._last_rep_recv_monotonic)
        ctrl_age_ms = self._ms_since(self._last_ctrl_recv_monotonic)
        self._log_event(
            "no_reply",
            "[VRMotionSource][Error] no VR reply after request "
            f"for {elapsed_ms:.1f} ms | h={self._future_horizon()}, "
            f"inflight={self._req_inflight}, start_pending={self._pending_start_request}, "
            f"active={self._vr_active}, req_seq={self._req_log_count}, "
            f"last_req_age_ms={last_req_age_ms}, last_req_h={self._last_req_horizon}, "
            f"last_req_start={self._last_req_start}, last_rep_age_ms={last_rep_age_ms}, "
            f"ctrl_age_ms={ctrl_age_ms}, post_step_gap_ms={self._latest_post_step_gap_ms}, "
            f"last_req_send_gap_ms={self._last_req_send_gap_ms}, "
            f"pressed={self._pressed_buttons(self._last_ctrl_buttons)}",
        )
        if elapsed_ms >= self.vr_debug_no_reply_warn_ms:
            recovered = self._recover_no_reply_if_needed(
                f"no usable VR reply for {elapsed_ms:.1f} ms"
            )
            if recovered:
                return
        if self.vr_fail_safe_no_reply_ms > 0.0 and elapsed_ms >= self.vr_fail_safe_no_reply_ms:
            self._fail_safe_stop(
                f"no usable VR reply for {elapsed_ms:.1f} ms "
                f">= {self.vr_fail_safe_no_reply_ms:.1f} ms | "
                f"req_seq={self._req_log_count}, last_req_age_ms={last_req_age_ms}, "
                f"last_req_h={self._last_req_horizon}, last_rep_age_ms={last_rep_age_ms}, "
                f"ctrl_age_ms={ctrl_age_ms}, post_step_gap_ms={self._latest_post_step_gap_ms}, "
                f"last_req_send_gap_ms={self._last_req_send_gap_ms}, "
                f"pressed={self._pressed_buttons(self._last_ctrl_buttons)}"
            )

    def _warn_stream_stale_if_needed(self) -> None:
        if self.vr_transport != "udp_stream":
            return
        if not self._vr_user_enabled:
            return
        now = time.monotonic()
        if self._last_rep_recv_monotonic is None:
            if self._oldest_unanswered_req_monotonic is None:
                self._oldest_unanswered_req_monotonic = now
            elapsed_ms = (now - self._oldest_unanswered_req_monotonic) * 1000.0
        else:
            elapsed_ms = (now - self._last_rep_recv_monotonic) * 1000.0
        if elapsed_ms < self.vr_debug_no_reply_warn_ms:
            return

        ctrl_age_ms = self._ms_since(self._last_ctrl_recv_monotonic)
        self._log_event(
            "udp_stream_stale",
            "[VRMotionSource][Error] no usable VR UDP stream packet "
            f"for {elapsed_ms:.1f} ms | h={self._future_horizon()}, "
            f"start_pending={self._pending_start_request}, active={self._vr_active}, "
            f"last_stream_seq={self._last_udp_stream_seq}, ctrl_age_ms={ctrl_age_ms}, "
            f"post_step_gap_ms={self._latest_post_step_gap_ms}, "
            f"pressed={self._pressed_buttons(self._last_ctrl_buttons)}",
        )
        recovered = self._recover_no_reply_if_needed(
            f"no usable VR UDP stream packet for {elapsed_ms:.1f} ms"
        )
        if recovered:
            return
        if self.vr_fail_safe_no_reply_ms > 0.0 and elapsed_ms >= self.vr_fail_safe_no_reply_ms:
            self._fail_safe_stop(
                f"no usable VR UDP stream packet for {elapsed_ms:.1f} ms "
                f">= {self.vr_fail_safe_no_reply_ms:.1f} ms | "
                f"last_stream_seq={self._last_udp_stream_seq}, ctrl_age_ms={ctrl_age_ms}, "
                f"post_step_gap_ms={self._latest_post_step_gap_ms}, "
                f"pressed={self._pressed_buttons(self._last_ctrl_buttons)}"
            )

    def on_fade_in(self):
        self.append_motion_from_tail("default")
        self._pending_start_request = False
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._vr_user_enabled = False
        self._prev_start_btn = False
        self._prev_stop_btn = False
        self._vr_active = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        self._vr_anchor_joint_pos = None
        self._vr_anchor_root_pos = None
        self._vr_anchor_root_quat = None
        self._vr_align_ready = False
        self._vr_fail_safe_triggered = False
        self._auto_restart_after_fail_safe = False

    def on_fade_out(self):
        self._vr_user_enabled = False
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._oldest_unanswered_req_monotonic = None
        self._vr_active = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        self._vr_anchor_joint_pos = None
        self._vr_anchor_root_pos = None
        self._vr_anchor_root_quat = None
        self._vr_align_ready = False
        self._pending_start_request = False
        self._vr_fail_safe_triggered = False
        self._auto_restart_after_fail_safe = False
        super().on_fade_out()

    def post_step(self):
        now = time.monotonic()
        if self._last_post_step_monotonic is not None:
            self._latest_post_step_gap_ms = (now - self._last_post_step_monotonic) * 1000.0
            if self._latest_post_step_gap_ms >= self.vr_debug_no_reply_warn_ms:
                self._log_event(
                    "robot_post_step_gap",
                    "[VRMotionSource][Warning] robot post_step gap "
                    f"{self._latest_post_step_gap_ms:.1f} ms | active={self._vr_active}, "
                    f"h={self._future_horizon()}, inflight={self._req_inflight}",
                    interval_s=0.0,
                )
        self._last_post_step_monotonic = now
        self._drain_control()
        if not self._vr_active and not self._vr_user_enabled:
            pico_ready = self._pico_video_ready()
            if self.vr_require_pico_video_ready and not pico_ready:
                self._log_event(
                    "vr_waiting_pico_video",
                    "[VRMotionSource][WAITING] Waiting for PICO Remote Vision video before RIGHT A can start. "
                    "Open PICO: WEBCAM Remote Vision -> Listen.",
                    interval_s=3.0,
                )
            else:
                self._log_event(
                    "vr_ready_press_a",
                    "[VRMotionSource][READY] PICO video is live. Hold RIGHT A to start collection. LEFT X stops VR control. RIGHT B skips to next random scene/task.",
                    interval_s=3.0,
                )
            if self.vr_transport == "udp_stream" and self._last_udp_stream_seq is None:
                if self._has_ctrl_pose_fallback():
                    self._log_event(
                        "vr_using_ctrl_fallback",
                        "[VRMotionSource][Fallback] Direct UDP stream not received; using ZMQ control-pose fallback.",
                        interval_s=3.0,
                    )
                else:
                    self._log_event(
                        "vr_waiting_udp_stream",
                        "[VRMotionSource][WAITING] No UDP stream or control-pose fallback received yet; "
                        "start bridge or check duplicate UDP listener with: ss -lunp | grep 28704",
                        interval_s=3.0,
                    )
        last_aligned_frame = self._drain_replies()
        if last_aligned_frame is not None:
            self._req_inflight = False
            self._req_inflight_steps_left = 0
            self._oldest_unanswered_req_monotonic = None
            self._pad_future_to_low_watermark(last_aligned_frame)
        elif self._req_inflight:
            self._req_inflight_steps_left -= 1
            if self._req_inflight_steps_left <= 0:
                self._req_inflight = False
                self._req_inflight_steps_left = 0
        if self.vr_transport == "udp_stream":
            self._warn_stream_stale_if_needed()
        else:
            self._send_request_if_needed()
            self._warn_no_reply_if_needed()
        self._warn_horizon_if_needed("post_step")

    def poll_control_without_robot_step(self) -> None:
        self.post_step()

    def is_pico_video_ready(self) -> bool:
        return self._pico_video_ready()

    def is_vr_active(self) -> bool:
        return bool(self._vr_active)

    def is_vr_start_pending(self) -> bool:
        return bool(self._pending_start_request)

    def is_vr_user_enabled(self) -> bool:
        return bool(self._vr_user_enabled)

    def is_vr_fail_safe_triggered(self) -> bool:
        return bool(self._vr_fail_safe_triggered)

    def latest_stream_seq(self) -> Optional[int]:
        return self._last_udp_stream_seq

    def latest_control_age_ms(self) -> Optional[float]:
        return self._ms_since(self._last_ctrl_recv_monotonic)

    def deactivate(self):
        self._vr_user_enabled = False
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._oldest_unanswered_req_monotonic = None
        self._vr_active = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        self._vr_anchor_joint_pos = None
        self._vr_anchor_root_pos = None
        self._vr_anchor_root_quat = None
        self._vr_align_ready = False
        self._pending_start_request = False
        self._vr_fail_safe_triggered = False
        self._auto_restart_after_fail_safe = False
        if self._req_sock is not None:
            try:
                self._req_sock.close(0)
            except Exception:
                pass
            self._req_sock = None
        if self._rep_sock is not None:
            try:
                self._rep_sock.close(0)
            except Exception:
                pass
            self._rep_sock = None
        if self._ctrl_sock is not None:
            try:
                self._ctrl_sock.close(0)
            except Exception:
                pass
            self._ctrl_sock = None
        if self._udp_stream_sock is not None:
            try:
                self._udp_stream_sock.close()
            except Exception:
                pass
            self._udp_stream_sock = None
