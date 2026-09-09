"""Run and record unsupported PICO whole-body teleoperation in lab-sim."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from .env import LabEnv, ROOT
from .pico_tracking import (
    BodyPDController,
    Dex3HandMapper,
    LabSimStateBridge,
    load_molmo_tracking,
)


BUTTON_FIELDS = (
    "left_key_one",
    "left_key_two",
    "right_key_one",
    "right_key_two",
    "left_trigger_value",
    "right_trigger_value",
    "left_grip_value",
    "right_grip_value",
    "left_index_trig",
    "right_index_trig",
    "left_grip",
    "right_grip",
)


def _numeric_button(buttons: dict[str, Any], name: str) -> float:
    try:
        value = float(buttons.get(name, 0.0))
        return value if np.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


class PicoTeleopSession:
    """Own one 50 Hz tracking session and its native lab-sim recording."""

    def __init__(
        self,
        output_dir: str | Path,
        molmo_root: str | Path | None = None,
        scene: str = "lab_g1_vision.xml",
        start_x: float = 1.72,
        start_y: float = -0.80,
        start_yaw: float = 0.0,
        live: bool = True,
        offline_motion: str | None = None,
        viewer: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scene = scene
        self.start_pose = np.asarray(
            [start_x, start_y, start_yaw], dtype=np.float64
        )
        if not np.isfinite(self.start_pose).all():
            raise ValueError("Start pose must be finite")
        self.live = bool(live)
        self.offline_motion = offline_motion
        self.use_viewer = bool(viewer)
        self.bindings = load_molmo_tracking(molmo_root)
        self.env = LabEnv(scene=scene, assisted=False)
        self.model = self.env.model
        self.data = self.env.data
        if not np.isclose(self.model.opt.timestep, 0.002):
            raise ValueError(
                f"PICO teleop requires 2 ms physics, got {self.model.opt.timestep}"
            )
        self.data.eq_active[:] = 0
        self.env.assisted = False
        self.hands = Dex3HandMapper(self.model, self.data)
        self.bridge = LabSimStateBridge(
            self.model, self.data, self.bindings, self.hands
        )
        self.body_pd = BodyPDController(
            self.model, self.data, self.bindings
        )
        self._reset_robot()
        self.runtime = self.bindings.runtime_cls(
            self.bridge,
            tracking_config_path=self.bindings.config_path,
            policy_path=self.bindings.policy_path,
            motion_source="vr" if self.live else "udp",
            enable_transport=self.live,
        )
        if self.live:
            source = self.runtime.policy.source
            if (
                getattr(source, "vr_transport", "") == "udp_stream"
                and getattr(source, "_udp_stream_sock", None) is None
            ):
                self.runtime.close()
                self.env.close()
                raise RuntimeError(
                    "Could not bind PICO UDP stream port 28704. Stop the old "
                    "MolmoSpaces/lab-sim teleop process and try again."
                )
        elif offline_motion:
            if not self.runtime.policy.source.append_motion_from_tail(
                offline_motion
            ):
                self.runtime.close()
                self.env.close()
                raise ValueError(f"Unknown offline motion {offline_motion!r}")
        self.records: list[dict[str, Any]] = []
        self._viewer = None
        self._closed = False
        self._initial_xy = self.data.qpos[
            self.bridge.base_qadr : self.bridge.base_qadr + 2
        ].copy()
        self._minimum_pelvis_height = float(
            self.data.qpos[self.bridge.base_qadr + 2]
        )

    def _reset_robot(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        qa = self.bridge.base_qadr
        x, y, yaw = self.start_pose
        self.data.qpos[qa : qa + 3] = (x, y, 0.78)
        self.data.qpos[qa + 3 : qa + 7] = (
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        )
        self.data.qpos[self.bridge.qadr] = self.bindings.default_body_qpos
        self.data.qpos[self.hands.qadr] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.eq_active[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self.bridge.update()

    def _object_position(self, name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            return np.full(3, np.nan)
        return self.data.xpos[body_id].copy()

    def _joint_scalar(self, name: str) -> float:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            return float("nan")
        return float(self.data.qpos[self.model.jnt_qposadr[joint_id]])

    def _reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        policy = self.runtime.policy
        if (
            policy.ref_joint_pos is None
            or policy.ref_root_pos is None
            or policy.ref_root_quat is None
        ):
            return (
                np.zeros(29, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.array([1, 0, 0, 0], dtype=np.float32),
            )
        index = min(policy.ref_idx, policy.ref_len - 1)
        return (
            policy.ref_joint_pos[index].copy(),
            policy.ref_root_pos[index].copy(),
            policy.ref_root_quat[index].copy(),
        )

    def _record(
        self,
        wall_time: float,
        body_target: np.ndarray,
        body_torque: np.ndarray,
    ) -> None:
        ref_joint, ref_pos, ref_quat = self._reference()
        source = self.runtime.policy.source
        self.records.append(
            {
                "time": float(self.data.time),
                "wall_time": float(wall_time),
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "reference_joint_pos": ref_joint,
                "reference_root_pos": ref_pos,
                "reference_root_quat": ref_quat,
                "body_target": np.asarray(body_target).copy(),
                "body_torque": np.asarray(body_torque).copy(),
                "hand_targets": self.hands.targets.copy(),
                "controller_button_values": np.asarray(
                    [
                        _numeric_button(self.bridge.latest_buttons, name)
                        for name in BUTTON_FIELDS
                    ],
                    dtype=np.float32,
                ),
                "stream_active": bool(getattr(source, "_vr_active", False)),
                "stream_seq": int(
                    getattr(source, "_last_udp_stream_seq", None) or -1
                ),
                "sample_box": self._object_position("sample_box"),
                "sample_bottle": self._object_position("sample_bottle"),
                "button_travel": self._joint_scalar("button_slide"),
                "drawer_open": self._joint_scalar("drawer_slide"),
            }
        )

    def _flush(self, stop_reason: str, started_wall: float) -> dict[str, Any]:
        if self.records:
            keys = tuple(self.records[0])
            arrays = {
                key: np.asarray([record[key] for record in self.records])
                for key in keys
            }
        else:
            self._record(0.0, np.zeros(29), np.zeros(29))
            arrays = {key: np.asarray([value])[:0] for key, value in self.records.pop().items()}
        arrays["body_joint_names"] = np.asarray(
            self.bindings.body_joint_names
        )
        arrays["policy_joint_names"] = np.asarray(
            self.bindings.policy_joint_names
        )
        arrays["hand_joint_names"] = np.asarray(self.hands.names)
        arrays["controller_button_names"] = np.asarray(BUTTON_FIELDS)
        np.savez_compressed(self.output_dir / "trajectory.npz", **arrays)

        current_xy = self.data.qpos[
            self.bridge.base_qadr : self.bridge.base_qadr + 2
        ]
        report = {
            "scene": self.scene,
            "molmo_root": str(self.bindings.root),
            "tracking_config": str(self.bindings.config_path),
            "tracking_policy": str(self.bindings.policy_path),
            "live_pico": self.live,
            "offline_motion": self.offline_motion,
            "assisted": False,
            "equality_active": bool(self.data.eq_active.any()),
            "physics_hz": 500,
            "policy_hz": 50,
            "start_pose_xy_yaw": self.start_pose.tolist(),
            "simulation_duration_s": float(self.data.time),
            "wall_duration_s": float(time.monotonic() - started_wall),
            "recorded_frames": len(self.records),
            "stop_reason": stop_reason,
            "minimum_pelvis_height_m": self._minimum_pelvis_height,
            "root_displacement_xy_m": float(
                np.linalg.norm(current_xy - self._initial_xy)
            ),
            "finite_state": bool(
                np.isfinite(self.data.qpos).all()
                and np.isfinite(self.data.qvel).all()
            ),
            "pause_reason": self.bridge.pause_reason,
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report

    def run(self, seconds: float = 0.0) -> dict[str, Any]:
        """Run until duration, right B, viewer close, fall, or interruption."""

        if seconds < 0 or not np.isfinite(seconds):
            raise ValueError("seconds must be finite and nonnegative")
        started_wall = time.monotonic()
        stop_reason = "duration" if seconds > 0 else "stopped"
        body_target = self.bridge.default_qpos_real.copy()
        body_torque = np.zeros(29, dtype=np.float64)
        try:
            if self.use_viewer:
                from mujoco import viewer as mujoco_viewer
                self._viewer = mujoco_viewer.launch_passive(self.model, self.data)
            while True:
                if seconds > 0 and self.data.time >= seconds - 1e-9:
                    stop_reason = "duration"
                    break
                if self._viewer is not None and not self._viewer.is_running():
                    stop_reason = "viewer_closed"
                    break

                body_target = self.runtime.step()
                for _ in range(10):
                    self.data.ctrl[:] = 0.0
                    body_torque = self.body_pd.compute(body_target)
                    self.data.ctrl[self.body_pd.actuator_ids] = body_torque
                    self.data.ctrl[
                        self.hands.actuator_ids
                    ] = self.hands.compute_torque()
                    mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
                self.bridge.update()
                pelvis_z = float(
                    self.data.qpos[self.bridge.base_qadr + 2]
                )
                self._minimum_pelvis_height = min(
                    self._minimum_pelvis_height, pelvis_z
                )
                self._record(
                    time.monotonic() - started_wall,
                    body_target,
                    body_torque,
                )

                if not (
                    np.isfinite(self.data.qpos).all()
                    and np.isfinite(self.data.qvel).all()
                ):
                    stop_reason = "nonfinite_state"
                    break
                if pelvis_z < 0.45:
                    stop_reason = "fall"
                    break
                if self.bridge.finish_requested:
                    stop_reason = "pico_right_b"
                    break
                if self._viewer is not None:
                    self._viewer.sync()
                if self.live or self._viewer is not None:
                    target_wall = float(self.data.time)
                    delay = target_wall - (time.monotonic() - started_wall)
                    if delay > 0:
                        time.sleep(delay)
        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
        except Exception as exc:
            stop_reason = f"error:{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                report = self._flush(stop_reason, started_wall)
            finally:
                self.close()
        return report

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self.runtime.close()
        self.env.close()


def run_pico_teleop(args) -> dict[str, Any]:
    live = not bool(args.offline_motion)
    session = PicoTeleopSession(
        output_dir=args.output_dir,
        molmo_root=args.molmo_root,
        scene=args.scene,
        start_x=args.start_x,
        start_y=args.start_y,
        start_yaw=args.start_yaw,
        live=live,
        offline_motion=args.offline_motion,
        viewer=args.viewer,
    )
    report = session.run(args.seconds)
    if args.render_after:
        render_pico_demo(args.output_dir)
    return report


def render_pico_demo(
    output_dir: str | Path,
    camera: str = "grid",
    width: int = 960,
    height: int = 600,
    fps: float = 25.0,
) -> Path:
    """Render a saved PICO trajectory without rerunning physics."""

    import imageio.v2 as imageio

    output = Path(output_dir).expanduser().resolve()
    metadata = json.loads((output / "metadata.json").read_text())
    with np.load(output / "trajectory.npz") as saved:
        times = saved["time"].copy()
        qpos = saved["qpos"].copy()
        qvel = saved["qvel"].copy()
    if len(times) == 0:
        raise ValueError("Cannot render an empty PICO trajectory")
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("Render width, height and fps must be positive")
    env = LabEnv(scene=metadata["scene"], assisted=False)
    env.data.eq_active[:] = 0
    path = output / "task.mp4"
    frame_times = np.arange(times[0], times[-1] + 1e-9, 1.0 / fps)
    indices = np.clip(np.searchsorted(times, frame_times), 0, len(times) - 1)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=2,
    )
    try:
        for index in indices:
            env.data.qpos[:] = qpos[index]
            env.data.qvel[:] = qvel[index]
            mujoco.mj_forward(env.model, env.data)
            if camera == "grid":
                panel_width = width // 2
                panel_height = height // 2
                panels = [
                    env.render(name, panel_width, panel_height)
                    for name in (
                        "reference",
                        "vision_left",
                        "vision_right",
                        "right_wrist",
                    )
                ]
                frame = np.vstack(
                    (np.hstack(panels[:2]), np.hstack(panels[2:]))
                )
            else:
                frame = env.render(camera, width, height)
            writer.append_data(frame)
    finally:
        writer.close()
        env.close()
    return path
