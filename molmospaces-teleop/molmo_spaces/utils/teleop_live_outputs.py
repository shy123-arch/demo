"""Environment-controlled live outputs for PICO teleoperation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from molmo_spaces.utils.teleop_camera_grid import TeleopCameraGridMonitor
from molmo_spaces.utils.virtual_camera_streamer import VirtualCameraStreamer


_MONITOR: TeleopCameraGridMonitor | None = None
_STREAMER: VirtualCameraStreamer | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class LiveTeleopOutputs:
    monitor: TeleopCameraGridMonitor | None = None
    streamer: VirtualCameraStreamer | None = None

    def update(self, observation: Any, *, task: Any = None, policy: Any = None, infos: Any = None) -> None:
        if self.monitor is not None:
            self.monitor.update(observation, task=task, policy=policy, infos=infos)
        if self.streamer is not None:
            self.streamer.submit_observation(observation, task=task)


def get_live_teleop_outputs_from_env() -> LiveTeleopOutputs | None:
    global _MONITOR, _STREAMER

    monitor = None
    streamer = None

    if _env_flag("MOLMO_TELEOP_MONITOR"):
        if _MONITOR is None:
            _MONITOR = TeleopCameraGridMonitor(
                tile_width=_env_int("MOLMO_TELEOP_MONITOR_TILE_WIDTH", 640),
                tile_height=_env_int("MOLMO_TELEOP_MONITOR_TILE_HEIGHT", 360),
                fps=_env_float("MOLMO_TELEOP_MONITOR_FPS", 15.0),
                window_name=os.getenv("MOLMO_TELEOP_MONITOR_WINDOW", "MolmoSpaces PICO Teleop"),
            )
        monitor = _MONITOR

    if _env_flag("MOLMO_PICO_VIRTUAL_CAMERA"):
        if _STREAMER is None:
            _STREAMER = VirtualCameraStreamer(
                camera_name=os.getenv("MOLMO_PICO_CAMERA_NAME", "head_camera"),
                left_camera_name=os.getenv("MOLMO_PICO_LEFT_CAMERA_NAME") or None,
                right_camera_name=os.getenv("MOLMO_PICO_RIGHT_CAMERA_NAME") or None,
                video_mode=os.getenv("MOLMO_PICO_VIDEO_MODE", "mono"),
                stereo_shift_px=_env_int("MOLMO_PICO_STEREO_SHIFT_PX", 8),
                target_ip=os.getenv("MOLMO_PICO_TARGET_IP") or None,
                prefer_control_ip=_env_flag("MOLMO_PICO_PREFER_CONTROL_IP"),
                host=os.getenv("MOLMO_PICO_CONTROL_HOST", "0.0.0.0"),
                port=_env_int("MOLMO_PICO_CONTROL_PORT", 13579),
                transport=os.getenv("MOLMO_PICO_STREAM_TRANSPORT") or None,
            )
            _STREAMER.start()
        streamer = _STREAMER

    if monitor is None and streamer is None:
        return None
    return LiveTeleopOutputs(monitor=monitor, streamer=streamer)


def pico_virtual_camera_ready() -> bool:
    if _STREAMER is None:
        return False
    return _STREAMER.has_active_session()
