"""Live four-camera monitor for G1+Dex1 teleoperation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np


CAMERA_TILES: tuple[tuple[str, str], ...] = (
    ("THIRD PERSON / MOLMO ENV", "third_person_camera"),
    ("HEAD CAMERA", "head_camera"),
    ("LEFT WRIST CAMERA", "left_wrist_camera"),
    ("RIGHT WRIST CAMERA", "right_wrist_camera"),
)


def _first_observation(observation: Any) -> dict[str, Any]:
    if isinstance(observation, list) and observation:
        return observation[0]
    if isinstance(observation, dict):
        return observation
    return {}


def _task_text(task: Any) -> str:
    if task is None:
        return ""
    try:
        return str(task.get_task_description())
    except Exception:
        return ""


def _success_text(infos: Any) -> str:
    info = infos[0] if isinstance(infos, list) and infos else infos
    if isinstance(info, dict) and info.get("success", False):
        return "SUCCESS"
    return ""


def _dex1_state(policy: Any) -> str:
    bridge = getattr(policy, "bridge", None)
    amounts = getattr(bridge, "dex1_open_amount", None)
    if not isinstance(amounts, dict) or not amounts:
        return "OPEN"
    value = float(np.mean([float(amounts.get(side, 1.0)) for side in ("left", "right")]))
    if value > 0.75:
        return "OPEN"
    if value < 0.25:
        return "CLOSED"
    return "PARTIAL"


def _to_uint8_rgb(frame: Any) -> np.ndarray | None:
    if not isinstance(frame, np.ndarray):
        return None
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        return None
    frame = frame[..., :3]
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        max_value = float(np.nanmax(frame)) if frame.size else 1.0
        scale = 255.0 if max_value <= 1.5 else 1.0
        return np.clip(frame * scale, 0, 255).astype(np.uint8)
    return np.clip(frame, 0, 255).astype(np.uint8)


class TeleopCameraGridMonitor:
    """Compose saved observation cameras into a live OpenCV window."""

    def __init__(
        self,
        *,
        tile_width: int = 640,
        tile_height: int = 360,
        fps: float = 15.0,
        window_name: str = "MolmoSpaces PICO Teleop",
    ) -> None:
        self.tile_width = int(tile_width)
        self.tile_height = int(tile_height)
        self.period_s = 1.0 / max(float(fps), 1.0)
        self.window_name = window_name
        self._last_update_s = 0.0
        self._disabled = False
        self._window_created = False

    def update(self, observation: Any, *, task: Any = None, policy: Any = None, infos: Any = None) -> None:
        if self._disabled:
            return
        now = time.monotonic()
        if now - self._last_update_s < self.period_s:
            return
        self._last_update_s = now

        obs = _first_observation(observation)
        frames = []
        gripper_state = _dex1_state(policy)
        task_description = _task_text(task)
        status = _success_text(infos)

        for label, camera_name in CAMERA_TILES:
            frame = _to_uint8_rgb(obs.get(camera_name))
            if frame is None:
                frame = np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
            tile = self._make_tile(frame, label, gripper_state, task_description, status)
            frames.append(tile)

        grid = np.vstack((np.hstack(frames[:2]), np.hstack(frames[2:])))
        self._show(grid)

    def _make_tile(
        self,
        frame_rgb: np.ndarray,
        label: str,
        gripper_state: str,
        task_description: str,
        status: str,
    ) -> np.ndarray:
        import cv2

        frame = cv2.resize(frame_rgb, (self.tile_width, self.tile_height), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 78), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label,
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"DEX1: {gripper_state}",
            (18, 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
        if status:
            cv2.putText(
                tile,
                status,
                (tile.shape[1] - 160, 61),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (80, 255, 120),
                2,
                cv2.LINE_AA,
            )
        if task_description:
            cv2.rectangle(tile, (0, tile.shape[0] - 38), (tile.shape[1], tile.shape[0]), (0, 0, 0), -1)
            cv2.putText(
                tile,
                task_description[:96],
                (18, tile.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        return tile

    def _show(self, grid_bgr: np.ndarray) -> None:
        try:
            import cv2

            if not self._window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, grid_bgr.shape[1], grid_bgr.shape[0])
                self._window_created = True
            cv2.imshow(self.window_name, grid_bgr)
            cv2.waitKey(1)
        except Exception as exc:
            self._disabled = True
            print(f"[TeleopCameraGridMonitor][disabled] {exc}")
