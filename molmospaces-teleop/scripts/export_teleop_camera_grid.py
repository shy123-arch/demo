#!/usr/bin/env python3
"""Export a saved G1+Dex1 teleop trajectory as a four-camera MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from molmo_spaces.robots.g1_dex1_constants import DEX1_FINGER_LOWER, DEX1_FINGER_UPPER


CAMERAS = [
    ("THIRD PERSON / MOLMO ENV", "third_person_camera"),
    ("HEAD CAMERA", "head_camera"),
    ("LEFT WRIST CAMERA", "left_wrist_camera"),
    ("RIGHT WRIST CAMERA", "right_wrist_camera"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Path to trajectories*.h5 or a house_* directory. Defaults to latest saved teleop H5.",
    )
    parser.add_argument("--traj-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tile-width", type=int, default=960)
    parser.add_argument("--tile-height", type=int, default=540)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def latest_h5() -> Path:
    root = Path.home() / ".cache" / "molmospaces" / "assets"
    paths = sorted(
        root.glob("**/g1_dex1_teleop_v1/**/house_*/trajectories*.h5"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No saved teleop trajectories found under {root}")
    return paths[0]


def resolve_h5(input_path: Path | None) -> Path:
    if input_path is None:
        return latest_h5()
    input_path = input_path.expanduser()
    if input_path.is_dir():
        paths = sorted(input_path.glob("trajectories*.h5"), key=lambda p: p.stat().st_mtime)
        if not paths:
            raise FileNotFoundError(f"No trajectories*.h5 found in {input_path}")
        return paths[-1]
    return input_path


def suffix_from_h5(h5_path: Path) -> str:
    stem = h5_path.stem
    if stem == "trajectories":
        return ""
    if stem.startswith("trajectories"):
        return stem[len("trajectories") :]
    return ""


def video_path_for_camera(h5_path: Path, traj_index: int, camera_name: str) -> Path:
    suffix = suffix_from_h5(h5_path)
    expected = h5_path.parent / f"episode_{traj_index:08d}_{camera_name}{suffix}.mp4"
    if expected.exists():
        return expected
    matches = sorted(h5_path.parent.glob(f"episode_{traj_index:08d}_{camera_name}*.mp4"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing saved video for camera {camera_name!r} near {h5_path}")


def placeholder_tile(width: int, height: int, camera_name: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"{camera_name} not saved",
        (24, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (160, 160, 160),
        2,
        cv2.LINE_AA,
    )
    return frame


def decode_json_scalar(value) -> dict:
    if isinstance(value, bytes):
        text = value.decode("utf-8").rstrip("\x00")
    elif isinstance(value, np.ndarray):
        text = bytes(value).decode("utf-8").rstrip("\x00")
    else:
        text = str(value)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def read_episode_metadata(h5_path: Path, traj_index: int) -> tuple[str, np.ndarray | None, np.ndarray | None]:
    with h5py.File(h5_path, "r") as h5:
        traj_name = f"traj_{traj_index}"
        if traj_name not in h5:
            raise KeyError(f"{traj_name!r} not found in {h5_path}; available={list(h5.keys())}")
        group = h5[traj_name]
        obs_scene = decode_json_scalar(group["obs_scene"][()]) if "obs_scene" in group else {}
        task_text = obs_scene.get("task_description") or obs_scene.get("text") or ""
        success = group["success"][()] if "success" in group else None
        actions = None
        for action_name in ("right_dex1", "left_dex1"):
            action_path = f"actions/{action_name}"
            if action_path in group:
                actions = group[action_path][()]
                break
    return task_text, success, actions


def index_for_frame(frame_index: int, frame_count: int, data_len: int) -> int:
    if data_len <= 1 or frame_count <= 1:
        return 0
    return min(data_len - 1, round(frame_index * (data_len - 1) / (frame_count - 1)))


def dex1_state(actions: np.ndarray | None, frame_index: int, frame_count: int) -> str:
    if actions is None or len(actions) == 0:
        return "OPEN"
    idx = index_for_frame(frame_index, frame_count, len(actions))
    value = float(np.mean(actions[idx]))
    denom = float(DEX1_FINGER_UPPER - DEX1_FINGER_LOWER)
    open_amount = 1.0 if abs(denom) < 1e-9 else (value - DEX1_FINGER_LOWER) / denom
    open_amount = float(np.clip(open_amount, 0.0, 1.0))
    if open_amount > 0.75:
        return "OPEN"
    if open_amount < 0.25:
        return "CLOSED"
    return "PARTIAL"


def success_state(success: np.ndarray | None, frame_index: int, frame_count: int) -> str:
    if success is None or len(success) == 0:
        return ""
    idx = index_for_frame(frame_index, frame_count, len(success))
    return "SUCCESS" if bool(success[idx]) else ""


def resize_cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(frame, (round(src_w * scale), round(src_h * scale)))
    y0 = max((resized.shape[0] - height) // 2, 0)
    x0 = max((resized.shape[1] - width) // 2, 0)
    return resized[y0 : y0 + height, x0 : x0 + width]


def label_tile(frame: np.ndarray, label: str, gripper_state: str, task_text: str, status: str) -> np.ndarray:
    tile = frame.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 78), (0, 0, 0), -1)
    cv2.putText(tile, label, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        tile,
        f"DEX1: {gripper_state}",
        (18, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (80, 220, 255),
        2,
        cv2.LINE_AA,
    )
    if task_text:
        text = task_text[:90]
        cv2.rectangle(tile, (0, tile.shape[0] - 42), (tile.shape[1], tile.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            tile,
            text,
            (18, tile.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    if status:
        cv2.putText(
            tile,
            status,
            (tile.shape[1] - 180, 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )
    return tile


def main() -> None:
    args = parse_args()
    h5_path = resolve_h5(args.input).resolve()
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    captures = []
    missing_cameras: list[str] = []
    for _label, camera_name in CAMERAS:
        try:
            path = video_path_for_camera(h5_path, args.traj_index, camera_name)
        except FileNotFoundError:
            missing_cameras.append(camera_name)
            captures.append((camera_name, None, None))
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            missing_cameras.append(camera_name)
            captures.append((camera_name, path, None))
            continue
        captures.append((camera_name, path, cap))

    opened_caps = [cap for _name, _path, cap in captures if cap is not None]
    if not opened_caps:
        raise FileNotFoundError(f"No saved camera videos found near {h5_path}")
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in opened_caps]
    frame_count = min(count for count in frame_counts if count > 0)
    if args.max_frames is not None:
        frame_count = min(frame_count, args.max_frames)
    fps = args.fps or opened_caps[0].get(cv2.CAP_PROP_FPS) or 50.0

    task_text, success, actions = read_episode_metadata(h5_path, args.traj_index)
    output = args.output
    if output is None:
        output = h5_path.parent / f"traj_{args.traj_index:08d}_camera_grid.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    size = (args.tile_width * 2, args.tile_height * 2)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")

    try:
        for frame_index in range(frame_count):
            tiles = []
            gripper_state = dex1_state(actions, frame_index, frame_count)
            status = success_state(success, frame_index, frame_count)
            for (label, _camera_name), (_name, _path, cap) in zip(CAMERAS, captures, strict=True):
                if cap is None:
                    frame = placeholder_tile(args.tile_width, args.tile_height, _camera_name)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = resize_cover(frame, args.tile_width, args.tile_height)
                tiles.append(label_tile(frame, label, gripper_state, task_text, status))
            if len(tiles) != 4:
                break
            writer.write(np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:]))))
    finally:
        writer.release()
        for _name, _path, cap in captures:
            if cap is not None:
                cap.release()

    print(f"[export_teleop_camera_grid] h5={h5_path}")
    print("[export_teleop_camera_grid] cameras:")
    for camera_name, path, _cap in captures:
        print(f"  {camera_name}: {path}")
    if missing_cameras:
        print(f"[export_teleop_camera_grid] missing_cameras={missing_cameras}")
    print(f"[export_teleop_camera_grid] frames={frame_count} fps={fps:.2f}")
    print(f"[export_teleop_camera_grid] output={output.resolve()}")


if __name__ == "__main__":
    main()
