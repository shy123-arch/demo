#!/usr/bin/env python3
"""Replay a saved G1+Dex1 teleop trajectory and render a four-camera MP4.

This is meant for the fast PICO collection path: collect with only the live
head camera, then render the expensive four-camera view after the episode.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import json
import pickle
import re
from pathlib import Path

import cv2
import h5py
import mujoco
import numpy as np

from molmo_spaces.configs.camera_configs import G1Dex1CameraSystem, G1Dex1FastCameraSystem
from molmo_spaces.data_generation.config.g1_dex1_teleop_config import G1Dex1TeleopDataGenConfig
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.robots.g1_dex1_constants import DEX1_FINGER_LOWER, DEX1_FINGER_UPPER
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler


CAMERAS = [
    ("THIRD PERSON / MOLMO ENV", "third_person_camera"),
    ("HEAD CAMERA", "head_camera"),
    ("LEFT WRIST CAMERA", "left_wrist_camera"),
    ("RIGHT WRIST CAMERA", "right_wrist_camera"),
]


def import_class_from_string(class_path: str) -> type:
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid class path: {class_path}. Expected 'module.ClassName' format.")
    module_path, class_name = parts
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


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
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS. Defaults to measured capture FPS for wall timing, or 25 for index timing.",
    )
    parser.add_argument(
        "--timing",
        choices=("wall", "index"),
        default="wall",
        help=(
            "wall=resample active teleop states to recorded wall-clock speed; "
            "index=one output frame per saved state."
        ),
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth saved state.")
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=360)
    parser.add_argument(
        "--keep-initial-gap",
        action="store_true",
        help="Keep the pre-control gap between the first reset frame and active teleop frames.",
    )
    parser.add_argument(
        "--initial-gap-threshold-s",
        type=float,
        default=2.0,
        help="Skip the first wall-time gap when it is larger than this and much larger than normal steps.",
    )
    parser.add_argument(
        "--trim-trailing-idle-s",
        type=float,
        default=3.0,
        help=(
            "Keep this many seconds after the last detected movement, then stop rendering. "
            "Use a negative value to disable tail trimming."
        ),
    )
    parser.add_argument(
        "--idle-motion-eps",
        type=float,
        default=1e-3,
        help="State delta threshold for trailing idle detection, using MuJoCo qpos when available.",
    )
    parser.add_argument(
        "--camera-profile",
        choices=("fast", "full"),
        default="fast",
        help="fast=320x240 RGB source cameras; full=640x480 with depth-capable config.",
    )
    parser.add_argument(
        "--third-person-mode",
        choices=("auto", "mjcf"),
        default="auto",
        help="auto=adaptive full-body camera for replay; mjcf=use robot tracking camera.",
    )
    parser.add_argument("--third-person-fov", type=float, default=92.0)
    return parser.parse_args()


def latest_h5() -> Path:
    roots = [
        ASSETS_DIR / "experiment_output" / "datagen",
        Path.home() / ".cache" / "molmospaces" / "assets",
    ]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(root.glob("**/g1_dex1_teleop_v1/**/house_*/trajectories*.h5"))
    paths = sorted(set(paths), key=lambda p: p.stat().st_mtime, reverse=True)
    if not paths:
        raise FileNotFoundError("No saved teleop trajectories found.")
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


def house_index_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.match(r"house_(\d+)(?:\D|$)", part)
        if match:
            return int(match.group(1))
    try:
        with h5py.File(path, "r") as h5:
            if "house_index" in h5.attrs:
                return int(h5.attrs["house_index"])
            if "source_h5" in h5.attrs:
                return house_index_from_path(Path(str(h5.attrs["source_h5"])))
    except Exception:
        pass
    return 0


def decode_json_scalar(value) -> dict:
    if isinstance(value, bytes):
        text = value.decode("utf-8").rstrip("\x00")
    elif isinstance(value, np.ndarray):
        text = bytes(value).decode("utf-8").rstrip("\x00")
    else:
        text = str(value)
    return json.loads(text)


def decode_json_row(row) -> dict:
    return decode_json_scalar(row)


def nearest_indices_for_times(wall_times: np.ndarray, target_times: np.ndarray) -> list[int]:
    insert_at = np.searchsorted(wall_times, target_times, side="left")
    indices = []
    last = len(wall_times) - 1
    for pos, target in zip(insert_at, target_times, strict=True):
        if pos <= 0:
            indices.append(0)
        elif pos > last:
            indices.append(last)
        else:
            before = pos - 1
            after = pos
            if abs(float(wall_times[after]) - float(target)) < abs(
                float(target) - float(wall_times[before])
            ):
                indices.append(after)
            else:
                indices.append(before)
    return indices


def active_wall_start_index(
    wall_times: np.ndarray,
    keep_initial_gap: bool,
    initial_gap_threshold_s: float,
) -> tuple[int, float]:
    if keep_initial_gap or len(wall_times) < 3:
        return 0, 0.0

    diffs = np.diff(wall_times)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(positive_diffs) == 0:
        return 0, 0.0

    median_step_s = float(np.median(positive_diffs))
    threshold_s = max(float(initial_gap_threshold_s), median_step_s * 10.0)
    first_gap_s = float(diffs[0])
    if first_gap_s > threshold_s:
        return 1, first_gap_s
    return 0, 0.0


def numeric_values(value) -> list[float]:
    if isinstance(value, dict):
        out: list[float] = []
        for child in value.values():
            out.extend(numeric_values(child))
        return out
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for child in value:
            out.extend(numeric_values(child))
        return out
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    return []


def motion_array(group, total_states: int) -> np.ndarray | None:
    if "env_states/mj_qpos" in group:
        return np.asarray(group["env_states/mj_qpos"][:total_states], dtype=np.float64)
    if "obs/agent/qpos" not in group:
        return None
    rows = []
    for row in group["obs/agent/qpos"][:total_states]:
        rows.append(numeric_values(decode_json_row(row)))
    if not rows:
        return None
    min_len = min(len(row) for row in rows)
    if min_len <= 0:
        return None
    return np.asarray([row[:min_len] for row in rows], dtype=np.float64)


def trailing_idle_end_index(
    group,
    total_states: int,
    start_index: int,
    trim_trailing_idle_s: float,
    idle_motion_eps: float,
) -> tuple[int, float]:
    if trim_trailing_idle_s < 0 or total_states < 3 or "wall_time_s" not in group:
        return total_states - 1, 0.0

    wall_times = np.asarray(group["wall_time_s"][:total_states], dtype=np.float64)
    states = motion_array(group, total_states)
    if states is None or len(states) < 2:
        return total_states - 1, 0.0

    deltas = np.nanmax(np.abs(np.diff(states, axis=0)), axis=1)
    moving_transitions = np.flatnonzero(deltas[start_index:] > float(idle_motion_eps))
    if moving_transitions.size:
        last_motion_state = min(total_states - 1, start_index + int(moving_transitions[-1]) + 1)
    else:
        last_motion_state = start_index

    keep_until = float(wall_times[last_motion_state]) + float(trim_trailing_idle_s)
    end_index = int(np.searchsorted(wall_times, keep_until, side="right") - 1)
    end_index = max(start_index, min(total_states - 1, end_index))
    trimmed_s = max(0.0, float(wall_times[-1] - wall_times[end_index]))
    return end_index, trimmed_s


def infer_output_fps(
    group,
    total_states: int,
    requested_fps: float | None,
    timing: str,
    keep_initial_gap: bool,
    initial_gap_threshold_s: float,
    end_index: int | None = None,
) -> float:
    if requested_fps is not None:
        return float(requested_fps)
    if timing == "wall" and "wall_time_s" in group and total_states > 1:
        wall_times = np.asarray(group["wall_time_s"][()], dtype=np.float64)
        wall_times = wall_times[:total_states]
        start_index, _skipped_gap_s = active_wall_start_index(
            wall_times=wall_times,
            keep_initial_gap=keep_initial_gap,
            initial_gap_threshold_s=initial_gap_threshold_s,
        )
        end = total_states - 1 if end_index is None else int(end_index)
        active_wall_times = wall_times[start_index : end + 1]
        duration = float(active_wall_times[-1] - active_wall_times[0])
        intervals = len(active_wall_times) - 1
        if duration > 0.0 and intervals > 0:
            return intervals / duration
    return 25.0


def build_frame_indices(
    group,
    total_states: int,
    fps: float,
    timing: str,
    stride: int,
    keep_initial_gap: bool,
    initial_gap_threshold_s: float,
    end_index: int | None = None,
) -> list[int]:
    stride = max(1, int(stride))
    if timing == "wall" and "wall_time_s" in group and total_states > 1:
        wall_times = np.asarray(group["wall_time_s"][()], dtype=np.float64)
        wall_times = wall_times[:total_states]
        start_index, skipped_gap_s = active_wall_start_index(
            wall_times=wall_times,
            keep_initial_gap=keep_initial_gap,
            initial_gap_threshold_s=initial_gap_threshold_s,
        )
        if start_index:
            print(
                "[ReplayRender] skipped initial inactive wall-time gap "
                f"{skipped_gap_s:.3f}s before active teleop replay"
            )
        end_i = total_states - 1 if end_index is None else int(end_index)
        active_wall_times = wall_times[start_index : end_i + 1]
        start = float(active_wall_times[0])
        end = float(active_wall_times[-1])
        duration = max(end - start, 0.0)
        output_frames = max(1, int(round(duration * fps)) + 1)
        target_times = start + np.arange(output_frames, dtype=np.float64) / fps
        target_times = np.minimum(target_times, end)
        relative_indices = nearest_indices_for_times(active_wall_times, target_times)
        return [i + start_index for i in relative_indices][::stride]
    return list(range(0, total_states, stride))


def load_saved_episode(group) -> tuple[object, dict]:
    obs_scene = decode_json_scalar(group["obs_scene"][()])
    frozen_config = obs_scene.get("frozen_config")
    if not frozen_config:
        raise KeyError("obs_scene does not contain frozen_config; cannot replay scene.")
    try:
        saved_episode = json.loads(frozen_config)
    except json.JSONDecodeError:
        saved_episode = pickle.loads(base64.b64decode(frozen_config))
    return saved_episode, obs_scene


def runtime_config_from_saved(saved_episode, h5_path: Path, camera_profile: str):
    cfg = G1Dex1TeleopDataGenConfig()
    cfg.use_passive_viewer = False
    cfg.num_workers = 1
    cfg.task_horizon = 1
    cfg.task_sampler_config.house_inds = [house_index_from_path(h5_path)]
    cfg.camera_config = G1Dex1FastCameraSystem() if camera_profile == "fast" else G1Dex1CameraSystem()

    runtime_robot_config = cfg.robot_config.model_copy(deep=True)
    saved_robot_config = getattr(saved_episode, "robot_config", None)
    if saved_robot_config is not None and getattr(saved_robot_config, "init_qpos", None) is not None:
        runtime_robot_config.init_qpos = saved_robot_config.init_qpos
    runtime_robot_config.init_qpos_noise_range = None
    cfg.robot_config = runtime_robot_config

    task_config = getattr(saved_episode, "task_config", None)
    task_cls_str = getattr(saved_episode, "task_cls_str", None)
    if task_config is None or task_cls_str is None:
        raise ValueError("frozen_config is missing task_config or task_cls_str")
    task_config = task_config.model_copy(deep=True)
    task_config.task_cls = import_class_from_string(task_cls_str)
    cfg.task_config = task_config
    cfg.task_type = "pick"
    return cfg


def build_task(saved_episode, h5_path: Path, camera_profile: str):
    cfg = runtime_config_from_saved(saved_episode, h5_path, camera_profile)
    sampler = PickTaskSampler(cfg)
    task = sampler.sample_task(house_index=house_index_from_path(h5_path))
    task.reset()
    return task


def apply_robot_state(task, qpos: dict, qvel: dict | None) -> None:
    robot = task.env.current_robot
    view = robot.robot_view
    for group_name, values in qpos.items():
        if group_name not in view.move_group_ids():
            continue
        group = view.get_move_group(group_name)
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == group.pos_dim:
            group.joint_pos = arr
    if qvel:
        for group_name, values in qvel.items():
            if group_name not in view.move_group_ids():
                continue
            group = view.get_move_group(group_name)
            arr = np.asarray(values, dtype=np.float64)
            if group.vel_dim and arr.size == group.vel_dim:
                group.joint_vel = arr
    mujoco.mj_forward(task.env.current_model, task.env.current_data)
    task.env.camera_manager.registry.update_all_cameras(task.env)


def apply_mujoco_state(task, mj_qpos: np.ndarray, mj_qvel: np.ndarray | None) -> None:
    model = task.env.current_model
    data = task.env.current_data

    qpos = np.asarray(mj_qpos, dtype=np.float64).reshape(-1)
    qpos_len = min(model.nq, qpos.size)
    data.qpos[:qpos_len] = qpos[:qpos_len]

    if mj_qvel is not None:
        qvel = np.asarray(mj_qvel, dtype=np.float64).reshape(-1)
        qvel_len = min(model.nv, qvel.size)
        data.qvel[:qvel_len] = qvel[:qvel_len]

    mujoco.mj_forward(model, data)
    task.env.camera_manager.registry.update_all_cameras(task.env)


THIRD_PERSON_LOCAL_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (-1.7, -1.0, 1.20),
    (-1.7, 1.0, 1.20),
    (-2.1, 0.0, 1.35),
    (-1.1, -1.7, 1.30),
    (-1.1, 1.7, 1.30),
    (0.0, -2.0, 1.35),
    (0.0, 2.0, 1.35),
    (1.5, -1.2, 1.30),
    (1.5, 1.2, 1.30),
    (-0.7, -0.7, 1.85),
    (-0.7, 0.7, 1.85),
    (0.7, -0.7, 1.85),
    (0.7, 0.7, 1.85),
)


def _base_pose_from_qpos(qpos: dict) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(qpos.get("base", []), dtype=np.float64)
    if base.size >= 7:
        return base[:3], base[3:7]
    return np.array([0.0, 0.0, 0.8], dtype=np.float64), np.array(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.size < 4:
        return 0.0
    norm = np.linalg.norm(quat[:4])
    if norm < 1e-9:
        return 0.0
    w, x, y, z = quat[:4] / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _local_offset_to_world(base_pos: np.ndarray, yaw: float, offset: tuple[float, float, float]) -> np.ndarray:
    ox, oy, oz = offset
    c = np.cos(yaw)
    s = np.sin(yaw)
    return base_pos + np.array([c * ox - s * oy, s * ox + c * oy, oz], dtype=np.float64)


def _lookat_vectors(camera_pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - camera_pos
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        forward = forward / norm
    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up_ref)
    if np.linalg.norm(right) < 1e-6:
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up_ref)
    right = right / max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    up = up / max(np.linalg.norm(up), 1e-9)
    return forward.astype(np.float32), up.astype(np.float32)


def _body_name_for_geom(model, geom_id: int) -> str:
    if geom_id < 0:
        return ""
    body_id = int(model.geom_bodyid[geom_id])
    return str(model.body(body_id).name or "")


def _is_robot_body(body_name: str) -> bool:
    lower = body_name.lower()
    return body_name.startswith("robot_0/") or "dex1" in lower or "g1" in lower


def _ray_candidate_score(task, camera_pos: np.ndarray, base_pos: np.ndarray, target: np.ndarray) -> float:
    model = task.env.current_model
    data = task.env.current_data
    geomgroup = np.ones(6, dtype=np.uint8)
    score = 0.0
    target_points = (
        target + np.array([0.0, 0.0, -0.35], dtype=np.float64),
        target,
        target + np.array([0.0, 0.0, 0.45], dtype=np.float64),
    )
    for point in target_points:
        point = point.copy()
        point[2] = max(0.10, point[2])
        ray = point - camera_pos
        dist = float(np.linalg.norm(ray))
        if dist < 1e-6:
            score -= 10.0
            continue
        direction = ray / dist
        geomid = np.array([-1], dtype=np.int32)
        hit_dist = float(
            mujoco.mj_ray(
                model,
                data,
                camera_pos.astype(np.float64),
                direction.astype(np.float64),
                geomgroup,
                1,
                -1,
                geomid,
            )
        )
        if hit_dist < 0.0 or hit_dist >= dist * 0.97:
            score += 1.0
            continue

        body_name = _body_name_for_geom(model, int(geomid[0]))
        if _is_robot_body(body_name):
            score += 2.0
        elif hit_dist >= dist * 0.82:
            score += 0.2
        else:
            score -= 6.0

    horizontal_dist = float(np.linalg.norm((camera_pos - base_pos)[:2]))
    score -= abs(horizontal_dist - 2.0) * 0.2
    score -= max(0.0, 0.7 - float(camera_pos[2])) * 3.0
    return score


def adaptive_third_person_pose(
    task,
    qpos: dict,
    preferred_index: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    base_pos, base_quat = _base_pose_from_qpos(qpos)
    yaw = _yaw_from_quat_wxyz(base_quat)
    target = base_pos + np.array([0.0, 0.0, 0.35], dtype=np.float64)
    candidates = [
        _local_offset_to_world(base_pos, yaw, offset) for offset in THIRD_PERSON_LOCAL_OFFSETS
    ]

    scored = [
        (_ray_candidate_score(task, camera_pos, base_pos, target), idx, camera_pos)
        for idx, camera_pos in enumerate(candidates)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_index, best_pos = scored[0]

    if preferred_index is not None and 0 <= preferred_index < len(candidates):
        preferred_pos = candidates[preferred_index]
        preferred_score = _ray_candidate_score(task, preferred_pos, base_pos, target)
        if preferred_score >= best_score - 1.0:
            best_index = preferred_index
            best_pos = preferred_pos

    forward, up = _lookat_vectors(best_pos, target)
    return best_pos.astype(np.float32), forward, up, best_index


def render_third_person_frame(
    task,
    qpos: dict,
    *,
    mode: str,
    fov: float,
    preferred_index: int | None,
) -> tuple[np.ndarray, int | None]:
    if mode == "mjcf":
        return task.env.render_rgb_frame("third_person_camera"), preferred_index

    pos, forward, up, selected_index = adaptive_third_person_pose(task, qpos, preferred_index)
    frame = task.env._render_frame(pos, forward, up, fov, segmentation=False)
    return frame, selected_index


def resize_cover(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(frame, (round(src_w * scale), round(src_h * scale)))
    y0 = max((resized.shape[0] - height) // 2, 0)
    x0 = max((resized.shape[1] - width) // 2, 0)
    return resized[y0 : y0 + height, x0 : x0 + width]


def dex1_state(qpos: dict) -> str:
    vals = []
    for key in ("left_dex1", "right_dex1"):
        if key in qpos:
            vals.extend(qpos[key])
    if not vals:
        return "OPEN"
    value = float(np.mean(vals))
    denom = float(DEX1_FINGER_UPPER - DEX1_FINGER_LOWER)
    open_amount = 1.0 if abs(denom) < 1e-9 else (value - DEX1_FINGER_LOWER) / denom
    open_amount = float(np.clip(open_amount, 0.0, 1.0))
    if open_amount > 0.75:
        return "OPEN"
    if open_amount < 0.25:
        return "CLOSED"
    return "PARTIAL"


def label_tile(frame: np.ndarray, label: str, gripper_state: str, task_text: str) -> np.ndarray:
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
        cv2.rectangle(tile, (0, tile.shape[0] - 42), (tile.shape[1], tile.shape[0]), (0, 0, 0), -1)
        cv2.putText(
            tile,
            task_text[:90],
            (18, tile.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return tile


def main() -> None:
    args = parse_args()
    h5_path = resolve_h5(args.input).resolve()
    output = args.output
    if output is None:
        output = h5_path.parent / f"traj_{args.traj_index:08d}_replay_camera_grid.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        traj_name = f"traj_{args.traj_index}"
        if traj_name not in h5:
            raise KeyError(f"{traj_name!r} not found in {h5_path}; available={list(h5.keys())}")
        group = h5[traj_name]
        saved_episode, obs_scene = load_saved_episode(group)
        qpos_rows = group["obs/agent/qpos"]
        qvel_rows = group["obs/agent/qvel"] if "obs/agent/qvel" in group else None
        mj_qpos_rows = group["env_states/mj_qpos"] if "env_states/mj_qpos" in group else None
        mj_qvel_rows = group["env_states/mj_qvel"] if "env_states/mj_qvel" in group else None
        total_states = len(qpos_rows)
        end_index = None
        if args.timing == "wall" and "wall_time_s" in group and total_states > 1:
            wall_times_for_range = np.asarray(group["wall_time_s"][:total_states], dtype=np.float64)
            start_index_for_trim, _ = active_wall_start_index(
                wall_times=wall_times_for_range,
                keep_initial_gap=args.keep_initial_gap,
                initial_gap_threshold_s=args.initial_gap_threshold_s,
            )
            end_index, trimmed_tail_s = trailing_idle_end_index(
                group=group,
                total_states=total_states,
                start_index=start_index_for_trim,
                trim_trailing_idle_s=args.trim_trailing_idle_s,
                idle_motion_eps=args.idle_motion_eps,
            )
            if trimmed_tail_s > 0.0:
                print(
                    "[ReplayRender] trimmed trailing idle tail "
                    f"{trimmed_tail_s:.3f}s after keeping {args.trim_trailing_idle_s:.3f}s idle"
                )
        output_fps = infer_output_fps(
            group=group,
            total_states=total_states,
            requested_fps=args.fps,
            timing=args.timing,
            keep_initial_gap=args.keep_initial_gap,
            initial_gap_threshold_s=args.initial_gap_threshold_s,
            end_index=end_index,
        )

        frame_indices = build_frame_indices(
            group=group,
            total_states=total_states,
            fps=output_fps,
            timing=args.timing,
            stride=args.stride,
            keep_initial_gap=args.keep_initial_gap,
            initial_gap_threshold_s=args.initial_gap_threshold_s,
            end_index=end_index,
        )
        if args.max_frames is not None:
            frame_indices = frame_indices[: args.max_frames]

        task = build_task(saved_episode, h5_path, args.camera_profile)
        size = (args.tile_width * 2, args.tile_height * 2)
        print(
            f"[ReplayRender] output_fps={output_fps:.3f} timing={args.timing} "
            f"output_frames={len(frame_indices)} source_states={total_states}"
        )
        if mj_qpos_rows is None:
            print(
                "[ReplayRender][WARNING] Full MuJoCo state is missing in this trajectory; "
                "offline replay restores robot joints only, so lifted/free objects may be wrong. "
                "Collect a new trajectory with the updated env_state sensor for faithful replay."
            )
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, size)
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output}")

        task_text = obs_scene.get("task_description") or obs_scene.get("text") or ""
        cached_src_i = None
        cached_tiles = None
        third_person_index = None
        try:
            for out_i, src_i in enumerate(frame_indices):
                if cached_src_i != src_i:
                    qpos = decode_json_row(qpos_rows[src_i])
                    if mj_qpos_rows is not None:
                        mj_qvel = mj_qvel_rows[src_i] if mj_qvel_rows is not None else None
                        apply_mujoco_state(task, mj_qpos_rows[src_i], mj_qvel)
                    else:
                        qvel = decode_json_row(qvel_rows[src_i]) if qvel_rows is not None else None
                        apply_robot_state(task, qpos, qvel)

                    tiles = []
                    gripper_state = dex1_state(qpos)
                    for label, camera_name in CAMERAS:
                        if camera_name == "third_person_camera":
                            frame, third_person_index = render_third_person_frame(
                                task,
                                qpos,
                                mode=args.third_person_mode,
                                fov=args.third_person_fov,
                                preferred_index=third_person_index,
                            )
                        else:
                            frame = task.env.render_rgb_frame(camera_name)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame = resize_cover(frame, args.tile_width, args.tile_height)
                        tiles.append(label_tile(frame, label, gripper_state, task_text))
                    cached_src_i = src_i
                    cached_tiles = tiles
                else:
                    tiles = cached_tiles
                writer.write(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
                if out_i == 0 or (out_i + 1) % 50 == 0 or out_i + 1 == len(frame_indices):
                    print(
                        f"[ReplayRender] {out_i + 1}/{len(frame_indices)} frames "
                        f"(source {src_i + 1}/{total_states})"
                    )
        finally:
            writer.release()
            with np.errstate(all="ignore"):
                task.close()

    print(f"[ReplayRender][SAVED] {output}")


if __name__ == "__main__":
    main()
