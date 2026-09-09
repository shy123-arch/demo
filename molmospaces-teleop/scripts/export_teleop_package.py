#!/usr/bin/env python3
"""Package saved PICO teleop trajectories with replay metadata.

The H5 already contains the frozen episode config and MuJoCo state needed for
scene/action replay in a MolmoSpaces checkout with the same assets available.
This script collects those files, checks the replay-critical fields, and writes
human-readable manifests next to the data.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from molmo_spaces.molmo_spaces_constants import ASSETS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help=(
            "Path to a trajectories*.h5 file, a house_* directory, or a run timestamp "
            "directory containing house_*/trajectories*.h5. Defaults to latest save marker."
        ),
    )
    parser.add_argument(
        "--latest-save-event",
        type=Path,
        default=Path("/mnt/f/molospace/run_outputs/latest_save_event.txt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/f/molospace/run_outputs/teleop_packages"),
    )
    parser.add_argument("--name", type=str, default=None, help="Package directory name.")
    parser.add_argument(
        "--traj-index",
        default="all",
        help="Trajectory index to render/package, or 'all'. Summaries are always written for all.",
    )
    parser.add_argument(
        "--render-grid",
        action="store_true",
        help="Also render four-camera replay MP4s into package/replays.",
    )
    parser.add_argument("--timing", choices=("wall", "index"), default="wall")
    parser.add_argument("--tar", action="store_true", help="Create a .tar.gz after packaging.")
    return parser.parse_args()


def parse_key_value_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def latest_h5_from_marker(marker_path: Path) -> Path:
    marker = parse_key_value_file(marker_path)
    if "hdf5" not in marker:
        raise FileNotFoundError(f"No hdf5= entry in {marker_path}")
    return Path(marker["hdf5"]).expanduser()


def latest_h5_from_cache() -> Path:
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


def resolve_h5_paths(input_path: Path | None, marker_path: Path) -> list[Path]:
    if input_path is None:
        try:
            return [latest_h5_from_marker(marker_path)]
        except FileNotFoundError:
            return [latest_h5_from_cache()]

    input_path = input_path.expanduser()
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    direct = sorted(input_path.glob("trajectories*.h5"), key=lambda p: p.name)
    nested = sorted(input_path.glob("house_*/trajectories*.h5"), key=lambda p: str(p))
    paths = direct + nested
    if not paths:
        raise FileNotFoundError(f"No trajectories*.h5 found under {input_path}")
    return paths


def house_index_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"house_(\d+)", part)
        if match:
            return int(match.group(1))
    return 0


def run_root_from_h5(path: Path) -> Path:
    parent = path.parent
    if re.fullmatch(r"house_\d+", parent.name):
        return parent.parent
    return parent


def decode_bytes(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if isinstance(value, str):
        return value.rstrip("\x00")
    if isinstance(value, np.ndarray):
        return bytes(np.asarray(value, dtype=np.uint8)).decode("utf-8", errors="replace").rstrip(
            "\x00"
        )
    return str(value)


def decode_json_scalar(value: Any) -> dict[str, Any]:
    text = decode_bytes(value)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    return str(value)


def decode_frozen_config(obs_scene: dict[str, Any]) -> tuple[Any | None, str | None]:
    frozen_config = obs_scene.get("frozen_config")
    if not frozen_config:
        return None, None
    if isinstance(frozen_config, str):
        try:
            return json.loads(frozen_config), "json"
        except json.JSONDecodeError:
            pass
        try:
            return pickle.loads(base64.b64decode(frozen_config)), "pickle_base64"
        except Exception:
            return None, "decode_failed"
    return None, "unsupported"


def frozen_summary(obs_scene: dict[str, Any]) -> dict[str, Any]:
    saved_episode, encoding = decode_frozen_config(obs_scene)
    summary: dict[str, Any] = {"encoding": encoding, "present": saved_episode is not None}
    if saved_episode is None:
        return summary

    if isinstance(saved_episode, dict):
        summary["type"] = "dict"
        summary["keys"] = sorted(saved_episode.keys())
        return summary

    task_config = getattr(saved_episode, "task_config", None)
    robot_config = getattr(saved_episode, "robot_config", None)
    camera_config = getattr(saved_episode, "camera_config", None)
    object_poses = getattr(task_config, "object_poses", None) or {}
    cameras = getattr(camera_config, "cameras", []) or []

    summary.update(
        {
            "type": type(saved_episode).__name__,
            "task_cls_str": getattr(saved_episode, "task_cls_str", None),
            "pickup_obj_name": getattr(task_config, "pickup_obj_name", None),
            "pickup_obj_start_pose": getattr(task_config, "pickup_obj_start_pose", None),
            "robot_base_pose": getattr(task_config, "robot_base_pose", None),
            "object_pose_count": len(object_poses),
            "object_pose_names_sample": list(object_poses.keys())[:20],
            "camera_names": [getattr(camera, "name", None) for camera in cameras],
            "robot_has_init_qpos": bool(getattr(robot_config, "init_qpos", None)),
        }
    )
    return safe_json_value(summary)


def dataset_tree(group: h5py.Group, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            rows.append(
                {
                    "path": f"{prefix}{name}",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            )

    group.visititems(visit)
    return rows


def bool_array_summary(array: np.ndarray | None) -> dict[str, Any]:
    if array is None:
        return {"present": False}
    values = np.asarray(array).astype(bool)
    true_indices = np.flatnonzero(values)
    return {
        "present": True,
        "any": bool(values.any()),
        "last": bool(values[-1]) if values.size else False,
        "count": int(values.sum()),
        "first_true_index": int(true_indices[0]) if true_indices.size else None,
        "length": int(values.size),
    }


def wall_time_summary(group: h5py.Group) -> dict[str, Any]:
    if "wall_time_s" not in group:
        return {"present": False}
    times = np.asarray(group["wall_time_s"][()], dtype=np.float64)
    if len(times) < 2:
        return {"present": True, "length": int(len(times))}
    duration = float(times[-1] - times[0])
    fps = float((len(times) - 1) / duration) if duration > 0 else None
    return {"present": True, "length": int(len(times)), "duration_s": duration, "fps": fps}


def collect_video_refs(group: h5py.Group, h5_path: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if "obs/sensor_data" not in group:
        return refs
    sensor_group = group["obs/sensor_data"]
    for camera_name, dataset in sensor_group.items():
        filename = decode_bytes(dataset[()])
        if not filename:
            continue
        video_path = h5_path.parent / filename
        refs.append(
            {
                "camera": camera_name,
                "filename": filename,
                "exists": video_path.exists(),
                "size_bytes": video_path.stat().st_size if video_path.exists() else None,
            }
        )
    return refs


def trajectory_indices(h5: h5py.File) -> list[int]:
    indices = []
    for key in h5.keys():
        match = re.fullmatch(r"traj_(\d+)", key)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def summarize_trajectory(h5_path: Path, traj_index: int) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as h5:
        traj_name = f"traj_{traj_index}"
        if traj_name not in h5:
            raise KeyError(f"{traj_name} not found in {h5_path}")
        group = h5[traj_name]
        obs_scene = decode_json_scalar(group["obs_scene"][()]) if "obs_scene" in group else {}
        success = group["success"][()] if "success" in group else None
        action_keys = sorted(row["path"] for row in dataset_tree(group["actions"])) if "actions" in group else []
        data_tree = dataset_tree(group)
        summary = {
            "h5": str(h5_path),
            "house_index": house_index_from_path(h5_path),
            "traj_index": traj_index,
            "task_type": obs_scene.get("task_type"),
            "task_description": obs_scene.get("task_description"),
            "policy_dt_ms": obs_scene.get("policy_dt_ms"),
            "success": bool_array_summary(success),
            "wall_time": wall_time_summary(group),
            "has_frozen_config": bool(obs_scene.get("frozen_config")),
            "has_mj_qpos": "env_states/mj_qpos" in group,
            "has_mj_qvel": "env_states/mj_qvel" in group,
            "has_agent_qpos": "obs/agent/qpos" in group,
            "action_keys": action_keys,
            "video_refs": collect_video_refs(group, h5_path),
            "frozen_config_summary": frozen_summary(obs_scene),
            "datasets": data_tree,
        }
        summary["replay_complete_check"] = {
            "ok": bool(
                summary["has_frozen_config"]
                and summary["has_mj_qpos"]
                and summary["has_mj_qvel"]
                and summary["has_agent_qpos"]
            ),
            "requires_same_molmospaces_assets": True,
            "notes": [
                "H5 contains frozen scene/task config plus MuJoCo qpos/qvel state.",
                "Receiver still needs a compatible MolmoSpaces checkout and asset cache/download access.",
            ],
        }
        return safe_json_value(summary)


def selected_traj_indices(all_indices: list[int], spec: str) -> list[int]:
    if spec == "all":
        return all_indices
    selected = [int(part.strip()) for part in spec.split(",") if part.strip()]
    missing = sorted(set(selected) - set(all_indices))
    if missing:
        raise ValueError(f"Requested traj indices not found: {missing}; available={all_indices}")
    return selected


def copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def render_grid(repo_root: Path, h5_path: Path, traj_index: int, output: Path, timing: str) -> None:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    env.setdefault("MOLMO_RENDER_SHADOWS", "0")
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "render_teleop_replay_camera_grid.py"),
        str(h5_path),
        "--traj-index",
        str(traj_index),
        "--timing",
        timing,
        "--output",
        str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def write_readme(package_dir: Path, manifest: dict[str, Any]) -> None:
    readme = package_dir / "README_REPLAY.md"
    lines = [
        "# MolmoSpaces PICO Teleop Package",
        "",
        "This package contains saved PICO teleop trajectories, metadata, and optional four-camera replay videos.",
        "",
        "Important: the H5 stores the frozen task config and MuJoCo qpos/qvel replay state, but the receiver still needs a compatible MolmoSpaces checkout and matching asset cache or download access.",
        "",
        "Replay one trajectory:",
        "",
        "```bash",
        "conda activate mlspaces",
        "cd /mnt/f/molospace/molmospaces-teleop/molmospaces-teleop",
        "MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA MUJOCO_GL=egl MOLMO_RENDER_SHADOWS=0 \\",
        "python scripts/render_teleop_replay_camera_grid.py /path/to/package/raw/house_N/trajectories_batch_1_of_1.h5 \\",
        "  --traj-index 0 --timing wall --output /path/to/replay.mp4",
        "```",
        "",
        "Files:",
        f"- manifest: {manifest['manifest_file']}",
        f"- trajectories summarized: {manifest['trajectory_count']}",
        f"- package created at: {manifest['created_time']}",
        "",
    ]
    readme.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    h5_paths = [path.resolve() for path in resolve_h5_paths(args.input, args.latest_save_event)]

    first_run_root = run_root_from_h5(h5_paths[0])
    package_name = args.name or f"teleop_package_{time.strftime('%Y%m%d_%H%M%S')}_{first_run_root.name}"
    package_dir = (args.output_dir / package_name).resolve()
    package_dir.mkdir(parents=True, exist_ok=True)

    raw_root = package_dir / "raw"
    replays_root = package_dir / "replays"
    summaries: list[dict[str, Any]] = []
    copied_files: list[str] = []
    rendered_files: list[str] = []

    for src_h5 in h5_paths:
        house_index = house_index_from_path(src_h5)
        house_dir = raw_root / f"house_{house_index}"
        dst_h5 = copy_if_exists(src_h5, house_dir / src_h5.name)
        if dst_h5 is None:
            raise FileNotFoundError(src_h5)
        copied_files.append(str(dst_h5.relative_to(package_dir)))

        for mp4 in sorted(src_h5.parent.glob("episode_*.mp4")):
            copied = copy_if_exists(mp4, house_dir / mp4.name)
            if copied is not None:
                copied_files.append(str(copied.relative_to(package_dir)))

        with h5py.File(src_h5, "r") as h5:
            all_indices = trajectory_indices(h5)
        render_indices = selected_traj_indices(all_indices, args.traj_index)

        for traj_index in all_indices:
            summary = summarize_trajectory(src_h5, traj_index)
            summary["package_h5"] = str(dst_h5.relative_to(package_dir))
            if args.render_grid and traj_index in render_indices:
                replay_name = f"house_{house_index}_traj_{traj_index:04d}_four_camera.mp4"
                replay_output = replays_root / replay_name
                render_grid(repo_root, dst_h5, traj_index, replay_output, args.timing)
                summary["four_camera_replay"] = str(replay_output.relative_to(package_dir))
                rendered_files.append(str(replay_output.relative_to(package_dir)))
            summaries.append(summary)

    run_root = first_run_root
    for cfg in sorted(run_root.glob("experiment_config_*.pkl")):
        copied = copy_if_exists(cfg, package_dir / "configs" / cfg.name)
        if copied is not None:
            copied_files.append(str(copied.relative_to(package_dir)))

    manifest = {
        "schema": "molmospaces_pico_teleop_package_v1",
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_h5_paths": [str(path) for path in h5_paths],
        "package_dir": str(package_dir),
        "trajectory_count": len(summaries),
        "copied_files": copied_files,
        "rendered_files": rendered_files,
        "trajectories": summaries,
    }
    manifest_path = package_dir / "manifest.json"
    manifest["manifest_file"] = str(manifest_path.relative_to(package_dir))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    jsonl_path = package_dir / "trajectories_manifest.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for summary in summaries:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    write_readme(package_dir, manifest)

    tar_path = None
    if args.tar:
        archive = shutil.make_archive(
            str(package_dir),
            "gztar",
            root_dir=package_dir.parent,
            base_dir=package_dir.name,
        )
        tar_path = Path(archive)

    print(f"[TeleopPackage][SAVED] {package_dir}")
    print(f"[TeleopPackage] manifest={manifest_path}")
    print(f"[TeleopPackage] trajectories={len(summaries)}")
    if rendered_files:
        print(f"[TeleopPackage] replays={len(rendered_files)}")
    if tar_path is not None:
        print(f"[TeleopPackage] archive={tar_path}")


if __name__ == "__main__":
    main()
