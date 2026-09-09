#!/usr/bin/env python3
"""Export successful PICO teleop trajectories as self-contained folders.

Each exported folder contains a single-trajectory H5, decoded action data,
robot/MuJoCo state arrays, metadata, scene/config copies when available, and an
optional four-camera replay rendered at recorded wall-clock timing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_teleop_package import (  # noqa: E402
    collect_video_refs,
    decode_bytes,
    decode_frozen_config,
    decode_json_scalar,
    frozen_summary,
    house_index_from_path,
    resolve_h5_paths,
    run_root_from_h5,
    safe_json_value,
    summarize_trajectory,
    trajectory_indices,
)
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR  # noqa: E402


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
        default=Path("/mnt/f/molospace/run_outputs/success_trajectory_exports"),
    )
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--render-grid", action="store_true")
    parser.add_argument("--timing", choices=("wall", "index"), default="wall")
    parser.add_argument("--tar", action="store_true")
    parser.add_argument(
        "--include-any-success",
        action="store_true",
        help="Export trajectories where success is true at any frame. Default requires final success.",
    )
    return parser.parse_args()


def is_success(summary: dict[str, Any], include_any_success: bool) -> bool:
    success = summary.get("success", {})
    if include_any_success:
        return bool(success.get("any"))
    return bool(success.get("last") or success.get("any"))


def sanitize_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "trajectory"


def dataset_to_array(dataset: h5py.Dataset) -> np.ndarray:
    return np.asarray(dataset[()])


def collect_datasets(group: h5py.Group, prefix: str = "") -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    def visit(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            key = f"{prefix}{name}".replace("/", "__")
            arrays[key] = dataset_to_array(obj)

    group.visititems(visit)
    return arrays


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json_value(value), ensure_ascii=False, indent=2), encoding="utf-8")


def copy_file_replace(src: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.chmod(target.stat().st_mode | stat.S_IWRITE | stat.S_IWUSR)
        target.unlink()
    shutil.copy2(src, target)


def decode_fixed_json_rows(dataset: h5py.Dataset) -> list[str]:
    rows = []
    data = np.asarray(dataset[()])
    if data.ndim == 1:
        data = data.reshape(1, -1)
    for row in data:
        text = decode_bytes(row)
        rows.append(text if text else "{}")
    return rows


def write_actions(src_group: h5py.Group, dst_dir: Path) -> list[str]:
    written: list[str] = []
    actions_dir = dst_dir / "actions"
    actions_dir.mkdir(parents=True, exist_ok=True)
    if "actions" not in src_group:
        return written

    actions = src_group["actions"]
    raw_arrays = collect_datasets(actions)
    if raw_arrays:
        raw_path = actions_dir / "actions_raw.npz"
        np.savez_compressed(raw_path, **raw_arrays)
        written.append(str(raw_path.relative_to(dst_dir)))

    for name, dataset in actions.items():
        if not isinstance(dataset, h5py.Dataset):
            continue
        jsonl_path = actions_dir / f"{sanitize_name(name)}.jsonl"
        rows = decode_fixed_json_rows(dataset)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(row + "\n")
        written.append(str(jsonl_path.relative_to(dst_dir)))
    return written


def write_robot_states(src_group: h5py.Group, dst_dir: Path) -> list[str]:
    written: list[str] = []
    states_dir = dst_dir / "robot_states"
    states_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key in ("env_states", "obs/agent"):
        if key in src_group:
            arrays.update(collect_datasets(src_group[key], prefix=f"{key}/"))
    for key in ("wall_time_s", "success"):
        if key in src_group:
            arrays[key] = dataset_to_array(src_group[key])
    if arrays:
        states_path = states_dir / "robot_mujoco_states.npz"
        np.savez_compressed(states_path, **arrays)
        written.append(str(states_path.relative_to(dst_dir)))
    return written


def copy_single_trajectory_h5(src_h5: Path, src_traj_index: int, dst_h5: Path) -> None:
    house_index = house_index_from_path(src_h5)
    dst_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        src.copy(f"traj_{src_traj_index}", dst, name="traj_0")
        dst.attrs["source_h5"] = str(src_h5)
        dst.attrs["source_traj_index"] = src_traj_index
        dst.attrs["house_index"] = house_index


def copy_run_configs(src_h5: Path, dst_dir: Path) -> list[str]:
    written: list[str] = []
    run_root = run_root_from_h5(src_h5)
    configs_dir = dst_dir / "configs"
    for pattern in ("experiment_config_*.pkl", "running_log.log"):
        for src in sorted(run_root.glob(pattern)):
            target = configs_dir / src.name
            copy_file_replace(src, target)
            written.append(str(target.relative_to(dst_dir)))
    return written


def candidate_scene_paths(house_index: int, obs_scene: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    scene_fields = (
        "scene_xml_path",
        "scene_path",
        "mjcf_path",
        "xml_path",
    )
    for field in scene_fields:
        value = obs_scene.get(field)
        if value:
            candidates.append(Path(str(value)).expanduser())

    scene_root = ASSETS_DIR / "scenes"
    for split in ("procthor-10k-train", "procthor-10k-val", "procthor-10k-test"):
        for prefix in ("train", "val", "test"):
            candidates.append(scene_root / split / f"{prefix}_{house_index}_ceiling.xml")
            candidates.append(scene_root / split / f"{prefix}_{house_index}.xml")

    if scene_root.exists():
        candidates.extend(sorted(scene_root.glob(f"**/*_{house_index}_ceiling.xml"))[:20])
        candidates.extend(sorted(scene_root.glob(f"**/*_{house_index}.xml"))[:20])

    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def xml_asset_refs(xml_path: Path) -> list[str]:
    try:
        text = xml_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    refs = sorted(set(re.findall(r'\bfile="([^"]+)"', text)))
    return refs


def copy_scene_files(src_group: h5py.Group, house_index: int, dst_dir: Path) -> dict[str, Any]:
    obs_scene = decode_json_scalar(src_group["obs_scene"][()]) if "obs_scene" in src_group else {}
    scene_dir = dst_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    saved_episode, encoding = decode_frozen_config(obs_scene)
    scene_info: dict[str, Any] = {
        "house_index": house_index,
        "frozen_config_encoding": encoding,
        "copied_scene_xml": None,
        "candidate_scene_paths": [],
        "asset_refs_file": None,
        "note": (
            "Full replay still needs the same MolmoSpaces code and asset cache/download access; "
            "this folder records scene XML and dependency references when available."
        ),
    }
    if saved_episode is not None:
        scene_info["frozen_episode_type"] = type(saved_episode).__name__

    for path in candidate_scene_paths(house_index, obs_scene):
        scene_info["candidate_scene_paths"].append(str(path))
        if not path.exists():
            continue
        target = scene_dir / path.name
        copy_file_replace(path, target)
        scene_info["copied_scene_xml"] = str(target.relative_to(dst_dir))
        refs = xml_asset_refs(path)
        refs_path = scene_dir / "asset_dependency_paths.txt"
        refs_path.write_text("\n".join(refs) + ("\n" if refs else ""), encoding="utf-8")
        scene_info["asset_refs_file"] = str(refs_path.relative_to(dst_dir))
        break

    write_json(scene_dir / "scene_manifest.json", scene_info)
    return scene_info


def copy_source_videos(src_group: h5py.Group, src_h5: Path, dst_dir: Path) -> list[str]:
    written: list[str] = []
    videos_dir = dst_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    referenced = collect_video_refs(src_group, src_h5)
    seen: set[Path] = set()
    for ref in referenced:
        src = src_h5.parent / str(ref["filename"])
        if src.exists() and src not in seen:
            target = videos_dir / src.name
            copy_file_replace(src, target)
            written.append(str(target.relative_to(dst_dir)))
            seen.add(src)
    for src in sorted(src_h5.parent.glob("episode_*.mp4")):
        if src in seen:
            continue
        target = videos_dir / src.name
        copy_file_replace(src, target)
        written.append(str(target.relative_to(dst_dir)))
        seen.add(src)
    return written


def render_grid(single_h5: Path, output: Path, timing: str) -> None:
    env = os.environ.copy()
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA")
    env.setdefault("MOLMO_RENDER_SHADOWS", "0")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_teleop_replay_camera_grid.py"),
        str(single_h5),
        "--traj-index",
        "0",
        "--timing",
        timing,
        "--trim-trailing-idle-s",
        "3",
        "--output",
        str(output),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def write_readme(dst_dir: Path, metadata: dict[str, Any]) -> None:
    lines = [
        f"# {metadata['trajectory_id']}",
        "",
        f"- task: {metadata.get('task_description')}",
        f"- house: {metadata.get('house')}",
        f"- success: {metadata.get('success_summary', {}).get('any')}",
        f"- source_h5: {metadata.get('source_h5')}",
        "",
        "Replay four-camera video from this folder:",
        "",
        "```bash",
        "conda activate mlspaces",
        "cd /mnt/f/molospace/molmospaces-teleop/molmospaces-teleop",
        "MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA MUJOCO_GL=egl MOLMO_RENDER_SHADOWS=0 \\",
        f"python scripts/render_teleop_replay_camera_grid.py {metadata['files']['single_trajectory_h5']} \\",
        "  --traj-index 0 --timing wall --trim-trailing-idle-s 3 --output replay.mp4",
        "```",
        "",
        "The package includes the frozen episode config, MuJoCo qpos/qvel, decoded actions, and available scene/config files. A receiver still needs the matching MolmoSpaces checkout and asset cache/download access for exact visual replay.",
        "",
    ]
    (dst_dir / "README_REPLAY.md").write_text("\n".join(lines), encoding="utf-8")


def write_package_readme(package_root: Path, manifest: dict[str, Any]) -> None:
    def rel(path: Any) -> str:
        if not path:
            return ""
        try:
            return str(Path(str(path)).relative_to(package_root))
        except Exception:
            return str(path)

    lines: list[str] = [
        "# MolmoSpaces PICO 成功轨迹导出包",
        "",
        "这个目录只包含被代码判定为 `success=True` 的 PICO 遥操作采集轨迹。每条轨迹一个独立文件夹，"
        "可以单独发送、单独回放，也可以整体作为一批数据交付。",
        "",
        "## 批次概览",
        "",
        f"- 创建时间：{manifest.get('created_time')}",
        f"- 扫描轨迹数：{manifest.get('scanned_trajectory_count')}",
        f"- 成功轨迹数：{manifest.get('success_trajectory_count')}",
        f"- 导出目录：`{manifest.get('package_root')}`",
        "",
        "## 成功轨迹清单",
        "",
        "| 序号 | 目录 | House | 任务 | 目标物体 | 帧数 | 时长(s) | 采集FPS |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]

    for i, item in enumerate(manifest.get("exports", []), start=1):
        wall = item.get("wall_time") or {}
        success = item.get("success_summary") or {}
        traj_dir = item.get("trajectory_id", "")
        lines.append(
            "| "
            f"{i} | `{traj_dir}` | `{item.get('house')}` | "
            f"{item.get('task_description')} | `{item.get('object_name')}` | "
            f"{success.get('length', '')} | "
            f"{float(wall.get('duration_s', 0.0)):.2f} | "
            f"{float(wall.get('fps', 0.0)):.2f} |"
        )

    lines.extend(
        [
            "",
            "## 目录结构",
            "",
            "每条轨迹目录结构如下：",
            "",
            "```text",
            "house_<N>_traj_0000_<task>/",
            "  trajectory.h5                         # 单条轨迹 H5，内部统一命名为 traj_0",
            "  metadata.json                         # 任务、场景、初始位姿、success、数据字段汇总",
            "  obs_scene.json                        # H5 中 obs_scene 的展开 JSON",
            "  README_REPLAY.md                      # 单条轨迹回放命令",
            "  actions/",
            "    actions_raw.npz                     # 原始动作数据数组",
            "    commanded_action.jsonl              # 每帧 commanded action，JSONL",
            "    ee_pose.jsonl                       # 每帧末端/身体 pose，JSONL",
            "    ee_twist.jsonl                      # 每帧 twist，JSONL",
            "    joint_pos.jsonl                     # 每帧关节位置，JSONL",
            "    joint_pos_rel.jsonl                 # 每帧相对关节命令，JSONL",
            "  robot_states/",
            "    robot_mujoco_states.npz             # MuJoCo qpos/qvel/time、机器人观测等",
            "  scene/",
            "    train_<N>_ceiling.xml               # 当前 house 的 MuJoCo 场景 XML",
            "    asset_dependency_paths.txt          # XML 里引用到的 mesh/texture 等资源路径",
            "    scene_manifest.json                 # 场景复制情况和依赖说明",
            "  configs/",
            "    experiment_config_*.pkl             # 本批次 datagen 配置",
            "    running_log.log                     # run 内部日志",
            "  videos/",
            "    episode_*_head_camera_*.mp4         # 采集时 PICO 看到的 head_camera 视频",
            "    four_camera_replay.mp4              # 离线四宫格回放视频",
            "```",
            "",
            "## H5 数据结构",
            "",
            "`trajectory.h5` 中只有一条轨迹，固定在 `traj_0` 下。关键字段：",
            "",
            "- `traj_0/obs_scene`：任务和 frozen config。frozen config 里保存了任务配置、机器人配置、目标物体、初始位姿等。",
            "- `traj_0/success`：每帧 success bool。导出脚本只导出 `success=True` 的轨迹。",
            "- `traj_0/wall_time_s`：采集时真实墙钟时间，用于按真实速度导出视频。",
            "- `traj_0/actions/*`：动作与遥操作控制数据。H5 内是固定宽度 uint8 JSON 行；`actions/*.jsonl` 是已经解码后的逐帧版本。",
            "- `traj_0/env_states/mj_qpos`：每帧 MuJoCo `qpos`，用于还原机器人、物体、关节等仿真状态。",
            "- `traj_0/env_states/mj_qvel`：每帧 MuJoCo `qvel`。",
            "- `traj_0/env_states/mj_time`：每帧 MuJoCo 仿真时间。",
            "- `traj_0/env_states/articulations/*`：场景中可动 articulated object 的状态。",
            "- `traj_0/obs/agent/qpos` 和 `traj_0/obs/agent/qvel`：机器人控制器观测到的关节状态。",
            "- `traj_0/obs/extra/robot_base_pose`：每帧机器人底盘世界位姿。",
            "",
            "## 初始位姿与场景还原",
            "",
            "`metadata.json` 的 `frozen_config_summary` 里重点看：",
            "",
            "- `pickup_obj_name`：目标物体名字。",
            "- `pickup_obj_start_pose`：目标物体初始世界位姿，格式为 `x,y,z + quat(w,x,y,z)`。",
            "- `robot_base_pose`：机器人初始世界位姿，格式为 `x,y,z + quat(w,x,y,z)`。",
            "- `object_pose_count` / `object_pose_names_sample`：场景中记录到的物体位姿概览。",
            "",
            "完整视觉还原需要：同版本 MolmoSpaces 代码、同一套 assets/cache、以及本包中的 `trajectory.h5`。"
            "本包已经带了场景 XML 和资源引用清单，但没有把所有 mesh/texture 资源全量复制进去。",
            "",
            "## 回放命令",
            "",
            "回放任意单条轨迹的四宫格视频：",
            "",
            "```bash",
            "conda activate mlspaces",
            "cd /mnt/f/molospace/molmospaces-teleop/molmospaces-teleop",
            "",
            "MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \\",
            "MUJOCO_GL=egl \\",
            "MOLMO_RENDER_SHADOWS=0 \\",
            "python scripts/render_teleop_replay_camera_grid.py /path/to/trajectory.h5 \\",
            "  --traj-index 0 \\",
            "  --timing wall \\",
            "  --trim-trailing-idle-s 3 \\",
            "  --output replay_four_camera.mp4",
            "```",
            "",
            "`--timing wall` 会按采集时的真实速度导出，不会强行固定成 25fps。"
            "`--trim-trailing-idle-s 3` 会在末尾静止后只保留 3 秒，避免无效尾段。",
            "",
            "## 批次文件",
            "",
            "- `manifest.json`：完整机器可读清单。",
            "- `manifest.jsonl`：一行一条成功轨迹，方便后续脚本处理。",
            "",
        ]
    )

    for item in manifest.get("exports", []):
        files = item.get("files") or {}
        lines.extend(
            [
                f"### {item.get('trajectory_id')}",
                "",
                f"- 单条 H5：`{rel(files.get('single_trajectory_h5'))}`",
                f"- 四宫格视频：`{rel(files.get('four_camera_replay'))}`",
                f"- Head camera 视频：`{', '.join(f'`{x}`' for x in files.get('source_videos', [])) or '无'}",
                "",
            ]
        )

    (package_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def export_one(
    src_h5: Path,
    traj_index: int,
    summary: dict[str, Any],
    package_root: Path,
    render: bool,
    timing: str,
) -> dict[str, Any]:
    house_index = house_index_from_path(src_h5)
    task_slug = sanitize_name(str(summary.get("task_description") or "task"))
    trajectory_id = f"house_{house_index}_traj_{traj_index:04d}_{task_slug}"
    dst_dir = package_root / trajectory_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    single_h5 = dst_dir / "trajectory.h5"
    copy_single_trajectory_h5(src_h5, traj_index, single_h5)

    with h5py.File(src_h5, "r") as h5:
        src_group = h5[f"traj_{traj_index}"]
        obs_scene = decode_json_scalar(src_group["obs_scene"][()]) if "obs_scene" in src_group else {}
        files = {
            "single_trajectory_h5": str(single_h5),
            "actions": write_actions(src_group, dst_dir),
            "robot_states": write_robot_states(src_group, dst_dir),
            "source_videos": copy_source_videos(src_group, src_h5, dst_dir),
            "configs": copy_run_configs(src_h5, dst_dir),
        }
        scene_info = copy_scene_files(src_group, house_index, dst_dir)

    replay_path = None
    if render:
        replay_path = dst_dir / "videos" / "four_camera_replay.mp4"
        render_grid(single_h5, replay_path, timing)
        files["four_camera_replay"] = str(replay_path)

    metadata = {
        "schema": "molmospaces_success_teleop_trajectory_v1",
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trajectory_id": trajectory_id,
        "source_h5": str(src_h5),
        "source_traj_index": traj_index,
        "house": f"house_{house_index}",
        "task_description": summary.get("task_description"),
        "task_type": summary.get("task_type"),
        "object_name": obs_scene.get("object_name"),
        "success_summary": summary.get("success"),
        "wall_time": summary.get("wall_time"),
        "frozen_config_summary": frozen_summary(obs_scene),
        "scene": scene_info,
        "replay_complete_check": summary.get("replay_complete_check"),
        "datasets": summary.get("datasets"),
        "video_refs": summary.get("video_refs"),
        "files": files,
    }
    write_json(dst_dir / "metadata.json", metadata)
    write_json(dst_dir / "obs_scene.json", obs_scene)
    write_readme(dst_dir, metadata)
    return safe_json_value(metadata)


def main() -> None:
    args = parse_args()
    h5_paths = [path.resolve() for path in resolve_h5_paths(args.input, args.latest_save_event)]
    run_name = run_root_from_h5(h5_paths[0]).name
    package_name = args.name or f"success_teleop_{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    package_root = (args.output_dir / package_name).resolve()
    package_root.mkdir(parents=True, exist_ok=True)

    exports: list[dict[str, Any]] = []
    scanned: list[dict[str, Any]] = []
    for src_h5 in h5_paths:
        with h5py.File(src_h5, "r") as h5:
            indices = trajectory_indices(h5)
        for traj_index in indices:
            summary = summarize_trajectory(src_h5, traj_index)
            scanned.append(
                {
                    "h5": str(src_h5),
                    "traj_index": traj_index,
                    "house": f"house_{house_index_from_path(src_h5)}",
                    "task_description": summary.get("task_description"),
                    "success": summary.get("success"),
                }
            )
            if not is_success(summary, args.include_any_success):
                continue
            exports.append(
                export_one(
                    src_h5=src_h5,
                    traj_index=traj_index,
                    summary=summary,
                    package_root=package_root,
                    render=args.render_grid,
                    timing=args.timing,
                )
            )

    manifest = {
        "schema": "molmospaces_success_teleop_export_v1",
        "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_inputs": [str(path) for path in h5_paths],
        "package_root": str(package_root),
        "success_trajectory_count": len(exports),
        "scanned_trajectory_count": len(scanned),
        "scanned": scanned,
        "exports": exports,
    }
    write_json(package_root / "manifest.json", manifest)
    with (package_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for item in exports:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_package_readme(package_root, manifest)

    archive = None
    if args.tar:
        archive = shutil.make_archive(
            str(package_root),
            "gztar",
            root_dir=package_root.parent,
            base_dir=package_root.name,
        )

    print(f"[SuccessTeleopExport] scanned={len(scanned)} success={len(exports)}")
    print(f"[SuccessTeleopExport] output={package_root}")
    if archive:
        print(f"[SuccessTeleopExport] archive={archive}")
    if not exports:
        print("[SuccessTeleopExport][WARN] no success=True trajectories found")


if __name__ == "__main__":
    main()
