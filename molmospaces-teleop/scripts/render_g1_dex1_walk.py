#!/usr/bin/env python3
"""Run the bundled walk reference in closed-loop MuJoCo and record an MP4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from molmo_spaces.controllers.joint_pd_torque import JointPDTorqueController
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.policy.motion_tracking import MotionTrackingRuntime, MujocoStateBridge
from molmo_spaces.robots.g1_dex1 import G1Dex1Robot
from molmo_spaces.robots.g1_dex1_constants import (
    BODY_KD,
    BODY_KP,
    DEFAULT_BODY_QPOS,
    DEX1_FINGER_LOWER,
    DEX1_FINGER_UPPER,
)
from molmo_spaces.robots.robot_views.g1_dex1_view import G1Dex1RobotView

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=REPO_ROOT / "assets" / "robots" / "g1_dex1" / "model.xml",
    )
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--motion", default="walk")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera", default="tracking")
    parser.add_argument(
        "--camera-grid",
        action="store_true",
        help="Record tracking, head, left-wrist, and right-wrist views in a 2x2 grid.",
    )
    parser.add_argument(
        "--gripper-cycle",
        action="store_true",
        help="Cycle both Dex1-1 jaws open -> closed -> open during the video.",
    )
    parser.add_argument("--no-video", action="store_true", help="Run physics and metrics only.")
    parser.add_argument(
        "--scene-xml",
        type=Path,
        help="Optional MolmoSpaces scene; G1 is injected with its standard MjSpec path.",
    )
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    return parser.parse_args()


class _RobotAssetConfig:
    """Minimal config contract required by Robot.add_robot_to_scene."""

    name = "g1_dex1"

    def __init__(self, model_xml: Path) -> None:
        self._model_xml = model_xml.resolve()

    def get_robot_xml_path(self) -> Path:
        return self._model_xml


def build_model(args: argparse.Namespace) -> tuple[mujoco.MjModel, str]:
    if args.scene_xml is None:
        return mujoco.MjModel.from_xml_path(str(args.model_xml.resolve())), args.camera

    scene_xml = args.scene_xml.expanduser().absolute()
    # Installed Molmo scenes live in a symlink tree where ../../objects resolves
    # correctly. Keep an already-linked path unchanged so this smoke renderer
    # does not need to import the full dataset stack just to compile the XML.
    linked_objects_dir = (scene_xml.parent / ".." / ".." / "objects").resolve()
    if not (scene_xml.exists() and linked_objects_dir.exists()):
        # A raw cache scene has broken ../../objects references; translate it to
        # the per-install symlink tree, installing the indexed scene if needed.
        from molmo_spaces.molmo_spaces_constants import (
            ASSETS_DIR,
            DATA_TYPE_TO_SOURCE_TO_VERSION,
        )

        scene_sources = DATA_TYPE_TO_SOURCE_TO_VERSION["scenes"]
        scene_source = next(
            (part for part in reversed(scene_xml.parts) if part in scene_sources), None
        )
        split_and_index = scene_xml.stem.split("_")
        scene_index = int(split_and_index[-1]) if split_and_index[-1].isdigit() else None
        if scene_source is None or scene_index is None:
            raise FileNotFoundError(scene_xml)
        linked_scene = ASSETS_DIR / "scenes" / scene_source / scene_xml.name
        if not linked_scene.exists():
            from molmo_spaces.utils.lazy_loading_utils import install_scene_from_source_index

            install_scene_from_source_index(scene_source, scene_index)
        scene_xml = linked_scene.absolute()

    old_cwd = Path.cwd()
    try:
        # MuJoCo 3.5.0 resolves MjSpec asset paths at compile time.  The real
        # Molmo scene XML uses ../../objects/... paths, so compile beside it.
        os.chdir(scene_xml.parent)
        scene_spec = mujoco.MjSpec.from_file(scene_xml.name)
        yaw_quat = R.from_euler("z", args.start_yaw).as_quat(scalar_first=True).tolist()
        G1Dex1Robot.add_robot_to_scene(
            _RobotAssetConfig(args.model_xml),
            scene_spec,
            prefix="robot_0/",
            pos=[args.start_x, args.start_y],
            quat=yaw_quat,
        )
        model = scene_spec.compile()
    finally:
        os.chdir(old_cwd)
    camera = "robot_0/tracking" if args.camera == "tracking" else args.camera
    return model, camera


def gripper_command(time_s: float, enabled: bool) -> tuple[float, str]:
    if not enabled:
        return DEX1_FINGER_UPPER, "OPEN"
    if time_s < 2.0:
        return DEX1_FINGER_UPPER, "OPEN"
    if time_s < 2.5:
        return DEX1_FINGER_LOWER, "CLOSING"
    if time_s < 5.0:
        return DEX1_FINGER_LOWER, "CLOSED"
    if time_s < 5.5:
        return DEX1_FINGER_UPPER, "OPENING"
    return DEX1_FINGER_UPPER, "OPEN"


def label_frame(frame: np.ndarray, label: str, gripper_state: str) -> np.ndarray:
    labeled = Image.fromarray(frame)
    draw = ImageDraw.Draw(labeled)
    draw.rectangle((0, 0, labeled.width, 54), fill=(0, 0, 0))
    draw.text((14, 8), label, fill=(255, 255, 255))
    draw.text((14, 31), f"DEX1: {gripper_state}", fill=(255, 220, 80))
    return np.asarray(labeled)


def render_camera_grid(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    cameras: list[tuple[str, str]],
    gripper_state: str,
) -> np.ndarray:
    frames = []
    for label, camera in cameras:
        renderer.update_scene(data, camera=camera)
        frames.append(label_frame(renderer.render(), label, gripper_state))
    return np.vstack((np.hstack(frames[:2]), np.hstack(frames[2:])))


def main() -> None:
    args = parse_args()
    model, camera = build_model(args)
    if not np.isclose(model.opt.timestep, 0.002):
        raise ValueError(f"Expected a 500 Hz model, got dt={model.opt.timestep}")

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    namespace = "robot_0/" if args.scene_xml is not None else ""
    view = G1Dex1RobotView(data, namespace=namespace)
    body = view.get_move_group("body")
    body.joint_pos = DEFAULT_BODY_QPOS
    for side in ("left_dex1", "right_dex1"):
        view.get_move_group(side).joint_pos = [DEX1_FINGER_UPPER] * 2
    mujoco.mj_forward(model, data)

    body_pd = JointPDTorqueController(body, BODY_KP, BODY_KD)
    dex_controllers = {
        side: JointPosController(view.get_move_group(side))
        for side in ("left_dex1", "right_dex1")
    }
    for controller in dex_controllers.values():
        controller.set_target(np.full(2, DEX1_FINGER_UPPER))

    bridge = MujocoStateBridge(view)
    runtime = MotionTrackingRuntime(
        bridge,
        policy_path=args.policy_path,
        motion_source="udp",
        enable_transport=False,
    )
    if not runtime.policy.source.append_motion_from_tail(args.motion):
        raise ValueError(f"Could not queue motion {args.motion!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.camera_grid and (args.width % 2 or args.height % 2):
        raise ValueError("--camera-grid requires even --width and --height")
    render_width = args.width // 2 if args.camera_grid else args.width
    render_height = args.height // 2 if args.camera_grid else args.height
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, render_width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, render_height)
    writer = None
    renderer = None
    if not args.no_video:
        writer = imageio.get_writer(
            args.output,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
            pixelformat="yuv420p",
            output_params=[
                "-profile:v",
                "baseline",
                "-level:v",
                "3.1",
                "-tag:v",
                "avc1",
                "-movflags",
                "+faststart",
            ],
        )
        renderer = mujoco.Renderer(model, height=render_height, width=render_width)

    grid_cameras = [
        ("THIRD PERSON / MOLMO ENV", camera),
        ("HEAD CAMERA", f"{namespace}head_camera"),
        ("LEFT WRIST CAMERA", f"{namespace}left_wrist_camera"),
        ("RIGHT WRIST CAMERA", f"{namespace}right_wrist_camera"),
    ]
    if args.camera_grid:
        missing = [
            name
            for _, name in grid_cameras
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0
        ]
        if missing:
            raise ValueError(f"Missing cameras for grid: {missing}")

    policy_period_steps = round(0.02 / model.opt.timestep)
    total_steps = round(args.seconds / model.opt.timestep)
    next_frame_time = 0.0
    body_target = DEFAULT_BODY_QPOS.copy()
    pelvis_z = []
    tilt = []
    start_xy = np.asarray(view.base.joint_pos[:2]).copy()
    frames = 0
    gripper_state = "OPEN"

    try:
        for step in range(total_steps):
            if step % policy_period_steps == 0:
                body_target = runtime.step()
                body_pd.set_target(body_target)

            body.ctrl = body_pd.compute_ctrl_inputs()
            gripper_target, gripper_state = gripper_command(data.time, args.gripper_cycle)
            for side, controller in dex_controllers.items():
                controller.set_target(np.full(2, gripper_target))
                view.get_move_group(side).ctrl = controller.compute_ctrl_inputs()
            mujoco.mj_step(model, data)

            base_qpos = np.asarray(view.base.joint_pos)
            pelvis_z.append(float(base_qpos[2]))
            up = R.from_quat(base_qpos[3:7], scalar_first=True).apply([0.0, 0.0, 1.0])
            tilt.append(float(np.arccos(np.clip(up[2], -1.0, 1.0))))

            if renderer is not None and data.time + 1e-12 >= next_frame_time:
                if args.camera_grid:
                    frame = render_camera_grid(
                        renderer, data, grid_cameras, gripper_state
                    )
                else:
                    renderer.update_scene(data, camera=camera)
                    frame = renderer.render()
                writer.append_data(frame)
                frames += 1
                next_frame_time += 1.0 / args.fps
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        runtime.close()

    end_xy = np.asarray(view.base.joint_pos[:2]).copy()
    summary = {
        "video": str(args.output.resolve()),
        "motion": args.motion,
        "scene_xml": str(args.scene_xml.resolve()) if args.scene_xml else None,
        "seconds": args.seconds,
        "frames": frames,
        "camera_grid": [name for _, name in grid_cameras] if args.camera_grid else None,
        "gripper_cycle": args.gripper_cycle,
        "final_gripper_qpos_m": {
            side: view.get_move_group(side).joint_pos.tolist()
            for side in ("left_dex1", "right_dex1")
        },
        "start_xy_m": start_xy.tolist(),
        "end_xy_m": end_xy.tolist(),
        "xy_displacement_m": float(np.linalg.norm(end_xy - start_xy)),
        "pelvis_z_min_m": float(np.min(pelvis_z)),
        "pelvis_z_final_m": float(pelvis_z[-1]),
        "max_tilt_deg": float(np.degrees(np.max(tilt))),
        "fell": bool(np.min(pelvis_z) < 0.45 or np.max(tilt) > np.radians(60.0)),
        "finite_state": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
