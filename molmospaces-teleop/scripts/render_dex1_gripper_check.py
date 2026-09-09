#!/usr/bin/env python3
"""Render Dex1-1 wrist-camera views while cycling the jaws open and closed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from molmo_spaces.robots.g1_dex1_constants import DEX1_FINGER_LOWER, DEX1_FINGER_UPPER

REPO_ROOT = Path(__file__).resolve().parents[1]


FINGER_JOINTS = {
    side: [f"{side}_dex1_finger_joint_1", f"{side}_dex1_finger_joint_2"]
    for side in ("left", "right")
}
CAMERAS = {side: f"{side}_wrist_camera" for side in ("left", "right")}


def _set_fingers(model: mujoco.MjModel, data: mujoco.MjData, position: float) -> None:
    for names in FINGER_JOINTS.values():
        for name in names:
            joint_id = model.joint(name).id
            data.qpos[model.jnt_qposadr[joint_id]] = position
    mujoco.mj_forward(model, data)


def _collision_geom(model: mujoco.MjModel, body_name: str) -> int:
    body_id = model.body(body_name).id
    first = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    candidates = [gid for gid in range(first, first + count) if int(model.geom_group[gid]) == 3]
    if len(candidates) != 1:
        raise ValueError(f"Expected one collision geom on {body_name}, found {candidates}")
    return candidates[0]


def _collision_gap(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> float:
    geom_1 = _collision_geom(model, f"{side}_dex1_finger_link_1")
    geom_2 = _collision_geom(model, f"{side}_dex1_finger_link_2")
    return float(mujoco.mj_geomDistance(model, data, geom_1, geom_2, 1.0, None))


def _render_pair(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    label: str,
    position: float,
    gaps: dict[str, float],
) -> np.ndarray:
    views = []
    for side in ("left", "right"):
        renderer.update_scene(data, camera=CAMERAS[side])
        rgb = renderer.render().copy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 70), (0, 0, 0), -1)
        cv2.putText(
            bgr,
            f"{side.upper()} WRIST | {label}",
            (18, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            f"joint={position:+.4f} m  collision gap={gaps[side] * 1000:.2f} mm",
            (18, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (80, 220, 255),
            1,
            cv2.LINE_AA,
        )
        views.append(bgr)
    return np.hstack(views)


def _position_at_time(t: float) -> tuple[float, str]:
    if t < 1.0:
        return DEX1_FINGER_UPPER, "OPEN"
    if t < 2.5:
        alpha = (t - 1.0) / 1.5
        return (1.0 - alpha) * DEX1_FINGER_UPPER + alpha * DEX1_FINGER_LOWER, "CLOSING"
    if t < 4.0:
        return DEX1_FINGER_LOWER, "FULLY CLOSED"
    if t < 5.5:
        alpha = (t - 4.0) / 1.5
        return (1.0 - alpha) * DEX1_FINGER_LOWER + alpha * DEX1_FINGER_UPPER, "OPENING"
    return DEX1_FINGER_UPPER, "OPEN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=REPO_ROOT / "assets" / "robots" / "g1_dex1" / "model.xml",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dex1_wrist_open_close.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model_xml.resolve()))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (1280, 480),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")

    snapshots: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    try:
        for frame_index in range(round(args.duration * args.fps)):
            t = frame_index / args.fps
            position, label = _position_at_time(t)
            _set_fingers(model, data, position)
            gaps = {side: _collision_gap(model, data, side) for side in ("left", "right")}
            frame = _render_pair(renderer, data, label, position, gaps)
            writer.write(frame)
            if label in ("OPEN", "FULLY CLOSED") and label not in snapshots:
                snapshots[label] = frame.copy()
                metrics[label.lower().replace(" ", "_")] = {
                    "joint_position_m": float(position),
                    "left_collision_gap_m": gaps["left"],
                    "right_collision_gap_m": gaps["right"],
                }
    finally:
        writer.release()
        renderer.close()

    comparison_path = args.output.with_name(f"{args.output.stem}_comparison.jpg")
    cv2.imwrite(str(comparison_path), np.vstack([snapshots["OPEN"], snapshots["FULLY CLOSED"]]))
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"[ok] video: {args.output.resolve()}")
    print(f"[ok] comparison: {comparison_path.resolve()}")
    print(f"[ok] metrics: {metrics_path.resolve()} | {metrics}")


if __name__ == "__main__":
    main()
