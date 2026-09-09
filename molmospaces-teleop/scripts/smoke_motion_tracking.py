#!/usr/bin/env python3
"""One-step bundled ONNX smoke test on the MolmoSpaces G1 state bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from molmo_spaces.policy.motion_tracking import MotionTrackingRuntime, MujocoStateBridge
from molmo_spaces.robots.g1_dex1_constants import DEFAULT_BODY_QPOS, DEX1_FINGER_UPPER
from molmo_spaces.robots.robot_views.g1_dex1_view import G1Dex1RobotView

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_xml",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "assets" / "robots" / "g1_dex1" / "model.xml",
    )
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--motion-source", choices=("udp", "vr"), default="udp")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model_xml.resolve()))
    data = mujoco.MjData(model)
    view = G1Dex1RobotView(data)
    view.get_move_group("body").joint_pos = DEFAULT_BODY_QPOS
    for side in ("left_dex1", "right_dex1"):
        view.get_move_group(side).joint_pos = [DEX1_FINGER_UPPER, DEX1_FINGER_UPPER]
    mujoco.mj_forward(model, data)

    bridge = MujocoStateBridge(view)
    runtime = MotionTrackingRuntime(
        bridge,
        policy_path=args.policy_path,
        motion_source=args.motion_source,
        enable_transport=args.motion_source == "vr",
    )
    target = runtime.step()
    if runtime.observation_dim not in {1590, 1623}:
        raise AssertionError(f"Unexpected observation dim: {runtime.observation_dim}")
    if target.shape != (29,):
        raise AssertionError(f"Unexpected target shape: {target.shape}")
    runtime.close()
    print(
        f"[ok] bundled ONNX step: obs={runtime.observation_dim}, "
        f"target_shape={target.shape}, finite={bool((target == target).all())}"
    )


if __name__ == "__main__":
    main()
