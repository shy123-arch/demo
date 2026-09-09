#!/usr/bin/env python3
"""Exercise the bundled UDP reference and PICO-to-Dex1 control path."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import mujoco

from molmo_spaces.policy.motion_tracking import MotionTrackingRuntime, MujocoStateBridge
from molmo_spaces.robots.g1_dex1_constants import DEFAULT_BODY_QPOS, DEX1_FINGER_UPPER
from molmo_spaces.robots.robot_views.g1_dex1_view import G1Dex1RobotView

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = REPO_ROOT / "assets" / "robots" / "g1_dex1" / "model.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    view = G1Dex1RobotView(data)
    view.get_move_group("body").joint_pos = DEFAULT_BODY_QPOS
    for side in ("left_dex1", "right_dex1"):
        view.get_move_group(side).joint_pos = [DEX1_FINGER_UPPER, DEX1_FINGER_UPPER]
    mujoco.mj_forward(model, data)

    bridge = MujocoStateBridge(view)
    runtime = MotionTrackingRuntime(bridge, motion_source="vr", enable_transport=True)
    payload = {
        "stream_seq": 1,
        "frames": [
            {
                "root_pos": [0.0, 0.0, 0.78],
                "root_quat": [1.0, 0.0, 0.0, 0.0],
                "dof_pos": DEFAULT_BODY_QPOS.tolist(),
            }
        ],
        "controller_buttons": {
            "right_key_one": True,
            "left_trigger_value": 1.0,
            "left_grip_value": 0.0,
            "right_trigger_value": 0.0,
            "right_grip_value": 0.0,
        },
        "retarget_age_ms": 0.0,
    }
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.sendto(json.dumps(payload).encode("utf-8"), ("127.0.0.1", 28704))
    time.sleep(0.03)
    try:
        for _ in range(5):
            runtime.step()
        source = runtime.policy.source
        if not source._vr_active:
            raise AssertionError("VR UDP stream did not activate the reference source")
        if bridge.dex1_open_amount != {"left": 0.0, "right": 1.0}:
            raise AssertionError(f"Unexpected PICO-to-Dex1 mapping: {bridge.dex1_open_amount}")
    finally:
        sender.close()
        runtime.close()

    print("[ok] UDP reference decoded and PICO trigger closed the left Dex1 jaw")


if __name__ == "__main__":
    main()
