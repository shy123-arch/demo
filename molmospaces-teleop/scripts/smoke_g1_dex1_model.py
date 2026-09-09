#!/usr/bin/env python3
"""Validate the bundled G1 + Dex1-1 MuJoCo joint contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from molmo_spaces.controllers.joint_pd_torque import JointPDTorqueController
from molmo_spaces.policy.motion_tracking.state_bridge import MujocoStateBridge
from molmo_spaces.robots.g1_dex1_constants import (
    BODY_JOINT_NAMES,
    BODY_KD,
    BODY_KP,
    DEX1_FINGER_LOWER,
    dex1_joint_names,
)
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
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model_xml.resolve()))
    expected = BODY_JOINT_NAMES + dex1_joint_names("left") + dex1_joint_names("right")
    missing_joints = [
        name
        for name in expected
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0
    ]
    missing_actuators = []
    for name in expected:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if not any(int(model.actuator_trnid[aid, 0]) == joint_id for aid in range(model.nu)):
            missing_actuators.append(name)

    old_dex3 = [
        name
        for side in ("left", "right")
        for name in (f"{side}_hand_thumb_0_joint", f"{side}_hand_index_0_joint")
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
    ]
    old_dex3_geoms = [
        name
        for side in ("left", "right")
        for name in (f"{side}_hand_palm_collision", f"{side}_hand_collision")
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0
    ]
    expected_cameras = ["head_camera", "left_wrist_camera", "right_wrist_camera"]
    missing_robot_cameras = [
        name
        for name in expected_cameras
        if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, name
        )
        < 0
    ]
    if missing_joints or missing_actuators or old_dex3 or old_dex3_geoms or missing_robot_cameras:
        raise AssertionError(
            f"contract failed: missing_joints={missing_joints}, "
            f"missing_actuators={missing_actuators}, old_dex3={old_dex3}, "
            f"old_dex3_geoms={old_dex3_geoms}, missing_robot_cameras={missing_robot_cameras}"
        )
    if (model.nq, model.nv, model.nu) != (40, 39, 33):
        raise AssertionError(f"Unexpected dimensions: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    if abs(model.opt.timestep - 0.002) > 1e-12:
        raise AssertionError(f"Expected 500 Hz timestep, got {model.opt.timestep}")
    for side in ("left", "right"):
        for name in dex1_joint_names(side):
            actuator_id = model.actuator(name).id
            if abs(model.actuator_ctrlrange[actuator_id, 0] - DEX1_FINGER_LOWER) > 1e-12:
                raise AssertionError(f"{name} does not use the close-gap correction")
            if abs(model.actuator_gainprm[actuator_id, 0] - 1000.0) > 1e-12:
                raise AssertionError(f"{name} does not use the small-object grasp stiffness")

    data = mujoco.MjData(model)
    view = G1Dex1RobotView(data)
    body = view.get_move_group("body")
    pd = JointPDTorqueController(body, BODY_KP, BODY_KD)
    if view.move_group_ids() != ["base", "body", "left_dex1", "right_dex1"]:
        raise AssertionError(f"Unexpected move groups: {view.move_group_ids()}")
    if pd.compute_ctrl_inputs().shape != (29,):
        raise AssertionError("29D body PD controller did not produce 29 torques")
    bridge = MujocoStateBridge(view)
    if bridge.qj_real.shape != (29,) or bridge.qj_isaac.shape != (29,):
        raise AssertionError("State bridge did not expose both 29D joint orders")
    if bridge.current_reference_anchor()["root_quat"].shape != (4,):
        raise AssertionError("State bridge root quaternion is invalid")
    bridge.set_hand_from_controller_buttons(
        {"left_trigger_value": 1.0, "right_trigger_value": 0.0}
    )
    if bridge.dex1_open_amount != {"left": 0.0, "right": 1.0}:
        raise AssertionError(f"Unexpected PICO-to-Dex1 mapping: {bridge.dex1_open_amount}")

    print(
        "[ok] G1+Dex1 contract: free base + 29 body joints + "
        "4 Dex1-1 finger joints, 33 actuators"
    )


if __name__ == "__main__":
    main()
