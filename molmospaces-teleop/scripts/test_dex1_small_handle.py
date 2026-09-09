#!/usr/bin/env python3
"""Check whether the left Dex1-1 can hold a thin spoon-handle proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.robots.g1_dex1_constants import DEX1_FINGER_LOWER, DEX1_FINGER_UPPER

REPO_ROOT = Path(__file__).resolve().parents[1]


def _build_test_model(
    xml_path: Path,
    handle_radius: float,
    handle_mass: float,
    finger_friction: float | None,
) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1="pelvis",
    )
    handle = spec.worldbody.add_body(name="spoon_handle")
    handle.add_freejoint(name="spoon_handle_free")
    handle.add_geom(
        name="spoon_handle_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[handle_radius, 0.04, 0.0],
        mass=handle_mass,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.2, 0.4, 0.9, 1.0],
    )
    model = spec.compile()
    model.opt.gravity[:] = 0.0
    if finger_friction is not None:
        for side in ("left", "right"):
            for finger in (1, 2):
                body_id = model.body(f"{side}_dex1_finger_link_{finger}").id
                first = int(model.body_geomadr[body_id])
                count = int(model.body_geomnum[body_id])
                for geom_id in range(first, first + count):
                    if int(model.geom_group[geom_id]) == 3:
                        model.geom_friction[geom_id, 0] = finger_friction
    return model


def _run_load(model: mujoco.MjModel, load: float, grasp_x: float) -> dict:
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        for finger in (1, 2):
            name = f"{side}_dex1_finger_joint_{finger}"
            joint_id = model.joint(name).id
            data.qpos[model.jnt_qposadr[joint_id]] = DEX1_FINGER_UPPER
            data.ctrl[model.actuator(name).id] = DEX1_FINGER_LOWER

    mujoco.mj_forward(model, data)
    gripper_id = model.body("left_dex1_base_link").id
    rotation = data.xmat[gripper_id].reshape(3, 3).copy()
    base_position = data.xpos[gripper_id].copy()
    handle_joint_id = model.joint("spoon_handle_free").id
    qpos_address = model.jnt_qposadr[handle_joint_id]
    data.qpos[qpos_address : qpos_address + 3] = base_position + rotation @ np.array(
        [grasp_x, 0.0, 0.0]
    )
    data.qpos[qpos_address + 3 : qpos_address + 7] = data.xquat[gripper_id]
    mujoco.mj_forward(model, data)

    for _ in range(round(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    handle_id = model.body("spoon_handle").id
    rotation = data.xmat[gripper_id].reshape(3, 3).copy()
    relative_before = rotation.T @ (data.xpos[handle_id] - data.xpos[gripper_id])
    normal_forces = []
    contact_pairs = []
    for contact_index, contact in enumerate(data.contact):
        geom_names = {model.geom(contact.geom1).name, model.geom(contact.geom2).name}
        if "spoon_handle_geom" in geom_names:
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, contact_index, force)
            normal_forces.append(float(force[0]))
            contact_pairs.append(sorted(geom_names))

    # Pull along the handle axis for one second. This is equivalent to gravity
    # on a load of (load / 9.81) kilograms but avoids disturbing the robot.
    data.xfrc_applied[handle_id, :3] = -load * rotation[:, 2]
    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    rotation_after = data.xmat[gripper_id].reshape(3, 3)
    relative_after = rotation_after.T @ (data.xpos[handle_id] - data.xpos[gripper_id])
    axial_slip_mm = abs(float((relative_after - relative_before)[2])) * 1000.0
    final_contacts = sum(
        "spoon_handle_geom" in {model.geom(c.geom1).name, model.geom(c.geom2).name}
        for c in data.contact
    )
    return {
        "axial_load_N": load,
        "equivalent_mass_g": load / 9.81 * 1000.0,
        "normal_forces_N": normal_forces,
        "contact_pairs": contact_pairs,
        "axial_slip_mm": axial_slip_mm,
        "final_contacts": final_contacts,
        "held": axial_slip_mm < 10.0 and final_contacts >= 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=REPO_ROOT / "assets" / "robots" / "g1_dex1" / "model.xml",
    )
    parser.add_argument("--diameter-mm", type=float, default=4.0)
    parser.add_argument("--mass-g", type=float, default=30.0)
    parser.add_argument("--grasp-x", type=float, default=0.14)
    parser.add_argument("--finger-friction", type=float)
    parser.add_argument("--output", type=Path, default=Path("outputs/dex1_small_handle_test.json"))
    args = parser.parse_args()

    model = _build_test_model(
        args.model_xml.resolve(),
        handle_radius=args.diameter_mm / 2000.0,
        handle_mass=args.mass_g / 1000.0,
        finger_friction=args.finger_friction,
    )
    load = args.mass_g / 1000.0 * 9.81
    result = {
        "handle_diameter_mm": args.diameter_mm,
        "handle_mass_g": args.mass_g,
        "grasp_x_m": args.grasp_x,
        "result": _run_load(model, load, args.grasp_x),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"[ok] {args.output.resolve()}")


if __name__ == "__main__":
    main()
