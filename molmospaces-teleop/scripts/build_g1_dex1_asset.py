#!/usr/bin/env python3
"""Build a free-base G1 asset with Unitree Dex1-1 grippers.

The source body is HoloTeleop's 29-DoF G1.  This builder removes both Dex3
hands and grafts the official Dex1-1 model from HIW-500-controoler.  Output is
a repository-local robot asset at ``assets/robots/g1_dex1``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mujoco
import numpy as np


# The official URDF lower limit is -0.020 m, but its two collision pads still
# have a 5.88 mm gap there.  Extend each simulated finger by 3 mm so an empty
# close reaches contact.  This is a simulation closure trim, not a hardware
# limit calibration.
FINGER_URDF_LOWER = -0.02
FINGER_LOWER = -0.023
FINGER_UPPER = 0.0245
FINGER_KP = 1000.0
FINGER_KD = 25.0
BASE_POS = (0.0415, 0.0, 0.0)
DEX3_CHILDREN = (
    "hand_mimic",
    "hand_thumb_0_link",
    "hand_middle_0_link",
    "hand_index_0_link",
)
DEX1_MESHES = (
    "Dex1_base_link.STL",
    "Dex1_finger_link_1.STL",
    "Dex1_finger_link_2.STL",
    "dex1_col_1.stl",
    "dex1_col_2.stl",
)
INERTIA = {
    "base": dict(
        mass=0.19138,
        ipos=(0.043853, -0.0001306, 1.4003e-05),
        diag=(7.4842e-05, 5.861e-05, 3.9364e-05),
    ),
    1: dict(
        mass=0.086783,
        ipos=(0.073453, -0.011173, 0.0086355),
        diag=(2.5583e-05, 1.4549e-05, 3.6013e-05),
    ),
    2: dict(
        mass=0.086783,
        ipos=(0.073453, 0.011173, -0.0086355),
        diag=(2.5583e-05, 1.4549e-05, 3.6013e-05),
    ),
}

# Approximate Dex1-1 wrist-camera mount copied from HIW-500-controoler.  The
# camera sits above the gripper base and looks forward/down into the jaw tips.
WRIST_CAMERA_POS = (0.05, 0.0, 0.075)
WRIST_CAMERA_FORWARD = (0.75, 0.0, -0.66)
WRIST_CAMERA_UP = (0.0, 0.0, 1.0)
WRIST_CAMERA_FOVY = 95.0
HEAD_CAMERA_POS = (0.06, 0.0, 0.45)
HEAD_CAMERA_EULER = (0.0, -0.8, -1.57)
HEAD_CAMERA_FOVY = 90.0


def _set_inertial(body, values: dict) -> None:
    body.explicitinertial = True
    body.mass = float(values["mass"])
    body.ipos = np.asarray(values["ipos"], dtype=float)
    body.iquat = np.asarray([1.0, 0.0, 0.0, 0.0])
    body.inertia = np.asarray(values["diag"], dtype=float)


def _remove_dex3(spec: mujoco.MjSpec, side: str) -> None:
    hand_prefix = f"{side}_hand_"
    for actuator in list(spec.actuators):
        target = str(getattr(actuator, "target", ""))
        if actuator.name.startswith(hand_prefix) or target.startswith(hand_prefix):
            spec.delete(actuator)

    wrist = spec.body(f"{side}_wrist_yaw_link")
    if wrist is None:
        raise ValueError(f"Missing {side} wrist in source G1 model")
    for geom in list(wrist.geoms):
        mesh_name = str(getattr(geom, "meshname", ""))
        old_collision = geom.name in {
            f"{side}_hand_palm_collision",
            f"{side}_hand_collision",
        }
        if (
            mesh_name.endswith("hand_palm_link")
            or mesh_name.endswith("rubber_hand")
            or old_collision
        ):
            spec.delete(geom)
    for suffix in DEX3_CHILDREN:
        body = spec.body(f"{side}_{suffix}")
        if body is not None:
            spec.delete(body)


def _register_dex1_meshes(spec: mujoco.MjSpec) -> None:
    existing = {mesh.name for mesh in spec.meshes}
    for filename in DEX1_MESHES:
        name = filename.rsplit(".", 1)[0]
        if name in existing:
            continue
        mesh = spec.add_mesh()
        mesh.name = name
        mesh.file = filename


def _graft_dex1(spec: mujoco.MjSpec, side: str) -> None:
    wrist = spec.body(f"{side}_wrist_yaw_link")
    base = wrist.add_body(name=f"{side}_dex1_base_link", pos=list(BASE_POS))
    _set_inertial(base, INERTIA["base"])
    base.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="Dex1_base_link",
        contype=0,
        conaffinity=0,
        group=2,
        rgba=[0.792, 0.820, 0.933, 1.0],
    )

    for finger_index, axis_sign in ((1, -1), (2, 1)):
        joint_name = f"{side}_dex1_finger_joint_{finger_index}"
        finger = base.add_body(name=f"{side}_dex1_finger_link_{finger_index}")
        _set_inertial(finger, INERTIA[finger_index])
        finger.add_joint(
            name=joint_name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=[0, axis_sign, 0],
            range=[FINGER_LOWER, FINGER_UPPER],
        )
        finger.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"Dex1_finger_link_{finger_index}",
            contype=0,
            conaffinity=0,
            group=2,
            rgba=[0.792, 0.820, 0.933, 1.0],
        )
        finger.add_geom(
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=f"dex1_col_{finger_index}",
            group=3,
            rgba=[0.3, 0.3, 0.3, 1.0],
        )

        actuator = spec.add_actuator(name=joint_name)
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.target = joint_name
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        # A 4 mm handle only stops each finger about 2 mm before the close
        # target. kp=100 generated ~0.16 N normal force per side and let a
        # 20--30 g spoon-equivalent load slip.  kp=500 improved contact force
        # but still let the 30 g proxy slide out. The higher stiffness provides
        # useful small-object preload while the 20 N effort cap remains the
        # final safety limit.  Damping is scaled with sqrt(kp) for a crisp,
        # non-oscillatory close at the 2 ms physics step.
        actuator.gainprm[0] = FINGER_KP
        actuator.biasprm[1] = -FINGER_KP
        actuator.biasprm[2] = -FINGER_KD
        actuator.ctrlrange = [FINGER_LOWER, FINGER_UPPER]
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.forcerange = [-20.0, 20.0]
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE


def _camera_quat(forward: tuple[float, float, float], up: tuple[float, float, float]) -> np.ndarray:
    """Return a MuJoCo camera quaternion from body-frame forward/up vectors."""
    camera_z = -np.asarray(forward, dtype=float)
    camera_z /= np.linalg.norm(camera_z)
    camera_x = np.cross(np.asarray(up, dtype=float), camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    rotation = np.column_stack((camera_x, camera_y, camera_z)).reshape(9)
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, rotation)
    return quat


def _euler_quat(euler: tuple[float, float, float]) -> np.ndarray:
    """Match MuJoCo's default intrinsic xyz Euler convention."""
    a, b, c = euler
    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)
    rotation_x = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    rotation_y = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rotation_z = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    quat = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quat, (rotation_x @ rotation_y @ rotation_z).reshape(9))
    return quat


def _add_wrist_cameras(spec: mujoco.MjSpec) -> None:
    quat = _camera_quat(WRIST_CAMERA_FORWARD, WRIST_CAMERA_UP)
    for side in ("left", "right"):
        base = spec.body(f"{side}_dex1_base_link")
        if base is None:
            raise ValueError(f"Missing {side} Dex1-1 base for wrist camera")
        base.add_camera(
            name=f"{side}_wrist_camera",
            pos=list(WRIST_CAMERA_POS),
            quat=list(quat),
            fovy=WRIST_CAMERA_FOVY,
        )


def _add_head_camera(spec: mujoco.MjSpec) -> None:
    torso = spec.body("torso_link")
    if torso is None:
        raise ValueError("Missing torso_link for head camera")
    torso.add_camera(
        name="head_camera",
        pos=list(HEAD_CAMERA_POS),
        quat=list(_euler_quat(HEAD_CAMERA_EULER)),
        fovy=HEAD_CAMERA_FOVY,
    )


def build_asset(g1_dir: Path, dex1_dir: Path, output_dir: Path) -> Path:
    source_xml = g1_dir / "g1.xml"
    source_meshes = g1_dir / "meshes"
    dex1_meshes = dex1_dir / "meshes"
    if not source_xml.is_file() or not source_meshes.is_dir():
        raise FileNotFoundError(f"Invalid source G1 asset directory: {g1_dir}")
    missing = [name for name in DEX1_MESHES if not (dex1_meshes / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Dex1-1 meshes in {dex1_meshes}: {missing}")

    output_meshes = output_dir / "meshes"
    output_meshes.mkdir(parents=True, exist_ok=True)
    for path in source_meshes.iterdir():
        if path.is_file():
            shutil.copy2(path, output_meshes / path.name)
    for name in DEX1_MESHES:
        shutil.copy2(dex1_meshes / name, output_meshes / name)

    output_xml = output_dir / "model.xml"
    shutil.copy2(source_xml, output_xml)
    spec = mujoco.MjSpec.from_file(str(output_xml))
    spec.option.timestep = 0.002
    for side in ("left", "right"):
        _remove_dex3(spec, side)
    _register_dex1_meshes(spec)
    for side in ("left", "right"):
        _graft_dex1(spec, side)
    _add_head_camera(spec)
    _add_wrist_cameras(spec)

    output_xml.write_text(spec.to_xml())
    model = mujoco.MjModel.from_xml_path(str(output_xml))
    print(
        f"[ok] {output_xml} | nq={model.nq} nv={model.nv} "
        f"nu={model.nu} timestep={model.opt.timestep}"
    )
    return output_xml


def _default_repo_paths() -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    teleop_root = repo.parents[1]
    return (
        teleop_root / "HoloTeleop" / "sim2real" / "assets" / "g1",
        repo.parent / "HIW-500-controoler" / "assets" / "dex1_1",
    )


def main() -> None:
    default_g1, default_dex1 = _default_repo_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--g1-dir", type=Path, default=default_g1)
    parser.add_argument("--dex1-dir", type=Path, default=default_dex1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "robots" / "g1_dex1",
    )
    args = parser.parse_args()
    build_asset(args.g1_dir.resolve(), args.dex1_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
