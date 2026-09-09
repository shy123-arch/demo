#!/usr/bin/env python3
"""Prepare and validate the portable Unitree G1 lab asset.

The checked-in ``assets/g1/robot.xml`` is the edited, portable MJCF.  This
utility copies only the STL meshes referenced by the local source MJCF and
then validates the checked-in model.  It has no third-party dependencies, so
it can also be used before installing MuJoCo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "assets" / "g1"
DEFAULT_SOURCE_XML = (
    REPO_ROOT.parent
    / "parameter-fusion"
    / "psi0-regularized-finetune-20260829-v1"
    / "training"
    / "Psi0-regularized"
    / "real"
    / "assets"
    / "g1"
    / "g1_body29_hand14.xml"
)


def _mesh_files(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    asset = root.find("asset")
    if asset is None:
        raise ValueError(f"source model has no <asset>: {xml_path}")
    names = [mesh.attrib["file"] for mesh in asset.findall("mesh")]
    if not names or any(Path(name).name != name for name in names):
        raise ValueError("source mesh paths must be simple filenames")
    if len(names) != len(set(names)):
        raise ValueError("source model contains duplicate mesh filenames")
    return names


def copy_meshes(source_xml: Path, output_root: Path) -> list[Path]:
    source_mesh_dir = source_xml.parent / "meshes"
    destination = output_root / "meshes"
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for filename in _mesh_files(source_xml):
        source = source_mesh_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"referenced source mesh is missing: {source}")
        target = destination / filename
        shutil.copy2(source, target)
        copied.append(target)
    return copied


def validate(robot_xml: Path, output_root: Path) -> dict[str, int]:
    root = ET.parse(robot_xml).getroot()
    if root.tag != "mujoco":
        raise ValueError(f"not an MJCF model: {robot_xml}")

    compiler = root.find("compiler")
    if compiler is None or compiler.attrib.get("meshdir") != "meshes":
        raise ValueError('standalone robot.xml must use compiler meshdir="meshes"')

    asset = root.find("asset")
    if asset is None:
        raise ValueError("robot.xml has no <asset>")
    mesh_names = [mesh.attrib["file"] for mesh in asset.findall("mesh")]
    missing = [name for name in mesh_names if not (output_root / "meshes" / name).is_file()]
    if missing:
        raise FileNotFoundError("missing portable meshes: " + ", ".join(missing))

    worldbody = root.find("worldbody")
    pelvis = None if worldbody is None else worldbody.find("body")
    if pelvis is None or pelvis.attrib.get("name") != "pelvis":
        raise ValueError("robot root must be worldbody/body[@name='pelvis']")
    if pelvis.attrib.get("pos") != "1.50 -0.80 0.793":
        raise ValueError("pelvis spawn pose must be 1.50 -0.80 0.793")
    if pelvis.find("joint[@type='free']") is None:
        raise ValueError("robot root must have a free joint")

    names = {node.attrib.get("name") for node in root.iter()}
    required = {
        "left_palm",
        "right_palm",
        "left_palm_site",
        "right_palm_site",
        "g1_head",
        "pelvis_support_anchor",
        "pelvis_support",
    }
    missing_required = sorted(required - names)
    if missing_required:
        raise ValueError("required frames/equality missing: " + ", ".join(missing_required))
    for side in ("left", "right"):
        site = root.find(f".//body[@name='{side}_palm']/site[@name='{side}_palm_site']")
        if site is None or site.attrib.get("pos") != "0.07 0 0":
            raise ValueError(f"{side}_palm_site must be 0.07 m along its palm frame")
    if root.find(".//geom[@name='floor']") is not None:
        raise ValueError("robot asset must not contain a room floor")

    motors = root.findall("actuator/motor")
    motor_names = [motor.attrib.get("name") for motor in motors]
    motor_joints = [motor.attrib.get("joint") for motor in motors]
    if len(motors) != 43 or len(set(motor_names)) != 43 or motor_names != motor_joints:
        raise ValueError("expected 43 one-to-one original motor mappings")

    return {"meshes": len(mesh_names), "motors": len(motors)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-xml",
        type=Path,
        default=DEFAULT_SOURCE_XML,
        help="local source MJCF used only to select meshes",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="directory containing robot.xml and meshes/",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate without copying source meshes",
    )
    args = parser.parse_args(argv)

    try:
        if not args.check:
            if not args.source_xml.is_file():
                raise FileNotFoundError(f"source MJCF not found: {args.source_xml}")
            copied = copy_meshes(args.source_xml, args.output_root)
            print(f"copied {len(copied)} referenced meshes to {args.output_root / 'meshes'}")
        summary = validate(args.output_root / "robot.xml", args.output_root)
        print(f"validated robot.xml: {summary['meshes']} meshes, {summary['motors']} motors")
    except (ET.ParseError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"prepare_robot.py: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
