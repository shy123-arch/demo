"""Build a passive rolling chair scene and independently calibrate its mechanics.

The chair is a free rigid assembly supported by five unactuated spherical
ball casters. Ball joints approximate omnidirectional office casters; they
have Coulomb bearing resistance and viscous damping. This does not model
swivel-fork alignment, trail, twin-wheel scrub or upholstery compliance.
The optional calibration applies external forces ONLY in an isolated
chair-and-floor model, never in a robot task rollout.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _vec(values):
    return " ".join(f"{v:.9g}" for v in values)


def make_chair(x=.65, y=-2.):
    body = ET.Element("body", name="push_chair", pos=_vec((x, y, 0)))
    ET.SubElement(body, "freejoint", name="push_chair_free")
    common = dict(contype="1", conaffinity="1", group="3",
                  friction=".7 .005 .0001", condim="4")
    specs = [
        ("seat", "cylinder", (0, 0, .68), (.31, .065), 2.8, "lab_stool_gray"),
        ("back", "box", (-.15, 0, .91), (.065, .24, .20), 2., "lab_stool_gray"),
        ("stem", "cylinder", (0, 0, .36), (.045, .25), 1.4, "lab_metal_dark"),
        ("hub", "cylinder", (0, 0, .12), (.11, .035), 1.3, "lab_metal_dark"),
    ]
    for name, kind, pos, size, mass, material in specs:
        ET.SubElement(body, "geom", name=f"push_chair_{name}", type=kind,
                      pos=_vec(pos), size=_vec(size), mass=str(mass),
                      material=material, **common)
    for i, degrees in enumerate((0, 72, 144, 216, 288)):
        a = math.radians(degrees)
        xy = (.34 * math.cos(a), .34 * math.sin(a))
        ET.SubElement(body, "geom", name=f"push_chair_leg_{i}", type="capsule",
                      fromto=_vec((0, 0, .14, *xy, .065)), size=".018",
                      mass=".3", material="lab_metal_dark", **common)
        caster = ET.SubElement(body, "body", name=f"push_chair_caster_{i}",
                               pos=_vec((*xy, .045)))
        ET.SubElement(caster, "joint", name=f"push_chair_caster_ball_{i}",
                      type="ball", damping=".003", frictionloss=".025")
        ET.SubElement(caster, "geom", name=f"push_chair_caster_ball_geom_{i}",
                      type="sphere", size=".045", mass=".2",
                      material="lab_caster", contype="1", conaffinity="1",
                      friction=".7 .005 .0001", condim="6", priority="2",
                      group="3")
    return body


def build(output: Path, x=.65, y=-2.):
    source = ROOT / "scenes/lab_g1_vision.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "reference_laboratory_unitree_g1_passive_chair")
    world = root.find("worldbody")
    old = [el for el in world if el.get("name", "").startswith("lab_stool_0_")]
    if len(old) != 14:
        raise ValueError(f"Expected 14 static stool geoms, found {len(old)}")
    for el in old:
        world.remove(el)
    world.append(make_chair(x, y))
    ET.indent(root, "  ")
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="unicode")
    return root


def calibration_model(scene_root):
    """Only the scene chair and its real floor; robot not present or supported."""
    root = ET.Element("mujoco", model="isolated_passive_chair_calibration")
    root.append(copy.deepcopy(scene_root.find("option")))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    asset = ET.SubElement(root, "asset")
    names = {"lab_stool_gray", "lab_metal_dark", "lab_caster", "lab_floor",
             "lab_floor_checker"}
    for el in scene_root.find("asset"):
        if el.get("name") in names:
            asset.append(copy.deepcopy(el))
    world = ET.SubElement(root, "worldbody")
    world.append(copy.deepcopy(scene_root.find("worldbody/geom[@name='lab_floor_slab']")))
    world.append(copy.deepcopy(scene_root.find("worldbody/body[@name='push_chair']")))
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def mechanics_trial(model, force_n, duration=10.):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = model.body("push_chair").id
    start = data.xpos[bid].copy()
    dt = model.opt.timestep
    rows = []
    force = np.zeros(6)
    floor = model.geom("lab_floor_slab").id
    total_floor_normal = 0.
    for _ in range(round(duration / dt)):
        data.qfrc_applied[:] = 0
        # Explicit OFFLINE calibration load, applied 2--3 s at the back.
        if 2. <= data.time < 3. and force_n:
            point = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ np.array([-.215, 0, .91])
            mujoco.mj_applyFT(model, data, np.array([force_n, 0., 0.]),
                             np.zeros(3), point, bid, data.qfrc_applied)
        mujoco.mj_step(model, data)
        if int(round(data.time / dt)) % 10 == 0:
            tilt = math.degrees(math.acos(np.clip(data.xmat[bid, 8], -1., 1.)))
            rows.append([data.time, *data.xpos[bid], tilt, *data.qvel[:6]])
    for ci in range(data.ncon):
        con = data.contact[ci]
        if floor in (con.geom1, con.geom2):
            mujoco.mj_contactForce(model, data, ci, force)
            total_floor_normal += float(force[0])
    arr = np.asarray(rows)
    delta = data.xpos[bid] - start
    return dict(force_n=force_n, force_time_s=[2., 3.], duration_s=duration,
                displacement_m=delta.tolist(), final_xy_displacement_m=float(np.linalg.norm(delta[:2])),
                max_xy_displacement_m=float(np.linalg.norm(arr[:, 1:3] - start[:2], axis=1).max()),
                max_tilt_deg=float(arr[:, 4].max()), final_tilt_deg=float(arr[-1, 4]),
                final_root_speed_m_s=float(np.linalg.norm(data.qvel[:3])),
                final_floor_normal_n=total_floor_normal,
                max_linear_speed_m_s=float(np.linalg.norm(arr[:, 5:8], axis=1).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scenes/lab_g1_chair.xml")
    parser.add_argument("--x", type=float, default=.65)
    parser.add_argument("--y", type=float, default=-2.)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "outputs/chair_tasks/scene_mechanics")
    args = parser.parse_args()
    root = build(args.output, args.x, args.y)
    full = mujoco.MjModel.from_xml_path(str(args.output))
    model = calibration_model(root)
    trials = [mechanics_trial(model, f) for f in (0., 2., 5., 8.)]
    report = dict(scene=str(args.output.relative_to(ROOT)),
                  source_scene="scenes/lab_g1_vision.xml", body="push_chair", freejoint="push_chair_free",
                  initial_position_m=[args.x, args.y, 0.],
                  initial_back_center_m=[args.x-.15, args.y, .91],
                  initial_back_rear_face_x_m=args.x-.215,
                  suggested_robot_station_m=[args.x-.65, args.y],
                  replacement="Removed 14 static lab_stool_0_* geoms; other scene objects preserved.",
                  mass_kg=float(model.body_mass.sum()),
                  rigid_frame_mass_kg=float(model.body_mass[model.body('push_chair').id]),
                  caster_mass_kg=.2, caster_count=5, caster_radius_m=.045,
                  caster_ball_joint_frictionloss_nm=.025, caster_ball_joint_damping_nm_s=.003,
                  caster_surface_friction=[.7, .005, .0001], caster_contact_dimensions=6,
                  approximation="Five spherical ball casters with independent passive 3-DOF rotations. Omits swivel-fork alignment, trail, twin-wheel scrub and upholstery compliance; no planar locks or root damping.",
                  no_task_assistance="No chair actuators, equality constraints, external-force sources or state-pose updates added to task scene.",
                  calibration_scope="Isolated chair and copied laboratory floor. No robot. The 2/5/8 N loads below are external OFFLINE calibration probes, never task successes.",
                  trials=trials, scene_nq=int(full.nq), scene_nv=int(full.nv))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["# Passive chair mechanics", "", report['approximation'], "",
             f"Initial chair center: ({args.x}, {args.y}, 0) m; rear back surface x = {args.x-.215:.3f} m, back center height = 0.91 m. Robot can start at ({args.x-.65}, {args.y}) facing +X. This workstation is preset.", "",
             "Total mass 10 kg (9 kg rigid frame, five 0.2 kg balls). Each 45 mm radius caster has a passive ball joint with 0.025 N m Coulomb bearing resistance and 0.003 N m s/rad viscous damping. Surface friction is 0.7 sliding / 0.005 m torsional / 0.0001 m rolling, condim 6. Root has six free degrees of freedom and no support/planar constraint.", "",
             report['calibration_scope'], "",
             "| Calibration load (2–3 s) | 10 s XY displacement | Max tilt | Final speed |",
             "| --- | --- | --- | --- |"]
    for t in trials:
        lines.append(f"| {t['force_n']:.0f} N | {t['final_xy_displacement_m']*1000:.4f} mm | {t['max_tilt_deg']:.4f}° | {t['final_root_speed_m_s']:.6f} m/s |")
    (args.report_dir / "MECHANICS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
