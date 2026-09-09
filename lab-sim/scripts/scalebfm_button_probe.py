#!/usr/bin/env python3
"""Physical ScaleBFM button-contact experiment, with no object assistance.

Only the reset places the robot at its station. All subsequent body commands
are five world-frame poses supplied to the official mode-4 policy. Fingers
remain open. Scene geometry and object dynamics remain untouched.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from lab_sim.env import LabEnv
from lab_sim.scalebfm import FIVE_BODIES, ScaleBFMController
from lab_sim.scalebfm_demo import PoseSequence


def build_plan(p, q, args):
    lift = p.copy(); lift[2] = (p[2, 0], args.wy, args.wz + .025)
    pre = lift.copy(); pre[2] = (args.wx - .075, args.wy, args.wz)
    push = pre.copy(); push[2, 0] = args.wx
    retreat = pre.copy()
    rotated = q.copy()
    rotated[2] = Rotation.from_euler("xyz", [args.roll, args.pitch, args.yaw], degrees=True).as_quat()[[3, 0, 1, 2]]
    states = [(0, "reset station", p, q), (1, "standing settle", p, q),
              (3, "lift hand above worktop", lift, rotated),
              (5, "align extended index finger", pre, rotated),
              (8, "press button through contact", push, rotated),
              (9, "hold physical press", push, rotated),
              (11, "release spring button", retreat, rotated),
              (12, "hold released", retreat, rotated),
              (13.5, "retract above worktop", lift, rotated),
              (15.5, "recover standing", p, q),
              (16, "standing settled", p, q)]
    return [{"time": t, "label": label, "poses": {name: {
        "position": pos[i].tolist(), "quaternion_wxyz": quat[i].tolist()}
        for i, name in enumerate(FIVE_BODIES)}} for t, label, pos, quat in states]


def geom_info(env, gid):
    return {"geom_id": int(gid), "geom_name": env.model.geom(int(gid)).name or None,
            "body_name": env.model.body(int(env.model.geom_bodyid[gid])).name}


def run(args):
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    env = LabEnv(assisted=False)
    controller = ScaleBFMController(env, device=args.device)
    controller.reset(env, xy=(args.x, args.y))
    p, q = controller.poses(env)
    frames = build_plan(p, q, args)
    seq = PoseSequence(frames)
    (out / "commands.json").write_text(json.dumps({"control_mode": 4,
        "reset_robot_xy": [args.x, args.y], "reset_robot_yaw": 0,
        "finger_posture": "all finger joints target zero (open)",
        "coordinate_frame": "world", "keyframes": frames}, indent=2) + "\n")
    button_geom = env.model.geom("button_collision").id
    assist = [env.model.actuator(n).id for n in ("button_assist", "drawer_assist")]
    button_qadr = env.model.joint("button_slide").qposadr[0]
    records = []; contacts = []; incidental_contacts = []; collision_counts = {}; last_phase = None
    failed = None; max_ctrl = 0.; max_applied = 0.; max_eq = 0
    best_distance = float("inf"); closest = None; force = np.zeros(6)
    started = time.monotonic()
    for step in range(round(args.seconds / .02)):
        t = env.data.time
        target_p, target_q = seq.sample(t + .02 * controller.offsets)
        try:
            controller.advance(env, target_p, target_q)
        except RuntimeError as exc:
            failed = str(exc)
            break
        actual_p, actual_q = controller.poses(env)
        current_p, current_q = seq.sample([env.data.time])
        value = float(env.data.qpos[button_qadr])
        max_ctrl = max(max_ctrl, float(np.max(np.abs(env.data.ctrl[assist]))))
        max_applied = max(max_applied, float(np.max(np.abs(env.data.xfrc_applied))),
                          float(np.max(np.abs(env.data.qfrc_applied))))
        max_eq = max(max_eq, int(np.count_nonzero(env.data.eq_active)))
        for ci in range(env.data.ncon):
            contact = env.data.contact[ci]
            a, b = int(contact.geom1), int(contact.geom2)
            if button_geom in (a, b):
                other = b if a == button_geom else a
                mujoco.mj_contactForce(env.model, env.data, ci, force)
                contacts.append({"time": float(env.data.time), "button_travel_m": value,
                    "other": geom_info(env, other), "position": contact.pos.tolist(),
                    "normal_force_N": float(force[0]), "penetration_m": float(contact.dist)})
            ainfo, binfo = geom_info(env, a), geom_info(env, b)
            if ("right_" in ainfo["body_name"] or "right_" in binfo["body_name"]) and (
                ainfo["body_name"] in ("world", "analyzer", "lab_drawer", "sample_box", "sample_bottle") or
                binfo["body_name"] in ("world", "analyzer", "lab_drawer", "sample_box", "sample_bottle")):
                key = str((ainfo, binfo))
                collision_counts[key] = collision_counts.get(key, 0) + 1
                if "right_hand_" in ainfo["body_name"] or "right_hand_" in binfo["body_name"]:
                    mujoco.mj_contactForce(env.model, env.data, ci, force)
                    incidental_contacts.append({"time": float(env.data.time), "geoms": [ainfo, binfo],
                        "position": contact.pos.tolist(), "normal_force_N": float(force[0])})
        for gid in range(env.model.ngeom):
            body = env.model.body(int(env.model.geom_bodyid[gid])).name
            if body.startswith("right_hand_") and env.model.geom_contype[gid]:
                distance = mujoco.mj_geomDistance(env.model, env.data, gid, button_geom, .2, None)
                if distance < best_distance:
                    best_distance = float(distance)
                    closest = {"time": float(env.data.time), "distance_m": best_distance,
                        "finger": geom_info(env, gid), "wrist_position": actual_p[2].tolist(),
                        "wrist_quaternion_wxyz": actual_q[2].tolist()}
        records.append((float(env.data.time), env.data.qpos.copy(), env.data.qvel.copy(),
            actual_p, actual_q, current_p[0], current_q[0], value, env.data.ctrl.copy()))
        phase = seq.phase(env.data.time)
        if phase != last_phase:
            print(f"t={env.data.time:.2f} {phase}: wrist={actual_p[2].round(4).tolist()} button={value:.5f} near={best_distance:.5f}", flush=True)
            last_phase = phase
    array = lambda index: np.asarray([r[index] for r in records])
    travel = array(7); positions = array(3)
    quat = array(4)[:, 0]
    pelvis_tilt = np.degrees(np.arccos(np.clip(1 - 2 * (quat[:, 1]**2 + quat[:, 2]**2), -1, 1)))
    report = {"task": "physical analyzer button press and release",
        "model": "official ScaleBFM M model_22200.pt", "control_mode": 4,
        "parameters": vars(args), "completed": failed is None and len(records) == round(args.seconds/.02),
        "failure": failed, "simulation_seconds": float(env.data.time),
        "button_max_travel_m": float(travel.max()), "button_final_travel_m": float(travel[-1]),
        "button_contact_samples": len(contacts), "button_contact_bodies": sorted({c["other"]["body_name"] for c in contacts}),
        "button_peak_normal_contact_force_N": max([c["normal_force_N"] for c in contacts], default=0.),
        "button_travel_over_8mm_seconds": float(np.sum(travel > .008)*.02),
        "success": bool(failed is None and travel.max() > .008 and abs(travel[-1]) < .002 and contacts),
        "minimum_pelvis_z": float(positions[:, 0, 2].min()),
        "maximum_pelvis_tilt_deg": float(pelvis_tilt.max()),
        "foot_max_xy_displacement_m": np.linalg.norm(positions[:, 3:, :2] - p[None, 3:, :2], axis=-1).max(0).tolist(),
        "body_support": False, "max_active_equalities": max_eq, "max_assistance_control": max_ctrl,
        "max_external_applied_force": max_applied,
        "reset_only_station_placement": True, "walking_to_station": False,
        "scene_or_collision_changes": False, "object_qpos_mutations_after_reset": False,
        "closest_finger": closest, "incidental_contact_sample_counts": collision_counts,
        "incidental_hand_scene_contact_samples": len(incidental_contacts),
        "incidental_hand_scene_peak_normal_force_N": max([c["normal_force_N"] for c in incidental_contacts], default=0.),
        "wall_seconds": time.monotonic()-started}
    np.savez_compressed(out / "trajectory.npz", time=array(0), qpos=array(1), qvel=array(2),
        actual_position=positions, actual_quaternion_wxyz=array(4),
        target_position=array(5), target_quaternion_wxyz=array(6),
        button_travel=travel, ctrl=array(8), body_names=np.asarray(FIVE_BODIES))
    (out / "contacts.json").write_text(json.dumps(contacts, indent=2) + "\n")
    (out / "incidental_contacts.json").write_text(json.dumps(incidental_contacts, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    env.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=1.65)
    parser.add_argument("--y", type=float, default=-.87)
    parser.add_argument("--wx", type=float, default=1.99)
    parser.add_argument("--wy", type=float, default=-1.095)
    parser.add_argument("--wz", type=float, default=.867)
    parser.add_argument("--roll", type=float, default=0)
    parser.add_argument("--pitch", type=float, default=0)
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--seconds", type=float, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/scalebfm_tasks/button_probe/final")
    run(parser.parse_args())
