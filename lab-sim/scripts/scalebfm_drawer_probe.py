"""Physical ScaleBFM drawer closing probes; reset-only drawer initialization."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_sim.env import LabEnv
from lab_sim.scalebfm import ScaleBFMController, FIVE_BODIES
from lab_sim.scalebfm_demo import PoseSequence


def keyframe(t, label, p, q):
    return {"time": t, "label": label, "poses": {
        n: {"position": p[i].tolist(), "quaternion_wxyz": q[i].tolist()}
        for i, n in enumerate(FIVE_BODIES)}}


def run_trial(env, controller, output, x=1.48, end_x=1.94, wrist_z=.74,
              pelvis_dx=.035, closed=False, single=False):
    output.mkdir(parents=True, exist_ok=True)
    controller.reset(env, xy=(x, -.8), yaw=0)
    drawer_qa = env.model.joint("drawer_slide").qposadr[0]
    env.data.qpos[drawer_qa] = .18  # Initial task setup, before all physics steps.
    mujoco.mj_forward(env.model, env.data)
    if closed:
        env.grip(True, "left")
        env.grip(True, "right")
    p, q = controller.poses(env)
    prepare = p.copy()
    hands = (2,) if single else (1, 2)
    for i in hands:
        prepare[i, [0, 2]] = (1.70, wrist_z)
    push = prepare.copy()
    push[0, 0] += pelvis_dx
    for i in hands:
        push[i, 0] = end_x
    frames = [keyframe(t, label, target, q) for t, label, target in [
        (0, "initial stance", p), (2, "settle", p),
        (4, "align hands with drawer front", prepare),
        (8, "push drawer closed", push), (9.5, "hold closing pressure", push),
        (12, "withdraw hands", p), (14, "finish balanced", p)]]
    plan = {"task": "drawer_close", "initial_xy": [x, -.8], "initial_yaw": 0,
            "initial_drawer_open_m": .18, "finger_targets": {
                n: float(env.target[i]) for i, n in enumerate(env.names) if "hand_" in n},
            "coordinate_frame": "world", "control_mode": 4,
            "bodies": list(FIVE_BODIES), "keyframes": frames}
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    seq = PoseSequence(frames)
    records = []
    contacts = Counter()
    other_robot_contacts = Counter()
    failure = None
    drawer_id = env.model.body("lab_drawer").id
    robot_bodies = set()
    for i in range(env.model.nbody):
        k = i
        while k:
            if env.model.body(k).name == "pelvis":
                robot_bodies.add(i)
                break
            k = int(env.model.body_parentid[k])
    for step in range(700):
        target, quat = seq.sample(env.data.time + .02 * controller.offsets)
        try:
            controller.advance(env, target, quat)
        except RuntimeError as exc:
            failure = str(exc)
            break
        force = np.zeros(6)
        for contact_i in range(env.data.ncon):
            con = env.data.contact[contact_i]
            geoms = (int(con.geom1), int(con.geom2))
            bodies = [int(env.model.geom_bodyid[g]) for g in geoms]
            names = [env.model.geom(g).name or env.model.body(b).name for g, b in zip(geoms, bodies)]
            if drawer_id in bodies and any(b in robot_bodies for b in bodies):
                mujoco.mj_contactForce(env.model, env.data, contact_i, force)
                if force[0] > .01:
                    contacts[" / ".join(names)] += 1
            elif any(b in robot_bodies for b in bodies) and not all(b in robot_bodies for b in bodies):
                other_robot_contacts[" / ".join(names)] += 1
        records.append((env.data.time, env.data.qpos.copy(), env.data.qvel.copy(),
                        env.data.xpos[env.model.body("pelvis").id].copy(),
                        env.data.qpos[drawer_qa]))
        if step % 100 == 99:
            print(f"{output.name} t={env.data.time:.1f} drawer={env.data.qpos[drawer_qa]:.4f} "
                  f"pelvis={env.data.qpos[:3].round(3).tolist()} "
                  f"right_palm={env.site('right_grasp').round(3).tolist()}", flush=True)
    drawer = np.array([r[4] for r in records])
    pelvis = np.array([r[3] for r in records])
    np.savez_compressed(output / "trajectory.npz", time=[r[0] for r in records],
                        qpos=[r[1] for r in records], qvel=[r[2] for r in records],
                        drawer_open_m=drawer, pelvis_position=pelvis)
    report = {"success": bool(failure is None and drawer[-1] < .025 and contacts),
              "failure": failure, "seconds": float(env.data.time),
              "initial_drawer_open_m": .18, "final_drawer_open_m": float(drawer[-1]),
              "minimum_drawer_open_m": float(drawer.min()),
              "minimum_pelvis_z": float(pelvis[:, 2].min()),
              "active_equalities": int(np.count_nonzero(env.data.eq_active)),
              "object_assist_controls": {n: float(env.data.ctrl[env.model.actuator(n).id])
                                         for n in ("drawer_assist", "button_assist")},
              "drawer_contact_samples": dict(contacts),
              "other_robot_contact_samples": dict(other_robot_contacts),
              "parameters": {"x": x, "end_x": end_x, "wrist_z": wrist_z,
                             "pelvis_dx": pelvis_dx, "closed": closed, "single": single}}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/scalebfm_drawer_probe")
    parser.add_argument("--x", type=float, default=1.48)
    parser.add_argument("--end-x", type=float, default=1.94)
    parser.add_argument("--wrist-z", type=float, default=.74)
    parser.add_argument("--pelvis-dx", type=float, default=.035)
    parser.add_argument("--closed", action="store_true")
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    env = LabEnv(assisted=False)
    controller = ScaleBFMController(env)
    try:
        run_trial(env, controller, Path(args.output_dir), x=args.x, end_x=args.end_x,
                  wrist_z=args.wrist_z, pelvis_dx=args.pelvis_dx,
                  closed=args.closed, single=args.single)
    finally:
        env.close()


if __name__ == "__main__":
    main()
