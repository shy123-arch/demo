"""Physical laboratory interaction trials driven by ScaleBFM five-point poses."""
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .env import LabEnv, ROOT
from .scalebfm import ScaleBFMController, FIVE_BODIES
from .scalebfm_demo import PoseSequence, frame


def make_keyframe(t, label, positions, quaternions):
    return {"time": t, "label": label, "poses": {
        n: {"position": positions[i].tolist(), "quaternion_wxyz": quaternions[i].tolist()}
        for i, n in enumerate(FIVE_BODIES)}}


class ContactAudit:
    """Observe every physical tick without applying forces or editing state."""

    def __init__(self, env):
        self.pairs = {}
        self.max_button = 0.
        self.button_trigger_time = 0.
        self.min_drawer = float(env.data.joint("drawer_slide").qpos[0])
        self.min_height = float(env.data.qpos[2])
        self.max_assist_control = 0.
        self.max_external_force = 0.
        self.active_equality_seen = False
        self.forces = np.zeros(6)
        self.assist_ids = [env.model.actuator(n).id for n in ("drawer_assist", "button_assist")]
        self.hand_ids = {i for i in range(env.model.nbody)
            if any((env.model.body(i).name or "").startswith(s) for s in (
                "left_hand_", "right_hand_", "left_wrist_", "right_wrist_"))}
        self.robot_ids = set()
        root = env.model.body("pelvis").id
        for i in range(env.model.nbody):
            a = i
            while a:
                if a == root:
                    self.robot_ids.add(i); break
                a = env.model.body_parentid[a]

    def tick(self, env):
        d, m = env.data, env.model
        self.max_button = max(self.max_button, float(d.joint("button_slide").qpos[0]))
        if d.joint("button_slide").qpos[0] > .008:
            self.button_trigger_time += m.opt.timestep
        self.min_drawer = min(self.min_drawer, float(d.joint("drawer_slide").qpos[0]))
        self.min_height = min(self.min_height, float(d.qpos[2]))
        self.max_assist_control = max(self.max_assist_control, float(np.max(np.abs(d.ctrl[self.assist_ids]))))
        self.max_external_force = max(self.max_external_force, float(np.max(np.abs(d.xfrc_applied))),
                                      float(np.max(np.abs(d.qfrc_applied))))
        self.active_equality_seen |= bool(d.eq_active.any())
        for ci in range(d.ncon):
            co = d.contact[ci]
            g1, g2 = int(co.geom1), int(co.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if (b1 in self.robot_ids) == (b2 in self.robot_ids):
                continue
            rg, og, rb, ob = (g1,g2,b1,b2) if b1 in self.robot_ids else (g2,g1,b2,b1)
            other_geom = m.geom(og).name or f"geom_{og}"
            if "floor" in other_geom:
                continue
            mujoco.mj_contactForce(m, d, ci, self.forces)
            normal = abs(float(self.forces[0]))
            if normal < .01:
                continue
            key = (m.body(rb).name, other_geom)
            if key not in self.pairs:
                self.pairs[key] = {"robot_body": key[0], "other_geom": key[1],
                    "other_body": m.body(ob).name, "hand_contact": rb in self.hand_ids,
                    "first_time": float(d.time), "last_time": float(d.time),
                    "max_normal_force_n": 0., "normal_impulse_ns": 0., "contact_samples": 0}
            item = self.pairs[key]
            item["last_time"] = float(d.time)
            item["max_normal_force_n"] = max(item["max_normal_force_n"], normal)
            item["normal_impulse_ns"] += normal * m.opt.timestep
            item["contact_samples"] += 1

    def hand_contact(self, target):
        return any(x["hand_contact"] and (x["other_body"] == target or target in x["other_geom"])
                   for x in self.pairs.values())


def run_plan(plan, output_dir, device="cpu", env=None, controller=None):
    """Run a finite experiment, saving full states and independently measured outcomes."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    owned = env is None
    env = env or LabEnv(assisted=False)
    controller = controller or ScaleBFMController(env, device=device)
    started = time.perf_counter()
    try:
        controller.reset(env, xy=plan["initial_xy"], yaw=plan.get("initial_yaw", 0))
        initial_opening = plan.get("initial_drawer_open_m", 0.)
        if initial_opening:
            env.data.joint("drawer_slide").qpos[0] = initial_opening
        for name, target in plan.get("finger_targets", {}).items():
            if "hand_" not in name:
                raise ValueError("Only Dex3 finger targets can supplement the ScaleBFM body policy")
            env.target[env.names.index(name)] = target
        mujoco.mj_forward(env.model, env.data)
        initial_box = env.body("sample_box")
        initial_bottle = env.body("sample_bottle")
        sequence = PoseSequence(plan["keyframes"])
        (out/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
        audit = ContactAudit(env)
        records = []
        failure = None
        previous_phase = None
        count = round(sequence.times[-1]/.02)
        for i in range(count):
            t = env.data.time
            p, q = sequence.sample(t + .02*controller.offsets)
            phase = sequence.phase(t)
            try:
                controller.advance(env, p, q, tick_callback=audit.tick)
            except RuntimeError as exc:
                failure = str(exc)
            records.append((float(env.data.time), env.data.qpos.copy(), env.data.qvel.copy(),
                            p[0], q[0], env.body("sample_box"), env.body("sample_bottle"),
                            float(env.data.joint("button_slide").qpos[0]),
                            float(env.data.joint("drawer_slide").qpos[0])))
            if phase != previous_phase:
                print(f"{plan['task']} t={env.data.time:.2f} {phase}; "
                      f"button={audit.max_button*1000:.1f} mm; drawer={records[-1][8]:.3f}; "
                      f"box={np.round(records[-1][5],3).tolist()}", flush=True)
                previous_phase = phase
            if failure:
                break
        final_box = env.body("sample_box")
        box_tilt = float(np.degrees(np.arccos(np.clip(env.data.body("sample_box").xmat.reshape(3,3)[2,2], -1, 1))))
        final_bottle = env.body("sample_bottle")
        final_drawer = float(env.data.joint("drawer_slide").qpos[0])
        final_button = float(env.data.joint("button_slide").qpos[0])
        completed = failure is None and len(records) == count
        unassisted = audit.max_assist_control == 0 and audit.max_external_force == 0 and not audit.active_equality_seen
        task = plan["task"]
        if task == "button":
            contact = audit.hand_contact("instrument_button")
            objective = contact and audit.max_button > .008 and final_button < .002
        elif task == "push_box":
            contact = audit.hand_contact("sample_box")
            goal = np.asarray(plan["box_goal_xy"])
            objective = contact and np.linalg.norm(final_box[:2]-goal) < plan.get("goal_tolerance_m", .025) and .825 < final_box[2] < .85 and box_tilt < 15
        elif task == "close_drawer":
            contact = audit.hand_contact("lab_drawer")
            objective = contact and initial_opening >= .1 and final_drawer < .025
        else:
            raise ValueError(f"Unknown task: {task}")
        target_body = {"button":"instrument_button", "push_box":"sample_box", "close_drawer":"lab_drawer"}[task]
        report = {
            "task": task, "model": "official ScaleBFM M/model_22200.pt", "control_mode": 4,
            "success": bool(completed and unassisted and objective), "completed": completed,
            "failure": failure, "hand_target_contact": contact,
            "simulation_seconds": float(env.data.time), "minimum_pelvis_z": audit.min_height,
            "max_button_travel_m": audit.max_button,
            "final_button_travel_m": final_button, "button_trigger_time_s": audit.button_trigger_time,
            "initial_drawer_open_m": initial_opening, "final_drawer_open_m": final_drawer,
            "minimum_drawer_open_m": audit.min_drawer,
            "initial_box_position": initial_box.tolist(), "final_box_position": final_box.tolist(),
            "box_displacement_m": (final_box-initial_box).tolist(),
            "box_goal_xy": plan.get("box_goal_xy"),
            "box_goal_error_m": float(np.linalg.norm(final_box[:2]-plan["box_goal_xy"])) if task == "push_box" else None,
            "box_final_tilt_deg": box_tilt,
            "bottle_displacement_m": (final_bottle-initial_bottle).tolist(),
            "unassisted": unassisted, "max_object_assist_control": audit.max_assist_control,
            "max_external_applied_force": audit.max_external_force,
            "active_equality_seen": audit.active_equality_seen,
            "contacts": list(audit.pairs.values()),
            "non_target_contacts": [x for x in audit.pairs.values() if x["other_body"] != target_body],
            "inference_median_ms": float(np.median(controller.inference_ms)),
            "wall_seconds": time.perf_counter()-started,
            "initialization": "Robot station and optional open drawer are set before the first physics step",
        }
        (out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        if records:
            np.savez_compressed(out/"trajectory.npz", time=np.asarray([r[0] for r in records]),
                qpos=np.asarray([r[1] for r in records]), qvel=np.asarray([r[2] for r in records]),
                target_position=np.asarray([r[3] for r in records]),
                target_quaternion_wxyz=np.asarray([r[4] for r in records]),
                sample_box=np.asarray([r[5] for r in records]), sample_bottle=np.asarray([r[6] for r in records]),
                button_travel=np.asarray([r[7] for r in records]), drawer_open=np.asarray([r[8] for r in records]))
        print(json.dumps({k:v for k,v in report.items() if k!='contacts'}, indent=2), flush=True)
        return report
    finally:
        if owned: env.close()


def render_run(output_dir, fps=10):
    """Render the saved physical rollout; never simulate a substitute motion."""
    import imageio.v2 as imageio
    out = Path(output_dir)
    plan = json.loads((out/"plan.json").read_text())
    report = json.loads((out/"report.json").read_text())
    sequence = PoseSequence(plan["keyframes"])
    z = np.load(out/"trajectory.npz")
    env = LabEnv(assisted=False)
    cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
    # View from the aisle towards the right-side workstation.
    cam.lookat[:] = (1.87, -1.02, .80)
    cam.distance = 2.15; cam.azimuth = 25; cam.elevation = -17
    try:
        with imageio.get_writer(str(out/"task.mp4"), fps=fps, codec="libx264", quality=7, macro_block_size=8) as writer:
            for i in range(0, len(z['time']), round(50/fps)):
                env.data.time = float(z['time'][i])
                env.data.qpos[:] = z['qpos'][i]; env.data.qvel[:] = z['qvel'][i]
                env.data.eq_active[:] = 0
                mujoco.mj_forward(env.model, env.data)
                im = frame(env, z['target_position'][i], f"{plan['task']}: {sequence.phase(env.data.time)}", cam)
                from PIL import ImageDraw
                draw = ImageDraw.Draw(im)
                if plan['task'] == 'button':
                    travel = 1000 * z['button_travel'][i]
                    metric = f"Button travel: {max(0,travel):.1f} mm | trigger: 8 mm"
                    color = (110, 245, 150) if travel > 8 else (240, 240, 240)
                elif plan['task'] == 'push_box':
                    dx = 1000 * (z['sample_box'][i,0] - report['initial_box_position'][0])
                    metric = f"Sample box moved: {dx:.1f} mm | desired: 60 mm"
                    color = (240, 240, 240)
                else:
                    opening = 1000 * z['drawer_open'][i]
                    metric = f"Drawer opening: {max(0,opening):.1f} mm | started at 180 mm"
                    color = (110, 245, 150) if opening < 25 else (240, 240, 240)
                draw.rectangle((0, im.height-24, im.width, im.height), fill=(18,28,33))
                draw.text((14, im.height-18), metric, fill=color)
                writer.append_data(np.asarray(im))
                if i == len(z['time'])//2 - len(z['time'])//2 % round(50/fps):
                    im.save(out/"interaction.png")
            im.save(out/"final.png")
    finally:
        env.close()


def run_tasks(args):
    tasks = ["button", "push_box", "close_drawer"] if args.task == "all" else [args.task]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = controller = None
    reports = {}
    try:
        if not args.render_only:
            env = LabEnv(assisted=False)
            controller = ScaleBFMController(env, device=args.device)
        for task in tasks:
            folder = output / task
            if args.render_only:
                reports[task] = json.loads((folder/"report.json").read_text())
            else:
                plan = json.loads((ROOT/"assets/scalebfm_tasks"/f"{task}.json").read_text())
                reports[task] = run_plan(plan, folder, device=args.device, env=env, controller=controller)
            if args.video or args.render_only:
                render_run(folder)
        summary = {"model":"official ScaleBFM M", "control_mode":4,
            "tasks": {name: {k:v for k,v in r.items() if k not in ("contacts", "non_target_contacts")}
                      for name,r in reports.items()},
            "successful_tasks": sum(r["success"] for r in reports.values()), "attempted_tasks":len(reports),
            "scope":"Independent tasks start at preset robot stations; this is not a navigation or grasping evaluation"}
        (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        print(f"ScaleBFM physical tasks: {summary['successful_tasks']}/{len(tasks)} succeeded; {output.resolve()}")
    finally:
        if env: env.close()
