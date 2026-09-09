"""Command and record five-point ScaleBFM control in the laboratory."""
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from .env import LabEnv
from .scalebfm import ScaleBFMController, FIVE_BODIES


class PoseSequence:
    """Smoothly interpolate explicitly commanded world-frame body poses."""

    def __init__(self, frames):
        self.frames = frames
        self.times = np.asarray([f["time"] for f in frames], dtype=float)
        if len(frames) < 2 or not np.isfinite(self.times).all() or self.times[0] != 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError("Pose sequence requires increasing times beginning at zero")
        self.positions = np.asarray([[f["poses"][n]["position"] for n in FIVE_BODIES] for f in frames], dtype=float)
        self.quaternions = np.asarray([[f["poses"][n]["quaternion_wxyz"] for n in FIVE_BODIES] for f in frames], dtype=float)
        if self.positions.shape != (len(frames), 5, 3) or self.quaternions.shape != (len(frames), 5, 4):
            raise ValueError("Each keyframe must contain all five positions and wxyz quaternions")
        if not np.isfinite(self.positions).all() or not np.isfinite(self.quaternions).all():
            raise ValueError("Pose keyframes must be finite")
        if not np.allclose(np.linalg.norm(self.quaternions, axis=-1), 1, atol=1e-3):
            raise ValueError("Pose quaternions must be normalized")
        self.fixed_orientation = np.array_equal(self.quaternions,
            np.broadcast_to(self.quaternions[0], self.quaternions.shape))
        self.slerps = []
        if not self.fixed_orientation:
            self.slerps = [[Slerp([0, 1], Rotation.from_quat(
                self.quaternions[k:k+2, b][:, [1, 2, 3, 0]]))
                for b in range(5)] for k in range(len(frames)-1)]

    def sample(self, times):
        times = np.clip(np.asarray(times), self.times[0], self.times[-1])
        i = np.clip(np.searchsorted(self.times, times, side="right") - 1, 0, len(self.times) - 2)
        s = (times - self.times[i]) / (self.times[i + 1] - self.times[i])
        # Quintic interpolation: zero velocity and acceleration at keyframes.
        s = np.clip(s * s * s * (10 + s * (-15 + 6 * s)), 0, 1)
        p = self.positions[i] + s[:, None, None] * (self.positions[i + 1] - self.positions[i])
        if self.fixed_orientation:
            return p, np.broadcast_to(self.quaternions[0], (len(times), 5, 4)).copy()
        q = np.empty((len(times), 5, 4))
        for k in range(len(times)):
            for b in range(5):
                q[k, b] = self.slerps[i[k]][b]([s[k]]).as_quat()[0, [3, 0, 1, 2]]
        return p, q

    def phase(self, t):
        idx = min(np.searchsorted(self.times, t, side="right"), len(self.frames)-1)
        return self.frames[idx].get("label", "five-point target")


def direct_program(positions, quaternions):
    """Assistant-authored targets; no recorded human motion or tracking hardware."""
    base = positions.copy()
    squat = base.copy(); squat[0, 2] -= .10
    right = base.copy(); right[2] += (.16, 0, .18)
    both = right.copy(); both[1] += (.16, 0, .18)
    shift = both.copy(); shift[0, 1] += .045
    frames = []
    for t, label, pos in (
        (0, "stand", base), (2, "stand", base),
        (4, "squat", squat), (5, "hold squat", squat), (7, "stand up", base),
        (9, "right hand forward", right), (10, "hold right hand", right),
        (12, "both hands forward", both), (13, "hold both hands", both),
        (15, "shift pelvis left", shift), (16, "hold shifted pose", shift),
        (18, "return to stand", base), (20, "stand", base),
    ):
        frames.append({"time": t, "label": label, "poses": {
            n: {"position": pos[i].tolist(), "quaternion_wxyz": quaternions[i].tolist()}
            for i, n in enumerate(FIVE_BODIES)}})
    return frames


def frame(env, target, phase, camera="robot", width=640, height=400):
    cam = camera
    if camera == "robot":
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = (0, -.8, .76)
        cam.distance = 2.5
        cam.azimuth = 25
        cam.elevation = -12
    if env.renderer is None or (env.renderer.width, env.renderer.height) != (width, height):
        if env.renderer is not None:
            env.renderer.close()
        env.renderer = mujoco.Renderer(env.model, height=height, width=width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = (1, 1, 1, 1, 0, 1)
    opt.sitegroup[:] = 0
    env.renderer.update_scene(env.data, camera=cam, scene_option=opt)
    # Keep software-rendered feedback practical; these flags only affect appearance.
    env.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    env.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    colors = [(1,.7,.1,1), (.15,.75,1,1), (1,.3,.4,1), (.25,1,.5,1), (.8,.4,1,1)]
    for p, color in zip(target, colors):
        scene = env.renderer.scene
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, [.024]*3,
                           p, np.eye(3).ravel(), color)
        scene.ngeom += 1
    im = Image.fromarray(env.renderer.render().copy())
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, width, 52), fill=(18, 28, 33))
    draw.text((14, 9), f"ScaleBFM M | FIVE-POINT BODY CONTROL | t={env.data.time:.2f}s", fill="white")
    draw.text((14, 30), f"{phase} | no support / no grasp weld | dots: commanded body poses", fill=(211,227,233))
    return im


def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = LabEnv(assisted=False)
    writer = viewer = None
    records = []
    failure = None
    started = time.perf_counter()
    try:
        controller = ScaleBFMController(env, device=args.device)
        controller.reset(env)
        initial_pos, initial_quat = controller.poses(env)
        if args.targets:
            payload = json.loads(Path(args.targets).read_text())
            frames = payload["keyframes"]
        else:
            frames = direct_program(initial_pos, initial_quat)
        sequence = PoseSequence(frames)
        seconds = args.seconds if args.seconds is not None else sequence.times[-1]
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("Duration must be positive and finite")
        (output / "commands.json").write_text(json.dumps({
            "coordinate_frame": "world", "units": "metres; quaternion wxyz",
            "control_mode": 4, "bodies": FIVE_BODIES, "keyframes": frames,
        }, indent=2) + "\n")
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(output / "five_point.mp4"), fps=10,
                                        codec="libx264", quality=7, macro_block_size=8)
        if args.viewer:
            import mujoco.viewer as mjv
            viewer = mjv.launch_passive(env.model, env.data)
            viewer.cam.lookat[:] = (0, -.8, .76)
            viewer.cam.distance = 2.5; viewer.cam.azimuth = 25; viewer.cam.elevation = -12
        last_phase = None
        for i in range(round(seconds / .02)):
            t = env.data.time
            phase = sequence.phase(t)
            targets, quats = sequence.sample(t + .02 * controller.offsets)
            try:
                controller.advance(env, targets, quats)
            except RuntimeError as exc:
                failure = str(exc)
                break
            actual_p, actual_q = controller.poses(env)
            now_p, now_q = sequence.sample([env.data.time])
            error = np.linalg.norm(actual_p - now_p[0], axis=-1)
            angle = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(actual_q * now_q[0], axis=-1)), 0, 1)))
            records.append((env.data.time, actual_p, actual_q, now_p[0], now_q[0], error, angle,
                            env.data.qpos.copy(), env.data.qvel.copy()))
            if phase != last_phase:
                print(f"t={env.data.time:.2f} {phase}; errors cm={np.round(error*100, 1).tolist()}", flush=True)
                last_phase = phase
            if writer and i % 5 == 0:
                writer.append_data(np.asarray(frame(env, now_p[0], phase, args.camera)))
            if i in (249, 499, 649, 799):
                frame(env, now_p[0], phase, args.camera).save(output / f"pose_{i+1:04d}.png")
            if viewer:
                if not viewer.is_running():
                    failure = "Viewer closed before sequence completed"
                    break
                viewer.sync()
                time.sleep(.02)
        if writer:
            writer.close(); writer = None
        if records:
            ts = np.asarray([r[0] for r in records])
            actual = np.asarray([r[1] for r in records])
            errors = np.asarray([r[5] for r in records])
            angles = np.asarray([r[6] for r in records])
            np.savez_compressed(output / "trajectory.npz", time=ts, body_names=np.asarray(FIVE_BODIES),
                actual_position=actual, actual_quaternion_wxyz=np.asarray([r[2] for r in records]),
                target_position=np.asarray([r[3] for r in records]),
                target_quaternion_wxyz=np.asarray([r[4] for r in records]),
                position_error_m=errors, orientation_error_deg=angles,
                qpos=np.asarray([r[7] for r in records]), qvel=np.asarray([r[8] for r in records]))
            report = {
                "model": "ScaleBFM humanoid_transformer_m/model_22200.pt",
                "inference": "official original PyTorch weights", "device": args.device,
                "control_mode": 4, "controlled_points": list(FIVE_BODIES),
                "command_source": str(args.targets) if args.targets else "assistant-authored world-frame pose keyframes",
                "simulation_seconds": float(env.data.time), "requested_seconds": float(seconds),
                "completed": failure is None and len(records) == round(seconds/.02),
                "failure": failure, "finite": bool(np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()),
                "base_support": False, "active_equalities": int(np.count_nonzero(env.data.eq_active)),
                "policy_hz": 50, "physics_hz": 1/env.model.opt.timestep,
                "minimum_pelvis_z": float(actual[:,0,2].min()),
                "maximum_pelvis_z": float(actual[:,0,2].max()),
                "position_rmse_m": dict(zip(FIVE_BODIES, np.sqrt((errors**2).mean(0)).tolist())),
                "position_max_error_m": dict(zip(FIVE_BODIES, errors.max(0).tolist())),
                "orientation_rmse_deg": dict(zip(FIVE_BODIES, np.sqrt((angles**2).mean(0)).tolist())),
                "body_motion_range_xyz_m": dict(zip(FIVE_BODIES, np.ptp(actual, axis=0).tolist())),
                "inference_median_ms": float(np.median(controller.inference_ms)),
                "inference_p95_ms": float(np.percentile(controller.inference_ms, 95)),
                "wall_seconds": time.perf_counter()-started,
            }
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            p, _ = sequence.sample([env.data.time])
            frame(env, p[0], sequence.phase(env.data.time), args.camera).save(output / "final.png")
            Image.fromarray(env.render("reference", 1280, 800)).save(output / "laboratory.png")
            print(json.dumps(report, indent=2), flush=True)
        if failure:
            raise RuntimeError(failure)
    finally:
        if writer: writer.close()
        if viewer: viewer.close()
        env.close()
