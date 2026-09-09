"""Offline replay of recorded camera/encoder/IMU ScaleBFM task rollouts.

This module is not imported by the runtime controller. It reads independent
evaluator data only after a rollout and never advances simulation physics.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = (18, 28, 33)
TEXT = (228, 238, 242)
GREEN = (110, 245, 150)


def _json(path, fallback):
    return json.loads(path.read_text()) if path.exists() else fallback


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_run(output_dir, fps=None):
    """Replay all three views at the recorded state rate (normally 50 FPS).

    Robot camera panels are reconstructed views, not additional observations
    received by the controller. Actual observation timestamps remain visible.
    """
    output = Path(output_dir)
    initialization = _json(output / "initialization.json", {})
    report = _json(output / "report.json", {})
    evaluation = report.get("independent_evaluation", _json(output / "evaluator.json", {}))
    controller = report.get("controller", _json(output / "controller_result.json", {}))
    task = controller.get("task", evaluation.get("task", "task"))
    observation_times = np.asarray([
        float(decision["time"])
        for decision in _json(output / "visual_decisions.json", [])
    ])
    with np.load(output / "trajectory.npz") as saved:
        trajectory = {k: saved[k] for k in saved.files}
    with np.load(output / "controller_trace.npz") as saved:
        trace = {k: saved[k] for k in saved.files}
    times = np.asarray(trajectory["time"])
    if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
            or np.any(np.diff(times) <= 0)):
        raise ValueError("Recording must contain at least two increasing state timestamps")
    timestep = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), timestep, rtol=1e-5, atol=1e-8):
        raise ValueError("Recording must have uniformly spaced state timestamps")
    recorded_fps = round(1 / timestep, 6)
    fps = recorded_fps if fps is None else float(fps)
    if not np.isfinite(fps) or fps <= 0 or fps > recorded_fps + 1e-6:
        raise ValueError(f"fps must be positive and at most the {recorded_fps:g} Hz recording rate")
    scene = initialization.get("scene", "lab_g1_vision.xml")
    scene_path = Path(scene)
    if not scene_path.is_absolute():
        scene_path = ROOT / scene_path if scene_path.parts[0] == "scenes" else ROOT / "scenes" / scene_path
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    # Fixture placement is reconstructed only for the offline observer view.
    # Free-object placement is already part of each recorded qpos state.
    model.body_pos[model.body("analyzer").id] += np.asarray(
        initialization.get("button_shift_xyz", (0, 0, 0)), dtype=float)
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = (1.87, -1.02, .80)
    if task in ('grasp_lift','pick_place','push_chair') or task.startswith('step_'):
        camera.lookat[:] = trajectory['qpos'][0,:3] + np.array([.25,-.08,.04])
    camera.distance = 2.15
    camera.azimuth = 25
    camera.elevation = -17
    if task == 'push_chair':
        camera.lookat[:] = trajectory['qpos'][0,:3] + np.array([.40, 0., -.12])
        camera.distance = 2.6
        camera.azimuth = 115
        camera.elevation = -16
    option = mujoco.MjvOption()
    option.geomgroup[:] = (1, 1, 1, 1, 0, 1)
    option.sitegroup[:] = 0
    title_font, font, small = _font(21), _font(15), _font(13)
    initial_box = np.asarray(evaluation.get("initial_box_position", trajectory["sample_box"][0]))
    initial_bottle = np.asarray(evaluation.get('initial_bottle_position',trajectory['sample_bottle'][0]))
    target_body = {"button": "instrument_button", "push_box": "sample_box",
                   'grasp_lift':'sample_bottle','pick_place':'sample_bottle',
                   'push_chair':'push_chair'}.get(task)
    contacts = [c for c in evaluation.get("contacts", [])
                if c.get("hand_contact") and c.get("other_body") == target_body]
    if task == "button":
        contact_index = int(np.argmax(trajectory["button_travel"]))
    elif task in ('grasp_lift','pick_place'):
        contact_index = int(np.argmax(trajectory['sample_bottle'][:,2]))
    elif contacts:
        contact_time = .5 * (contacts[0]["first_time"] + contacts[0]["last_time"])
        contact_index = int(np.argmin(abs(times - contact_time)))
    else:
        contact_index = len(times) // 2
    if np.isclose(fps, recorded_fps, rtol=0, atol=1e-6):
        fps = recorded_fps
        # One output frame per state avoids float timestamp skips/duplicates.
        indices = np.arange(len(times))
    else:
        frame_times = np.arange(times[0], times[-1] + 1e-7, 1 / fps)
        indices = np.rint((frame_times - times[0]) / timestep).astype(int)
        indices = indices.clip(0, len(times) - 1)
        if indices[-1] != len(times) - 1:
            indices = np.r_[indices, len(times) - 1]
    renderer = mujoco.Renderer(model, width=640, height=480)
    rendered = {}

    def render_view(view):
        renderer.update_scene(data, camera=view, scene_option=option)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        return Image.fromarray(renderer.render().copy())

    def make_frame(index):
        now = float(times[index])
        data.qpos[:] = trajectory["qpos"][index]
        data.qvel[:] = trajectory["qvel"][index]
        data.time = now
        data.eq_active[:] = 0
        data.ctrl[:] = 0
        data.xfrc_applied[:] = 0
        data.qfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        canvas = Image.new("RGB", (960, 664), BACKGROUND)
        canvas.paste(render_view(camera), (0, 80))
        draw = ImageDraw.Draw(canvas)
        title = ('Encoders + IMU -> ScaleBFM | RGB monitoring' if task.startswith('step_')
                 else 'Camera + encoders + IMU -> ScaleBFM')
        draw.text((14, 10), title, font=title_font, fill=TEXT)
        task_label = {"button": "Press analyzer button", "push_box": "Push sample box",
                      'grasp_lift':'Grasp and lift sample bottle',
                      'pick_place':'Grasp, transport and release bottle',
                      'push_chair':'Push a free chair with passive rolling casters',
                      'step_forward':'Commanded forward stepping',
                      'step_side':'Commanded sideways stepping'}.get(task, task)
        draw.text((14, 40), f"{task_label}  |  recorded state t = {now:.2f} s", font=font, fill=TEXT)
        draw.text((14, 62), f"Offline replay | all views {fps:g} FPS", font=small, fill=(163, 193, 208))
        for side_i, (side, name) in enumerate((("Left", "vision_left"), ("Right", "vision_right"))):
            top = 80 + 240 * side_i
            raw = render_view(name).resize((320, 240), Image.Resampling.LANCZOS)
            canvas.paste(raw, (640, top))
            draw.rectangle((640, top, 960, top + 23), fill=BACKGROUND)
            draw.text((648, top + 3), f"{side} camera replay  t = {now:.2f} s",
                      font=small, fill=TEXT)
        past = observation_times[observation_times <= now + 1e-7]
        if len(past):
            last_observation = float(past[-1])
            observation_label = f"Controller last RGB: {last_observation:.2f} s | age {max(0., now - last_observation):.2f} s"
        else:
            observation_label = "Controller RGB: no observation yet"
        draw.text((430, 62), observation_label, font=small, fill=(163, 193, 208))
        ti = int(np.clip(np.searchsorted(trace["time"], now + 1e-7, side="right") - 1,
                         0, len(trace["time"]) - 1))
        phase = str(trace["phase"][ti]).replace("_", " ")
        draw.text((14, 571), f"Controller: {phase}", font=font, fill=TEXT)
        if task == "button":
            travel_mm = max(0., 1000 * float(trajectory["button_travel"][index]))
            metric = f"button travel {travel_mm:.1f} mm  |  press threshold 8 mm"
            color = GREEN if travel_mm > 8 else TEXT
        elif task == "push_box":
            displacement = trajectory["sample_box"][index] - initial_box
            moved_mm = 1000 * np.linalg.norm(displacement[:2])
            metric = f"box displacement {moved_mm:.1f} mm  |  target 40 mm"
            color = GREEN if moved_mm >= 40 else TEXT
        elif task in ('grasp_lift','pick_place'):
            delta=trajectory['sample_bottle'][index]-initial_bottle
            metric=f"bottle center lift {1000*delta[2]:.1f} mm | transport {1000*np.linalg.norm(delta[:2]):.1f} mm"
            color=GREEN if delta[2]>.03 else TEXT
        elif task.startswith('step_'):
            delta=trajectory['qpos'][index,:3]-trajectory['qpos'][0,:3]
            metric=f"root motion x {1000*delta[0]:.1f} mm | y {1000*delta[1]:.1f} mm"
            color=TEXT
        elif task == 'push_chair':
            delta = trajectory['push_chair'][index] - trajectory['push_chair'][0]
            chair_tilt = np.degrees(np.arccos(np.clip(
                data.body('push_chair').xmat.reshape(3, 3)[2, 2], -1, 1)))
            metric = f"chair forward {1000 * delta[0]:.1f} mm | target 80 mm | tilt {chair_tilt:.1f} deg"
            color = GREEN if delta[0] >= .08 else TEXT
        else:
            metric, color = "recorded state", TEXT
        draw.text((14, 599), f"Independent evaluator (offline): {metric}", font=font, fill=color)
        if index == len(times) - 1:
            physical = "passed" if evaluation.get("success") else "not passed"
            visual = "confirmed" if controller.get("visual_success") else "unconfirmed"
            detail = (f"Final: physical evaluation {physical}; commanded relative gait, RGB monitoring"
                      if task.startswith('step_') else
                      f"Final: physical evaluation {physical}; controller visual success {visual}")
        else:
            detail = "Evaluator measurements are absent from the controller inputs"
        draw.text((14, 628), detail, font=small, fill=(163, 193, 208))
        return canvas

    temporary_video = output / "task.rendering.mp4"
    try:
        with imageio.get_writer(str(temporary_video), fps=fps,
                                codec="libx264", quality=7, macro_block_size=8,
                                ffmpeg_params=["-movflags", "+faststart"]) as writer:
            for frame_i, index in enumerate(indices):
                image = make_frame(int(index))
                writer.append_data(np.asarray(image))
                if int(index) in (0, contact_index, len(times) - 1):
                    rendered[int(index)] = image
                if frame_i % 100 == 0:
                    print(f"{output.name}: rendered {frame_i + 1}/{len(indices)} frames", flush=True)
        for index, filename in ((0, "initial.png"), (contact_index, "contact.png"),
                                (len(times) - 1, "final.png")):
            image = rendered.get(index)
            if image is None:
                image = make_frame(index)
            image.save(output / filename)
            if filename == "contact.png":
                image.save(output / "interaction.png")
        temporary_video.replace(output / "task.mp4")
        metadata = {"offline_only": True, "physics_steps": 0, "fps": fps,
                    "frame_count": len(indices), "video": "task.mp4",
                    "recorded_state_fps": recorded_fps,
                    "recorded_state_count": len(times),
                    "all_views_synchronized": True,
                    "sensor_panels": "left/right camera views reconstructed from the same recorded state as the observer",
                    "sensor_panels_fps": fps,
                    "sensor_panels_are_controller_observations": False,
                    "controller_observation_times_s": observation_times.tolist(),
                    "observer_truth_overlays": "independent evaluator metrics only",
                    "contact_screenshot_time_s": float(times[contact_index]),
                    "recorded_duration_s": float(times[-1] - times[0]),
                    "scene": scene}
        (output / "render_report.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
    finally:
        renderer.close()
        temporary_video.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=None,
                        help="output FPS; default matches the recorded state rate (normally 50)")
    args = parser.parse_args()
    render_run(args.output_dir, args.fps)


if __name__ == "__main__":
    main()
