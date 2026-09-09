# PICO Teleoperation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Connect the proven MolmoSpaces PICO/GMR tracking stack to the unsupported lab-sim G1 and save native replayable demos.

**Architecture:** Add a lab-specific state bridge that satisfies MotionTrackingRuntime without changing the PICO sender or tracking policy. Run the policy at 50 Hz, apply body and Dex3 PD torques at 500 Hz, and record one synchronized NPZ row per policy tick for offline video rendering.

**Tech Stack:** Python 3.12, MuJoCo 3.3.7, NumPy, SciPy, ONNX Runtime, PyYAML, pyzmq, imageio.

**Spec:** docs/superpowers/specs/2026-09-09-pico-teleop-design.md

## Global Constraints

- The pelvis is always free and unsupported; assistance and equality constraints are disabled.
- Reuse the existing MolmoSpaces PICO/GMR, VRMotionSource, tracking config, and ONNX checkpoint.
- Physics runs at 500 Hz and policy inference at 50 Hz.
- Output is lab-sim trajectory.npz, metadata JSON, and optional offline MP4; no HDF5.
- Validate all 29 body joints by name before stepping.
- Keep the existing PICO button meanings: right A start/resume, left X stop/hold, right B finish demo.
- Run only focused tests and one short offline walk smoke test.
- lab-sim is not a Git worktree, so commit steps are intentionally omitted.

---

### Task 1: Tracking state bridge and Dex3 mapper

**Files:**
- Create: lab_sim/pico_tracking.py
- Test: tests/test_pico_tracking.py

**Interfaces:**
- Produces: resolve_molmo_root(path: str | Path | None) -> Path
- Produces: load_molmo_tracking(path) -> MolmoTrackingBindings
- Produces: LabSimStateBridge(model, data, bindings)
- Produces: Dex3HandMapper(model, data)
- Produces: BodyPDController(model, data, bindings)

- [x] **Step 1: Write focused contract tests**

Create tests that compile scenes/lab_g1_vision.xml and assert:

    bridge.qj_real.shape == (29,)
    bridge.qj_isaac.shape == (29,)
    bridge.current_reference_anchor()["root_quat"].shape == (4,)
    mapper.targets.shape == (14,)
    open_targets are all zero
    close targets remain inside each joint range
    left/right trigger commands only change their own hand

- [x] **Step 2: Run the tests and confirm the module is absent**

Run:

    MUJOCO_GL=osmesa .venv/bin/python -m unittest tests.test_pico_tracking -v

Expected: import failure for lab_sim.pico_tracking.

- [x] **Step 3: Implement Molmo checkout loading**

Resolve, in order: explicit --molmo-root, MOLMO_TELEOP_ROOT, then the sibling
demo/molospace/molmospaces-teleop/molmospaces-teleop checkout. Validate:

    configs/motion_tracking_live.yaml
    assets/policies/motion_tracking/policy.onnx
    assets/policies/motion_tracking/policy.onnx.data

Prepend the resolved checkout to sys.path and import MotionTrackingRuntime plus
BODY_JOINT_NAMES, POLICY_JOINT_NAMES, DEFAULT_BODY_QPOS, BODY_KP, BODY_KD, and
BODY_TO_POLICY_INDICES.

- [x] **Step 4: Implement LabSimStateBridge**

Map qpos, qvel, actuator torque and the free joint by exact names. Read
imu-pelvis-angular-velocity and imu-pelvis-linear-acceleration when present,
falling back to free-joint velocity only for the angular rate. Implement:

    update() -> None
    current_reference_anchor() -> dict[str, np.ndarray]
    reset_root_xy_origin(reason: str) -> None
    set_hand_from_controller_buttons(buttons: dict) -> None
    handle_vr_record_buttons(buttons: dict) -> None

The record-button handler latches a finish request on the rising edge of
right_key_two. The hand handler delegates to Dex3HandMapper.

- [x] **Step 5: Implement Dex3HandMapper and BodyPDController**

For each hand compute close_amount as:

    trigger = clip(side_trigger_value, 0, 1)
    grip = clip(side_grip_value, 0, 1)
    close_amount = 0.5 + 0.5 * trigger if grip > 1e-4 else trigger

Expand close_amount linearly from zero to 55 percent of the farther signed
joint limit, independently for the seven named joints on each side.

BodyPDController transfers shared joint armature, damping, and frictionloss
from assets/robots/g1_dex1/model.xml, applies BODY_KP/BODY_KD, and clips torque
to lab-sim joint actuator-force limits.

- [x] **Step 6: Run the focused tests**

Run:

    MUJOCO_GL=osmesa .venv/bin/python -m unittest tests.test_pico_tracking -v

Expected: all Task 1 tests pass.

---

### Task 2: Teleoperation loop and native recording

**Files:**
- Create: lab_sim/pico_teleop.py
- Test: tests/test_pico_teleop.py

**Interfaces:**
- Consumes: Task 1 bridge, mapper, body PD controller, and Molmo bindings.
- Produces: PicoTeleopSession
- Produces: run_pico_teleop(args) -> dict
- Produces: render_pico_demo(output_dir, camera, width, height, fps) -> Path

- [x] **Step 1: Write recorder and offline-mode tests**

Use a temporary output directory. Run a 0.1 second transport-disabled session
with the default reference and assert:

    trajectory.npz exists
    time, qpos, qvel, body_target, body_torque, hand_targets exist
    all arrays have equal nonzero leading length
    metadata.json reports assisted=false, physics_hz=500, policy_hz=50
    every state and command array is finite

- [x] **Step 2: Run the tests and confirm the session is absent**

Run:

    MUJOCO_GL=osmesa .venv/bin/python -m unittest tests.test_pico_teleop -v

Expected: import failure for lab_sim.pico_teleop.

- [x] **Step 3: Implement PicoTeleopSession initialization**

Compile LabEnv(scene="lab_g1_vision.xml", assisted=False), disable every
equality constraint, place the pelvis at the requested start XY/yaw, set the
29 body joints to DEFAULT_BODY_QPOS and all Dex3 joints to zero, then construct:

    LabSimStateBridge
    MotionTrackingRuntime(motion_source="vr", enable_transport=True)
    BodyPDController
    native NPZ recorder

Support transport-disabled mode for offline smoke tests and a queued named
motion such as walk.

- [x] **Step 4: Implement the 500/50 Hz loop**

At each 20 ms policy boundary:

    body_target = runtime.step()
    record current reference, buttons and stream health

For each of ten 2 ms physics ticks:

    data.ctrl[:] = 0
    body_torque = body_pd.compute(body_target)
    finger_torque = kp * (hand_target - q) - kd * dq
    apply clipped body and finger torques
    mujoco.mj_step(model, data)

Exit cleanly on duration, right-B finish request, viewer close, KeyboardInterrupt,
non-finite state, or pelvis z below 0.45 m. Always close transport and flush
recording in finally.

- [x] **Step 5: Implement NPZ and metadata output**

Save:

    time, wall_time, qpos, qvel
    reference_joint_pos, reference_root_pos, reference_root_quat
    body_target, body_torque, hand_targets
    controller_button_values, stream_active
    sample_box, sample_bottle, button_travel, drawer_open

Save joint-name arrays and a metadata.json with scene, start pose, policy/config
paths, rates, stop reason, duration, minimum pelvis height, displacement, and
assistance/equality audit.

- [x] **Step 6: Implement optional viewer and offline MP4**

The viewer uses mujoco.viewer.launch_passive and real-time wall-clock pacing.
Offline rendering restores each saved qpos into the same scene and writes the
selected camera to task.mp4 with imageio. Rendering never runs inside the
500 Hz torque loop unless explicitly requested.

- [x] **Step 7: Run the recorder test**

Run:

    MUJOCO_GL=osmesa .venv/bin/python -m unittest tests.test_pico_teleop -v

Expected: all Task 2 tests pass.

---

### Task 3: CLI, usage documentation, and short smoke test

**Files:**
- Modify: run.py
- Modify: README.md
- Modify: docs/superpowers/plans/2026-09-09-pico-teleop.md

**Interfaces:**
- Consumes: run_pico_teleop and render_pico_demo from Task 2.
- Produces: run.py pico-teleop and run.py pico-render commands.

- [x] **Step 1: Add CLI commands**

Add pico-teleop arguments:

    --molmo-root
    --output-dir
    --scene
    --start-x
    --start-y
    --start-yaw
    --seconds
    --viewer
    --offline-motion
    --render-after

Add pico-render arguments for output directory, camera, FPS, width and height.

- [x] **Step 2: Add the exact operator workflow to README**

Document two terminals:

    Terminal 1:
    cd demo/molospace/molmospaces-teleop/molmospaces-teleop
    scripts/run_pose_bridge.sh

    Terminal 2:
    cd demo/lab-sim
    .venv/bin/python run.py pico-teleop --viewer --output-dir outputs/pico_demo

Document A/X/B and trigger/grip controls, UDP 28704 conflict handling, and:

    .venv/bin/python run.py pico-render --output-dir outputs/pico_demo

- [x] **Step 3: Run syntax and targeted tests**

Run:

    .venv/bin/python -m py_compile run.py lab_sim/pico_tracking.py lab_sim/pico_teleop.py
    MUJOCO_GL=osmesa .venv/bin/python -m unittest tests.test_pico_tracking tests.test_pico_teleop -v

Expected: compile succeeds and focused tests pass.

- [x] **Step 4: Run one short unsupported walk smoke test**

Run:

    MUJOCO_GL=osmesa .venv/bin/python run.py pico-teleop --offline-motion walk --seconds 2 --output-dir /tmp/lab-sim-pico-smoke

Expected: process exits successfully, all state remains finite, assistance is
false, and trajectory.npz is readable. This is the only physics smoke test.

- [x] **Step 5: Render the saved smoke trajectory**

Run:

    MUJOCO_GL=osmesa .venv/bin/python run.py pico-render --output-dir /tmp/lab-sim-pico-smoke --width 640 --height 400

Expected: a readable MP4 with at least one frame.

- [x] **Step 6: Mark completed plan checkboxes**

Update this plan only for steps actually completed. Report the live-PICO test
as pending because it requires the user's headset and XRoboToolkit service.

## Execution status

Implementation, focused unit tests, unsupported offline walk smoke, live
UDP/ZMQ receiver initialization, NPZ loading, and MP4 decoding are complete.
The only pending verification is an operator-driven PICO session with the
headset and XRoboToolkit service supplying real frames.
