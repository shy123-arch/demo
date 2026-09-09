# PICO Teleoperation for lab-sim

## Goal

Run the existing MolmoSpaces PICO/GMR whole-body teleoperation stack against
the lab-sim G1 model. The first version must use a free, unsupported pelvis,
track the operator at 50 Hz, walk through physical foot contacts, control both
Dex3 hands, and save a native trajectory.npz plus review videos.

HDF5 and training-pipeline integration are outside this version.

## Architecture

The existing PICO Unity application, XRoboToolkit PC service, and
teleop/pose_bridge.py remain unchanged. The bridge continues publishing the
latest GMR frame over UDP port 28704 and control events over ZMQ port 28703.

lab-sim adds a small adapter around the existing MolmoSpaces
MotionTrackingRuntime and bundled ONNX checkpoint:

    PICO -> XRoboToolkit -> pose_bridge/GMR
         -> UDP/ZMQ -> MolmoSpaces VRMotionSource
         -> 29D ONNX tracking policy
         -> lab-sim 29D PD torque control at 500 Hz
         -> free-base G1 + Dex3 in the lab scene

The adapter imports the motion-tracking package and assets from a configurable
MolmoSpaces checkout. It validates all 29 joint names before simulation starts;
there is no positional fallback for missing or reordered joints.

## Components

### LabSimStateBridge

Expose the lab-sim MuJoCo state using the interface required by
MotionTrackingRuntime: body joint position/velocity/torque, policy-order
arrays, pelvis position and wxyz quaternion, pelvis angular velocity, linear
acceleration, and the current reference anchor. Root XY is reset at each PICO
start so walking is relative to the robot's current lab position.

The lab model receives the tracking deployment's joint armature, damping,
friction, PD gains, and torque limits. Dex3 mass and inertia stay native to
lab-sim.

### Dex3 hand mapping

Each PICO controller independently drives one Dex3 hand. Trigger provides a
continuous open-to-close scalar. Holding grip boosts the scalar into the
stronger half of the closing range, matching the established MolmoSpaces
semantics. The scalar is expanded into mirrored, joint-limit-aware thumb,
index, and middle-finger targets. Stopping VR control holds the last safe hand
target.

### Teleoperation session

A new run.py pico-teleop command creates the lab scene with assistance and all
equality constraints disabled. It initializes the tracking policy at the
existing standing pose and runs:

- ONNX inference and PICO reference consumption at 50 Hz;
- body and finger PD torque application at 500 Hz;
- an optional passive MuJoCo viewer with real-time pacing;
- finite-state, pelvis-height, and stream-health checks.

PICO controls retain their existing meanings:

- right A: start or resume tracking;
- left X: stop tracking and hold safely;
- right B: end the current recording cleanly;
- triggers/grips: left and right Dex3 closure.

Loss of fresh PICO data uses the existing VRMotionSource hold fail-safe rather
than replaying stale motion.

## Recording

One row is recorded at every 50 Hz control tick. trajectory.npz contains at
least:

- simulation time and wall-clock time;
- full MuJoCo qpos and qvel;
- 29D PICO/GMR reference pose and root pose when available;
- 29D tracking-policy target;
- 29D applied body torque;
- 14D Dex3 target;
- controller buttons and stream-health fields;
- task-object state already recorded by lab-sim.

The output directory also contains session metadata and MP4s from selected lab
cameras. Video is synchronized by simulation timestamps and may be rendered
offline from the saved qpos, preserving control-loop speed.

## Failure handling

Startup fails before stepping if the Molmo checkout, ONNX external-data file,
tracking config, required sensors, actuators, or joint mappings are missing.
Runtime non-finite state, pelvis height below the safety threshold, explicit
user stop, or PICO stream timeout terminates or holds according to the
configured fail-safe and still flushes the recording.

UDP port conflicts produce an actionable error instead of silently running
without PICO input.

## Verification

Verification proceeds without PICO first, using the bundled walking reference
through the same state bridge and PD loop:

1. joint and observation-contract tests;
2. Dex3 mapping and button-edge tests;
3. unsupported standing smoke test;
4. closed-loop walk smoke test in the lab scene;
5. recording schema and offline replay test;
6. live PICO checklist confirming A/X/B, walking, turning, arm motion, both
   hands, stale-stream hold, and a saved replayable demo.

Success requires finite state, no support equality, pelvis height above the
fall threshold, measurable root translation during walking, and a valid
trajectory.npz plus video.
