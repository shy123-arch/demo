# G1 + Dex1-1 motion tracking teleoperation

This repository contains a standalone G1 + Dex1-1 teleoperation integration for
MolmoSpaces. Runtime code, the ONNX checkpoint, one walk reference, and the final
MuJoCo robot asset are bundled in this checkout; sibling source repositories and a
pre-populated MolmoSpaces cache are not required.

## Control contract

- The tracking policy consumes a 1590D observation/history vector at 50 Hz.
- Its normalized 29D output is scaled, joint-name remapped, and added to the default
  G1 pose before the 500 Hz joint-space PD controller applies it.
- The pelvis is a physical free joint. Locomotion comes from leg torques, not mocap.
- Left and right Dex1-1 jaws are separate 2D action groups driven by PICO trigger/grip
  values; they do not change the tracking policy action dimension.
- Head, left wrist, right wrist, and third-person camera configurations are available
  for teleoperation capture.

Dex1 uses a simulation-only `-0.023 m` lower limit. The official `-0.020 m` limit left
a measured 5.88 mm collision-pad gap, so simulation adds 3 mm travel per finger. Do
not apply this correction to real hardware without calibration. Finger servos use
`kp=1000`, `kd=25`, and the original 20 N actuator force limit.

## Bundled resources

```text
assets/
├── motions/walk.npz
├── policies/motion_tracking/
│   ├── policy.json
│   ├── policy.onnx
│   └── policy.onnx.data
└── robots/g1_dex1/
    ├── model.xml
    └── meshes/
```

Verify the tracked resource hashes from the repository root:

```bash
cd assets
sha256sum -c SHA256SUMS
cd ..
```

The ONNX external-data file must remain next to `policy.onnx` with the exact name
`policy.onnx.data`.

## Offline verification

Use an environment with the repository installed in editable mode and MuJoCo enabled:

```bash
export PYTHONPATH=.
MUJOCO_GL=egl python scripts/smoke_g1_dex1_model.py
MUJOCO_GL=egl python scripts/smoke_motion_tracking.py
MUJOCO_GL=egl python scripts/smoke_vr_transport.py
```

Run a closed-loop walk without rendering:

```bash
MUJOCO_GL=egl python scripts/render_g1_dex1_walk.py \
  outputs/walk.mp4 --seconds 4 --no-video --gripper-cycle
```

Generate the four-camera video by removing `--no-video` and adding `--camera-grid`:

```bash
MUJOCO_GL=egl python scripts/render_g1_dex1_walk.py \
  outputs/walk_camera_grid.mp4 --seconds 8 --camera-grid --gripper-cycle
```

Dex1-specific checks are available separately:

```bash
MUJOCO_GL=egl python scripts/render_dex1_gripper_check.py
MUJOCO_GL=egl python scripts/test_dex1_small_handle.py
```

## PICO prerequisites

Online input still requires external hardware/runtime components:

1. PICO headset and trackers;
2. XRoboToolkit PC Service;
3. the callback-enabled `xrobotoolkit_sdk` Python binding;
4. `general_motion_retargeting` (GMR).

Check the Python environment before starting:

```bash
python -c 'import xrobotoolkit_sdk, general_motion_retargeting; print("PICO dependencies OK")'
```

These external components are not copied into this repository. The bridge itself is
bundled under `teleop/`.

## Start live teleoperation

Start only the PICO-to-reference bridge:

```bash
BRIDGE_PYTHON=python scripts/run_pose_bridge.sh
```

The default low-latency path sends the latest retargeted reference to
`127.0.0.1:28704/udp`. Override the destination for split-host deployment:

```bash
UDP_STREAM_HOST=192.168.1.20 UDP_STREAM_PORT=28704 scripts/run_pose_bridge.sh
```

Start the bridge and MolmoSpaces data collection together:

```bash
PYTHON=python BRIDGE_PYTHON=python MUJOCO_GL=egl scripts/run_live_teleop.sh
```

The launcher uses
`molmo_spaces.data_generation.config.g1_dex1_teleop_config:G1Dex1TeleopDataGenConfig`.
It terminates the bridge when simulation exits. The tracking source enters its
configured hold/fail-safe path when reference packets stop.

`PYTHON` selects the MolmoSpaces interpreter; `BRIDGE_PYTHON` may point to a separate
environment containing GMR and XRoboToolkit if those packages cannot coexist in the
simulation environment.

Ports used by the integration:

| Port | Protocol | Purpose |
|---:|---|---|
| 28701 | ZMQ/TCP | optional reference request channel |
| 28702 | ZMQ/TCP | optional reference reply channel |
| 28703 | ZMQ/TCP | controller button channel |
| 28704 | UDP | default low-latency pose + controller stream |

## Data collection configuration

`G1Dex1TeleopDataGenConfig` uses 20 ms policy steps and 2 ms controller/simulation
steps. Episodes are capped at 60 seconds and keep failed trials, which is useful for
raw teleoperation collection. The policy action dictionary contains:

```text
body:       float[29]
left_dex1:  float[2]
right_dex1: float[2]
```

MolmoSpaces records the configured head, wrist, and third-person camera observations
together with robot state and actions through its normal trajectory pipeline.

## Rebuilding assets

Normal use must not run the asset builder. The repository already includes the final
MJCF and referenced meshes. `scripts/build_g1_dex1_asset.py` remains a development
tool for intentional rebuilds and requires explicit upstream G1 and Dex1 source paths.

See `THIRD_PARTY_NOTICES.md` and `assets/robots/g1_dex1/SOURCE.md` before publishing or
redistributing the migrated code, checkpoint, motion, or robot resources.
