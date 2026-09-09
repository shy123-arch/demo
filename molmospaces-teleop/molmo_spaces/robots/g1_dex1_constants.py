"""Shared G1 + Dex1 joint and controller constants.

The body order below is the Unitree/low-level ("real") order used by
the source controller configuration. The ONNX action uses a different policy
order; the policy adapter must remap by name before sending a target to the
``body`` move group.
"""

from __future__ import annotations

import numpy as np

BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

POLICY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

DEFAULT_BODY_QPOS = np.array(
    [
        -0.28,
        0.0,
        0.0,
        0.5,
        -0.23,
        0.0,
        -0.28,
        0.0,
        0.0,
        0.5,
        -0.23,
        0.0,
        0.0,
        0.0,
        0.0,
        0.35,
        0.16,
        0.0,
        0.87,
        0.0,
        0.0,
        0.0,
        0.35,
        -0.16,
        0.0,
        0.87,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

BODY_KP = np.array(
    [
        99.09842777666113,
        99.09842777666113,
        40.17923847137318,
        99.09842777666113,
        28.50124619574858,
        28.50124619574858,
        99.09842777666113,
        99.09842777666113,
        40.17923847137318,
        99.09842777666113,
        28.50124619574858,
        28.50124619574858,
        40.17923847137318,
        28.50124619574858,
        28.50124619574858,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        8.611032447370201,
        8.611032447370201,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        14.25062309787429,
        8.611032447370201,
        8.611032447370201,
    ],
    dtype=np.float64,
)

BODY_KD = np.array(
    [
        6.3088018534966395,
        6.3088018534966395,
        2.5578897650279457,
        6.3088018534966395,
        1.814445686584846,
        1.814445686584846,
        6.3088018534966395,
        6.3088018534966395,
        2.5578897650279457,
        6.3088018534966395,
        1.814445686584846,
        1.814445686584846,
        2.5578897650279457,
        1.814445686584846,
        1.814445686584846,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.548195351665136,
        0.548195351665136,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.907222843292423,
        0.548195351665136,
        0.548195351665136,
    ],
    dtype=np.float64,
)

# Official Dex1-1 URDF lower limit.  At this position, the supplied collision
# meshes retain a measured 5.88 mm gap between pads.
DEX1_FINGER_URDF_LOWER = -0.02
# Sim-only closure trim: 3 mm extra travel per finger closes the empty jaw.
# Do not copy this value to hardware without a real Dex1-1 calibration.
DEX1_FINGER_LOWER = -0.023
DEX1_FINGER_UPPER = 0.0245
DEX1_SIDES = ("left", "right")


def dex1_joint_names(side: str) -> list[str]:
    if side not in DEX1_SIDES:
        raise ValueError(f"Unknown Dex1-1 side: {side!r}")
    return [f"{side}_dex1_finger_joint_1", f"{side}_dex1_finger_joint_2"]


def name_remap_indices(source_names: list[str], target_names: list[str]) -> np.ndarray:
    """Return indices that reorder an array from ``source`` to ``target`` names."""

    index = {name: i for i, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in index]
    if missing:
        raise ValueError(f"Missing joints while building remap: {missing}")
    return np.asarray([index[name] for name in target_names], dtype=np.int64)


POLICY_TO_BODY_INDICES = name_remap_indices(POLICY_JOINT_NAMES, BODY_JOINT_NAMES)
BODY_TO_POLICY_INDICES = name_remap_indices(BODY_JOINT_NAMES, POLICY_JOINT_NAMES)
