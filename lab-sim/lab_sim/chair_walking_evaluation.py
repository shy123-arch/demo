"""Post-run evidence that stepping and physically pushing overlap in time."""
import json
from pathlib import Path

import mujoco
import numpy as np


def walking_contact_overlap(output_dir):
    out = Path(output_dir)
    root = Path(__file__).resolve().parents[1]
    scene = Path(json.loads((out / 'initialization.json').read_text())['scene'])
    if not scene.is_absolute():
        scene = root / 'scenes' / scene
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    with np.load(out / 'trajectory.npz') as z:
        times, poses = z['time'].copy(), z['qpos'].copy()
    with np.load(out / 'chair_audit.npz') as z:
        audit = {k: z[k].copy() for k in z.files}
    floor = model.geom('lab_floor_slab').id
    floor_z = model.geom_pos[floor, 2] + model.geom_size[floor, 2]
    feet = [model.body(side + '_ankle_roll_link').id for side in ('left', 'right')]
    geoms = [np.flatnonzero((model.geom_bodyid == foot) &
             (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)) for foot in feet]
    clearance = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        clearance.append([np.min(data.geom_xpos[g, 2] - model.geom_size[g, 0]) - floor_z
                          for g in geoms])
    clearance = np.asarray(clearance)
    # Hold the previous 50 Hz sole sample; contacts remain recorded at 500 Hz.
    index = np.searchsorted(times, audit['time'], side='right') - 1
    valid = index >= 0
    airborne = (clearance[np.maximum(index, 0)] > .005) & valid[:, None]
    active = (audit['time'] >= 2.) & audit['hand_contact'] & (audit['vx'] > .005)
    overlap = airborne & active[:, None]
    dt = float(np.median(np.diff(audit['time'])))
    duration = overlap.sum(axis=0) * dt
    travel = float(np.sum(np.maximum(np.diff(audit['x'], prepend=audit['x'][0]), 0.) *
                          np.any(overlap, axis=1)))
    passed = bool(np.all(duration >= .08 - 1e-9) and travel >= .01)
    return {
        'walking_hand_contact_and_chair_motion_overlap_s_left_right': duration.tolist(),
        'chair_forward_travel_during_airborne_hand_push_m': travel,
        'walking_push_temporal_overlap': passed,
        'walking_overlap_criteria': {
            'minimum_sole_clearance_m': .005,
            'minimum_chair_forward_speed_m_s': .005,
            'minimum_overlap_s_each_foot': .08,
            'minimum_chair_travel_during_overlap_m': .01,
            'sole_sample_hz': 50,
            'contact_sample_hz': 500,
            'postrun_only': True,
        },
    }
