"""Offline foot-motion audit; never used to generate robot commands."""
from pathlib import Path
import json
import mujoco
import numpy as np


def foot_motion(output_dir,axis=0,requested=.1):
    root=Path(__file__).resolve().parents[1]
    with np.load(Path(output_dir)/'trajectory.npz') as recording:
        time=recording['time'].copy();qpos=recording['qpos'].copy()
    initialization_path=Path(output_dir)/'initialization.json'
    initialization=json.loads(initialization_path.read_text()) if initialization_path.exists() else {}
    model=mujoco.MjModel.from_xml_path(str(root/'scenes'/initialization.get('scene','lab_g1_vision.xml')))
    data=mujoco.MjData(model)
    feet=[model.body(side+'_ankle_roll_link').id for side in ('left','right')]
    geoms=[np.flatnonzero((model.geom_bodyid==body)&(model.geom_type==mujoco.mjtGeom.mjGEOM_SPHERE)) for body in feet]
    floor=model.geom('lab_floor_slab').id
    floor_z=float(model.geom_pos[floor,2]+model.geom_size[floor,2])
    positions=[];clearance=[]
    for pose in qpos:
        data.qpos[:]=pose
        mujoco.mj_kinematics(model,data)
        positions.append(data.xpos[feet].copy())
        clearance.append([float(np.min(data.geom_xpos[g,2]-model.geom_size[g,0])-floor_z) for g in geoms])
    positions=np.asarray(positions);clearance=np.asarray(clearance)
    delta=positions[-1]-positions[0]
    active=time>=2.
    duration=np.sum((clearance>.005)&active[:,None],axis=0)*float(np.median(np.diff(time)))
    criteria=bool(np.all(delta[:,axis]>.5*requested) and np.all(duration>=.08-1e-9))
    return {'foot_displacement_m_left_right':delta.tolist(),
            'max_foot_bottom_clearance_m_left_right':np.max(clearance[active],axis=0).tolist(),
            'foot_airborne_above_5mm_s_left_right':duration.tolist(),
            'both_feet_stepped':criteria,
            'foot_motion_criteria':{'minimum_foot_axis_progress_fraction':.5,
                                    'minimum_sole_clearance_m':.005,
                                    'minimum_airborne_s_per_foot':.08,
                                    'offline_audit_hz':50}}
