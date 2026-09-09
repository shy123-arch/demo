"""Unsupported relative stepping with ScaleBFM and support-foot odometry.

Commands are robot-relative gait motions. This is a short commanded stepping
probe, not autonomous visual navigation. RGB is saved for observation; no scene
poses, root translation or contacts enter the runtime controller.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .scalebfm_proprio import ProprioScaleBFM, _quat_apply, _quat_mul
from .vision_robot_io import VisionRobotIO
from .vision_servo import SensorPort, VisualServo


class SteppingScaleBFM(ProprioScaleBFM):
    def __init__(self,*args,**kwargs):
        self.support=None
        self.inertial_source=None
        self.velocity_correction=None
        super().__init__(*args,**kwargs)

    def set_support(self,side):
        if side not in (None,0,1):raise ValueError('support is left=0, right=1 or both=None')
        self._foot_anchors=self._body_positions[self._ankle_indices].copy()
        self.support=side

    def _set_sensor_state(self,sensors,*,initial=False):
        if initial:
            super()._set_sensor_state(sensors,initial=True)
            self._previous_rotated=self._body_positions.copy()
            return
        self.joint_pos,self.joint_vel,imu,self.gyro=sensors
        orientation=_quat_mul(self._inertial_to_initial,imu)
        if np.dot(orientation,self.root_quaternion)<0:orientation=-orientation
        self.root_quaternion=orientation/np.linalg.norm(orientation)
        local_pos,local_quat=self._forward_kinematics(self.joint_pos)
        rotated=_quat_apply(self.root_quaternion,local_pos)
        estimates=self._foot_anchors-rotated[self._ankle_indices]
        self.root_position=estimates.mean(axis=0) if self.support is None else estimates[self.support].copy()
        if self.inertial_source is not None:
            self.root_position[:2]=self.inertial_source()['position'][:2]
            if self.support is None and self.velocity_correction is not None:
                foot_velocity=(rotated[self._ankle_indices]-self._previous_rotated[self._ankle_indices])/.02
                self.velocity_correction(-np.mean(foot_velocity,axis=0))
        self._previous_rotated=rotated.copy()
        self._body_positions=rotated+self.root_position
        self._body_quaternions=_quat_mul(self.root_quaternion,local_quat)
        self.foot_anchor_residual=np.linalg.norm(self._body_positions[self._ankle_indices]-self._foot_anchors,axis=1)


class SteppingServo(VisualServo):
    def snapshot(self,label):
        left,right=self.port.stereo()
        index=len(self.decisions)
        for side,im in [('left',left),('right',right)]:
            Image.fromarray(im).save(self.out/f'{index:02d}_{label}_{side}.png')
        self.decisions.append({'time':self.time,'label':label,'use':'RGB monitoring only; relative gait command'})
        self._save()

    def gait(self,length=.08,pairs=1,sway=.045,lift=.035,seconds=.8,direction='forward'):
        result={'task':'step_'+direction,'status':'started','visual_success':False,
                'scope':'commanded relative steps; RGB monitoring; no autonomous navigation',
                'state_estimate':('accelerometer + gyro XY with encoder velocity correction during commanded double support; support foot for height'
                                  if self.policy.inertial_source is not None else
                                  'IMU + encoders, commanded support-foot switching'),
                'requested_step_length_m':length,'step_pairs':pairs,
                'requested_foot_lift_m':lift,'sway_m':sway,'swing_seconds':seconds}
        axis=0 if direction=='forward' else 1
        try:
            self.move(self.command_p,seconds=2.,label='stand')
            self.snapshot('initial')
            origin=self.command_p.copy()
            for cycle in range(pairs):
                for swing in ((0,1) if direction=='side' else (1,0)):
                    support=1-swing
                    shifted=self.command_p.copy()
                    midpoint=.5*(shifted[3]+shifted[4])
                    delta=midpoint[:2]-shifted[0,:2]
                    delta[1]+=sway if support==0 else -sway
                    shifted[:3,:2]+=delta
                    self.policy.set_support(None)
                    self.move(shifted,seconds=1.,label=f'shift_to_{support}_{cycle}')
                    self.policy.set_support(support)
                    goal=shifted.copy()
                    goal[3+swing,axis]=origin[3+swing,axis]+(cycle+1)*length
                    up=goal.copy();up[3+swing,2]+=lift
                    up[3+swing,axis]=.5*(shifted[3+swing,axis]+goal[3+swing,axis])
                    up[:3,axis]+=.25*length
                    self.move(up,seconds=seconds,label=f'swing_{swing}_up_{cycle}')
                    goal[:3]=up[:3]
                    goal[:3,axis]+=.25*length
                    self.move(goal,seconds=seconds,label=f'swing_{swing}_land_{cycle}')
                    self.move(self.command_p,seconds=.3,label='settle_new_foot')
                    self.policy.set_support(None)
                    self.snapshot(f'after_step_{cycle}_{swing}')
            centered=self.command_p.copy()
            midpoint=.5*(centered[3]+centered[4])
            centered[:3,:2]+=midpoint[:2]-centered[0,:2]
            self.move(centered,seconds=1.5,label='center_after_steps')
            if self.policy.inertial_source is not None:
                desired_root=centered[0].copy()
                for correction_index in range(4):
                    error=desired_root[:2]-self.policy.root_position[:2]
                    if np.linalg.norm(error)<.025:break
                    corrected=self.command_p.copy()
                    corrected[:3,:2]+=np.clip(error,-.02,.02)
                    self.move(corrected,seconds=.7,label=f'encoder_imu_goal_correction_{correction_index}')
            self.move(self.command_p,seconds=1.,label='final_stand')
            self.snapshot('final')
            result['status']='relative_gait_completed'
        except (RuntimeError,ValueError) as exc:
            result.update(status='stopped',failure=str(exc))
        finally:
            result['simulation_seconds']=self.time
            result['estimated_root_displacement_m']=self.policy.root_position.tolist()
            (self.out/'controller_result.json').write_text(json.dumps(result,indent=2)+'\n')
            if self.motion:
                np.savez_compressed(self.out/'controller_trace.npz',time=[x[0] for x in self.motion],
                    target_position=[x[1] for x in self.motion],target_quaternion=[x[2] for x in self.motion],
                    estimated_position=[x[3] for x in self.motion],estimated_quaternion=[x[4] for x in self.motion],
                    foot_residual=[x[5] for x in self.motion],phase=[x[6] for x in self.motion])
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--length',type=float,default=.08)
    parser.add_argument('--pairs',type=int,default=1)
    parser.add_argument('--sway',type=float,default=.045)
    parser.add_argument('--lift',type=float,default=.06)
    parser.add_argument('--swing-seconds',type=float,default=.7)
    parser.add_argument('--direction',choices=['forward','side'],default='forward')
    parser.add_argument('--inertial-xy',action='store_true')
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    policy=SteppingScaleBFM()
    io=VisionRobotIO(policy,station=(0.,-2.),output_dir=out)
    if args.inertial_xy:
        policy.inertial_source=io.imu_odometry
        policy.velocity_correction=io.correct_imu_velocity
    try:
        servo=SteppingServo(policy,SensorPort(io.sensors,io.step,io.images),out)
        result=servo.gait(args.length,args.pairs,args.sway,args.lift,args.swing_seconds,args.direction)
        # Only after controller termination, read the independent stored truth.
        evaluation=io.evaluate('standing')
        with np.load(out/'trajectory.npz') as z:
            displacement=z['qpos'][-1,:3]-z['qpos'][0,:3]
            final_q=z['qpos'][-1,3:7]
        up=_quat_apply(final_q,np.array([0.,0.,1.]))
        axis=0 if args.direction=='forward' else 1
        evaluation.update(task=result['task'],root_displacement_m=displacement.tolist(),
                          final_tilt_deg=float(np.degrees(np.arccos(np.clip(up[2],-1,1)))),
                          commanded_displacement_m=args.length*args.pairs)
        requested=args.length*args.pairs
        position_error=abs(displacement[axis]-requested)
        tolerance=max(.04,.25*abs(requested))
        objective=bool(displacement[axis]>.6*requested and position_error<=tolerance
                       and abs(displacement[1-axis])<.05)
        evaluation.update(objective_achieved=objective,goal_axis_error_m=float(position_error),
                          goal_tolerance_m=tolerance,
                          collision_free=not bool(evaluation['non_target_contacts']),
                          estimated_root_error_m=(displacement-policy.root_position).tolist())
        evaluation['success']=bool(result['status']=='relative_gait_completed' and evaluation['upright']
                                   and evaluation['unassisted'] and objective
                                   and evaluation['collision_free']
                                   and up[2]>np.cos(np.radians(15)))
        from .mobility_evaluation import foot_motion
        evaluation.update(foot_motion(out,axis,requested))
        evaluation['success'] &= evaluation['both_feet_stepped']
        evaluation['objective_achieved'] &= evaluation['both_feet_stepped']
        summary={'controller':result,'independent_evaluation':evaluation,
                 'model':'official ScaleBFM M/model_22200.pt','control_mode':4}
        (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'evaluator.json').write_text(json.dumps(evaluation,indent=2)+'\n')
        print(json.dumps({'controller':result,'evaluation':evaluation},indent=2),flush=True)
    finally:io.close()


if __name__=='__main__':main()
