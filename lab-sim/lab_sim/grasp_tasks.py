"""Camera-guided unsupported bottle grasp and transport experiments."""
import argparse
import json
from pathlib import Path

import numpy as np

from .grasp_perception import BottleStereo
from .scalebfm_proprio import ProprioScaleBFM, _quat_apply
from .vision_robot_io import VisionRobotIO
from .vision_servo import SensorPort, VisualServo


class GraspServo(VisualServo):
    def __init__(self, policy, port, output_dir, hand_command, *, blind=False,
                 grasp_offset=None, close_scale=1., transport_y=.10):
        super().__init__(policy,port,output_dir,blind=blind)
        self.vision=BottleStereo()
        self.hand_command=hand_command
        self.grasp_offset=None if grasp_offset is None else np.asarray(grasp_offset,dtype=float)
        self.close_scale=float(close_scale)
        self.transport_y=float(transport_y)
        self.hand_log=[]

    def command_hand(self,values,label):
        self.hand_command(values)
        self.hand_log.append(dict(time=self.time,label=label,targets=values))
        (self.out/'hand_commands.json').write_text(json.dumps(self.hand_log,indent=2)+'\n')

    def _grasp(self,task,result):
        # Robot-specific presets are independent of scene state.
        from .grasp_geometry import RIGHT_HAND_JOINT_NAMES, grasp_targets
        open_targets=dict.fromkeys(RIGHT_HAND_JOINT_NAMES,0.)
        self.command_hand(open_targets,'open')
        self.move(self.command_p,seconds=2.,label='stand_using_encoders_and_imu')
        before=self.observe(task,'initial')
        if before['estimated_point'] is None:
            result['status']='no_visual_target_hold';return
        target=np.asarray(before['estimated_point'])
        result['initial_visual_point']=target.tolist()
        result['visual_diameter_m']=before['diagnostics']['diameter_estimate_m']
        preshape=grasp_targets(result['visual_diameter_m'])
        offset=preshape.center_in_wrist if self.grasp_offset is None else self.grasp_offset
        quaternion=preshape.wrist_wxyz
        closed_targets=dict(zip(RIGHT_HAND_JOINT_NAMES,preshape.closed_q))
        result['command_grasp_offset_m']=offset.tolist()
        wrist_goal=target-_quat_apply(quaternion,offset)
        raised=self.command_p[2].copy()
        raised[2]=max(raised[2],target[2]+.16)
        self.wrist(raised,1.5,'raise_open_hand',quaternion)
        fresh=self.observe(task,'after_lift')
        if fresh['estimated_point'] is None:
            result['approach_uses_cached_rgb_target']=True
        else:
            target=np.asarray(fresh['estimated_point'])
        wrist_goal=target-_quat_apply(quaternion,offset)
        above=wrist_goal.copy();above[2]+= .16
        self.wrist(above,1.5,'open_hand_above_visual_bottle')
        self._align_wrist(above,'align_above',rounds=3)
        fresh=self.observe(task,'above_bottle')
        if fresh['estimated_point'] is not None:
            refreshed=np.asarray(fresh['estimated_point'])
            if np.linalg.norm(refreshed-target)<.06:
                target=refreshed
                wrist_goal=target-_quat_apply(quaternion,offset)
        self.wrist(wrist_goal,2.,'lower_open_gripper_around_bottle')
        self._align_wrist(wrist_goal,'align_grasp',rounds=3)
        preclose=self.observe(task,'before_closing')
        if preclose['estimated_point'] is not None:
            refreshed=np.asarray(preclose['estimated_point'])
            if np.linalg.norm(refreshed-target)<.04:
                new_goal=refreshed-_quat_apply(quaternion,offset)
                self._align_wrist(new_goal,'visual_grasp_correction',rounds=2)
                target=refreshed
        result['manipulation_started']=True
        for i in range(5):
            fraction=(i+1)/5
            command={name:float(open_targets[name]+fraction*(angle*self.close_scale-open_targets[name]))
                     for name,angle in closed_targets.items()}
            self.command_hand(command,f'close_{i}')
            self.move(self.command_p,seconds=.3,label=f'close_fingers_{i}')
        self.move(self.command_p,seconds=.5,label='settle_grip')
        self.observe(task,'closed_grip')
        lifted=self.command_p[2].copy();lifted[2]+=.10
        self.wrist(lifted,2.,'lift_bottle_attempt')
        self.move(self.command_p,seconds=1.,label='hold_lift')
        lifted_view=self.observe(task,'lifted')
        if lifted_view['estimated_point'] is not None:
            measured_lift=float(np.asarray(lifted_view['estimated_point'])[2]-target[2])
        else:
            measured_lift=None
        result['visual_lift_m']=measured_lift
        # Visual success is conservative under hand occlusion; no contact/truth
        # or physical evaluator signal is available to this controller.
        result['visual_success']=bool(measured_lift is not None and measured_lift>.03)
        if task=='pick_place':
            moved=self.command_p[2].copy();moved[1]+=self.transport_y
            self.wrist(moved,2.,'transport_held_bottle')
            transported=self.observe(task,'after_transport')
            if transported['estimated_point'] is not None:
                later_lift=float(np.asarray(transported['estimated_point'])[2]-target[2])
                result['visual_lift_after_transport_m']=later_lift
                result['visual_success']=bool(result['visual_success'] or later_lift>.03)
            lowered=self.command_p[2].copy();lowered[2]-=.10
            self.wrist(lowered,2.,'lower_to_initial_visual_support_height')
            self.move(self.command_p,seconds=.5,label='settle_before_release')
            self.command_hand(open_targets,'release')
            self.move(self.command_p,seconds=1.,label='open_fingers_release')
            retreat=self.command_p[2].copy();retreat[2]+=.15
            self.wrist(retreat,1.5,'withdraw_open_hand')
            self.move(self.command_p,seconds=1.,label='placed_object_settle')
            final=self.observe(task,'after_release')
            result['final_visual_point']=final['estimated_point']
            if final['estimated_point'] is not None:
                delta=np.asarray(final['estimated_point'])-np.asarray(before['estimated_point'])
                result['visual_transport_delta_m']=delta.tolist()
                result['visual_success']=bool(result['visual_success'] and np.linalg.norm(delta[:2])>=.08
                                              and abs(delta[2])<.025)
            else:result['visual_success']=False
        result['status']='visual_grasp_observed' if result['visual_success'] else 'grasp_attempt_completed_visual_unconfirmed'

    def run(self,task):
        result={'task':task,'environment_inputs':['left_rgb','right_rgb'],
                'robot_inputs':['joint_encoders','pelvis_gyro_integrated_attitude'],
                'visual_success':False,'status':'started','manipulation_started':False,
                'command_grasp_offset_m':None if self.grasp_offset is None else self.grasp_offset.tolist()}
        try:
            self._grasp(task,result)
        except (RuntimeError,ValueError) as exc:
            result.update(status='stopped',failure=str(exc))
        finally:
            result['simulation_seconds']=self.time
            result['inference_median_ms']=float(np.median(self.policy.inference_ms)) if self.policy.inference_ms else None
            (self.out/'controller_result.json').write_text(json.dumps(result,indent=2)+'\n')
            if self.motion:
                np.savez_compressed(self.out/'controller_trace.npz',
                    time=[x[0] for x in self.motion],target_position=[x[1] for x in self.motion],
                    target_quaternion=[x[2] for x in self.motion],estimated_position=[x[3] for x in self.motion],
                    estimated_quaternion=[x[4] for x in self.motion],foot_residual=[x[5] for x in self.motion],
                    phase=[x[6] for x in self.motion])
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['grasp_lift','pick_place'],default='grasp_lift')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--station',type=float,nargs=2,default=[1.62,-.72])
    parser.add_argument('--grasp-offset',type=float,nargs=3,default=None)
    parser.add_argument('--bottle-shift',type=float,nargs=3,default=[0,0,0])
    parser.add_argument('--close-scale',type=float,default=1.)
    parser.add_argument('--transport-y',type=float,default=.10)
    parser.add_argument('--blind',action='store_true')
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    policy=ProprioScaleBFM()
    io=VisionRobotIO(policy,station=args.station,output_dir=out,bottle_shift=args.bottle_shift)
    try:
        servo=GraspServo(policy,SensorPort(io.sensors,io.step,io.images),out,io.set_hand_targets,
                         blind=args.blind,grasp_offset=args.grasp_offset,close_scale=args.close_scale,
                         transport_y=args.transport_y)
        result=servo.run(args.task)
        evaluation=io.evaluate(args.task)
        summary={'controller':result,'independent_evaluation':evaluation,
                 'control_mode':4,'model':'official ScaleBFM M/model_22200.pt'}
        (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({'controller':result,'evaluation':evaluation},indent=2),flush=True)
    finally:io.close()


if __name__=='__main__':main()
