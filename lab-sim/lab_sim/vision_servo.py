"""Camera-driven workstation actions; this module cannot access the simulator.

The port carries RGB, joint encoders and an IMU. Metric environment targets
derive from current or earlier RGB measurements and bounded task offsets.
Robot dimensions and the calibrated
camera fixture are fixed known hardware parameters. No world task plans are read.
"""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from .scalebfm_proprio import _quat_apply
from .stereo_perception import StereoPerception


@dataclass(frozen=True)
class SensorPort:
    sensors: object
    step: object
    stereo: object


class VisualServo:
    def __init__(self, policy, port, output_dir, *, blind=False):
        self.policy, self.port = policy, port
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.vision = StereoPerception()
        self.blind = blind
        self.time = 0.
        self.decisions, self.motion = [], []
        policy.reset(**port.sensors())
        self.home_p, self.home_q = policy.poses()
        self.command_p, self.command_q = self.home_p.copy(), self.home_q.copy()
        # Fixed calibration: columns are optical right/down/forward in torso.
        y = np.array([.342, 0, .940]); y /= np.linalg.norm(y)
        self.camera_rotation = np.column_stack(([0, -1, 0], -y, np.cross([0,-1,0], y) * -1))
        self.camera_translation = np.array([.12, .0325, .30])

    def camera_to_local(self, points):
        torso_p, torso_q = self.policy.torso_pose()
        return torso_p + _quat_apply(torso_q, np.asarray(points) @ self.camera_rotation.T + self.camera_translation)

    def observe(self, task, label):
        left, right = self.port.stereo()
        if self.blind:
            left, right = np.zeros_like(left), np.zeros_like(right)
        detection = self.vision.detect(left, right, task)
        point = detection['camera_point']
        detection['estimated_point'] = None if point is None else self.camera_to_local(point).tolist()
        corners = detection['diagnostics'].get('camera_corners')
        if corners is not None:
            detection['estimated_corners'] = self.camera_to_local(corners).tolist()
        face = detection['diagnostics'].get('white_side_face',{}).get('camera_point')
        if face is not None:
            detection['estimated_front_face_point'] = self.camera_to_local(face).tolist()
        index = len(self.decisions)
        prefix = f'{index:02d}_{label}'
        Image.fromarray(left).save(self.out/f'{prefix}_left.png')
        Image.fromarray(right).save(self.out/f'{prefix}_right.png')
        preview = Image.fromarray(left)
        draw = ImageDraw.Draw(preview)
        if detection['left_pixel'] is not None:
            x,y = detection['left_pixel'];draw.ellipse((x-7,y-7,x+7,y+7),outline=(255,70,30),width=2)
        draw.text((8,8),f'{label} | RGB only | t={self.time:.2f}',fill=(255,80,0))
        preview.save(self.out/f'{prefix}_annotated.png')
        entry = dict(time=self.time, label=label, detection=detection)
        self.decisions.append(entry)
        self._save()
        print(json.dumps({'time':round(self.time,2),'observation':label,
            'point':detection['estimated_point'],'confidence':detection['confidence'],
            'status':detection['diagnostics']['status']}),flush=True)
        return detection

    def _save(self):
        (self.out/'visual_decisions.json').write_text(json.dumps(self.decisions,indent=2)+'\n')

    def move(self, destination, orientation=None, seconds=1., label='move'):
        start_p, start_q = self.command_p.copy(), self.command_q.copy()
        end_p = np.asarray(destination).copy()
        end_q = start_q.copy() if orientation is None else np.asarray(orientation).copy()
        n = max(1, round(seconds/.02));duration=n*.02
        rotations = [Slerp([0,1],Rotation.from_quat(np.stack((start_q[b],end_q[b]))[:,[1,2,3,0]])) for b in range(5)]
        print(f't={self.time:.2f} {label} -> wrist {np.round(end_p[2],4).tolist()}',flush=True)
        for i in range(n):
            s = np.clip((i*.02 + self.policy.offsets*.02)/duration,0,1)
            s = np.clip(s**3*(10+s*(-15+6*s)),0,1)
            p = start_p[None]+s[:,None,None]*(end_p-start_p)[None]
            q = np.stack([r(s).as_quat()[:,[3,0,1,2]] for r in rotations],axis=1)
            desired,_ = self.policy.infer(p,q)
            self.port.step(desired)
            sensors = self.port.sensors()
            self.policy.update(**sensors)
            self.time += .02
            measured_p, measured_q = self.policy.poses()
            self.motion.append((self.time,p[0].copy(),q[0].copy(),measured_p,measured_q,
                                self.policy.foot_anchor_residual.copy(),label))
            up = _quat_apply(self.policy.root_quaternion,np.array([0,0,1.]))
            if up[2] < np.cos(np.radians(35)):
                raise RuntimeError('IMU tilt exceeded 35 degrees')
        self.command_p, self.command_q = end_p, end_q

    def wrist(self, point, seconds, label, quaternion=None):
        p,q = self.command_p.copy(),self.command_q.copy()
        p[2]=point
        if quaternion is not None:q[2]=quaternion
        self.move(p,q,seconds,label)

    def _align_wrist(self, target, label, rounds=2):
        # Correct policy tracking error using FK from encoders; bounded correction.
        for i in range(rounds):
            error = target-self.policy.poses()[0][2]
            if np.linalg.norm(error)<.008:break
            correction=np.clip(error,-.025,.025)
            self.wrist(self.command_p[2]+correction,.7,f'{label}_{i}')

    def run(self, task):
        result = {'task':task,'environment_inputs':['left_rgb','right_rgb'],
                  'robot_inputs':['joint_encoders','pelvis_gyro_integrated_attitude'],
                  'visual_success':False,'status':'started','manipulation_started':False}
        try:
            self.move(self.command_p,seconds=2.,label='stand_using_encoders_and_imu')
            before = self.observe(task,'initial')
            if before['estimated_point'] is None:
                result['status']='no_visual_target_hold'
                return result
            result['initial_visual_point']=before['estimated_point']
            if task=='button': self._button(before,result)
            elif task=='push_box': self._box(before,result)
            else: raise ValueError(task)
            return result
        except (RuntimeError,ValueError) as exc:
            result['status']='stopped';result['failure']=str(exc)
            return result
        finally:
            result['simulation_seconds']=self.time
            result['inference_median_ms']=float(np.median(self.policy.inference_ms))
            result['maximum_foot_anchor_residual_m']=float(np.max([np.max(x[5]) for x in self.motion])) if self.motion else 0
            (self.out/'controller_result.json').write_text(json.dumps(result,indent=2)+'\n')
            if self.motion:
                np.savez_compressed(self.out/'controller_trace.npz',
                    time=[x[0] for x in self.motion],target_position=[x[1] for x in self.motion],
                    target_quaternion=[x[2] for x in self.motion],estimated_position=[x[3] for x in self.motion],
                    estimated_quaternion=[x[4] for x in self.motion],foot_residual=[x[5] for x in self.motion],phase=[x[6] for x in self.motion])

    def _button(self,before,result):
        target=np.asarray(before['estimated_point'])
        tcp=np.array([.216,-.0062,.0285]) # Open index tip, robot geometry only.
        rotation=np.array([1.,0,0,0])
        wrist_goal=target-tcp
        raised=self.command_p[2].copy();raised[2]=max(raised[2],target[2]+.14)
        self.wrist(raised,1.5,'lift_in_aisle',rotation)
        fresh=self.observe('button','after_lift')
        if fresh['estimated_point'] is None:
            result['status']='target_lost_before_approach';return
        target=np.asarray(fresh['estimated_point']);wrist_goal=target-tcp
        standoff=wrist_goal.copy();standoff[0]-=.055
        above=standoff.copy();above[2]=raised[2]
        self.wrist(above,1.5,'move_above_visual_button')
        self.wrist(standoff,1.5,'lower_to_visual_button')
        self._align_wrist(standoff,'align_encoder_tcp')
        fresh=self.observe('button','pre_contact')
        if fresh['estimated_point'] is not None:
            updated=np.asarray(fresh['estimated_point'])
            if np.linalg.norm(updated-target)<.05:target=updated
        wrist_goal=target-tcp
        result['manipulation_started']=True
        result['command_source']='stereo button center plus known index-tip offset'
        press_observations=[]
        missing=0
        # A task-directed forward press, bounded to 25 mm beyond the visible face.
        for i,offset in enumerate((-.02,0.,.012,.025)):
            current=self.policy.poses()[0][2]
            desired=wrist_goal+np.array([offset,0,0])
            correction=np.clip(desired-current,-.025,.025)
            self.wrist(self.command_p[2]+correction,1.,f'bounded_press_{i}')
            obs=self.observe('button',f'press_{i}')
            if obs['estimated_point'] is not None:
                press_observations.append(obs['estimated_point'])
                missing=0
                # A successful visible movement ends pressing early. The
                # threshold is a task command, not evaluator feedback.
                if np.asarray(obs['estimated_point'])[0]-target[0]>.010:
                    break
            else:
                missing+=1
                if missing>=2:break
        retreat=self.command_p[2].copy();retreat[0]-=.10
        self.wrist(retreat,1.3,'horizontal_release')
        self.move(self.command_p,seconds=.5,label='release_settle')
        after=self.observe('button','after_release')
        result['final_visual_point']=after['estimated_point']
        result['press_visual_points']=press_observations
        travel=max([p[0]-target[0] for p in press_observations],default=0.)
        result['visual_button_travel_estimate_m']=travel
        release_error=None if after['estimated_point'] is None else float(np.linalg.norm(np.asarray(after['estimated_point'])-target))
        result['visual_release_error_m']=release_error
        result['visual_success']=bool(travel>.008 and release_error is not None and release_error<.004)
        result['status']='visual_press_and_release_observed' if result['visual_success'] else 'press_attempt_completed_visual_press_unverified'
        result['visual_measurement_note']='Stereo position change, without calibrated millimetre accuracy; independent physical scoring is separate.'

    def _box(self,before,result):
        target=np.asarray(before['estimated_point'])
        corners=before.get('estimated_corners')
        if corners is None:
            result['status']='box_edges_unavailable_hold';return
        corners=np.asarray(corners)
        # Visible near lid edge, with a small task probe below it. This probe is
        # not an assumed box dimension and is explicitly logged.
        edge=corners[np.argsort(corners[:,0])[:2]].mean(axis=0)
        contact=edge-np.array([0,0,.016])
        if before.get('estimated_front_face_point') is not None:
            contact=np.asarray(before['estimated_front_face_point'])
        rotation=np.array([np.sqrt(.5),np.sqrt(.5),0,0])
        tcp=_quat_apply(rotation,np.array([.216,-.0062,0]))
        wrist_goal=contact-tcp
        raised=self.command_p[2].copy();raised[2]=max(raised[2],target[2]+.16)
        self.wrist(raised,1.5,'lift_in_aisle',rotation)
        standoff=wrist_goal.copy();standoff[0]-=.045
        above=standoff.copy();above[2]=raised[2]
        self.wrist(above,1.5,'move_above_visual_box_edge')
        self.wrist(standoff,1.5,'lower_to_visual_box_side')
        self._align_wrist(standoff,'align_encoder_tcp')
        result['manipulation_started']=True
        result['contact_visual_point']=contact.tolist()
        result['command_source']='stereo lid near edge; 16mm bounded probe below edge, or measured front face'
        # Recompute fingertip-to-face error from fresh RGB and encoder FK after
        # every increment, while keeping the visual displacement goal fixed.
        current_contact=contact.copy()
        missing=0
        for i in range(9):
            measured_p,measured_q=self.policy.poses()
            current_tcp=measured_p[2]+_quat_apply(measured_q[2],np.array([.216,-.0062,0]))
            desired_tcp=current_contact+np.array([.015,0,0])
            error=np.clip(desired_tcp-current_tcp,-.025,.025)
            self.wrist(self.command_p[2]+error,1.,f'camera_corrected_box_push_{i}')
            obs=self.observe('push_box',f'push_{i}')
            if obs['estimated_point'] is None:
                missing+=1
                if missing>=2:break
                continue
            missing=0
            delta=np.asarray(obs['estimated_point'])-target
            if delta[0]>.055:break
            face=obs.get('estimated_front_face_point')
            if face is not None:
                measured_contact=np.asarray(face)
                if np.linalg.norm(measured_contact-current_contact)<.08:
                    current_contact=measured_contact
            else:
                # Relative lid movement preserves the already observed face
                # offset; stop if the lid itself is persistently occluded.
                current_contact=contact+delta
        retreat=self.command_p[2].copy();retreat[0]-=.12
        self.wrist(retreat,1.5,'horizontal_release_before_lifting')
        self.move(self.command_p,seconds=.5,label='box_settle')
        after=self.observe('push_box','after_release')
        result['final_visual_point']=after['estimated_point']
        if after['estimated_point'] is not None:
            delta=np.asarray(after['estimated_point'])-target
            result['visual_box_displacement_m']=delta.tolist()
            result['visual_success']=bool(delta[0]>=.040 and abs(delta[2])<.025)
            result['visual_success_note']='Heuristic >=40mm forward RGB displacement; stereo uncertainty is not calibrated.'
        result['status']='push_attempt_completed'
