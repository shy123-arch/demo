"""RGB-guided chair pushing while taking relative ScaleBFM steps.

Runtime environment inputs are stereo RGB only. The initial robot/chair
workstation is preset by the simulator adapter. Gait uses encoders and IMU;
no contact, chair pose, root truth, or evaluator signal enters this controller.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .scalebfm_mobility import SteppingScaleBFM
from .vision_robot_io import VisionRobotIO
from .vision_servo import SensorPort, VisualServo
from .scalebfm_proprio import _quat_mul


class ChairServo(VisualServo):
    def __init__(self, *args, length=.15, pairs=2, sway=.12, lift=.20,
                 swing_seconds=1., toe_up_deg=15., apex_hold=.3, **kwargs):
        super().__init__(*args, **kwargs)
        from .chair_perception import ChairStereo
        self.vision = ChairStereo()
        self.length, self.pairs = length, pairs
        self.sway, self.lift, self.swing_seconds = sway, lift, swing_seconds
        self.toe_up_deg, self.apex_hold = toe_up_deg, apex_hold

    def hand_goals(self, detection):
        # Contact points come from stereo top-edge measurements, inset toward
        # the middle and 13 cm below the visible edge (a bounded task offset).
        top = np.asarray(detection['estimated_point'])
        corners = detection.get('estimated_corners')
        span = .12
        if corners is not None:
            span = min(.15, .27 * float(np.ptp(np.asarray(corners)[:, 1])))
        goals = np.repeat(top[None], 2, axis=0)
        goals[:, 2] -= .13
        goals[:, 1] += (span, -span)
        # Known robot geometry: the open index fingertip in an upright wrist.
        goals -= np.array([[.216, .0062, .0285], [.216, -.0062, .0285]])
        return goals

    def two_hands(self, goals, seconds, label):
        p, q = self.command_p.copy(), self.command_q.copy()
        p[1:3] = goals
        q[1:3] = (1., 0., 0., 0.)
        self.move(p, q, seconds, label)

    def align_hands(self, goals, rounds=3):
        for i in range(rounds):
            error = goals - self.policy.poses()[0][1:3]
            if np.max(np.abs(error)) < .009:
                break
            self.two_hands(self.command_p[1:3] + np.clip(error, -.025, .025),
                           .6, f'encoder_hand_alignment_{i}')

    def walk_push(self, result):
        origin = self.command_p.copy()
        for cycle in range(self.pairs):
            for swing in (1, 0):
                support = 1 - swing
                # Observe from double support with the torso centered laterally,
                # keeping the chair corners in the common stereo field of view.
                centered = self.command_p.copy()
                centered[0, 1] = .5 * (centered[3, 1] + centered[4, 1])
                self.policy.set_support(None)
                self.move(centered, seconds=.8, label='center_for_stereo')
                observed = self.observe('push_chair', f'before_step_{cycle}_{swing}')
                if observed['estimated_point'] is None:
                    result['status'] = 'target_lost_stop_walking'
                    return
                measured = self.hand_goals(observed)
                measured[:, 0] += .012
                error = measured - self.policy.poses()[0][1:3]
                corrected = self.command_p[1:3] + np.clip(error, -.025, .025)
                self.two_hands(corrected, .5, 'refresh_visual_hand_contact')

                shifted = self.command_p.copy()
                midpoint = .5 * (shifted[3] + shifted[4])
                shifted[0, :2] = midpoint[:2]
                shifted[0, 1] += self.sway if support == 0 else -self.sway
                self.policy.set_support(None)
                self.move(shifted, seconds=1., label=f'weight_shift_{support}_{cycle}')
                self.policy.set_support(support)
                goal = shifted.copy()
                goal[3 + swing, 0] = origin[3 + swing, 0] + (cycle + 1) * self.length
                up = goal.copy()
                up[3 + swing, 2] += self.lift
                up[3 + swing, 0] = .5 * (shifted[3 + swing, 0] + goal[3 + swing, 0])
                up[:3, 0] += .25 * self.length
                landing_q = self.command_q.copy()
                up_q = landing_q.copy()
                angle = np.radians(-self.toe_up_deg)
                up_q[3 + swing] = _quat_mul(
                    self.home_q[3 + swing], np.array([np.cos(angle / 2), 0., np.sin(angle / 2), 0.]))
                self.move(up, up_q, seconds=self.swing_seconds, label=f'push_swing_{swing}_up_{cycle}')
                if self.apex_hold > 0:
                    self.move(up, up_q, seconds=self.apex_hold, label=f'push_swing_{swing}_clearance_hold_{cycle}')
                goal[:3] = up[:3]
                goal[:3, 0] += .25 * self.length
                self.move(goal, landing_q, seconds=self.swing_seconds, label=f'push_swing_{swing}_land_{cycle}')
                self.move(self.command_p, seconds=.3, label='settle_new_foot')
                self.policy.set_support(None)

        centered = self.command_p.copy()
        centered[0, :2] = .5 * (centered[3, :2] + centered[4, :2])
        self.move(centered, seconds=1.5, label='center_after_push_steps')
        result['status'] = 'walking_push_sequence_completed'

    def run(self, task='push_chair'):
        result = dict(task=task, status='started', visual_success=False,
                      manipulation_started=False, environment_inputs=['left_rgb', 'right_rgb'],
                      robot_inputs=['joint_encoders', 'native_accelerometer', 'native_gyro'],
                      scope='preset nearby workstation; RGB guided hands plus relative forward stepping',
                      requested_step_length_m=self.length, step_pairs=self.pairs,
                      commanded_sway_m=self.sway, commanded_foot_lift_m=self.lift,
                      commanded_toe_up_deg=self.toe_up_deg, apex_hold_s=self.apex_hold,
                      requested_root_progress_m=self.length*self.pairs)
        try:
            self.move(self.command_p, seconds=2., label='stand_using_encoders_and_imu')
            before = self.observe(task, 'initial')
            if before['estimated_point'] is None:
                result['status'] = 'no_visual_target_hold'
                self.move(self.command_p, seconds=2., label='blind_hold')
                return result
            result['initial_visual_point'] = before['estimated_point']
            goals = self.hand_goals(before)
            raised = self.command_p[1:3].copy()
            raised[:, 2] = goals[:, 2]
            self.two_hands(raised, 1.5, 'raise_both_open_hands')
            fresh = self.observe(task, 'after_raise')
            if fresh['estimated_point'] is None:
                result['status'] = 'target_lost_before_contact'
                return result
            goals = self.hand_goals(fresh)
            goals[:, 0] -= .035
            self.two_hands(goals, 1.5, 'approach_visual_chair_back')
            self.align_hands(goals)
            goals[:, 0] += .045
            self.align_hands(goals, rounds=3)
            result['manipulation_started'] = True
            self.walk_push(result)
            retreat = self.command_p[1:3].copy()
            retreat[:, 0] -= .12
            self.two_hands(retreat, 1.5, 'release_chair_back')
            self.move(self.command_p, seconds=2., label='stand_after_release')
            after = self.observe(task, 'final')
            result['final_visual_point'] = after['estimated_point']
            if after['estimated_point'] is not None:
                delta = np.asarray(after['estimated_point']) - np.asarray(before['estimated_point'])
                result['visual_chair_delta_m'] = delta.tolist()
                result['visual_success'] = bool(delta[0] >= .10 and abs(delta[1]) < .10)
        except (RuntimeError, ValueError) as exc:
            result.update(status='stopped', failure=str(exc))
        finally:
            result['simulation_seconds'] = self.time
            result['estimated_root_displacement_m'] = self.policy.root_position.tolist()
            result['inference_median_ms'] = float(np.median(self.policy.inference_ms)) if self.policy.inference_ms else None
            (self.out/'controller_result.json').write_text(json.dumps(result, indent=2)+'\n')
            if self.motion:
                np.savez_compressed(self.out/'controller_trace.npz', time=[x[0] for x in self.motion],
                    target_position=[x[1] for x in self.motion], target_quaternion=[x[2] for x in self.motion],
                    estimated_position=[x[3] for x in self.motion], estimated_quaternion=[x[4] for x in self.motion],
                    foot_residual=[x[5] for x in self.motion], phase=[x[6] for x in self.motion])
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--station', nargs=2, type=float, default=[-.08, -2.])
    parser.add_argument('--length', type=float, default=.15)
    parser.add_argument('--pairs', type=int, default=2)
    parser.add_argument('--sway', type=float, default=.12)
    parser.add_argument('--lift', type=float, default=.20)
    parser.add_argument('--swing-seconds', type=float, default=1.)
    parser.add_argument('--toe-up-deg', type=float, default=15.)
    parser.add_argument('--apex-hold', type=float, default=.3)
    parser.add_argument('--blind', action='store_true')
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    policy = SteppingScaleBFM()
    io = VisionRobotIO(policy, station=args.station, output_dir=out, scene='lab_g1_chair.xml')
    policy.inertial_source = io.imu_odometry
    policy.velocity_correction = io.correct_imu_velocity
    try:
        servo = ChairServo(policy, SensorPort(io.sensors, io.step, io.images), out,
                           blind=args.blind, length=args.length, pairs=args.pairs,
                           sway=args.sway, lift=args.lift, swing_seconds=args.swing_seconds,
                           toe_up_deg=args.toe_up_deg, apex_hold=args.apex_hold)
        result = servo.run()
        evaluation = io.evaluate('push_chair')
        from .mobility_evaluation import foot_motion
        evaluation.update(foot_motion(out, 0, args.length*args.pairs))
        from .chair_walking_evaluation import walking_contact_overlap
        evaluation.update(walking_contact_overlap(out))
        evaluation['chair_push_success'] = evaluation['success']
        evaluation['success'] = bool(evaluation['success'] and evaluation['both_feet_stepped']
                                    and evaluation['walking_push_temporal_overlap']
                                    and not evaluation['non_target_contacts']
                                    and not evaluation['chair_final_hand_contact']
                                    and result['status'] == 'walking_push_sequence_completed')
        evaluation['walking_push_success'] = evaluation['success']
        evaluation['objective_achieved'] = evaluation['success']
        summary = dict(controller=result, independent_evaluation=evaluation,
                       control_mode=4, model='official ScaleBFM M/model_22200.pt')
        (out/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
        (out/'evaluator.json').write_text(json.dumps(evaluation, indent=2)+'\n')
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        io.close()


if __name__ == '__main__':
    main()
