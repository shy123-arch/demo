"""Run camera + encoder + IMU control, then independently evaluate physics."""
import argparse
import json
from pathlib import Path

from .scalebfm_proprio import ProprioScaleBFM
from .vision_robot_io import VisionRobotIO
from .vision_servo import SensorPort, VisualServo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=['button','push_box'],default='button')
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--blind',action='store_true')
    parser.add_argument('--button-shift',nargs=3,type=float,default=[0,0,0])
    parser.add_argument('--box-shift',nargs=2,type=float,default=[0,0])
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    policy=ProprioScaleBFM()
    io=VisionRobotIO(policy,button_shift=args.button_shift,box_shift=args.box_shift,output_dir=out)
    try:
        servo=VisualServo(policy,SensorPort(io.sensors,io.step,io.images),out,blind=args.blind)
        result=servo.run(args.task)
        # Evaluation ends the hardware episode and cannot feed another action.
        evaluation=io.evaluate(args.task)
        summary={'controller':result,'independent_evaluation':evaluation,
                 'control_mode':4,'model':'official ScaleBFM M/model_22200.pt'}
        (out/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({'controller_status':result['status'],
            'visual_success':result['visual_success'],'physical_success':evaluation['success'],
            'button_travel_mm':evaluation['max_button_travel_m']*1000,
            'box_displacement_mm':evaluation['box_displacement_xy_m']*1000,
            'box_tilt':evaluation['box_final_tilt_deg'],'unassisted':evaluation['unassisted']},indent=2),flush=True)
    finally:io.close()


if __name__=='__main__':main()
