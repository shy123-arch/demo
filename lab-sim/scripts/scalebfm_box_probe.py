#!/usr/bin/env python3
"""Develop and measure a ScaleBFM-driven physical sample-box push."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_sim.env import LabEnv
from lab_sim.scalebfm import ScaleBFMController
from lab_sim.scalebfm_tasks import make_keyframe, run_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push-x", type=float, default=1.93)
    parser.add_argument("--wrist-z", type=float, default=.855)
    parser.add_argument("--output", default="outputs/scalebfm_box_probe/trial_01")
    args = parser.parse_args()
    env = LabEnv(assisted=False)
    c = ScaleBFMController(env)
    c.reset(env, xy=(1.62,-1.015))
    base_p, base_q = c.poses(env)
    frames = []
    for t, label, wrist in (
        (0,"initial standing",None), (1.5,"stand",None),
        (3.5,"lift hand in aisle",(1.68,-1.225,1.04)),
        (5,"move above table",(1.80,-1.225,1.04)),
        (6.5,"lower behind sample box",(1.80,-1.225,args.wrist_z)),
        (8.5,"approach box",(1.855,-1.225,args.wrist_z)),
        (10.5,"push box along table",(args.push_x,-1.225,args.wrist_z)),
        (11.5,"hold at target",(args.push_x,-1.225,args.wrist_z)),
        (13.5,"withdraw horizontally",(1.76,-1.225,args.wrist_z)),
        (15,"lift clear of box",(1.76,-1.225,1.04)),
        (17,"retract above aisle",(1.64,-1.225,1.04)),
        (19,"return hand",None), (20,"settle",None),
    ):
        p, q = base_p.copy(), base_q.copy()
        if wrist is not None:
            p[2] = wrist; q[2] = (1,0,0,0)
        frames.append(make_keyframe(t,label,p,q))
    plan = {"task":"push_box", "initial_xy":[1.62,-1.015], "initial_yaw":0,
            "box_goal_xy":[2.16,-1.23], "goal_tolerance_m":.025,
            "keyframes":frames}
    try:
        run_plan(plan,args.output,env=env,controller=c)
    finally:
        env.close()


if __name__ == "__main__": main()
