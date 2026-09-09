#!/usr/bin/env python3
"""Run, inspect, render or connect Psi0 to the G1 laboratory."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    render=sub.add_parser("render",help="Render reference / overview / workstation / head cameras")
    render.add_argument("--camera",default="reference")
    render.add_argument("--output",default="outputs/reference.png")
    render.add_argument("--width",type=int,default=1280)
    render.add_argument("--height",type=int,default=800)
    view=sub.add_parser("view",help="Interactive MuJoCo viewer; R resets, 1/2/3 run demos")
    demo=sub.add_parser("demo",help="Run explicit assisted scene demonstrations")
    demo.add_argument("--task",choices=["pick_place","workflow","patrol","all"],default="all")
    demo.add_argument("--video",action="store_true")
    demo.add_argument("--viewer",action="store_true")
    demo.add_argument("--camera",default="workstation")
    psi=sub.add_parser("psi0",help="Psi0 SIMPLE HTTP policy bridge")
    psi.add_argument("--url",default="http://127.0.0.1:22085/act")
    psi.add_argument("--instruction",default="Pick up the blue sample bottle and place it in the tray.")
    psi.add_argument("--chunks",type=int,default=10)
    psi.add_argument("--dry-run",action="store_true",help="Validate protocol and mappings without model inference")
    psi.add_argument("--video",action="store_true")
    psi.add_argument("--controller",choices=["assisted","gr00t"],default="assisted")
    walk=sub.add_parser("walk",help="Unassisted ONNX locomotion with 500 Hz PD feedback")
    walk.add_argument("--seconds",type=float,default=12)
    walk.add_argument("--patrol",action="store_true")
    walk.add_argument("--video",action="store_true")
    walk.add_argument("--width",type=int,default=640)
    walk.add_argument("--height",type=int,default=400)
    bfm=sub.add_parser("scalebfm",help="ScaleBFM M five-point body pose control (mode 4)")
    bfm.add_argument("--targets",help="JSON world-frame pose keyframes; default is the direct body-control demonstration")
    bfm.add_argument("--seconds",type=float,default=None)
    bfm.add_argument("--device",default="cpu",help="PyTorch device, e.g. cpu or cuda:7")
    bfm.add_argument("--video",action="store_true")
    bfm.add_argument("--viewer",action="store_true")
    bfm.add_argument("--camera",choices=["robot","reference"],default="robot")
    bfm.add_argument("--output-dir",default="outputs/scalebfm_five_point")
    physical=sub.add_parser("scalebfm-tasks",help="Physical button, sample-box and drawer tasks with ScaleBFM")
    physical.add_argument("--task",choices=["all","button","push_box","close_drawer"],default="all")
    physical.add_argument("--output-dir",default="outputs/scalebfm_tasks/validated")
    physical.add_argument("--device",default="cpu")
    physical.add_argument("--video",action="store_true")
    physical.add_argument("--render-only",action="store_true",help="Render previously saved physical trajectories")
    pico=sub.add_parser("pico-teleop",help="Unsupported PICO whole-body teleoperation and native demo recording")
    pico.add_argument("--molmo-root",default=None,help="MolmoSpaces teleop checkout; defaults to sibling demo/molospace")
    pico.add_argument("--output-dir",default=None)
    pico.add_argument("--scene",default="lab_g1_vision.xml")
    pico.add_argument("--start-x",type=float,default=1.72)
    pico.add_argument("--start-y",type=float,default=-.80)
    pico.add_argument("--start-yaw",type=float,default=0.)
    pico.add_argument("--seconds",type=float,default=0.,help="0 runs until B, viewer close or Ctrl-C")
    pico.add_argument("--viewer",action="store_true")
    pico.add_argument("--offline-motion",default=None,help="Run a bundled reference such as walk instead of PICO")
    pico.add_argument("--render-after",action="store_true")
    pico_render=sub.add_parser("pico-render",help="Render a saved PICO trajectory without rerunning physics")
    pico_render.add_argument("--output-dir",required=True)
    pico_render.add_argument("--camera",default="grid")
    pico_render.add_argument("--fps",type=float,default=25.)
    pico_render.add_argument("--width",type=int,default=960)
    pico_render.add_argument("--height",type=int,default=600)
    args=parser.parse_args()
    if args.command!="view" and not getattr(args,"viewer",False):
        os.environ.setdefault("MUJOCO_GL","osmesa" if not os.environ.get("DISPLAY") else "glfw")
    import numpy as np
    import mujoco
    from PIL import Image,ImageDraw
    from lab_sim.env import LabEnv,ROOT
    from lab_sim.tasks import TaskRunner
    os.chdir(ROOT)
    (ROOT/"outputs").mkdir(exist_ok=True)
    if args.command=="pico-teleop":
        from lab_sim.pico_teleop import run_pico_teleop
        if args.output_dir is None:
            args.output_dir=str(ROOT/"outputs"/time.strftime("pico_demo_%Y%m%d_%H%M%S"))
        report=run_pico_teleop(args)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        print(f"trajectory: {Path(args.output_dir).resolve()/'trajectory.npz'}")
        return
    if args.command=="pico-render":
        from lab_sim.pico_teleop import render_pico_demo
        path=render_pico_demo(args.output_dir,args.camera,args.width,args.height,args.fps)
        print(path)
        return
    if args.command=="scalebfm":
        from lab_sim.scalebfm_demo import run
        run(args)
        return
    if args.command=="scalebfm-tasks":
        from lab_sim.scalebfm_tasks import run_tasks
        run_tasks(args)
        return
    env=LabEnv()
    viewer=None
    writer=None
    try:
        if args.command=="render":
            env.step(nstep=150)
            path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(env.render(args.camera,args.width,args.height)).save(path)
            print(path.resolve());return
        if args.command=="view" or getattr(args,"viewer",False):
            import mujoco.viewer
            pending=[]
            def key(k):
                if k in (49,50,51):pending.append(("pick_place","workflow","patrol")[k-49])
                elif k in (82,114):pending.append("reset")
            viewer=mujoco.viewer.launch_passive(env.model,env.data,key_callback=key)
            viewer.opt.geomgroup[:]=(1,1,1,1,0,1)
            viewer.opt.sitegroup[:]=0
            viewer.cam.type=mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid=env.model.camera("reference").id
        if args.command=="view":
            print("R: reset | 1: pick/place | 2: drawer + analyzer | 3: assisted patrol")
            while viewer.is_running():
                if pending:
                    task=pending.pop(0);env.reset()
                    if task!="reset":
                        def sync(phase,e):
                            viewer.sync();time.sleep(.02)
                        print(TaskRunner(env,sync).run(task))
                else:
                    env.step();viewer.sync();time.sleep(.02)
            return
        if args.command=="demo":
            tasks=["pick_place","workflow","patrol"] if args.task=="all" else [args.task]
            reports={}
            for task in tasks:
                env.reset(task)
                if args.video:
                    import imageio.v2 as imageio
                    writer=imageio.get_writer(f"outputs/{task}.mp4",fps=10,codec="libx264",quality=7,macro_block_size=8)
                count=0
                def callback(phase,e):
                    nonlocal count
                    if viewer:
                        viewer.sync();time.sleep(.02)
                    if writer and count%5==0:
                        cam="reference" if task=="patrol" else args.camera
                        im=Image.fromarray(e.render(cam,960,600))
                        draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,36),fill=(18,28,33))
                        draw.text((14,10),f"G1 LAB | ASSISTED DEMO | {phase}",fill=(235,244,245))
                        writer.append_data(np.asarray(im))
                    count+=1
                report=TaskRunner(env,callback).run(task)
                reports[task]=report
                Path(f"outputs/{task}_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
                print(json.dumps({task:report},ensure_ascii=False,indent=2))
                if writer:writer.close();writer=None
                Image.fromarray(env.render("reference",1280,800)).save(f"outputs/{task}_final.png")
            if args.task=="all":
                Path("outputs/task_report.json").write_text(json.dumps(reports,indent=2),encoding="utf-8")
        elif args.command=="walk":
            from lab_sim.walking import WalkingController,walk_route
            controller=WalkingController(env)
            if args.video:
                import imageio.v2 as imageio
                writer=imageio.get_writer("outputs/walking.mp4",fps=10,codec="libx264",macro_block_size=8)
            count=0
            def walking_frame(phase,e):
                nonlocal count
                if writer and count%5==0:
                    im=Image.fromarray(e.render("reference",args.width,args.height))
                    draw=ImageDraw.Draw(im);draw.rectangle((0,0,args.width,36),fill=(18,28,33))
                    draw.text((14,10),f"G1 LAB | PHYSICAL ONNX WALKING | {phase}",fill=(235,244,245))
                    writer.append_data(np.asarray(im))
                count+=1
            report=walk_route(env,controller,args.seconds,args.patrol,walking_frame)
            Path("outputs/walking_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(json.dumps(report,indent=2))
        elif args.command=="psi0":
            from lab_sim.psi0 import Psi0Adapter,encode_numpy,decode_numpy
            controller=None
            if args.controller=="gr00t":
                from lab_sim.walking import WalkingController
                controller=WalkingController(env)
                controller.reset_standing(env)
            adapter=Psi0Adapter(env,args.url,controller=controller)
            if args.dry_run:
                payload=adapter.observation(args.instruction)
                encoded=json.dumps(payload,default=encode_numpy)
                decoded=json.loads(encoded,object_hook=decode_numpy)
                assert decoded["state"]["states"].shape==(1,32)
                assert np.array_equal(decoded["image"]["rgb_head_stereo_left"],payload["image"]["rgb_head_stereo_left"])
                Image.fromarray(payload["image"]["rgb_head_stereo_left"]).save("outputs/psi0_head.png")
                Path("outputs/psi0_request.json").write_text(encoded,encoding="utf-8")
                action=np.r_[env.data.qpos[adapter.qadr],adapter.rpyh,0,0,0,0]
                adapter.apply(action)
                print("Psi0 SIMPLE protocol OK: RGB uint8 (480,640,3), state (1,32), action (36). No inference was run.")
                return
            print(f"Psi0 + {args.controller} control. No grasp weld is applied automatically.")
            if args.video:
                import imageio.v2 as imageio
                writer=imageio.get_writer("outputs/psi0.mp4",fps=30,codec="libx264")
            for i in range(args.chunks):
                actions=adapter.query(args.instruction)
                for action in actions:
                    adapter.apply(action)
                    if writer:writer.append_data(env.render("workstation",960,600))
                print(f"chunk {i+1}: {len(actions)} actions, simulation t={env.data.time:.2f}")
    finally:
        if writer:writer.close()
        if viewer:viewer.close()
        env.close()


if __name__=="__main__":
    main()
