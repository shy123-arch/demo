"""Repeatable assisted task demonstrations, with physical object dynamics.

The assistance is explicit: a pelvis support, a proximity-gated grasp weld,
and optional drawer/button forces. These demos validate scene mechanics and
interfaces; they are not learned-policy success measurements.
"""
import math
import numpy as np
from .navigation import plan_path


def blend(t):
    return t*t*(3-2*t)


class TaskRunner:
    def __init__(self, env, callback=None):
        self.env = env
        self.callback = callback or (lambda phase, env: None)
        self.dt = .02
        self.phase = "ready"
        self.metrics = {}

    def tick(self):
        self.env.step(nstep=10)
        self.callback(self.phase, self.env)

    def hold(self, seconds, phase):
        self.phase = phase
        for _ in range(round(seconds/self.dt)):
            self.tick()

    def hand(self, goal, seconds=1.5, phase="reach"):
        self.phase = phase
        start = self.env.site("right_grasp")
        goal = np.array(goal)
        for t in np.linspace(0, 1, max(2, round(seconds/self.dt))):
            self.env.solve_ik(start + blend(t)*(goal-start), iterations=15)
            self.tick()

    def base(self, xy, yaw=0, seconds=1, height=.79):
        start = self.env.base_pose.copy()
        start_yaw = 2*math.atan2(start[6], start[3])
        delta_yaw = (yaw-start_yaw+math.pi)%(2*math.pi)-math.pi
        for t in np.linspace(0, 1, max(2, round(seconds/self.dt))):
            s=blend(t)
            z=start[2]+s*(height-start[2])
            self.env.base_pose[2] = z
            knee=2*math.acos(np.clip((z-.18)/.61, .5, .995))
            for side in ("left", "right"):
                for n,v in (("hip_pitch",-knee/2),("knee",knee),("ankle_pitch",-knee/2)):
                    self.env.target[self.env.names.index(f"{side}_{n}_joint")]=v
            self.env.set_base(*(start[:2]+s*(np.array(xy)-start[:2])), start_yaw+s*delta_yaw)
            self.tick()

    def pick_place(self):
        e=self.env
        self.hold(.3,"settle")
        bottle=e.site("sample_bottle_grasp")
        self.hand(bottle+[0,0,.12],phase="approach bottle")
        self.hand(bottle+[0,0,.025],phase="grasp bottle")
        e.attach("sample_bottle",max_distance=.10)
        self.hold(.3,"assisted grasp")
        self.hand(e.site("right_grasp")+[0,0,.16],phase="lift bottle")
        self.metrics["lift_height_m"] = float(e.body("sample_bottle")[2]-.873)
        self.phase="carry sample"
        self.base((1.72,-.30), seconds=2)
        self.hand((2.14,-.42,1.03),phase="over tray")
        self.hand((2.14,-.42,.975),phase="lower into tray")
        e.release("sample_bottle")
        self.hand((2.08,-.40,1.18),phase="release and retract")
        self.hold(1,"settle in tray")
        p=e.body("sample_bottle")
        self.metrics.update(bottle_position=p.tolist(), placed_in_tray=bool(abs(p[0]-2.14)<.14 and abs(p[1]+.42)<.14 and .82<p[2]<1.02))
        return self.metrics

    def workflow(self):
        e=self.env
        self.phase="approach drawer"
        self.base((1.68,-.65), seconds=1.5, height=.67)
        handle=e.site("drawer_handle_site")
        self.hand(handle+[-.04,0,.02],seconds=2,phase="reach drawer handle")
        jid=e.model.joint("drawer_slide").id
        qa,va=e.model.jnt_qposadr[jid],e.model.jnt_dofadr[jid]
        aid=e.model.actuator("drawer_assist").id
        self.phase="open drawer (assisted handle force)"
        for desired in np.linspace(0,.29,125):
            e.data.ctrl[aid]=np.clip(180*(desired-e.data.qpos[qa])-15*e.data.qvel[va],-45,45)
            e.set_base(1.68-desired,-.65,0)
            e.solve_ik(e.site("drawer_handle_site")+[-.025,0,.02],iterations=15)
            self.tick()
        self.metrics["drawer_open_m"]=float(e.data.qpos[qa])
        self.hold(.4,"drawer open")
        self.phase="close drawer"
        for desired in np.linspace(.29,0,125):
            e.data.ctrl[aid]=np.clip(180*(desired-e.data.qpos[qa])-15*e.data.qvel[va],-45,45)
            e.set_base(1.68-desired,-.65,0)
            e.solve_ik(e.site("drawer_handle_site")+[-.025,0,.02],iterations=15)
            self.tick()
        e.data.ctrl[aid]=0
        self.hand((1.90,-.9,.92),phase="retract from drawer")
        self.phase="finish closing drawer"
        for _ in range(100):
            e.data.ctrl[aid]=np.clip(-300*e.data.qpos[qa]-20*e.data.qvel[va],-45,45)
            self.tick()
        e.data.ctrl[aid]=0
        self.metrics["drawer_closed_m"]=float(e.data.qpos[qa])
        self.phase="approach analyzer"
        self.base((1.78,-.98),seconds=1.5)
        self.hand(e.site("button_site")+[-.06,0,.03],phase="reach analyzer button")
        self.hand(e.site("button_site")+[-.01,0,0],phase="press analyzer button")
        distance=float(np.linalg.norm(e.site("right_grasp")-e.site("button_site")))
        self.metrics["button_reach_error_m"]=distance
        button_id=e.model.actuator("button_assist").id
        if distance<.12:
            e.data.ctrl[button_id]=8
            self.hold(.4,"press button (assisted contact force)")
        self.metrics["button_travel_m"]=float(e.data.joint("button_slide").qpos[0])
        self.metrics["analyzer_triggered"]=bool(self.metrics["button_travel_m"]>.008)
        e.data.ctrl[button_id]=0
        self.hand((2.0,-1.1,1.18),phase="retract from analyzer")
        return self.metrics

    def patrol(self):
        e=self.env
        # Start with arms folded away from furniture.
        for side in ("left","right"):
            e.target[e.names.index(f"{side}_shoulder_pitch_joint")]=.25
            e.target[e.names.index(f"{side}_elbow_joint")]=1.2
        self.phase="leave workstation"
        self.base((.55,-.80),seconds=2)
        stations=[(0,-2.8),(-.6,-.8),(0,2.5),(.45,.7)]
        paths=[]
        for i,goal in enumerate(stations):
            self.phase=f"patrol station {i+1} (assisted base)"
            path=plan_path(e.model,e.data,e.base_pose[:2],goal)
            paths.append([list(p) for p in path])
            for p in path[1:]:
                delta=np.array(p)-e.base_pose[:2]
                yaw=math.atan2(delta[1],delta[0])
                self.base(p,yaw,seconds=max(.15,np.linalg.norm(delta)/.5))
            self.hold(.4,f"inspect station {i+1}")
            e.visited.add(i)
        self.metrics.update(visited_stations=sorted(e.visited),planned_paths=paths,patrol_complete=len(e.visited)==4)
        return self.metrics

    def run(self, task):
        if not self.env.assisted:
            raise RuntimeError("Scripted tasks require --assisted; use a policy for physics-only control")
        result=getattr(self,task)()
        result["mode"]="assisted demonstration"
        result["simulation_seconds"]=float(self.env.data.time)
        return result
