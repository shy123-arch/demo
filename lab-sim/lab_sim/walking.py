"""50 Hz learned locomotion targets with 500 Hz joint feedback."""
import math
import numpy as np
import mujoco
from . import locomotion as L


class WalkingController:
    def __init__(self, env):
        self.policy=L.G1WalkingPolicy()
        self.names=L.TARGET_JOINTS
        self.qa=np.array([env.model.joint(n).qposadr[0] for n in self.names])
        self.va=np.array([env.model.joint(n).dofadr[0] for n in self.names])
        self.aa=np.array([env.model.actuator(n).id for n in self.names])
        self.kp=np.r_[L.LOWER_KP,L.ARM_KP,L.HAND_KP]
        self.kd=np.r_[L.LOWER_KD,L.ARM_KD,L.HAND_KD]
        self.upper_indices=[env.names.index(n) for n in L.ARM_JOINTS]
        self.hand_indices=[env.names.index(n) for n in L.HAND_JOINTS]
        self.phase=0
        self.decimation=round(.02/env.model.opt.timestep)
        self.target=env.data.qpos[self.qa].copy()

    def reset_standing(self, env, xy=(0,-2.8), yaw=math.pi/2):
        env.reset()
        env.set_assistance(False)
        env.data.qpos[:3]=(*xy,.79)
        env.data.qpos[3:7]=(math.cos(yaw/2),0,0,math.sin(yaw/2))
        for name,q in zip(L.LOWER_JOINTS,L.LOWER_NEUTRAL):
            env.data.qpos[env.model.joint(name).qposadr[0]]=q
        for name in L.ARM_JOINTS+L.HAND_JOINTS:
            env.data.qpos[env.model.joint(name).qposadr[0]]=0
        env.target=env.data.qpos[env.qadr].copy()
        mujoco.mj_forward(env.model,env.data)
        self.policy.reset();self.phase=0

    def advance(self, env, velocity=(0,0,0), ticks=10):
        for _ in range(ticks):
            if self.phase%self.decimation==0:
                self.target=self.policy.apply(env.model,env.data,velocity,
                    upper_target=env.target[self.upper_indices],hand_target=env.target[self.hand_indices])
            # Feedback must be recomputed on every physics tick. Holding a
            # torque for a whole 20 ms policy period is unstable on this model.
            env.data.ctrl[self.aa]=self.kp*(self.target-env.data.qpos[self.qa])-self.kd*env.data.qvel[self.va]
            mujoco.mj_step(env.model,env.data)
            self.phase+=1
        if not np.isfinite(env.data.qpos).all() or env.data.qpos[2]<.45:
            raise RuntimeError(f"G1 lost balance at t={env.data.time:.3f}; stopping rollout")

    def step(self, env, command, ticks):
        """Psi0 whole-body bridge: arm/hand targets are already set on env."""
        q=env.data.qpos[3:7]
        yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))
        error=(command["target_yaw"]-yaw+math.pi)%(2*math.pi)-math.pi
        self.policy.height_cmd=float(np.clip(command["torso_rpyh"][3],.60,.80))
        self.policy.rpy_cmd=np.clip(command["torso_rpyh"][:3],-.3,.3).astype(np.float32)
        velocity=(np.clip(command["vx"],-.25,.25),np.clip(command["vy"],-.15,.15),np.clip(1.5*error,-.45,.45))
        self.advance(env,velocity,ticks)


def walk_route(env, controller, seconds=12, patrol=False, callback=None):
    from .navigation import plan_path
    callback=callback or (lambda phase,env:None)
    controller.reset_standing(env)
    start=env.data.qpos[:3].copy()
    minimum_z=.79
    visited=[]
    path=[];goal_index=0;path_index=0
    goals=[(-.6,-.8),(0,2.5),(.45,.7),(0,-2.8)]
    for i in range(round(seconds/.02)):
        if i<100:
            velocity=(0,0,0);phase="balance"
        elif patrol:
            if not path:
                path=plan_path(env.model,env.data,env.data.qpos[:2],goals[goal_index])
                path_index=1
            p=env.data.qpos[:2]
            while path_index<len(path)-1 and np.linalg.norm(np.array(path[path_index])-p)<.25:
                path_index+=1
            delta=np.array(path[path_index])-p
            q=env.data.qpos[3:7]
            yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]*q[2]+q[3]*q[3]))
            target_yaw=math.atan2(delta[1],delta[0])
            error=(target_yaw-yaw+math.pi)%(2*math.pi)-math.pi
            velocity=(.22*max(0,math.cos(error)),0,np.clip(1.6*error,-.5,.5))
            phase=f"physical patrol station {goal_index+1}"
            if np.linalg.norm(p-np.array(goals[goal_index]))<.22:
                visited.append(goal_index);goal_index+=1;path=[]
                if goal_index==len(goals):break
        else:
            velocity=(.15,0,0) if env.data.qpos[1]<2.3 else (0,0,0)
            phase="physical walking"
        controller.advance(env,velocity,10)
        minimum_z=min(minimum_z,float(env.data.qpos[2]))
        callback(phase,env)
    return {"mode":"unassisted ONNX locomotion", "simulation_seconds":float(env.data.time),
            "start_xyz":start.tolist(),"end_xyz":env.data.qpos[:3].tolist(),"minimum_pelvis_z":minimum_z,
            "visited_stations":visited,"patrol_complete":bool(patrol and len(visited)==4),"finite":bool(np.isfinite(env.data.qpos).all())}
