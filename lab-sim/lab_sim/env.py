"""Low-level simulation API. Robot actions are motor torques in actuator order."""
from pathlib import Path
import math
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


class LabEnv:
    def __init__(self, scene="lab_g1.xml", assisted=True):
        self.model = mujoco.MjModel.from_xml_path(str(ROOT / "scenes" / scene))
        self.data = mujoco.MjData(self.model)
        self.assisted = assisted
        self.renderer = None
        self.robot_actuators = np.array([i for i in range(self.model.nu)
            if self.model.actuator(i).name.endswith("_joint")], dtype=int)
        self.robot_joints = self.model.actuator_trnid[self.robot_actuators, 0]
        self.qadr = self.model.jnt_qposadr[self.robot_joints]
        self.vadr = self.model.jnt_dofadr[self.robot_joints]
        self.names = [self.model.joint(int(j)).name for j in self.robot_joints]
        self.kp = np.array([15 if "hand_" in n else 65 if any(s in n for s in ("shoulder", "elbow", "wrist")) else 180 for n in self.names], float)
        self.kd = np.array([.4 if "hand_" in n else 3 if any(s in n for s in ("shoulder", "elbow", "wrist")) else 5 for n in self.names], float)
        self.reset()

    def reset(self, task="pick_place"):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (1.72, -.80, .79)
        self.data.qpos[3:7] = (1, 0, 0, 0)
        for side in ("left", "right"):
            for name, val in (("hip_pitch", -.10), ("knee", .20), ("ankle_pitch", -.10), ("shoulder_pitch", -.80), ("elbow", 0.0)):
                self.data.qpos[self.model.joint(f"{side}_{name}_joint").qposadr[0]] = val
        self.base_pose = self.data.qpos[:7].copy()
        self.target = self.data.qpos[self.qadr].copy()
        self.data.eq_active[:] = 0
        self.set_assistance(self.assisted)
        self.task = task
        self.visited = set()
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def set_assistance(self, enabled):
        """A pelvis weld supports scripted arm demos; disable for whole-body control."""
        self.assisted = bool(enabled)
        try:
            self.data.eq_active[self.model.equality("base_support").id] = enabled
            self.data.mocap_pos[0] = self.data.qpos[:3]
            self.data.mocap_quat[0] = self.data.qpos[3:7]
        except KeyError:
            if enabled:
                raise RuntimeError("Scene is missing base_support equality")

    def set_base(self, x, y, yaw=0.0):
        """Relocate the support target. This is an assisted preview, not walking."""
        if not self.assisted:
            raise RuntimeError("set_base requires assisted mode; use motor torques for walking")
        self.base_pose[:3] = (x, y, self.base_pose[2])
        self.base_pose[3:] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        self.data.mocap_pos[0] = self.base_pose[:3]
        self.data.mocap_quat[0] = self.base_pose[3:]

    def pd_torque(self):
        torque = self.kp * (self.target - self.data.qpos[self.qadr]) - self.kd * self.data.qvel[self.vadr]
        torque += self.data.qfrc_bias[self.vadr]
        limits = self.model.jnt_actfrcrange[self.robot_joints]
        limited = self.model.jnt_actfrclimited[self.robot_joints].astype(bool)
        torque[limited] = np.clip(torque[limited], limits[limited, 0], limits[limited, 1])
        return torque

    def step(self, action=None, nstep=10):
        """Apply one torque per robot actuator (N*m) for nstep physics ticks."""
        if action is not None:
            action = np.asarray(action, dtype=float)
            if action.shape != (len(self.robot_actuators),) or not np.isfinite(action).all():
                raise ValueError(f"Expected {len(self.robot_actuators)} finite torques")
        for _ in range(nstep):
            self.data.ctrl[self.robot_actuators] = self.pd_torque() if action is None else action
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def site(self, name):
        return self.data.site(name).xpos.copy()

    def body(self, name):
        return self.data.body(name).xpos.copy()

    def solve_ik(self, point, side="right", iterations=80):
        """Position-only damped least-squares IK; returns desired arm joint angles."""
        ids = [self.model.joint(f"{side}_{n}_joint").id for n in ARM_JOINTS]
        qadr, vadr = self.model.jnt_qposadr[ids], self.model.jnt_dofadr[ids]
        temp = mujoco.MjData(self.model)
        temp.qpos[:] = self.data.qpos
        # Seed at previous target to keep trajectories continuous.
        for j, qa in zip(ids, qadr):
            temp.qpos[qa] = self.target[list(self.robot_joints).index(j)]
        site_id = self.model.site(f"{side}_grasp").id
        jac = np.zeros((3, self.model.nv))
        for _ in range(iterations):
            mujoco.mj_forward(self.model, temp)
            error = np.asarray(point) - temp.site_xpos[site_id]
            if np.linalg.norm(error) < .002:
                break
            mujoco.mj_jacSite(self.model, temp, jac, None, site_id)
            J = jac[:, vadr]
            delta = J.T @ np.linalg.solve(J @ J.T + .002 * np.eye(3), error)
            temp.qpos[qadr] = np.clip(temp.qpos[qadr] + np.clip(delta, -.15, .15), self.model.jnt_range[ids, 0] + .01, self.model.jnt_range[ids, 1] - .01)
        for j, qa in zip(ids, qadr):
            self.target[list(self.robot_joints).index(j)] = temp.qpos[qa]
        mujoco.mj_forward(self.model, temp)
        return float(np.linalg.norm(np.asarray(point) - temp.site_xpos[site_id]))

    def grip(self, closed=True, side="right"):
        for i, name in enumerate(self.names):
            if name.startswith(f"{side}_hand_"):
                lo, hi = self.model.jnt_range[self.robot_joints[i]]
                self.target[i] = (lo if abs(lo) > abs(hi) else hi) * .55 if closed else 0

    def attach(self, body_name="sample_bottle", side="right", max_distance=.08):
        """Explicit grasp assistance: activate a weld only near the palm site."""
        if not self.assisted:
            raise RuntimeError("Grasp assistance is disabled in physics mode")
        distance = np.linalg.norm(self.site(f"{side}_grasp") - self.site(f"{body_name}_grasp"))
        if distance > max_distance:
            raise RuntimeError(f"Palm too far from {body_name}: {distance:.3f} m")
        eid = self.model.equality(f"grasp_{body_name}").id
        a, b = self.model.eq_obj1id[eid], self.model.eq_obj2id[eid]
        # Weld relpose = object pose expressed in wrist coordinates, preserving pose.
        R = self.data.xmat[a].reshape(3, 3)
        self.model.eq_data[eid, 3:6] = R.T @ (self.data.xpos[b] - self.data.xpos[a])
        invq = self.data.xquat[a].copy(); invq[1:] *= -1
        relq = np.zeros(4)
        mujoco.mju_mulQuat(relq, invq, self.data.xquat[b])
        self.model.eq_data[eid, 6:10] = relq
        self.data.eq_active[eid] = 1
        self.grip(True, side)

    def release(self, body_name="sample_bottle"):
        self.data.eq_active[self.model.equality(f"grasp_{body_name}").id] = 0
        self.grip(False)

    def observe(self):
        result = {"time": float(self.data.time), "qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy(),
                  "robot_qpos": self.data.qpos[self.qadr].copy(), "robot_qvel": self.data.qvel[self.vadr].copy(),
                  "actuator_names": self.names.copy(), "base_position": self.data.qpos[:3].copy()}
        return result

    def render(self, camera="overview", width=1280, height=800, depth=False):
        if self.renderer is None or (self.renderer.width, self.renderer.height) != (width, height):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = (1, 1, 1, 1, 0, int(camera != "top"))
        opt.sitegroup[:] = 0
        self.renderer.update_scene(self.data, camera=camera, scene_option=opt)
        if depth:
            self.renderer.enable_depth_rendering()
        frame = self.renderer.render().copy()
        if depth:
            self.renderer.disable_depth_rendering()
        return frame

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
