"""Official ScaleBFM M policy with five body-pose targets in the G1 lab.

Commands are world-frame poses (metres, wxyz quaternions), not joint angles.
The learned policy controls all 29 body joints; Dex3 fingers hold their pose.
"""
import json
import time
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch

from .env import ROOT
from .vendor.scalebfm_network import HumanoidTransformer, TaskEmbedder
from .vendor.scalebfm_export import (
    HumanoidTransformerPolicyWrapperWithMode, build_mode_mappings, parse_xml,
    quat_apply_inverse, quat_mul_inverse_left,
)

ASSETS = ROOT / "assets/g1/policy/scalebfm_m"
FIVE_BODIES = ("pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link",
               "left_ankle_roll_link", "right_ankle_roll_link")


class ScaleBFMController:
    """50 Hz pose tracking with PD feedback on every MuJoCo physics tick."""

    def __init__(self, env, device="cpu", assets=ASSETS):
        self.assets = Path(assets)
        self.metadata = json.loads((self.assets / "metadata.json").read_text())
        self.device = torch.device(device)
        # Small batch CPU inference benefits from avoiding a large thread pool.
        if self.device.type == "cpu":
            torch.set_num_threads(1)
        meta = self.metadata
        arch = meta["policy_architecture"]
        actor = HumanoidTransformer(
            arch["prop_obs_dim"], arch["action_dim"], arch["output_dim"],
            arch["embedding_dim"], arch["num_heads"], arch["ff_dim"], arch["num_layers"],
        )
        embedder = TaskEmbedder(arch["task_obs_dim"], arch["embedding_dim"],
                                arch["reduced_task_dim"], arch["task_embedder_hidden_dims"])
        state = torch.load(self.assets / "model_22200.pt", map_location="cpu",
                           weights_only=True)["model_state_dict"]
        actor.load_state_dict({k[len("actor."):]: v for k, v in state.items()
                               if k.startswith("actor.")}, strict=True)
        embedder.load_state_dict({k[len("actor_task_embedder."):]: v for k, v in state.items()
                                  if k.startswith("actor_task_embedder.")}, strict=True)
        mode_table = torch.load(self.assets / "mode_table.pt", map_location=self.device,
                                weights_only=True)
        bodies, joints, parents, axes, translations, rotations = parse_xml(
            self.assets / "kinematics.xml", self.device)
        self.selected_names = meta["selected_body_names"]
        self.five_indices = [self.selected_names.index(n) for n in FIVE_BODIES]
        assert set(torch.nonzero(mode_table[4]).flatten().tolist()) == set(self.five_indices)
        self.default = np.asarray(meta["default_dof_pos"])
        self.policy = HumanoidTransformerPolicyWrapperWithMode(
            SimpleNamespace(actor=actor, actor_task_embedder=embedder),
            build_mode_mappings(mode_table, meta["mode_feature_dims"], True), mode_table,
            self.tensor(self.default)[None], self.tensor(meta["action_scale"])[None],
            meta["history_buffer_size"], len(meta["future_idx"]),
            translations, rotations / rotations.norm(dim=-1, keepdim=True), parents, axes,
            torch.tensor([bodies.index(n) for n in self.selected_names], device=self.device),
            torch.tensor([meta["joint_names"].index(n) for n in joints], device=self.device),
        ).to(self.device).eval()
        self.mode = torch.tensor([4], dtype=torch.long, device=self.device)
        self.offsets = np.asarray(meta["future_idx"])
        self.time_offsets = torch.as_tensor(self.offsets, dtype=torch.long,
                                            device=self.device)[None, :, None]
        self.names = meta["joint_names"]
        self.qadr = np.array([env.model.joint(n).qposadr[0] for n in self.names])
        self.vadr = np.array([env.model.joint(n).dofadr[0] for n in self.names])
        self.action_qadr = np.array([env.model.joint(n).qposadr[0] for n in meta["action_names"]])
        self.action_vadr = np.array([env.model.joint(n).dofadr[0] for n in meta["action_names"]])
        self.aadr = np.array([env.model.actuator(n).id for n in meta["action_names"]])
        self.body_ids = np.array([env.model.body(n).id for n in self.selected_names])
        self.kp = np.asarray(meta["stiffness"])
        self.kd = np.asarray(meta["damping"])
        self.limits = np.asarray(meta["torque_limit"])
        self.ticks = round(.02 / env.model.opt.timestep)
        if not np.isclose(self.ticks * env.model.opt.timestep, .02):
            raise ValueError("Physics timestep must divide the 20 ms policy interval")
        # Apply the official deployment joint dynamics in this environment only.
        deployment = ET.parse(self.assets / "deployment.xml").getroot()
        defaults = deployment.find("default/joint").attrib
        for node in deployment.findall(".//worldbody//joint"):
            name = node.get("name")
            if name not in self.names:
                continue
            va = env.model.joint(name).dofadr[0]
            env.model.dof_armature[va] = float(node.get("armature", defaults.get("armature", 0)))
            env.model.dof_damping[va] = float(node.get("damping", defaults.get("damping", 0)))
            env.model.dof_frictionloss[va] = float(node.get("frictionloss", defaults.get("frictionloss", 0)))
        mujoco.mj_setConst(env.model, env.data)
        self.hand_idx = np.array([i for i, n in enumerate(env.names) if "hand_" in n])
        self.inference_ms = []

    def tensor(self, value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)

    def reset(self, env, xy=(0, -.8), yaw=0):
        env.reset()
        env.set_assistance(False)
        env.data.eq_active[:] = 0
        env.data.qpos[:3] = (*xy, .76)
        env.data.qpos[3:7] = (np.cos(yaw/2), 0, 0, np.sin(yaw/2))
        env.data.qpos[self.qadr] = self.default
        env.data.qvel[:] = 0
        env.data.ctrl[:] = 0
        env.target = env.data.qpos[env.qadr].copy()
        mujoco.mj_forward(env.model, env.data)
        self.initial_pos = env.data.xpos[self.body_ids].copy()
        self.initial_quat = env.data.xquat[self.body_ids].copy()
        h = self.metadata["history_buffer_size"]
        self.history = [self.tensor(a)[None, None].repeat(1, h, 1) for a in (
            env.data.qpos[3:7], env.data.qvel[3:6], env.data.qpos[self.qadr],
            env.data.qvel[self.vadr], np.zeros(len(self.aadr)),
        )]
        self.inference_ms = []
        return self.initial_pos.copy(), self.initial_quat.copy()

    def poses(self, env):
        """Current five poses in the public pelvis, wrists, ankles order."""
        ids = self.body_ids[self.five_indices]
        return env.data.xpos[ids].copy(), env.data.xquat[ids].copy()

    def advance(self, env, positions, quaternions, tick_callback=None):
        """Apply six future frames of five world-frame poses, then step 20 ms."""
        pos, quat = np.asarray(positions), np.asarray(quaternions)
        if pos.shape != (6, 5, 3) or quat.shape != (6, 5, 4):
            raise ValueError("Expected positions (6,5,3) and wxyz quaternions (6,5,4)")
        if not np.isfinite(pos).all() or not np.isfinite(quat).all():
            raise ValueError("Target poses must be finite")
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-3):
            raise ValueError("Target quaternions must be normalized")
        target_pos = np.broadcast_to(self.initial_pos, (6, 14, 3)).copy()
        target_quat = np.broadcast_to(self.initial_quat, (6, 14, 4)).copy()
        target_pos[:, self.five_indices] = pos
        target_quat[:, self.five_indices] = quat
        root_quat = self.history[0][:, -1, None, None].expand(1, 6, 14, 4)
        target_local_pos = quat_apply_inverse(root_quat,
            self.tensor(target_pos - env.data.qpos[:3])[None])
        target_local_quat = quat_mul_inverse_left(root_quat, self.tensor(target_quat)[None])
        started = time.perf_counter()
        with torch.inference_mode():
            target, raw_action = self.policy(*self.history, target_local_pos,
                                              target_local_quat, self.mode, self.time_offsets)
        desired = target[0].cpu().numpy()
        self.inference_ms.append(1000 * (time.perf_counter() - started))
        if not np.isfinite(desired).all():
            raise RuntimeError("ScaleBFM produced a non-finite action")
        for _ in range(self.ticks):
            torque = self.kp * (desired - env.data.qpos[self.action_qadr]) - self.kd * env.data.qvel[self.action_vadr]
            env.data.ctrl[self.aadr] = np.clip(torque, -self.limits, self.limits)
            hi = self.hand_idx
            env.data.ctrl[env.robot_actuators[hi]] = (
                env.kp[hi] * (env.target[hi] - env.data.qpos[env.qadr[hi]])
                - env.kd[hi] * env.data.qvel[env.vadr[hi]])
            mujoco.mj_step(env.model, env.data)
            if tick_callback is not None:
                tick_callback(env)
            if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all() or env.data.qpos[2] < .45:
                raise RuntimeError(f"G1 lost balance at simulation t={env.data.time:.3f}")
        # mj_step leaves Cartesian poses one tick behind; refresh for commands and logs.
        mujoco.mj_forward(env.model, env.data)
        values = [self.tensor(a) for a in (env.data.qpos[3:7], env.data.qvel[3:6],
                                          env.data.qpos[self.qadr], env.data.qvel[self.vadr])]
        values.append(raw_action[0])
        for i, value in enumerate(values):
            self.history[i] = torch.cat((self.history[i][:, 1:], value[None, None]), dim=1)
        return desired.copy()
