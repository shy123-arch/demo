"""Psi0 SIMPLE HTTP protocol, matching Psi0/src/psi/deploy/psi0_serve_simple.py.

State is [left hand (thumb,middle,index), right hand (thumb,index,middle),
left arm, right arm, last commanded torso rpyh]. No normalization here:
the Psi0 server applies the checkpoint's own statistics.
"""
import base64
import json
import uuid
import urllib.request
import numpy as np
from .env import ARM_JOINTS

HAND_NAMES = ([f"left_hand_{p}_joint" for p in (
    "thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
    + [f"right_hand_{p}_joint" for p in (
    "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")])
ARM_NAMES = [f"{s}_{j}_joint" for s in ("left", "right") for j in ARM_JOINTS]


def encode_numpy(obj):
    if isinstance(obj, np.ndarray):
        obj = np.ascontiguousarray(obj)
        return {"__numpy__": base64.b64encode(obj.tobytes()).decode(), "dtype": obj.dtype.str, "shape": list(obj.shape)}
    raise TypeError(type(obj).__name__)


def decode_numpy(obj):
    if "__numpy__" in obj:
        dtype = np.dtype(obj["dtype"])
        if dtype.hasobject:
            raise ValueError("Object arrays are not a valid policy response")
        return np.frombuffer(base64.b64decode(obj["__numpy__"]), dtype=dtype).reshape(obj["shape"]).copy()
    return obj


class Psi0Adapter:
    def __init__(self, env, url="http://127.0.0.1:22085/act", controller=None):
        self.env, self.url, self.controller = env, url, controller
        self.rpyh = np.array([0, 0, 0, .75], dtype=np.float32)
        self.session_id = f"lab-{uuid.uuid4().hex[:12]}"
        self.step_index = 0
        self._time_budget = 0.0
        self.qadr = np.array([env.model.joint(n).qposadr[0] for n in HAND_NAMES + ARM_NAMES])
        self.target_indices = [env.names.index(n) for n in HAND_NAMES + ARM_NAMES]

    def observation(self, instruction, render=True):
        rgb = self.env.render("head", 640, 480) if render else np.zeros((480, 640, 3), dtype=np.uint8)
        states = np.concatenate([self.env.data.qpos[self.qadr], self.rpyh]).astype(np.float32)[None]
        history={"session_id":self.session_id,"step_index":self.step_index,"episode_index":0}
        # Psi0 RTC checks key presence, so even reset=False would reset it.
        if self.step_index==0:
            history["reset"]=True
        return {"image": {"rgb_head_stereo_left": rgb}, "instruction": instruction,
                "state": {"states": states}, "history": history,
                "condition": {}, "gt_action": None, "dataset_name": "simple", "timestamp": str(self.env.data.time)}

    def query(self, instruction, timeout=120):
        payload = self.observation(instruction)
        req = urllib.request.Request(self.url, data=json.dumps(payload, default=encode_numpy).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read(), object_hook=decode_numpy)
        if not isinstance(result, dict) or "action" not in result:
            raise RuntimeError(f"Psi0 server did not return actions: {str(result)[:400]}")
        actions = np.asarray(result["action"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 36 or len(actions) == 0 or not np.isfinite(actions).all():
            raise ValueError(f"Expected finite Psi0 SIMPLE actions (T,36), got {actions.shape}")
        return actions

    def apply(self, action, control_dt=1/30):
        """Apply absolute joints. Whole-body commands require an explicit controller.

        With pelvis assistance enabled, navigation is a support-target preview;
        it does not claim dynamically balanced locomotion.
        """
        a = np.asarray(action, dtype=float)
        if a.shape != (36,) or not np.isfinite(a).all():
            raise ValueError("Expected a finite 36-dimensional Psi0 action")
        if not self.env.assisted and self.controller is None:
            raise RuntimeError("Physics mode requires a whole-body controller for Psi0 navigation/rpyh")
        for value, name, ti in zip(a[:28], HAND_NAMES + ARM_NAMES, self.target_indices):
            j = self.env.model.joint(name).id
            self.env.target[ti] = np.clip(value, *self.env.model.jnt_range[j])
        self.rpyh[:] = a[28:32]
        command = {"vx": float(a[32]), "vy": float(a[33]), "turning_flag": float(a[34]),
                   "target_yaw": float(a[35]), "torso_rpyh": a[28:32].copy()}
        if not np.isfinite(control_dt) or control_dt < self.env.model.opt.timestep:
            raise ValueError("control_dt must be at least one physics timestep")
        self._time_budget += control_dt
        ticks = int((self._time_budget + 1e-10) / self.env.model.opt.timestep)
        self._time_budget -= ticks * self.env.model.opt.timestep
        if self.controller is not None:
            self.controller.step(self.env, command, ticks)
        else:
            # Assisted reference implementation for bridge and scene verification.
            for n, value in zip(("waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"), a[28:31]):
                j = self.env.model.joint(n).id
                self.env.target[self.env.names.index(n)] = np.clip(value, *self.env.model.jnt_range[j])
            yaw = 2 * np.arctan2(self.env.base_pose[6], self.env.base_pose[3])
            yaw_error = (a[35] - yaw + np.pi) % (2*np.pi) - np.pi
            yaw += np.clip(yaw_error, -.6*control_dt, .6*control_dt)
            velocity = np.clip(a[32:34], -.3, .3)
            dx = (np.cos(yaw)*velocity[0] - np.sin(yaw)*velocity[1]) * control_dt
            dy = (np.sin(yaw)*velocity[0] + np.cos(yaw)*velocity[1]) * control_dt
            # Keep support inside the central aisle; scene collisions remain active.
            x = np.clip(self.env.base_pose[0] + dx, -1.78, 1.78)
            y = np.clip(self.env.base_pose[1] + dy, -3.4, 2.7)
            self.env.base_pose[2] = np.clip(a[31], .65, .82)
            knee = 2*np.arccos(np.clip((self.env.base_pose[2]-.18)/.61, .5, .995))
            for side in ("left", "right"):
                for joint, value in (("hip_pitch", -knee/2), ("knee", knee), ("ankle_pitch", -knee/2)):
                    self.env.target[self.env.names.index(f"{side}_{joint}_joint")] = value
            self.env.set_base(x, y, yaw)
            self.env.step(nstep=ticks)
        self.step_index += 1
        return command
