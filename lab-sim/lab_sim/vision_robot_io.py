"""Private simulator adapter with a camera/encoder/gyro-only control boundary.

The control API is ``sensors()``, ``images()``, ``set_hand_targets(targets)``,
``hand_sensors()`` and ``step(desired29)``. Scene
initialization, independent physical scoring and trajectory recording live on
the simulator side of that boundary. Calling ``evaluate`` ends control, so
its privileged measurements cannot be used to continue the same rollout.

The simulated IMU attitude is integrated from its native body-frame gyro,
starting from the explicitly upright reset. No root orientation truth is
copied into the sensor packet, and no depth or segmentation image is used.
"""
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .scalebfm_tasks import ContactAudit
from .chair_evaluation import ChairAudit

ROOT = Path(__file__).resolve().parents[1]


class _BottleAudit:
    """Private per-physics-tick evidence; no values return to the controller."""

    def __init__(self, model, data):
        self.model = model
        self.body_id = model.body("sample_bottle").id
        self.geom_id = model.geom("sample_bottle_collision").id
        self.table_id = model.geom("lab_right_workstation_counter").id
        self.initial_position = data.body(self.body_id).xpos.copy()
        table_rotation = data.geom(self.table_id).xmat.reshape(3, 3)
        self.support_z = float(data.geom(self.table_id).xpos[2] +
            np.abs(table_rotation[2]) @ model.geom_size[self.table_id])
        self.finger_ids = {i for i in range(model.nbody)
            if any(part in (model.body(i).name or "")
                   for part in ("_hand_thumb_", "_hand_index_", "_hand_middle_"))}
        self.hand_ids = {i for i in range(model.nbody)
            if "_hand_" in (model.body(i).name or "")} | {
            model.body(name).id for name in ("left_wrist_yaw_link", "right_wrist_yaw_link")}
        self.rows = []
        self.force = np.zeros(6)
        self.support_geoms_seen = set()
        self.final_support_geoms = []
        self.final_finger_bodies = []
        self.tick(data)

    def tick(self, data):
        m = self.model
        position = data.body(self.body_id).xpos.copy()
        axis_z = float(data.geom(self.geom_id).xmat.reshape(3, 3)[2, 2])
        radius, half_height = m.geom_size[self.geom_id, :2]
        half_extent_z = half_height * abs(axis_z) + radius * np.sqrt(max(0., 1 - axis_z**2))
        bottom_z = float(data.geom(self.geom_id).xpos[2] - half_extent_z)
        hand, finger, table, non_hand = False, False, False, False
        hand_force, support_force = 0., 0.
        support_geoms, finger_bodies = set(), set()
        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if (b1 == self.body_id) == (b2 == self.body_id):
                continue
            other_geom, other_body = (g2, b2) if b1 == self.body_id else (g1, b1)
            mujoco.mj_contactForce(m, data, ci, self.force)
            normal = abs(float(self.force[0]))
            if normal < .01:
                continue
            if other_body in self.hand_ids:
                hand = True
                hand_force += normal
                if other_body in self.finger_ids:
                    finger = True
                    finger_bodies.add(m.body(other_body).name)
            else:
                # Conservatively count any load-bearing non-hand contact as
                # support, including robot arms and the environment.
                non_hand = True
                table |= other_geom == self.table_id
                support_force += normal
                support_geoms.add(m.geom(other_geom).name or f"geom_{other_geom}")
        self.support_geoms_seen.update(support_geoms)
        self.final_support_geoms = sorted(support_geoms)
        self.final_finger_bodies = sorted(finger_bodies)
        tilt = float(np.degrees(np.arccos(np.clip(axis_z, -1, 1))))
        self.rows.append((float(data.time), *position, bottom_z,
            bottom_z - self.support_z, tilt, hand, finger, non_hand, table,
            hand_force, support_force))

    def arrays(self):
        values = np.asarray(self.rows)
        names = ("time", "x", "y", "z", "bottom_z", "clearance_m", "tilt_deg",
                 "hand_contact", "finger_contact", "non_hand_support", "table_contact",
                 "hand_normal_force_n", "non_hand_normal_force_n")
        result = {name: values[:, i] for i, name in enumerate(names)}
        for name in ("hand_contact", "finger_contact", "non_hand_support", "table_contact"):
            result[name] = result[name].astype(bool)
        return result

    def report(self):
        a = self.arrays()
        dt = float(self.model.opt.timestep)
        airborne = a["clearance_m"] >= .03
        clean_hold = airborne & a["finger_contact"] & ~a["non_hand_support"]
        # The reset snapshot is not an integrated tick and carries no duration.
        airborne[0] = clean_hold[0] = False
        longest, current, transport = 0., 0., 0.
        episode_positions = []
        positions = np.column_stack((a["x"], a["y"], a["z"]))
        for held, position in zip(clean_hold, positions):
            if held:
                current += dt
                if not episode_positions:
                    episode_positions.append(position[:2].copy())
                transport = max(transport, float(np.linalg.norm(
                    position[:2] - episode_positions[0])))
                longest = max(longest, current)
            else:
                current = 0.
                episode_positions = []
        delta = positions[-1] - self.initial_position
        lifted = bool(longest + 1e-9 >= .5 and not np.any(airborne & a["non_hand_support"]))
        released = bool(not a["hand_contact"][-1])
        on_table = bool(a["table_contact"][-1] and abs(a["clearance_m"][-1]) <= .005)
        upright = bool(a["tilt_deg"][-1] < 15)
        return {
            "initial_bottle_position": self.initial_position.tolist(),
            "final_bottle_position": positions[-1].tolist(),
            "bottle_displacement_m": delta.tolist(),
            "bottle_final_planar_displacement_m": float(np.linalg.norm(delta[:2])),
            "bottle_max_center_lift_m": float(np.max(a["z"] - self.initial_position[2])),
            "bottle_max_lift_m": float(np.max(a["clearance_m"])),
            "bottle_initial_support_height_m": self.support_z,
            "bottle_final_bottom_clearance_m": float(a["clearance_m"][-1]),
            "bottle_airborne_duration_s": float(np.sum(airborne) * dt),
            "bottle_airborne_hand_contact_overlap_s": float(np.sum(airborne & a["hand_contact"]) * dt),
            "bottle_airborne_finger_contact_overlap_s": float(np.sum(airborne & a["finger_contact"]) * dt),
            "bottle_airborne_non_hand_support_s": float(np.sum(airborne & a["non_hand_support"]) * dt),
            "bottle_longest_unassisted_finger_hold_s": longest,
            "bottle_max_planar_transport_during_hold_m": transport,
            "bottle_final_tilt_deg": float(a["tilt_deg"][-1]),
            "bottle_final_hand_contact": bool(a["hand_contact"][-1]),
            "bottle_final_finger_contact": bool(a["finger_contact"][-1]),
            "bottle_final_finger_bodies": self.final_finger_bodies,
            "bottle_final_non_hand_support": bool(a["non_hand_support"][-1]),
            "bottle_final_support_geoms": self.final_support_geoms,
            "bottle_support_geoms_seen": sorted(self.support_geoms_seen),
            "bottle_final_released": released,
            "bottle_final_on_table": on_table,
            "bottle_final_upright": upright,
            "bottle_lift_achieved": lifted,
            "bottle_pick_place_achieved": bool(lifted and released and on_table and upright
                and np.linalg.norm(delta[:2]) >= .08 and transport >= .08),
            "bottle_evaluation_criteria": {
                "bottom_clearance_m": .03, "continuous_finger_hold_s": .5,
                "minimum_contact_normal_force_n": .01,
                "non_hand_contact_while_airborne_allowed": False,
                "pick_place_final_planar_displacement_m": .08,
                "pick_place_transport_during_continuous_hold_m": .08,
                "pick_place_final_tilt_deg_below": 15,
                "pick_place_final_table_clearance_tolerance_m": .005,
                "audit_frequency_hz": 1 / dt,
            },
        }


class VisionRobotIO:
    """A 500 Hz torque interface and calibrated 640 x 480 RGB stereo fixture."""

    def __init__(self, policy, station=(1.62, -1.015), box_shift=(0, 0),
                 button_shift=(0, 0, 0), output_dir=None, bottle_shift=(0, 0, 0),
                 scene=None, chair_shift=(0, 0)):
        station = self.__vector(station, 2, "station")
        box_shift = self.__vector(box_shift, 2, "box_shift")
        button_shift = self.__vector(button_shift, 3, "button_shift")
        bottle_shift = self.__vector(bottle_shift, 3, "bottle_shift")
        chair_shift = self.__vector(chair_shift, 2, "chair_shift")
        scene_path = Path(scene or "lab_g1_vision.xml")
        if not scene_path.is_absolute():
            scene_path = ROOT / scene_path if scene_path.parts[0] == "scenes" else ROOT / "scenes" / scene_path
        self.__model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.__has_chair = mujoco.mj_name2id(
            self.__model, mujoco.mjtObj.mjOBJ_BODY, "push_chair") >= 0
        if np.any(chair_shift) and not self.__has_chair:
            raise ValueError("chair_shift requires a scene containing push_chair")
        self.__data = mujoco.MjData(self.__model)
        self.__renderer = None
        self.__closed = False
        self.__evaluated = False
        self.__out = Path(output_dir) if output_dir is not None else None
        self.__records = []
        self.__initialization = {
            "scene": str(scene_path.relative_to(ROOT / "scenes"))
                if scene_path.is_relative_to(ROOT / "scenes") else str(scene_path),
            "station_xy": station.tolist(),
            "box_shift_xy": box_shift.tolist(), "button_shift_xyz": button_shift.tolist(),
            "bottle_shift_xyz": bottle_shift.tolist(),
            "chair_shift_xy": chair_shift.tolist(),
            "movable_chair_present": self.__has_chair,
            "initial_root_height_m": .76, "initial_root_upright": True,
            "imu_attitude_source": "native pelvis gyro integrated at 500 Hz from upright reset",
            "controller_environment_inputs": "RGB stereo only",
            "controller_proprioception": "joint encoders and pelvis gyro / integrated attitude",
            "camera_depth_or_segmentation": False,
        }
        if self.__out:
            self.__out.mkdir(parents=True, exist_ok=True)
            (self.__out / "initialization.json").write_text(
                json.dumps(self.__initialization, indent=2) + "\n")
        meta = policy.metadata
        self.__joint_names = list(meta["joint_names"])
        self.__qadr = np.array([self.__model.joint(n).qposadr[0] for n in self.__joint_names])
        self.__vadr = np.array([self.__model.joint(n).dofadr[0] for n in self.__joint_names])
        action_names = meta["action_names"]
        self.__action_qadr = np.array([self.__model.joint(n).qposadr[0] for n in action_names])
        self.__action_vadr = np.array([self.__model.joint(n).dofadr[0] for n in action_names])
        self.__aadr = np.array([self.__model.actuator(n).id for n in action_names])
        self.__kp = np.asarray(meta["stiffness"], dtype=float)
        self.__kd = np.asarray(meta["damping"], dtype=float)
        self.__limits = np.asarray(meta["torque_limit"], dtype=float)
        self.__hand_actuators = np.array([i for i in range(self.__model.nu)
            if "hand_" in (self.__model.actuator(i).name or "")], dtype=int)
        finger_joints = self.__model.actuator_trnid[self.__hand_actuators, 0]
        self.__hand_names = [self.__model.joint(int(j)).name for j in finger_joints]
        self.__hand_indices = {name: i for i, name in enumerate(self.__hand_names)}
        self.__hand_ranges = self.__model.jnt_range[finger_joints].copy()
        self.__hand_targets = np.zeros(len(finger_joints))
        self.__hand_qadr = self.__model.jnt_qposadr[finger_joints]
        self.__hand_vadr = self.__model.jnt_dofadr[finger_joints]
        # Scene fingers are direct unit-gain joint motors. Clip the PD command
        # to all configured actuator and joint force limits before application.
        if not np.allclose(self.__model.actuator_gear[self.__hand_actuators, 0], 1):
            raise ValueError("Finger PD requires direct unit-gear motors")
        self.__hand_ctrl_low = np.full(len(finger_joints), -np.inf)
        self.__hand_ctrl_high = np.full(len(finger_joints), np.inf)
        for limited, ranges in (
            (self.__model.actuator_ctrllimited[self.__hand_actuators],
             self.__model.actuator_ctrlrange[self.__hand_actuators]),
            (self.__model.actuator_forcelimited[self.__hand_actuators],
             self.__model.actuator_forcerange[self.__hand_actuators]),
            (self.__model.jnt_actfrclimited[finger_joints],
             self.__model.jnt_actfrcrange[finger_joints]),
        ):
            active = limited.astype(bool)
            self.__hand_ctrl_low[active] = np.maximum(self.__hand_ctrl_low[active], ranges[active, 0])
            self.__hand_ctrl_high[active] = np.minimum(self.__hand_ctrl_high[active], ranges[active, 1])
        self.__dt = float(self.__model.opt.timestep)
        self.__ticks = round(.02 / self.__dt)
        if not np.isclose(self.__dt, .002) or not np.isclose(self.__ticks * self.__dt, .02):
            raise ValueError("VisionRobotIO requires a 2 ms physics timestep")
        # These are deployment robot parameters, not environmental observations.
        deployment = ET.parse(Path(policy.assets) / "deployment.xml").getroot()
        defaults = deployment.find("default/joint").attrib
        for node in deployment.findall(".//worldbody//joint"):
            name = node.get("name")
            if name not in self.__joint_names:
                continue
            va = self.__model.joint(name).dofadr[0]
            for field in ("armature", "damping", "frictionloss"):
                getattr(self.__model, "dof_" + field)[va] = float(
                    node.get(field, defaults.get(field, 0)))
        # Randomized fixtures are set before control starts. They are never
        # supplied to perception or to the body policy.
        self.__model.body_pos[self.__model.body("analyzer").id] += button_shift
        mujoco.mj_setConst(self.__model, self.__data)
        mujoco.mj_resetData(self.__model, self.__data)
        root_qa = self.__model.joint("floating_base_joint").qposadr[0]
        self.__data.qpos[root_qa:root_qa + 3] = (*station, .76)
        self.__data.qpos[root_qa + 3:root_qa + 7] = (1, 0, 0, 0)
        self.__data.qpos[self.__qadr] = meta["default_dof_pos"]
        self.__data.qpos[self.__hand_qadr] = 0
        box_qa = self.__model.joint("sample_box_free").qposadr[0]
        self.__data.qpos[box_qa:box_qa + 2] += box_shift
        bottle_qa = self.__model.joint("sample_bottle_free").qposadr[0]
        self.__data.qpos[bottle_qa:bottle_qa + 3] += bottle_shift
        if self.__has_chair:
            chair_qa = self.__model.joint("push_chair_free").qposadr[0]
            self.__data.qpos[chair_qa:chair_qa + 2] += chair_shift
        self.__data.qvel[:] = 0
        self.__data.ctrl[:] = 0
        self.__data.eq_active[:] = 0
        self.__data.xfrc_applied[:] = 0
        self.__data.qfrc_applied[:] = 0
        mujoco.mj_forward(self.__model, self.__data)
        self.__imu_wxyz = np.array([1., 0, 0, 0])
        self.__gyro = self.__native_gyro()
        self.__imu_site_offset = self.__model.site('imu_in_pelvis').pos.copy()
        self.__imu_site_displacement = np.zeros(3)
        self.__imu_velocity = np.zeros(3)
        self.__accelerometer = self.__data.sensor('imu-pelvis-linear-acceleration').data.copy()
        self.__previous_world_acceleration = self.__accelerometer + np.array([0.,0.,-9.81])
        # The audit has its own private access to truth and never returns a
        # value to the control loop. evaluate() exposes it only after stopping.
        self.__audit_env = SimpleNamespace(model=self.__model, data=self.__data)
        self.__audit = ContactAudit(self.__audit_env)
        self.__initial_box = self.__data.body("sample_box").xpos.copy()
        self.__initial_bottle = self.__data.body("sample_bottle").xpos.copy()
        self.__bottle_audit = _BottleAudit(self.__model, self.__data)
        self.__chair_audit = ChairAudit(self.__model, self.__data) if self.__has_chair else None
        self.__chair_actuators = np.array([i for i in range(self.__model.nu)
            if self.__model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
            and self.__has_chair
            and int(self.__model.jnt_bodyid[self.__model.actuator_trnid[i, 0]])
                in self.__chair_audit.chair_ids], dtype=int)
        self.__max_chair_actuator_control = 0.
        self.__record()

    @staticmethod
    def __vector(value, size, name):
        result = np.asarray(value, dtype=float)
        if result.shape != (size,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must contain {size} finite values")
        return result.copy()

    def __require_open(self):
        if self.__closed:
            raise RuntimeError("VisionRobotIO is closed")

    def __native_gyro(self):
        return self.__data.sensor("imu-pelvis-angular-velocity").data.copy()

    def sensors(self):
        """Return copies of robot encoders and the gyro-based IMU estimate."""
        self.__require_open()
        return {
            "joint_pos": self.__data.qpos[self.__qadr].copy(),
            "joint_vel": self.__data.qvel[self.__vadr].copy(),
            "imu_wxyz": self.__imu_wxyz.copy(),
            "gyro": self.__gyro.copy(),
        }

    def set_hand_targets(self, targets):
        """Set named finger positions in radians; unspecified targets persist.

        Unknown joints and nonfinite/non-scalar values are rejected atomically.
        Finite positions are clipped to the robot's joint limits. This method
        issues no simulator step and returns no environment feedback.
        """
        self.__require_open()
        if self.__evaluated:
            raise RuntimeError("Evaluation ended this rollout; truth cannot feed further control")
        if not isinstance(targets, Mapping):
            raise TypeError("targets must map hand joint names to scalar radians")
        updated = self.__hand_targets.copy()
        for name, value in targets.items():
            if name not in self.__hand_indices:
                raise ValueError(f"Unknown hand joint: {name!r}")
            scalar = np.asarray(value, dtype=float)
            if scalar.shape != () or not np.isfinite(scalar):
                raise ValueError(f"Target for {name} must be a finite scalar")
            i = self.__hand_indices[name]
            updated[i] = np.clip(float(scalar), *self.__hand_ranges[i])
        self.__hand_targets[:] = updated

    def hand_sensors(self):
        """Return only finger position/velocity encoders, keyed by joint name."""
        self.__require_open()
        return {
            "joint_pos": dict(zip(self.__hand_names, self.__data.qpos[self.__hand_qadr].tolist())),
            "joint_vel": dict(zip(self.__hand_names, self.__data.qvel[self.__hand_vadr].tolist())),
        }

    def __integrate_attitude(self, gyro):
        # A body-frame incremental rotation right-multiplies the orientation.
        angle = np.linalg.norm(gyro) * self.__dt
        half = .5 * angle
        dq = np.r_[np.cos(half), gyro * (.5 * self.__dt * np.sinc(half / np.pi))]
        q = self.__imu_wxyz
        self.__imu_wxyz = np.r_[q[0] * dq[0] - np.dot(q[1:], dq[1:]),
            q[0] * dq[1:] + dq[0] * q[1:] + np.cross(q[1:], dq[1:])]
        self.__imu_wxyz /= np.linalg.norm(self.__imu_wxyz)

    def __rotate_imu(self, vector):
        q=self.__imu_wxyz
        t=2*np.cross(q[1:],vector)
        return vector+q[0]*t+np.cross(q[1:],t)

    def __integrate_translation(self):
        # Free inertial dead reckoning, from a zero-velocity reset. Specific
        # force is native accelerometer data; gravity and IMU mounting are known
        # calibration. No base qpos/qvel or scene target is used here.
        self.__accelerometer = self.__data.sensor('imu-pelvis-linear-acceleration').data.copy()
        acceleration=self.__rotate_imu(self.__accelerometer)+np.array([0.,0.,-9.81])
        mean=.5*(self.__previous_world_acceleration+acceleration)
        self.__imu_site_displacement += self.__imu_velocity*self.__dt + .5*mean*self.__dt**2
        self.__imu_velocity += mean*self.__dt
        self.__previous_world_acceleration=acceleration

    def imu_odometry(self):
        """Return an IMU-only relative position estimate, not simulator pose.

        This short, ideal-sensor experiment does not model real IMU bias/drift.
        Position compensates the known pelvis-to-IMU mounting lever arm.
        """
        self.__require_open()
        position=(self.__imu_site_displacement+self.__imu_site_offset
                  -self.__rotate_imu(self.__imu_site_offset))
        return {'position':position.copy(),'site_velocity':self.__imu_velocity.copy(),
                'accelerometer':self.__accelerometer.copy()}

    def correct_imu_velocity(self, encoder_root_velocity):
        """Fuse a robot-FK velocity estimate during commanded double support.

        The caller derives this vector from encoders and IMU, assuming planted
        feet. No simulator contact, free-base velocity, or world pose is read.
        """
        self.__require_open()
        if self.__evaluated:raise RuntimeError('Evaluation ended this rollout')
        velocity=self.__vector(encoder_root_velocity,3,'encoder_root_velocity')
        site_velocity=velocity+self.__rotate_imu(np.cross(self.__gyro,self.__imu_site_offset))
        self.__imu_velocity[:2]=.5*self.__imu_velocity[:2]+.5*site_velocity[:2]

    def step(self, desired29):
        """Apply 29 body PD targets and the current finger targets for 20 ms."""
        self.__require_open()
        if self.__evaluated:
            raise RuntimeError("Evaluation ended this rollout; truth cannot feed further control")
        desired = self.__vector(desired29, 29, "desired29")
        for _ in range(self.__ticks):
            gyro_before = self.__gyro
            self.__data.ctrl[:] = 0
            torque = self.__kp * (desired - self.__data.qpos[self.__action_qadr])
            torque -= self.__kd * self.__data.qvel[self.__action_vadr]
            self.__data.ctrl[self.__aadr] = np.clip(torque, -self.__limits, self.__limits)
            finger_torque = 15 * (self.__hand_targets - self.__data.qpos[self.__hand_qadr])
            finger_torque -= .4 * self.__data.qvel[self.__hand_vadr]
            self.__data.ctrl[self.__hand_actuators] = np.clip(
                finger_torque, self.__hand_ctrl_low, self.__hand_ctrl_high)
            mujoco.mj_step(self.__model, self.__data)
            # Refresh native sensor readings after integration; this also makes
            # the independent audit and subsequent RGB frames time-consistent.
            mujoco.mj_forward(self.__model, self.__data)
            self.__gyro = self.__native_gyro()
            self.__integrate_attitude(.5 * (gyro_before + self.__gyro))
            self.__integrate_translation()
            self.__audit.tick(self.__audit_env)
            self.__bottle_audit.tick(self.__data)
            if self.__chair_audit is not None:
                self.__chair_audit.tick(self.__data)
                if len(self.__chair_actuators):
                    self.__max_chair_actuator_control = max(self.__max_chair_actuator_control,
                        float(np.max(np.abs(self.__data.ctrl[self.__chair_actuators]))))
        self.__record()

    def images(self):
        """Return synchronized left/right RGB images, without privileged buffers."""
        self.__require_open()
        if self.__renderer is None:
            self.__renderer = mujoco.Renderer(self.__model, width=640, height=480)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = (1, 1, 1, 1, 0, 1)
        opt.sitegroup[:] = 0
        images = []
        for camera in ("vision_left", "vision_right"):
            self.__renderer.update_scene(self.__data, camera=camera, scene_option=opt)
            self.__renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self.__renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
            images.append(self.__renderer.render().copy())
        return tuple(images)

    def __record(self):
        if self.__out is None:
            return
        d = self.__data
        row = (float(d.time), d.qpos.copy(), d.qvel.copy(),
            d.body("sample_box").xpos.copy(), d.body("sample_bottle").xpos.copy(),
            float(d.joint("button_slide").qpos[0]), float(d.joint("drawer_slide").qpos[0]),
            self.__imu_wxyz.copy(), self.__gyro.copy(), self.__hand_targets.copy())
        if self.__has_chair:
            row += (d.body("push_chair").xpos.copy(), d.body("push_chair").xquat.copy())
        self.__records.append(row)

    def __save_recording(self):
        if self.__out is None or not self.__records:
            return
        names = ("time", "qpos", "qvel", "sample_box", "sample_bottle",
                 "button_travel", "drawer_open", "imu_wxyz", "gyro", "hand_targets")
        if self.__has_chair:
            names += ("push_chair", "push_chair_quaternion_wxyz")
        arrays = {name: np.asarray([r[i] for r in self.__records])
                  for i, name in enumerate(names)}
        np.savez_compressed(self.__out / "trajectory.npz", **arrays)
        np.savez_compressed(self.__out / "bottle_audit.npz", **self.__bottle_audit.arrays())
        if self.__chair_audit is not None:
            np.savez_compressed(self.__out / "chair_audit.npz", **self.__chair_audit.arrays())

    def evaluate(self, task):
        """End the rollout and report independent physical truth for assessment."""
        self.__require_open()
        if task not in ("button", "push_box", "standing", "grasp_lift", "pick_place", "push_chair"):
            raise ValueError("task must be button, push_box, standing, grasp_lift, pick_place or push_chair")
        if task == "push_chair" and self.__chair_audit is None:
            raise ValueError("push_chair evaluation requires a movable chair scene")
        self.__evaluated = True
        d, audit = self.__data, self.__audit
        final_box = d.body("sample_box").xpos.copy()
        box_delta = final_box - self.__initial_box
        box_tilt = float(np.degrees(np.arccos(np.clip(
            d.body("sample_box").xmat.reshape(3, 3)[2, 2], -1, 1))))
        final_button = float(d.joint("button_slide").qpos[0])
        target_body = {"button": "instrument_button", "push_box": "sample_box",
                       "grasp_lift": "sample_bottle", "pick_place": "sample_bottle",
                       "push_chair": "push_chair"}.get(task)
        bottle_report = self.__bottle_audit.report()
        chair_report = self.__chair_audit.report() if self.__chair_audit is not None else {}
        contact = audit.hand_contact(target_body) if target_body else False
        if task == "push_chair":
            contact = chair_report["chair_objective_checks"]["hand_contact"]
        unassisted = (audit.max_assist_control == 0 and audit.max_external_force == 0
                      and not audit.active_equality_seen and self.__max_chair_actuator_control == 0)
        upright = bool(audit.min_height > .45 and np.isfinite(d.qpos).all())
        if task == "button":
            objective = contact and audit.max_button > .008 and final_button < .002
        elif task == "push_box":
            objective = (contact and np.linalg.norm(box_delta[:2]) >= .04
                         and .825 < final_box[2] < .85 and box_tilt < 15)
        elif task == "grasp_lift":
            objective = bottle_report["bottle_lift_achieved"]
        elif task == "pick_place":
            objective = bottle_report["bottle_pick_place_achieved"]
        elif task == "push_chair":
            objective = chair_report["chair_push_achieved"]
        else:
            objective = upright
        true_quat = d.qpos[self.__model.joint("floating_base_joint").qposadr[0] + 3:][:4]
        attitude_error = 2 * np.arccos(np.clip(abs(np.dot(true_quat, self.__imu_wxyz)), 0, 1))
        report = {
            "task": task, "success": bool(objective and upright and unassisted),
            "objective_achieved": bool(objective), "duration_s": float(d.time),
            "upright": upright, "minimum_pelvis_height_m": audit.min_height,
            "hand_target_contact": contact, "max_button_travel_m": audit.max_button,
            "final_button_travel_m": final_button,
            "button_trigger_time_s": audit.button_trigger_time,
            "initial_box_position": self.__initial_box.tolist(),
            "final_box_position": final_box.tolist(), "box_displacement_m": box_delta.tolist(),
            "box_displacement_xy_m": float(np.linalg.norm(box_delta[:2])),
            "box_final_tilt_deg": box_tilt,
            "bottle_displacement_m": (d.body("sample_bottle").xpos - self.__initial_bottle).tolist(),
            "final_drawer_open_m": float(d.joint("drawer_slide").qpos[0]),
            "minimum_drawer_open_m": audit.min_drawer,
            "unassisted": bool(unassisted), "max_object_assist_control": audit.max_assist_control,
            "max_external_applied_force": audit.max_external_force,
            "active_equality_seen": bool(audit.active_equality_seen),
            "max_chair_actuator_control": self.__max_chair_actuator_control,
            "contacts": list(audit.pairs.values()),
            "non_target_contacts": [x for x in audit.pairs.values()
                if x["other_body"] != target_body and not
                    (task == "push_chair" and x["other_body"].startswith("push_chair"))],
            "imu_attitude_error_rad_postrun_only": float(attitude_error),
            "evaluation_feedback_to_controller": False,
            "hand_joint_names": self.__hand_names,
            "final_hand_targets_rad": dict(zip(self.__hand_names, self.__hand_targets.tolist())),
            "hand_pd_stiffness_nm_per_rad": 15., "hand_pd_damping_nms_per_rad": .4,
        }
        report.update(bottle_report)
        report.update(chair_report)
        self.__save_recording()
        if self.__out:
            (self.__out / "evaluator.json").write_text(json.dumps(report, indent=2) + "\n")
        return report

    def close(self):
        if not self.__closed:
            self.__save_recording()
            if self.__renderer is not None:
                self.__renderer.close()
                self.__renderer = None
            self.__closed = True
