"""Private physical chair-push audit, never exposed during robot control.

The chair and every descendant body (including passive casters) are one target.
World coordinates and contact forces here are independent scoring evidence.
"""
import mujoco
import numpy as np


def _subtree(model, root):
    result = set()
    for body in range(model.nbody):
        ancestor = body
        while ancestor:
            if ancestor == root:
                result.add(body)
                break
            ancestor = int(model.body_parentid[ancestor])
    return result


class ChairAudit:
    """Record chair/robot state and load-bearing contact at each 2 ms tick."""

    def __init__(self, model, data):
        self.model = model
        self.body_id = model.body("push_chair").id
        self.chair_ids = _subtree(model, self.body_id)
        self.robot_id = model.body("pelvis").id
        self.robot_ids = _subtree(model, self.robot_id)
        self.hand_ids = {i for i in self.robot_ids
            if "_hand_" in (model.body(i).name or "")} | {
            model.body(side + "_wrist_yaw_link").id for side in ("left", "right")}
        self.floor_id = model.geom("lab_floor_slab").id
        self.velocity_address = model.joint("push_chair_free").dofadr[0]
        self.initial_position = data.body(self.body_id).xpos.copy()
        self.initial_quaternion = data.body(self.body_id).xquat.copy()
        self.rows = []
        self.force = np.zeros(6)
        self.contacts = {}
        self.tick(data)

    def tick(self, data):
        m = self.model
        chair = data.body(self.body_id)
        tilt = np.degrees(np.arccos(np.clip(chair.xmat.reshape(3, 3)[2, 2], -1, 1)))
        robot_tilt = np.degrees(np.arccos(np.clip(
            data.body(self.robot_id).xmat.reshape(3, 3)[2, 2], -1, 1)))
        velocity = data.qvel[self.velocity_address:self.velocity_address + 3].copy()
        flags = {name: False for name in ("hand", "nonhand_robot", "floor", "environment")}
        forces = {name: 0. for name in flags}
        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if (b1 in self.chair_ids) == (b2 in self.chair_ids):
                continue
            chair_geom, other_geom, other_body = ((g1, g2, b2)
                if b1 in self.chair_ids else (g2, g1, b1))
            mujoco.mj_contactForce(m, data, ci, self.force)
            normal = abs(float(self.force[0]))
            if normal < .01:
                continue
            if other_body in self.hand_ids:
                kind = "hand"
            elif other_body in self.robot_ids:
                kind = "nonhand_robot"
            elif other_geom == self.floor_id:
                kind = "floor"
            else:
                kind = "environment"
            flags[kind] = True
            forces[kind] += normal
            names = (m.geom(chair_geom).name or f"geom_{chair_geom}",
                     m.geom(other_geom).name or f"geom_{other_geom}")
            key = names
            if key not in self.contacts:
                self.contacts[key] = {
                    "chair_geom": names[0], "other_geom": names[1],
                    "other_body": m.body(other_body).name or "world", "kind": kind,
                    "first_time_s": float(data.time), "last_time_s": float(data.time),
                    "contact_samples": 0, "max_normal_force_n": 0.,
                    "normal_impulse_ns": 0.,
                }
            entry = self.contacts[key]
            entry["last_time_s"] = float(data.time)
            entry["contact_samples"] += 1
            entry["max_normal_force_n"] = max(entry["max_normal_force_n"], normal)
            if data.time > 0:
                entry["normal_impulse_ns"] += normal * m.opt.timestep
        self.rows.append((float(data.time), *chair.xpos, *chair.xquat, *velocity,
            float(tilt), float(robot_tilt), float(data.body(self.robot_id).xpos[2]),
            *flags.values(), *forces.values()))

    def arrays(self):
        values = np.asarray(self.rows)
        names = ("time", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz",
                 "tilt_deg", "robot_tilt_deg", "robot_pelvis_height_m", "hand_contact",
                 "nonhand_robot_contact", "floor_contact", "environment_contact",
                 "hand_normal_force_n", "nonhand_robot_normal_force_n",
                 "floor_normal_force_n", "environment_normal_force_n")
        result = {name: values[:, i] for i, name in enumerate(names)}
        for name in ("hand_contact", "nonhand_robot_contact", "floor_contact", "environment_contact"):
            result[name] = result[name].astype(bool)
        return result

    def report(self):
        a = self.arrays()
        dt = float(self.model.opt.timestep)
        positions = np.column_stack((a["x"], a["y"], a["z"]))
        quaternions = np.column_stack((a["qw"], a["qx"], a["qy"], a["qz"]))
        delta = positions[-1] - self.initial_position
        speed = np.hypot(a["vx"], a["vy"])
        finite = bool(np.isfinite(np.asarray(self.rows)).all())
        duration = lambda key: float(np.sum(a[key][1:]) * dt)
        checks = {
            "finite_states": finite,
            "forward_displacement": bool(delta[0] >= .08),
            "hand_contact": bool(np.any(a["hand_contact"][1:])),
            "no_nonhand_robot_contact": bool(not np.any(a["nonhand_robot_contact"])),
            "chair_upright_throughout": bool(np.max(a["tilt_deg"]) < 15),
            "chair_final_upright": bool(a["tilt_deg"][-1] < 5),
            "chair_final_stopped": bool(speed[-1] < .03),
            "chair_final_floor_supported": bool(a["floor_contact"][-1]),
            "robot_upright_throughout": bool(np.max(a["robot_tilt_deg"]) < 25),
        }
        return {
            "initial_chair_position": self.initial_position.tolist(),
            "initial_chair_quaternion_wxyz": self.initial_quaternion.tolist(),
            "final_chair_position": positions[-1].tolist(),
            "final_chair_quaternion_wxyz": quaternions[-1].tolist(),
            "chair_displacement_m": delta.tolist(),
            "chair_forward_displacement_m": float(delta[0]),
            "chair_planar_displacement_m": float(np.linalg.norm(delta[:2])),
            "chair_max_forward_displacement_m": float(np.max(positions[:, 0] - positions[0, 0])),
            "chair_max_tilt_deg": float(np.max(a["tilt_deg"])),
            "chair_final_tilt_deg": float(a["tilt_deg"][-1]),
            "chair_max_planar_speed_m_s": float(np.max(speed)),
            "chair_final_planar_speed_m_s": float(speed[-1]),
            "chair_final_hand_contact": bool(a["hand_contact"][-1]),
            "chair_hand_contact_duration_s": duration("hand_contact"),
            "chair_nonhand_robot_contact_duration_s": duration("nonhand_robot_contact"),
            "chair_floor_contact_duration_s": duration("floor_contact"),
            "chair_environment_contact_duration_s": duration("environment_contact"),
            "chair_final_floor_supported": bool(a["floor_contact"][-1]),
            "chair_max_hand_normal_force_n": float(np.max(a["hand_normal_force_n"])),
            "chair_max_nonhand_robot_normal_force_n": float(np.max(a["nonhand_robot_normal_force_n"])),
            "chair_hand_normal_impulse_ns": float(np.sum(a["hand_normal_force_n"][1:]) * dt),
            "chair_contacts": list(self.contacts.values()),
            "robot_max_tilt_deg": float(np.max(a["robot_tilt_deg"])),
            "robot_final_tilt_deg": float(a["robot_tilt_deg"][-1]),
            "chair_objective_checks": checks,
            "chair_push_achieved": bool(all(checks.values())),
            "chair_evaluation_criteria": {
                "minimum_forward_displacement_m": .08,
                "forward_axis_world": [1., 0., 0.],
                "chair_max_tilt_deg_below": 15., "chair_final_tilt_deg_below": 5.,
                "chair_final_planar_speed_m_s_below": .03,
                "robot_max_tilt_deg_below": 25.,
                "hand_contact_required": True, "nonhand_robot_contact_allowed": False,
                "final_floor_support_required": True,
                "minimum_contact_normal_force_n": .01,
                "audit_frequency_hz": 1. / dt,
                "unassisted_required": True,
                "unassisted_check": "VisionRobotIO independent assistance/equality/force audit",
            },
        }
