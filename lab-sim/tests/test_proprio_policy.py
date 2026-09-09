"""Sensor-boundary and kinematic checks for the visual controller's body policy."""
import copy
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from lab_sim.scalebfm_proprio import ASSETS, FIVE_BODIES, ProprioScaleBFM


class ProprioPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = ProprioScaleBFM()
        # Independent FK oracle: only the robot model, with an explicit sensor
        # interface. The policy never receives this model or its data object.
        xml = ET.Element("mujoco")
        ET.SubElement(xml, "compiler", angle="radian")
        world = ET.SubElement(xml, "worldbody")
        body = copy.deepcopy(ET.parse(ASSETS / "kinematics.xml").find("worldbody/body"))
        for node in body.iter():
            node.attrib.pop("class", None)
            node.attrib.pop("childclass", None)
            for child in list(node):
                if child.tag == "geom":
                    node.remove(child)
        world.append(body)
        sensors = ET.SubElement(xml, "sensor")
        ET.SubElement(sensors, "framequat", name="orientation", objtype="site",
                      objname="imu_in_pelvis")
        ET.SubElement(sensors, "gyro", name="gyro", site="imu_in_pelvis")
        for i, name in enumerate(cls.policy.names):
            ET.SubElement(sensors, "jointpos", name=f"q{i}", joint=name)
            ET.SubElement(sensors, "jointvel", name=f"v{i}", joint=name)
        cls.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        cls.data = mujoco.MjData(cls.model)
        cls.qadr = np.array([cls.model.joint(n).qposadr[0] for n in cls.policy.names])

    def setUp(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.policy.default
        mujoco.mj_forward(self.model, self.data)

    def sensor_packet(self):
        return (np.array([self.data.sensor(f"q{i}").data[0] for i in range(29)]),
                np.array([self.data.sensor(f"v{i}").data[0] for i in range(29)]),
                self.data.sensor("orientation").data.copy(),
                self.data.sensor("gyro").data.copy())

    def hold_action(self):
        p, q = self.policy.poses()
        return self.policy.infer(np.broadcast_to(p, (6, 5, 3)),
                                 np.broadcast_to(q, (6, 5, 4)))

    def test_sensor_only_world_translation_and_heading_independence(self):
        self.data.qvel[3:6] = (.05, -.10, .02)
        mujoco.mj_forward(self.model, self.data)
        self.policy.reset(*self.sensor_packet())
        original_poses = self.policy.poses()
        original_action = self.hold_action()
        # The hidden physical world is translated and rotated. Only sensor
        # readings cross into the controller, never this translation vector.
        self.data.qpos[:3] = (17, -22, 4)
        yaw = 1.1
        self.data.qpos[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        mujoco.mj_forward(self.model, self.data)
        self.policy.reset(*self.sensor_packet())
        translated_action = self.hold_action()
        np.testing.assert_allclose(self.policy.root_position, 0, atol=1e-12)
        for before, after in zip(original_poses, self.policy.poses()):
            np.testing.assert_allclose(before, after, atol=1e-12)
        for before, after in zip(original_action, translated_action):
            np.testing.assert_allclose(before, after, atol=2e-5, rtol=2e-5)

    def test_robot_fk_agrees_with_independent_mujoco_robot_model(self):
        rng = np.random.default_rng(11)
        self.data.qpos[self.qadr] += rng.uniform(-.1, .1, 29)
        self.data.qpos[:3] = (-8, 3, 1.2)
        # Tilt about x leaves initial heading zero and tests gravity alignment.
        self.data.qpos[3:7] = (np.cos(.08), np.sin(.08), 0, 0)
        mujoco.mj_forward(self.model, self.data)
        self.policy.reset(*self.sensor_packet())
        positions, quaternions = self.policy.poses()
        for i, name in enumerate(FIVE_BODIES):
            expected = self.data.body(name).xpos - self.data.qpos[:3]
            np.testing.assert_allclose(positions[i], expected, atol=2e-7)
            np.testing.assert_allclose(quaternions[i], self.data.body(name).xquat,
                                       atol=2e-7)
        torso_pos, torso_quat = self.policy.torso_pose()
        np.testing.assert_allclose(torso_pos,
            self.data.body("torso_link").xpos - self.data.qpos[:3], atol=2e-7)
        np.testing.assert_allclose(torso_quat, self.data.body("torso_link").xquat,
                                   atol=2e-7)

    def test_encoder_squat_estimates_root_descent_and_preserves_action_history(self):
        self.policy.reset(*self.sensor_packet())
        original_feet = self.policy.poses()[0][3:].copy()
        _, raw_action = self.hold_action()
        for side in ("left", "right"):
            for joint, delta in (("hip_pitch", -.05), ("knee", .10), ("ankle_pitch", -.05)):
                self.data.joint(f"{side}_{joint}_joint").qpos[0] += delta
        mujoco.mj_forward(self.model, self.data)
        self.policy.update(*self.sensor_packet())
        self.assertLess(self.policy.root_position[2], -.005)
        np.testing.assert_allclose(self.policy.poses()[0][3:], original_feet, atol=1e-7)
        np.testing.assert_allclose(self.policy.history[4][0, -1].cpu(), raw_action)
        self.assertEqual(self.policy.history[4].shape, (1, 3, 29))


if __name__ == "__main__":
    unittest.main()
