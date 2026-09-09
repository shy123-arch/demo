"""Integration checks for the official five-point policy and pose commands."""
import unittest
import numpy as np

from lab_sim.env import LabEnv
from lab_sim.scalebfm import ScaleBFMController, FIVE_BODIES
from lab_sim.scalebfm_demo import PoseSequence, direct_program


class ScaleBFMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = LabEnv(assisted=False)
        cls.controller = ScaleBFMController(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.controller.reset(self.env)

    def hold_targets(self):
        pos, quat = self.controller.poses(self.env)
        return np.broadcast_to(pos, (6, 5, 3)), np.broadcast_to(quat, (6, 5, 4))

    def test_world_translation_and_heading_do_not_change_initial_action(self):
        c, env = self.controller, self.env
        pos, quat = self.hold_targets()
        original = c.advance(env, pos, quat)
        c.reset(env, xy=(.4, -.3), yaw=np.pi/2)
        pos, quat = self.hold_targets()
        transformed = c.advance(env, pos, quat)
        np.testing.assert_allclose(transformed, original, atol=2e-5, rtol=2e-5)
        self.assertEqual(original.shape, (29,))
        self.assertEqual(len(set(c.aadr)), 29)
        self.assertFalse(set(c.aadr) & set(env.robot_actuators[c.hand_idx]))

    def test_five_point_standing_is_physical_and_targets_are_validated(self):
        c, env = self.controller, self.env
        pos, quat = self.hold_targets()
        for _ in range(100):
            c.advance(env, pos, quat)
        self.assertFalse(env.assisted)
        self.assertFalse(env.data.eq_active.any())
        self.assertGreater(env.data.qpos[2], .7)
        self.assertLess(np.max(np.linalg.norm(c.poses(env)[0] - pos[0], axis=-1)), .07)
        with self.assertRaises(ValueError): c.advance(env, pos[:, :4], quat)
        with self.assertRaises(ValueError): c.advance(env, pos, quat * 2)
        with self.assertRaises(ValueError): c.advance(env, pos * np.nan, quat)

    def test_pose_sequence_boundaries_and_rotation_interpolation(self):
        pos, quat = self.controller.poses(self.env)
        frames = direct_program(pos, quat)
        # Exercise changing rotations and accumulated floating-point frame times.
        for f in frames[1:]:
            f["poses"][FIVE_BODIES[1]]["quaternion_wxyz"] = [np.cos(.1), 0, 0, np.sin(.1)]
        sequence = PoseSequence(frames)
        times = np.r_[-1, np.arange(0, 20.11, .02), 21]
        p, q = sequence.sample(times)
        self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-12)
        np.testing.assert_allclose(p[0], pos)
        np.testing.assert_allclose(p[-1], pos)


if __name__ == "__main__":
    unittest.main()
