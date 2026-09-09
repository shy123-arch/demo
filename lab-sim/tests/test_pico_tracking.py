"""Focused checks for the MolmoSpaces-to-lab-sim tracking adapter."""
import unittest

import mujoco
import numpy as np

from lab_sim.env import ROOT
from lab_sim.pico_tracking import (
    BodyPDController,
    Dex3HandMapper,
    LabSimStateBridge,
    load_molmo_tracking,
)


class PicoTrackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(
            str(ROOT / "scenes" / "lab_g1_vision.xml")
        )
        cls.data = mujoco.MjData(cls.model)
        cls.bindings = load_molmo_tracking()

    def setUp(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def test_state_bridge_has_exact_29d_contract(self):
        hands = Dex3HandMapper(self.model, self.data)
        bridge = LabSimStateBridge(self.model, self.data, self.bindings, hands)
        self.assertEqual(bridge.qj_real.shape, (29,))
        self.assertEqual(bridge.qj_isaac.shape, (29,))
        anchor = bridge.current_reference_anchor()
        self.assertEqual(anchor["joint_pos"].shape, (29,))
        self.assertEqual(anchor["root_pos"].shape, (3,))
        self.assertEqual(anchor["root_quat"].shape, (4,))
        self.assertTrue(np.isfinite(np.r_[bridge.qj_real, bridge.qj_isaac]).all())

    def test_dex3_mapping_is_mirrored_and_independent(self):
        hands = Dex3HandMapper(self.model, self.data)
        np.testing.assert_allclose(hands.targets, 0)
        hands.set_from_controller_buttons({"left_trigger_value": 1.0})
        left = hands.targets[:7].copy()
        right = hands.targets[7:].copy()
        self.assertGreater(np.linalg.norm(left), 0.1)
        np.testing.assert_allclose(right, 0)
        hands.set_from_controller_buttons({"right_trigger_value": 1.0})
        self.assertGreater(np.linalg.norm(hands.targets[7:]), 0.1)
        for target, (lo, hi) in zip(hands.targets, hands.ranges):
            self.assertGreaterEqual(target, lo)
            self.assertLessEqual(target, hi)

    def test_body_pd_output_is_finite_and_limited(self):
        hands = Dex3HandMapper(self.model, self.data)
        bridge = LabSimStateBridge(self.model, self.data, self.bindings, hands)
        pd = BodyPDController(self.model, self.data, self.bindings)
        torque = pd.compute(bridge.default_qpos_real)
        self.assertEqual(torque.shape, (29,))
        self.assertTrue(np.isfinite(torque).all())
        self.assertTrue(np.all(torque >= pd.torque_low))
        self.assertTrue(np.all(torque <= pd.torque_high))

    def test_nonfinite_hand_input_holds_previous_target(self):
        hands = Dex3HandMapper(self.model, self.data)
        hands.set_from_controller_buttons({"left_trigger_value": 0.5})
        previous = hands.targets[:7].copy()
        hands.set_from_controller_buttons({"left_trigger_value": float("nan")})
        np.testing.assert_array_equal(hands.targets[:7], previous)
        self.assertTrue(np.isfinite(hands.compute_torque()).all())


if __name__ == "__main__":
    unittest.main()
