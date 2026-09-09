"""Short recorder check for the PICO teleoperation session."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lab_sim.pico_teleop import PicoTeleopSession


class PicoTeleopTests(unittest.TestCase):
    def test_offline_session_saves_native_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            session = PicoTeleopSession(
                output_dir=output,
                live=False,
                offline_motion=None,
            )
            report = session.run(seconds=0.1)
            self.assertEqual(report["stop_reason"], "duration")
            trajectory = output / "trajectory.npz"
            self.assertTrue(trajectory.is_file())
            with np.load(trajectory) as saved:
                required = (
                    "time",
                    "qpos",
                    "qvel",
                    "body_target",
                    "body_torque",
                    "hand_targets",
                )
                lengths = [len(saved[name]) for name in required]
                self.assertGreater(lengths[0], 0)
                self.assertEqual(len(set(lengths)), 1)
                for name in required:
                    self.assertTrue(np.isfinite(saved[name]).all(), name)
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertFalse(metadata["assisted"])
            self.assertEqual(metadata["physics_hz"], 500)
            self.assertEqual(metadata["policy_hz"], 50)
            self.assertFalse(metadata["equality_active"])


if __name__ == "__main__":
    unittest.main()
