import json
import unittest
import numpy as np
import mujoco
from lab_sim.env import LabEnv
from lab_sim.psi0 import Psi0Adapter,HAND_NAMES,ARM_NAMES,encode_numpy,decode_numpy
from lab_sim.navigation import plan_path,obstacles


class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=LabEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.assisted=True
        self.env.reset()

    def test_passive_objects_settle_and_support_is_optional(self):
        e=self.env
        e.step(nstep=1000)
        self.assertTrue(np.isfinite(e.data.qpos).all())
        self.assertLess(np.linalg.norm(e.data.qpos[:3]-[1.72,-.8,.79]),.01)
        self.assertAlmostEqual(e.body("sample_bottle")[2],.873,delta=.01)
        np.testing.assert_allclose(e.body("sample_box"),[2.10,-1.23,.838],atol=.015)
        self.assertEqual(len(e.robot_actuators),43)
        e.set_assistance(False)
        self.assertFalse(e.data.eq_active.any())
        with self.assertRaises(RuntimeError):e.attach()

    def test_psi0_finger_order_and_roundtrip(self):
        e=self.env;a=Psi0Adapter(e)
        self.assertEqual(HAND_NAMES[3],"left_hand_middle_0_joint")
        self.assertEqual(HAND_NAMES[10],"right_hand_index_0_joint")
        for i,n in enumerate(HAND_NAMES+ARM_NAMES):
            e.data.qpos[e.model.joint(n).qposadr[0]]=i/100
        payload=a.observation("test",render=False)
        np.testing.assert_allclose(payload["state"]["states"][0,:28],np.arange(28)/100,atol=1e-7)
        result=json.loads(json.dumps(payload,default=encode_numpy),object_hook=decode_numpy)
        np.testing.assert_array_equal(result["state"]["states"],payload["state"]["states"])
        self.assertEqual(result["image"]["rgb_head_stereo_left"].dtype,np.uint8)
        with self.assertRaises(ValueError):a.apply(np.full(36,np.nan))
        with self.assertRaises(ValueError):a.apply(np.zeros(35))

    def test_psi0_apply_uses_joint_names_and_physics_mode_requires_controller(self):
        e=self.env;a=Psi0Adapter(e)
        action=np.r_[e.data.qpos[a.qadr],0,0,0,.79,0,0,0,0]
        action[3]=-.2  # Left middle, not left index.
        action[10]=.25 # Right index, not right middle.
        a.apply(action)
        self.assertNotIn("reset",a.observation("test",render=False)["history"])
        self.assertAlmostEqual(e.target[e.names.index("left_hand_middle_0_joint")],-.2)
        self.assertAlmostEqual(e.target[e.names.index("right_hand_index_0_joint")],.25)
        e.set_assistance(False)
        with self.assertRaises(RuntimeError):a.apply(action)

    def test_navigation_routes_avoid_furniture(self):
        e=self.env
        bounds=obstacles(e.model,e.data)
        route=plan_path(e.model,e.data,(0,-2.8),(0,2.5))
        self.assertGreater(len(route),5)
        for p in route:
            self.assertFalse(any(np.all(np.array(p)>=lo) and np.all(np.array(p)<=hi) for lo,hi in bounds))
        with self.assertRaises(ValueError):plan_path(e.model,e.data,(2.6,-2),(0,0))


if __name__=="__main__":unittest.main()
