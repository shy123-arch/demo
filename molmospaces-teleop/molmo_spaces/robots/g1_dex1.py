"""Free-base G1 robot with motion tracking and Dex1-1 grippers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mujoco import MjData

from molmo_spaces.controllers.joint_pd_torque import JointPDTorqueController
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.dummy_parallel_kinematics import DummyParallelKinematics
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.g1_dex1_constants import BODY_KD, BODY_KP

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig


class G1Dex1Robot(Robot):
    """Control surface used by the bundled motion-tracking policy.

    ``body`` accepts absolute 29D joint targets in hardware joint order.
    ``left_dex1`` and ``right_dex1`` each accept two identical finger targets.
    The free base is state-only and moves through legged dynamics.
    """

    def __init__(self, mj_data: MjData, exp_config: MlSpacesExpConfig) -> None:
        super().__init__(mj_data, exp_config)
        cfg = exp_config.robot_config
        self._robot_view = cfg.robot_view_factory(mj_data, cfg.robot_namespace)
        self._kinematics = MlSpacesKinematics(cfg)
        self._parallel_kinematics = DummyParallelKinematics(cfg, self._kinematics)
        self._controllers = {
            "body": JointPDTorqueController(
                self._robot_view.get_move_group("body"), BODY_KP, BODY_KD
            ),
            "left_dex1": JointPosController(self._robot_view.get_move_group("left_dex1")),
            "right_dex1": JointPosController(self._robot_view.get_move_group("right_dex1")),
        }

    @property
    def namespace(self) -> str:
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self):
        return self._robot_view

    @property
    def kinematics(self):
        return self._kinematics

    @property
    def parallel_kinematics(self):
        return self._parallel_kinematics

    @property
    def controllers(self):
        return self._controllers

    def reset(self) -> None:
        for move_group_id, qpos in self.exp_config.robot_config.init_qpos.items():
            group = self._robot_view.get_move_group(move_group_id)
            group.joint_pos = np.asarray(qpos, dtype=np.float64)
            if group.vel_dim:
                group.joint_vel = np.zeros(group.vel_dim, dtype=np.float64)
        for controller in self._controllers.values():
            controller.reset()

    def set_world_pose(self, robot_world_pose: np.ndarray | list[float]) -> None:
        """Place G1 on the floor when callers provide planar ``(x, y, yaw)``.

        The generic implementation turns a planar pose into ``z=0``.  That is
        correct for a wheeled base whose root frame lies on the floor, but G1's
        free joint is on the pelvis.  Preserve its current standing height for
        planar placement; explicit 7D/4x4 poses still control all six DoFs.
        """

        pose = np.asarray(robot_world_pose, dtype=np.float64)
        if pose.shape == (3,):
            x, y, yaw = pose
            pelvis_z = float(self._robot_view.base.pose[2, 3])
            robot_world_pose = np.array(
                [x, y, pelvis_z, np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                dtype=np.float64,
            )
        super().set_world_pose(robot_world_pose)

    @staticmethod
    def robot_model_root_name() -> str:
        return "pelvis"
