"""Joint-space PD controller for torque-actuated MuJoCo joints."""

from __future__ import annotations

import numpy as np

from molmo_spaces.controllers.abstract import AbstractPositionController
from molmo_spaces.robots.robot_views.abstract import MoveGroup


class JointPDTorqueController(AbstractPositionController):
    """Track absolute joint-position targets with torque commands.

    The bundled G1 MJCF uses torque motors. Its low-level controller applies
    ``kp * (q_target - q) - kd * dq`` at every MuJoCo physics step, so the
    generic position-actuator controller is not suitable for the 29D body.
    """

    def __init__(self, robot_move_group: MoveGroup, kp: np.ndarray, kd: np.ndarray) -> None:
        super().__init__(robot_move_group)
        self.kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        self.kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        if robot_move_group.pos_dim != robot_move_group.vel_dim:
            raise ValueError("JointPDTorqueController requires scalar hinge/slide joints")
        if robot_move_group.n_actuators != robot_move_group.pos_dim:
            raise ValueError("JointPDTorqueController requires one actuator per joint")
        if self.kp.shape != (robot_move_group.pos_dim,) or self.kd.shape != (
            robot_move_group.pos_dim,
        ):
            raise ValueError(
                f"Gain shape mismatch: kp={self.kp.shape}, kd={self.kd.shape}, "
                f"joints={robot_move_group.pos_dim}"
            )
        self._target = robot_move_group.joint_pos.copy()
        self._stationary = True

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def target_pos(self) -> np.ndarray:
        return self._target.copy()

    @property
    def stationary(self) -> bool:
        return self._stationary

    def set_target(self, target_joint_positions) -> None:
        target = np.asarray(target_joint_positions, dtype=np.float64).reshape(-1)
        if target.shape != (self.robot_move_group.pos_dim,):
            raise ValueError(
                f"Target shape {target.shape} does not match {self.robot_move_group.pos_dim} joints"
            )
        limits = self.robot_move_group.joint_pos_limits
        self._target = np.clip(target, limits[:, 0], limits[:, 1])
        self._stationary = False

    def set_to_stationary(self) -> None:
        self._target = self.robot_move_group.joint_pos.copy()
        self._stationary = True

    def compute_ctrl_inputs(self) -> np.ndarray:
        q = self.robot_move_group.joint_pos
        dq = self.robot_move_group.joint_vel
        torque = self.kp * (self._target - q) - self.kd * dq
        limits = self.robot_move_group.ctrl_limits
        return np.clip(torque, limits[:, 0], limits[:, 1])

    def reset(self) -> None:
        self.set_to_stationary()
