"""Configuration module for MolmoSpaces experiments.

This module provides configuration classes organized by category:
- abstract_config: Base Config class
- abstract_exp_config: Base experiment configuration
- camera_configs: Camera-related configurations
- robot_configs: Robot-related configurations
- task_configs: Task-related configurations
- task_sampler_configs: Task sampler-related configurations
- policy_configs: Policy-related configurations
"""

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.camera_configs import (
    CameraConfig,
    CameraSystemConfig,
    FixedExocentricCameraConfig,
    FrankaDroidCameraSystem,
    FrankaRandomizedD405D455CameraSystem,
    G1Dex1CameraSystem,
    G1Dex1FastCameraSystem,
    G1Dex1PicoLiveCameraSystem,
    G1Dex1StereoCameraSystem,
    MjcfCameraConfig,
    RandomizedExocentricCameraConfig,
    RBY1GoProD455CameraSystem,
    RBY1MjcfCameraSystem,
    RobotMountedCameraConfig,
)
from molmo_spaces.configs.policy_configs import BasePolicyConfig, MotionTrackingPolicyConfig
from molmo_spaces.configs.robot_configs import BaseRobotConfig, FrankaRobotConfig, G1Dex1RobotConfig
from molmo_spaces.configs.task_configs import BaseMujocoTaskConfig, PickTaskConfig
from molmo_spaces.configs.task_sampler_configs import (
    BaseMujocoTaskSamplerConfig,
    PickTaskSamplerConfig,
)

__all__ = [
    "Config",
    "MlSpacesExpConfig",
    # Camera configs - new unified system
    "CameraSystemConfig",
    "CameraConfig",
    "MjcfCameraConfig",
    "RobotMountedCameraConfig",
    "FixedExocentricCameraConfig",
    "RandomizedExocentricCameraConfig",
    "RBY1MjcfCameraSystem",
    "RBY1GoProD455CameraSystem",
    "FrankaRandomizedD405D455CameraSystem",
    "FrankaDroidCameraSystem",
    "G1Dex1CameraSystem",
    "G1Dex1FastCameraSystem",
    "G1Dex1PicoLiveCameraSystem",
    "G1Dex1StereoCameraSystem",
    # Robot configs
    "BaseRobotConfig",
    "FrankaRobotConfig",
    "G1Dex1RobotConfig",
    # Task configs
    "BaseMujocoTaskConfig",
    "PickTaskConfig",
    # Task sampler configs
    "BaseMujocoTaskSamplerConfig",
    "PickTaskSamplerConfig",
    # Policy configs
    "BasePolicyConfig",
    "MotionTrackingPolicyConfig",
    "ObjectManipulationPlannerPolicyConfig",
]
