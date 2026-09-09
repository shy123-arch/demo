"""Standalone motion-tracking runtime backed by the bundled ONNX policy."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from .core import TrackingPolicyRaw
from .paths import REPO_ROOT, resolve_asset_path
from .state_bridge import MujocoStateBridge
from .utils import DictToClass

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "motion_tracking_live.yaml"
DEFAULT_POLICY_PATH = REPO_ROOT / "assets" / "policies" / "motion_tracking" / "policy.onnx"


class MotionTrackingRuntime:
    """Build training-compatible observations and run the bundled 29D policy."""

    def __init__(
        self,
        bridge: MujocoStateBridge,
        tracking_config_path: str | Path | None = None,
        policy_path: str | Path | None = None,
        motion_source: str = "udp",
        enable_transport: bool = False,
    ) -> None:
        self.bridge = bridge
        config_path = Path(tracking_config_path or DEFAULT_CONFIG_PATH).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Tracking config not found: {config_path}")
        config_dict = yaml.safe_load(config_path.read_text())

        env_policy_dir = os.environ.get("MOLMO_TRACKING_CKPT_DIR")
        env_policy_path = os.environ.get("MOLMO_TRACKING_POLICY_PATH")
        if env_policy_dir and env_policy_path:
            raise ValueError("Set only one of MOLMO_TRACKING_CKPT_DIR or MOLMO_TRACKING_POLICY_PATH")
        if env_policy_dir:
            configured_policy = Path(env_policy_dir) / "policy.onnx"
        else:
            configured_policy = env_policy_path or policy_path or config_dict.get("policy_path") or DEFAULT_POLICY_PATH
        resolved_policy_path = resolve_asset_path(configured_policy).resolve()
        if not resolved_policy_path.is_file():
            raise FileNotFoundError(f"ONNX checkpoint not found: {resolved_policy_path}")
        external_data_path = resolved_policy_path.with_name(resolved_policy_path.name + ".data")
        if not external_data_path.is_file():
            raise FileNotFoundError(f"ONNX external data not found: {external_data_path}")

        source = str(motion_source).strip().lower()
        if source not in {"udp", "vr"}:
            raise ValueError("motion_source must be 'udp' or 'vr'")
        config_dict["policy_path"] = str(resolved_policy_path)
        config_dict["motion_source"] = source
        config_dict["runtime_reference_enable"] = False
        if source == "udp":
            config_dict["udp_enable"] = bool(enable_transport)
        elif not enable_transport:
            raise ValueError("VR motion source requires enable_transport=True")

        self.config_path = config_path
        self.policy_path = resolved_policy_path
        self.policy = TrackingPolicyRaw("tracking", DictToClass(config_dict), bridge)
        self.reset()

    @property
    def observation_dim(self) -> int:
        return int(self.policy.num_obs)

    def reset(self) -> None:
        self.bridge.update()
        self.policy.reset_reference_to_state(self.bridge.current_reference_anchor(), name="reset")
        self.policy.fade_in()

    def step(self) -> np.ndarray:
        """Return an absolute 29D body target in hardware joint order."""

        self.bridge.update()
        self.policy.update_obs()
        action_delta_real = self.policy.compute_action()
        self.policy.post_step()
        target = self.bridge.default_qpos_real + np.asarray(action_delta_real, dtype=np.float32)
        if target.shape != (29,) or not np.all(np.isfinite(target)):
            raise RuntimeError(f"Invalid body target: shape={target.shape}")
        return target

    def close(self) -> None:
        self.policy.deactivate()
