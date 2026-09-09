"""G1 + Dex1-1 motion-tracking data collection configuration."""

from __future__ import annotations

import os
import random
from pathlib import Path

from molmo_spaces.configs.base_pick_config import PickBaseConfig
from molmo_spaces.configs.camera_configs import (
    G1Dex1CameraSystem,
    G1Dex1FastCameraSystem,
    G1Dex1PicoLiveCameraSystem,
    G1Dex1StereoCameraSystem,
)
from molmo_spaces.configs.policy_configs import MotionTrackingPolicyConfig
from molmo_spaces.configs.robot_configs import G1Dex1RobotConfig
from molmo_spaces.configs.task_sampler_configs import PickTaskSamplerConfig
from molmo_spaces.data_generation.config_registry import register_config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR, get_scenes
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(f"[G1Dex1TeleopConfig][WARN] ignoring invalid {name}={value!r}", flush=True)
        return default


def _parse_house_indices(value: str) -> list[int]:
    indices: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [p.strip() for p in part.split(":")]
            if len(pieces) not in {2, 3}:
                raise ValueError(f"Invalid house range {part!r}; expected start:stop[:step]")
            start = int(pieces[0])
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            indices.extend(range(start, stop, step))
        elif "-" in part and not part.startswith("-"):
            start_s, end_s = [p.strip() for p in part.split("-", 1)]
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            indices.extend(range(start, end + step, step))
        else:
            indices.append(int(part))

    seen: set[int] = set()
    unique_indices: list[int] = []
    for idx in indices:
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        unique_indices.append(idx)
    if not unique_indices:
        raise ValueError(f"No valid house indices parsed from {value!r}")
    return unique_indices


def _available_house_indices(scene_dataset: str, data_split: str) -> list[int]:
    mapping = get_scenes(scene_dataset, data_split)
    split_mapping = mapping[data_split]
    return sorted(int(k) for k, v in split_mapping.items() if v is not None)


def _apply_pick_scene_env_overrides(config) -> None:
    """Optional env overrides for local PICO data collection."""
    sampler_config = config.task_sampler_config

    horizon = _env_int("MOLMO_PICK_TASK_HORIZON")
    if horizon is not None:
        config.task_horizon = horizon

    samples_per_house = _env_int("MOLMO_PICK_SAMPLES_PER_HOUSE")
    if samples_per_house is not None:
        sampler_config.samples_per_house = max(1, samples_per_house)

    explicit = os.environ.get("MOLMO_PICK_HOUSE_INDS")
    house_range = os.environ.get("MOLMO_PICK_HOUSE_RANGE")
    random_houses = _env_flag("MOLMO_PICK_RANDOM_HOUSES", False)
    if not explicit and not house_range and not random_houses:
        return

    if explicit:
        house_pool = _parse_house_indices(explicit)
    elif house_range:
        house_pool = _parse_house_indices(house_range)
    else:
        house_pool = _available_house_indices(config.scene_dataset, config.data_split)

    available = set(_available_house_indices(config.scene_dataset, config.data_split))
    filtered_pool = [idx for idx in house_pool if idx in available]
    if not filtered_pool:
        raise ValueError(
            "No usable houses after applying scene filters. "
            f"pool={house_pool[:20]} scene_dataset={config.scene_dataset!r} split={config.data_split!r}"
        )

    if random_houses:
        count = _env_int("MOLMO_PICK_HOUSE_COUNT", 1)
        count = max(1, int(count or 1))
        seed = _env_int("MOLMO_PICK_HOUSE_SEED")
        rng = random.Random(seed)
        if count <= len(filtered_pool):
            house_inds = rng.sample(filtered_pool, count)
        else:
            house_inds = filtered_pool[:]
            rng.shuffle(house_inds)
        print(
            "[G1Dex1TeleopConfig] random pick houses enabled: "
            f"house_inds={house_inds} pool_size={len(filtered_pool)} seed={seed}",
            flush=True,
        )
    else:
        house_inds = filtered_pool
        print(
            "[G1Dex1TeleopConfig] pick houses overridden: "
            f"house_inds={house_inds} pool_size={len(filtered_pool)}",
            flush=True,
        )

    sampler_config.house_inds = house_inds


@register_config("G1Dex1TeleopDataGenConfig")
class G1Dex1TeleopDataGenConfig(PickBaseConfig):
    """Collect pick episodes using the bundled whole-body tracking policy.

    The policy supplies 29D body targets at 50 Hz. Dex1-1 commands come from
    the same PICO trigger/grip packet and run in parallel with the body policy.
    """

    policy_dt_ms: float = 20.0
    ctrl_dt_ms: float = 2.0
    sim_dt_ms: float = 2.0
    # Local four-camera teleop currently runs at about 6 Hz wall-clock on the
    # target WSL laptop, so 360 policy steps is roughly a one-minute demo.
    task_horizon: int = 360
    num_workers: int = 1
    filter_for_successful_trajectories: bool = False

    robot_config: G1Dex1RobotConfig = G1Dex1RobotConfig()
    camera_config: G1Dex1CameraSystem = G1Dex1CameraSystem()
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
        house_inds=[0],
        samples_per_house=1,
        episodes_per_batch=1,
        max_total_attempts_multiplier=1,
        filter_for_grasps=False,
        robot_base_z_override=0.793,
        robot_safety_radius=0.45,
        base_pose_sampling_radius_range=(0.55, 0.95),
        robot_placement_rotation_range_rad=0.0,
        check_robot_placement_visibility=False,
    )
    policy_config: MotionTrackingPolicyConfig = MotionTrackingPolicyConfig(
        motion_source="vr",
        enable_transport=True,
    )
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "g1_dex1_teleop_v1"

    @property
    def tag(self) -> str:
        return "g1_dex1_teleop_datagen"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        _apply_pick_scene_env_overrides(self)


@register_config("G1Dex1TeleopViewerConfig")
class G1Dex1TeleopViewerConfig(G1Dex1TeleopDataGenConfig):
    """Interactive viewer variant for live PICO teleop debugging."""

    use_passive_viewer: bool = True
    viewer_cam_dict: dict = {
        "distance": 4.0,
        "azimuth": 45.0,
        "elevation": -20.0,
        "lookat": [0.0, 0.0, 0.8],
    }


@register_config("G1Dex1TeleopFastCaptureConfig")
class G1Dex1TeleopFastCaptureConfig(G1Dex1TeleopDataGenConfig):
    """Low-cost non-viewer four-camera capture for PICO teleop."""

    use_passive_viewer: bool = False
    camera_config: G1Dex1FastCameraSystem = G1Dex1FastCameraSystem()


@register_config("G1Dex1TeleopFastViewerConfig")
class G1Dex1TeleopFastViewerConfig(G1Dex1TeleopViewerConfig):
    """Low-cost interactive viewer variant for local PICO teleop debugging."""

    camera_config: G1Dex1FastCameraSystem = G1Dex1FastCameraSystem()


@register_config("G1Dex1TeleopPicoLiveConfig")
class G1Dex1TeleopPicoLiveConfig(G1Dex1TeleopDataGenConfig):
    """Minimal local/server config for PICO remote vision plus trajectory capture."""

    # PICO live uses one RGB camera and now runs near 25 FPS on the NVIDIA WSL path.
    # 1500 policy steps is roughly a one-minute operator demo.
    task_horizon: int = 1500
    use_passive_viewer: bool = False
    camera_config: G1Dex1PicoLiveCameraSystem = G1Dex1PicoLiveCameraSystem()


@register_config("G1Dex1TeleopStereoViewerConfig")
class G1Dex1TeleopStereoViewerConfig(G1Dex1TeleopViewerConfig):
    """Viewer variant with extra head-mounted stereo cameras."""

    camera_config: G1Dex1StereoCameraSystem = G1Dex1StereoCameraSystem()
