#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-/home/william/miniconda3/envs/mlspaces/bin/python}"
viewer="${MOLMO_TELEOP_VIEWER:-0}"
default_config="molmo_spaces.data_generation.config.g1_dex1_teleop_config:G1Dex1TeleopDataGenConfig"
viewer_config="molmo_spaces.data_generation.config.g1_dex1_teleop_config:G1Dex1TeleopViewerConfig"
fast_capture_config="molmo_spaces.data_generation.config.g1_dex1_teleop_config:G1Dex1TeleopFastCaptureConfig"
fast_viewer_config="molmo_spaces.data_generation.config.g1_dex1_teleop_config:G1Dex1TeleopFastViewerConfig"
pico_live_config="molmo_spaces.data_generation.config.g1_dex1_teleop_config:G1Dex1TeleopPicoLiveConfig"
stereo_viewer_config="molmo_spaces.data_generation.config.g1_dex1_teleop_config:G1Dex1TeleopStereoViewerConfig"
if [[ -n "${MOLMO_TELEOP_CONFIG:-}" ]]; then
    config="${MOLMO_TELEOP_CONFIG}"
elif [[ "${MOLMO_TELEOP_PICO_LIVE:-0}" == "1" ]]; then
    config="${pico_live_config}"
    viewer="0"
    export MOLMO_TELEOP_MONITOR=0
elif [[ "${MOLMO_TELEOP_FAST_CAPTURE:-0}" == "1" ]]; then
    config="${fast_capture_config}"
elif [[ "${MOLMO_TELEOP_FAST:-0}" == "1" ]]; then
    config="${fast_viewer_config}"
elif [[ "${MOLMO_TELEOP_STEREO:-0}" == "1" || "${MOLMO_PICO_VIDEO_MODE:-mono}" == "stereo_sbs" ]]; then
    config="${stereo_viewer_config}"
elif [[ "${viewer}" == "1" ]]; then
    config="${viewer_config}"
else
    config="${default_config}"
fi
udp_stream_port="${VR_UDP_STREAM_PORT:-28704}"
pico_control_port="${MOLMO_PICO_CONTROL_PORT:-13579}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export MOLMO_TELEOP_REALTIME_VIDEO="${MOLMO_TELEOP_REALTIME_VIDEO:-0}"
if [[ "${MOLMO_PICO_VIRTUAL_CAMERA:-0}" == "1" ]]; then
    export MOLMO_REQUIRE_PICO_VIDEO_READY="${MOLMO_REQUIRE_PICO_VIDEO_READY:-1}"
else
    export MOLMO_REQUIRE_PICO_VIDEO_READY="${MOLMO_REQUIRE_PICO_VIDEO_READY:-0}"
fi
if [[ "${MOLMO_TELEOP_PICO_LIVE:-0}" == "1" ]]; then
    export MOLMO_WAIT_FOR_TELEOP_START="${MOLMO_WAIT_FOR_TELEOP_START:-1}"
else
    export MOLMO_WAIT_FOR_TELEOP_START="${MOLMO_WAIT_FOR_TELEOP_START:-0}"
fi
if [[ -z "${MOLMO_SAVE_EVENT_FILE:-}" && -d /mnt/f/molospace/run_outputs ]]; then
    export MOLMO_SAVE_EVENT_FILE="/mnt/f/molospace/run_outputs/latest_save_event.txt"
fi
if [[ "${MOLMO_TELEOP_PICO_LIVE:-0}" == "1" ]]; then
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
elif [[ -z "${MUJOCO_GL:-}" && ("${viewer}" == "1" || "${config}" == *ViewerConfig*) ]]; then
    export MUJOCO_GL="glfw"
else
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

if command -v ss >/dev/null 2>&1; then
    existing="$(ss -H -lunp 2>/dev/null | grep -E "[:*]${udp_stream_port}([[:space:]]|$)" || true)"
    if [[ -n "${existing}" ]]; then
        echo "[run_molmo_teleop][ERROR] UDP :${udp_stream_port} is already bound. Stop the old MolmoSpaces process first:" >&2
        echo "${existing}" >&2
        exit 2
    fi
    if [[ "${MOLMO_PICO_VIRTUAL_CAMERA:-0}" == "1" ]]; then
        existing_tcp="$(ss -H -ltnp 2>/dev/null | grep -E "[:*]${pico_control_port}([[:space:]]|$)" || true)"
        if [[ -n "${existing_tcp}" ]]; then
            echo "[run_molmo_teleop][ERROR] TCP :${pico_control_port} is already bound. Stop the old MolmoSpaces/PICO video process first:" >&2
            echo "${existing_tcp}" >&2
            echo "[run_molmo_teleop][ERROR] Check with: ss -ltnp | grep ':${pico_control_port}'" >&2
            exit 3
        fi
    fi
fi

echo "[run_molmo_teleop] python=${python_bin}"
echo "[run_molmo_teleop] config=${config}"
echo "[run_molmo_teleop] MUJOCO_GL=${MUJOCO_GL}"
echo "[run_molmo_teleop] MESA_D3D12_DEFAULT_ADAPTER_NAME=${MESA_D3D12_DEFAULT_ADAPTER_NAME:-<unset>}"
echo "[run_molmo_teleop] viewer=${viewer}"
echo "[run_molmo_teleop] pico_live=${MOLMO_TELEOP_PICO_LIVE:-0}"
echo "[run_molmo_teleop] fast=${MOLMO_TELEOP_FAST:-0}"
echo "[run_molmo_teleop] monitor=${MOLMO_TELEOP_MONITOR:-0} tile=${MOLMO_TELEOP_MONITOR_TILE_WIDTH:-640}x${MOLMO_TELEOP_MONITOR_TILE_HEIGHT:-360}@${MOLMO_TELEOP_MONITOR_FPS:-15}"
echo "[run_molmo_teleop] realtime_video=${MOLMO_TELEOP_REALTIME_VIDEO}"
echo "[run_molmo_teleop] pico_virtual_camera=${MOLMO_PICO_VIRTUAL_CAMERA:-0} camera=${MOLMO_PICO_CAMERA_NAME:-head_camera} video_mode=${MOLMO_PICO_VIDEO_MODE:-mono} control_port=${MOLMO_PICO_CONTROL_PORT:-13579}"
echo "[run_molmo_teleop] pico_stream=${MOLMO_PICO_STREAM_WIDTH:-<app/default>}x${MOLMO_PICO_STREAM_HEIGHT:-<app/default>}@${MOLMO_PICO_STREAM_FPS:-<app/default>} bitrate=${MOLMO_PICO_STREAM_BITRATE:-<app/default>} transport=${MOLMO_PICO_STREAM_TRANSPORT:-<app/auto>}"
echo "[run_molmo_teleop] controls=RIGHT_A start, LEFT_X stop/pause, ${MOLMO_TELEOP_NEXT_BUTTON:-right_key_two} next_random_scene_task"
echo "[run_molmo_teleop] pick_random_houses=${MOLMO_PICK_RANDOM_HOUSES:-0} house_count=${MOLMO_PICK_HOUSE_COUNT:-<default>} house_range=${MOLMO_PICK_HOUSE_RANGE:-<unset>} house_inds=${MOLMO_PICK_HOUSE_INDS:-<config/default>}"
echo "[run_molmo_teleop] require_pico_video_ready=${MOLMO_REQUIRE_PICO_VIDEO_READY}"
echo "[run_molmo_teleop] wait_for_teleop_start=${MOLMO_WAIT_FOR_TELEOP_START}"
if [[ "${MOLMO_PICO_VIRTUAL_CAMERA:-0}" == "1" && "${MOLMO_TELEOP_MONITOR:-0}" == "1" ]]; then
    echo "[run_molmo_teleop][WARN] PICO streaming and desktop four-camera monitor are both enabled; disable MOLMO_TELEOP_MONITOR for better teleop FPS."
fi
if [[ "${MOLMO_PICO_VIRTUAL_CAMERA:-0}" == "1" && "${MOLMO_TELEOP_PICO_LIVE:-0}" != "1" ]]; then
    echo "[run_molmo_teleop][WARN] PICO streaming is enabled without MOLMO_TELEOP_PICO_LIVE=1; four-camera capture usually runs at about 6-8 FPS on this laptop."
fi
if [[ "${MOLMO_PICO_VIRTUAL_CAMERA:-0}" == "1" && "${MOLMO_PICO_STREAM_FPS:-25}" -lt 20 ]]; then
    echo "[run_molmo_teleop][WARN] MOLMO_PICO_STREAM_FPS=${MOLMO_PICO_STREAM_FPS}; this intentionally caps the PICO video below 20 FPS."
fi
if [[ "${MUJOCO_GL}" == "egl" && "${MESA_D3D12_DEFAULT_ADAPTER_NAME:-}" != *NVIDIA* ]]; then
    echo "[run_molmo_teleop][WARN] MUJOCO_GL=egl but MESA_D3D12_DEFAULT_ADAPTER_NAME is not NVIDIA; WSL may render on the integrated GPU."
fi
echo "[run_molmo_teleop] tracking_ckpt_dir=${MOLMO_TRACKING_CKPT_DIR:-<config default>} tracking_policy_path=${MOLMO_TRACKING_POLICY_PATH:-<config default>}"
echo "[run_molmo_teleop] save_event_file=${MOLMO_SAVE_EVENT_FILE:-<house output _SAVE_COMPLETE.txt only>}"
echo "[run_molmo_teleop] wait for: [VRMotionSource] Connected transport=udp_stream and [VRMotionSource][READY]"
echo "[run_molmo_teleop] start marker after holding RIGHT A: TELEOP_VR_STARTED"
echo "[run_molmo_teleop] save marker after demo finishes: TELEOP_SAVE_COMPLETE"

exec "${python_bin}" -m molmo_spaces.data_generation.main "${config}" "$@"
