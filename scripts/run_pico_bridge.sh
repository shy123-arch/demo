#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BRIDGE_PYTHON="${BRIDGE_PYTHON:-${project_root}/.venv-bridge/bin/python}"
export LD_LIBRARY_PATH="${project_root}/third_party/XRoboToolkit-PC-Service-Pybind/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${BRIDGE_PYTHON}" -c 'import general_motion_retargeting, xrobotoolkit_sdk as x; assert all(hasattr(x, n) for n in ("register_frame_callback", "clear_frame_callback", "has_frame_callback")), "Install bundled callback-enabled SDK using scripts/setup_pico.sh"'
exec bash "${project_root}/molmospaces-teleop/scripts/run_pose_bridge.sh" "$@"
