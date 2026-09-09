#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
config="${MOLMO_TELEOP_CONFIG:-molmo_spaces.data_generation.config.g1_dex1_teleop_config:G1Dex1TeleopDataGenConfig}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

bridge_pid=""
cleanup() {
    if [[ -n "${bridge_pid}" ]] && kill -0 "${bridge_pid}" 2>/dev/null; then
        kill "${bridge_pid}" 2>/dev/null || true
        wait "${bridge_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[run_live_teleop] all-in-one mode; for split terminals use bash scripts/run_molmo_teleop.sh and bash scripts/run_pose_bridge.sh"
"${repo_root}/scripts/run_pose_bridge.sh" &
bridge_pid=$!

sleep 2
if ! kill -0 "${bridge_pid}" 2>/dev/null; then
    wait "${bridge_pid}"
    exit 1
fi

exec "${python_bin}" -m molmo_spaces.data_generation.main "${config}" "$@"
