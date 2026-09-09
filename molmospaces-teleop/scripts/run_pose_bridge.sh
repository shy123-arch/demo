#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${BRIDGE_PYTHON:-${PYTHON:-python}}"
udp_stream_host="${UDP_STREAM_HOST:-127.0.0.1}"
udp_stream_port="${UDP_STREAM_PORT:-28704}"
wait_for_molmo_udp="${WAIT_FOR_MOLMO_UDP:-0}"
wait_timeout_s="${WAIT_FOR_MOLMO_UDP_TIMEOUT_S:-180}"
udp_stream_qos="${UDP_STREAM_QOS:-0}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

has_udp_listener() {
    command -v ss >/dev/null 2>&1 || return 1
    ss -H -lun 2>/dev/null | grep -E "[:*]${udp_stream_port}([[:space:]]|$)" >/dev/null 2>&1
}

if [[ -n "${udp_stream_host}" ]]; then
    echo "[run_pose_bridge] UDP stream target udp://${udp_stream_host}:${udp_stream_port}"
    if [[ "${udp_stream_host}" =~ ^(127\.0\.0\.1|localhost|0\.0\.0\.0)$ ]]; then
        if [[ "${wait_for_molmo_udp}" == "1" ]]; then
            echo "[run_pose_bridge] waiting for MolmoSpaces UDP receiver on :${udp_stream_port} (timeout ${wait_timeout_s}s)"
            deadline=$((SECONDS + wait_timeout_s))
            while ! has_udp_listener; do
                if (( SECONDS >= deadline )); then
                    echo "[run_pose_bridge][ERROR] no UDP listener on :${udp_stream_port}; start bash scripts/run_molmo_teleop.sh first" >&2
                    exit 2
                fi
                sleep 1
            done
            echo "[run_pose_bridge] detected UDP receiver on :${udp_stream_port}"
        elif ! has_udp_listener; then
            echo "[run_pose_bridge][WARN] no local UDP listener on :${udp_stream_port} yet; start MolmoSpaces main before pressing RIGHT A"
        fi
    fi
fi

udp_qos_arg="--no-udp_stream_qos"
[[ "${udp_stream_qos}" == "1" ]] && udp_qos_arg="--udp_stream_qos"
exec "${python_bin}" -u -m teleop.pose_bridge \
    --robot unitree_g1 \
    --actual_human_height "${ACTUAL_HUMAN_HEIGHT:-1.6}" \
    --ctrl_fps 50 \
    --retarget_fps "${RETARGET_FPS:-60}" \
    --retarget_worker_ready_timeout_s "${RETARGET_WORKER_READY_TIMEOUT_S:-60}" \
    --lookback_ms "${LOOKBACK_MS:-25}" \
    --retarget_buffer_window_s 0.5 \
    --log_interval_s "${LOG_INTERVAL_S:-3}" \
    --req_bind_addr 'tcp://*:28701' \
    --rep_bind_addr 'tcp://*:28702' \
    --ctrl_bind_addr 'tcp://*:28703' \
    --udp_stream_host "${udp_stream_host}" \
    --udp_stream_port "${udp_stream_port}" \
    --udp_stream_fps "${UDP_STREAM_FPS:-50}" \
    "${udp_qos_arg}" \
    --min_link_height 0.0 \
    --min_link_height_align_strategy startup_fixed \
    --min_link_height_bootstrap_frames 10 \
    "$@"
