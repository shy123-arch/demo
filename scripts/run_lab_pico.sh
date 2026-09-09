#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MOLMO_TELEOP_ROOT="${project_root}/molmospaces-teleop"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
cd "${project_root}/lab-sim"
exec "${LAB_PYTHON:-${project_root}/lab-sim/.venv/bin/python}" run.py pico-teleop "$@"
