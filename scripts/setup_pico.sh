#!/usr/bin/env bash
# Ubuntu 22.04/24.04 x86_64; apt prerequisites are documented in README.md.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"
"${python_bin}" -m venv "${project_root}/lab-sim/.venv"
"${project_root}/lab-sim/.venv/bin/python" -m pip install -r "${project_root}/lab-sim/requirements.txt"
"${python_bin}" -m venv "${project_root}/.venv-bridge"
bridge_python="${project_root}/.venv-bridge/bin/python"
"${bridge_python}" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"${bridge_python}" -m pip install -r "${project_root}/requirements-bridge.txt"
"${bridge_python}" -m pip install --no-deps --no-build-isolation -e "${project_root}/third_party/GMR"
sdk_root="${project_root}/third_party/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK"
binding_root="${project_root}/third_party/XRoboToolkit-PC-Service-Pybind"
cmake -S "${sdk_root}" -B "${sdk_root}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${sdk_root}/build" --parallel 2
mkdir -p "${binding_root}/lib"
cp "${sdk_root}/build/libPXREARobotSDK.so" "${binding_root}/lib/"
export CMAKE_PREFIX_PATH="$("${bridge_python}" -m pybind11 --cmakedir)${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
"${bridge_python}" -m pip install --no-build-isolation "${binding_root}"
export LD_LIBRARY_PATH="${binding_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${bridge_python}" -c 'import general_motion_retargeting, xrobotoolkit_sdk as x; assert hasattr(x, "register_frame_callback"); print("PICO bridge dependencies ready")'
echo 'Next: start XRoboToolkit PC Service, connect/calibrate PICO, then use the two launchers in README.md.'
