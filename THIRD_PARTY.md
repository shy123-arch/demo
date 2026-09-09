# Bundled dependencies

Original licenses and notices are retained in each directory.

- `lab-sim/assets/g1`: Unitree models and NVIDIA policies; see local LICENSE/NOTICE files.
- `molmospaces-teleop`: snapshot of the user-uploaded, locally modified MolmoSpaces teleop project, including its tracking policy, configuration and walk reference. This is not a pristine upstream revision.
- `XRoboToolkit-Unity-Client`: snapshot supplied with the user's MolmoSpaces archive, including PICO integration SDK.
- `third_party/GMR`: https://github.com/YanjieZe/GMR at `bb1bbe40774794fceb2a7c579a3464a28e68c844`; G1 assets, Python package and supporting code. Other robots' mesh assets are omitted. This is a fetched upstream version, not the unavailable `GMR-current` installation on the original teleop PC.
- `third_party/XRoboToolkit-PC-Service`: https://github.com/XR-Robotics/XRoboToolkit-PC-Service at `85bac4dbc1fd5cef42c74a160d9c30aa3491f122`.
- `third_party/XRoboToolkit-PC-Service-Pybind`: https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind at `c64ccf6acd577a333e03b66fafe8efeeceb511b1`, with local frame callback bindings and GIL-safe service shutdown for `teleop/pose_bridge.py`. This recreates the required API; it is not the original PC's unavailable patch.

Python packages and system libraries are installed separately using the documented setup. Headset applications must be built/installed on the actual PICO device. Generated data, Python environments and Git history from the supplied archives are not included.
