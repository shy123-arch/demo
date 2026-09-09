"""Generate the static MuJoCo laboratory scene used by the lab-sim demo.

The generated file is an include fragment (``<mujocoinclude>``), so it can be
included from a larger model that supplies a robot and any dynamic props.  The
scene is deliberately made from MuJoCo primitives: this keeps it portable and
lets the robot collide with the room and the broad furniture surfaces without
requiring an external mesh package.

Coordinate convention: z is up, the front/entrance is y=-4.3, the back wall is
y=+4.3, and the tall window wall is x=-3.4.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence
from xml.etree import ElementTree as ET


Vec3 = Sequence[float]


def _f(value: float) -> str:
    """Format a number compactly while keeping XML deterministic."""

    if abs(value) < 5e-7:
        value = 0.0
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _v(values: Iterable[float]) -> str:
    return " ".join(_f(float(value)) for value in values)


def _attrs(**kwargs: object) -> MutableMapping[str, str]:
    return {key: (value if isinstance(value, str) else _f(float(value))) for key, value in kwargs.items() if value is not None}


def _asset(root: ET.Element) -> ET.Element:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    return asset


def _worldbody(root: ET.Element) -> ET.Element:
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    return worldbody


def _material(asset: ET.Element, name: str, rgba: str, *, texture: str | None = None, emission: float = 0.0,
              reflectance: float = 0.0, shininess: float = 0.25, specular: float = 0.25,
              metallic: float = 0.0, roughness: float = 0.55) -> None:
    attrs = {
        "name": name,
        "rgba": rgba,
        "emission": _f(emission),
        "reflectance": _f(reflectance),
        "shininess": _f(shininess),
        "specular": _f(specular),
        "metallic": _f(metallic),
        "roughness": _f(roughness),
    }
    if texture:
        attrs["texture"] = texture
    ET.SubElement(asset, "material", attrs)


def _texture(asset: ET.Element, name: str, rgb1: str, rgb2: str, *, builtin: str = "checker") -> None:
    ET.SubElement(
        asset,
        "texture",
        {
            "name": name,
            "type": "2d",
            "builtin": builtin,
            "width": "256",
            "height": "256",
            "rgb1": rgb1,
            "rgb2": rgb2,
        },
    )


def _geom(
    parent: ET.Element,
    name: str,
    geom_type: str,
    pos: Vec3,
    *,
    size: Vec3 | None = None,
    material: str | None = None,
    rgba: str | None = None,
    euler: Vec3 | None = None,
    quat: Vec3 | None = None,
    fromto: Sequence[float] | None = None,
    contype: int = 0,
    conaffinity: int = 0,
    group: int = 0,
    **extra: str,
) -> ET.Element:
    attrs: MutableMapping[str, str] = {
        "name": name,
        "type": geom_type,
        "pos": _v(pos),
        "contype": str(contype),
        "conaffinity": str(conaffinity),
        "group": str(group),
    }
    if size is not None:
        attrs["size"] = _v(size)
    if material is not None:
        attrs["material"] = material
    if rgba is not None:
        attrs["rgba"] = rgba
    if euler is not None:
        attrs["euler"] = _v(euler)
    if quat is not None:
        attrs["quat"] = _v(quat)
    if fromto is not None:
        attrs["fromto"] = _v(fromto)
    attrs.update(extra)
    return ET.SubElement(parent, "geom", attrs)


def _box(parent: ET.Element, name: str, pos: Vec3, half: Vec3, *, material: str | None = None,
         rgba: str | None = None, euler: Vec3 | None = None, collision: bool = False,
         group: int = 0, **extra: str) -> ET.Element:
    return _geom(parent, name, "box", pos, size=half, material=material, rgba=rgba, euler=euler,
                 contype=1 if collision else 0, conaffinity=1 if collision else 0, group=group, **extra)


def _cylinder(parent: ET.Element, name: str, pos: Vec3, radius: float, half_height: float, *,
              material: str | None = None, rgba: str | None = None, euler: Vec3 | None = None,
              collision: bool = False, group: int = 0, **extra: str) -> ET.Element:
    return _geom(parent, name, "cylinder", pos, size=(radius, half_height), material=material, rgba=rgba,
                 euler=euler, contype=1 if collision else 0, conaffinity=1 if collision else 0,
                 group=group, **extra)


def _sphere(parent: ET.Element, name: str, pos: Vec3, radius: float, *, material: str | None = None,
            rgba: str | None = None, collision: bool = False, group: int = 0, **extra: str) -> ET.Element:
    return _geom(parent, name, "sphere", pos, size=(radius,), material=material, rgba=rgba,
                 contype=1 if collision else 0, conaffinity=1 if collision else 0, group=group, **extra)


def _capsule(parent: ET.Element, name: str, start: Vec3, end: Vec3, radius: float, *,
             material: str | None = None, rgba: str | None = None, collision: bool = False,
             group: int = 0, **extra: str) -> ET.Element:
    return _geom(parent, name, "capsule", (0.0, 0.0, 0.0), size=(radius,), material=material, rgba=rgba,
                 fromto=(*start, *end), contype=1 if collision else 0, conaffinity=1 if collision else 0,
                 group=group, **extra)


def _site(parent: ET.Element, name: str, pos: Vec3, size: Vec3, *, rgba: str, site_type: str = "box") -> ET.Element:
    return ET.SubElement(parent, "site", {"name": name, "type": site_type, "pos": _v(pos), "size": _v(size), "rgba": rgba})


def _look_at_xyaxes(position: Vec3, target: Vec3, up: Vec3 = (0.0, 0.0, 1.0)) -> str:
    """Return MuJoCo's six-value xyaxes camera orientation for a look-at view."""

    px, py, pz = position
    tx, ty, tz = target
    dx, dy, dz = tx - px, ty - py, tz - pz
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-8:
        raise ValueError("camera position and target must differ")
    # MuJoCo's camera looks down its local -Z axis.  x cross y = z.
    view = (dx / length, dy / length, dz / length)
    zaxis = tuple(-value for value in view)
    dot = sum(up[i] * zaxis[i] for i in range(3))
    yraw = tuple(up[i] - dot * zaxis[i] for i in range(3))
    ylength = math.sqrt(sum(value * value for value in yraw))
    if ylength < 1e-8:
        up = (0.0, 1.0, 0.0)
        dot = sum(up[i] * zaxis[i] for i in range(3))
        yraw = tuple(up[i] - dot * zaxis[i] for i in range(3))
        ylength = math.sqrt(sum(value * value for value in yraw))
    yaxis = tuple(value / ylength for value in yraw)
    xaxis = (
        yaxis[1] * zaxis[2] - yaxis[2] * zaxis[1],
        yaxis[2] * zaxis[0] - yaxis[0] * zaxis[2],
        yaxis[0] * zaxis[1] - yaxis[1] * zaxis[0],
    )
    return _v((*xaxis, *yaxis))


def _add_camera(parent: ET.Element, name: str, position: Vec3, target: Vec3, *, fovy: float = 50.0) -> None:
    ET.SubElement(parent, "camera", {
        "name": name,
        "pos": _v(position),
        "xyaxes": _look_at_xyaxes(position, target),
        "fovy": _f(fovy),
    })


def _add_drawer_front(parent: ET.Element, name: str, x: float, y: float, z: float, width: float, *,
                      depth: float = 0.018, height: float = 0.18, material: str = "lab_cabinet") -> None:
    _box(parent, name + "_panel", (x, y, z), (width, depth, height), material=material, group=2)
    _box(parent, name + "_handle", (x, y - depth - 0.018, z), (width * 0.18, 0.018, 0.012),
         material="lab_metal", group=2)


def _add_side_drawer_front(parent: ET.Element, name: str, x: float, y: float, z: float, depth: float,
                           width: float, *, side: str, height: float = 0.18,
                           material: str = "lab_cabinet") -> None:
    """Add a drawer front on a bench's aisle-facing x side.

    The left run faces +x and the right run faces -x.  Keeping these panels on
    the inward sides makes the cabinet seams and handles visible from the
    robot's central aisle instead of hiding them on the room-facing y side.
    """

    sign = 1.0 if side == "plus_x" else -1.0
    _box(parent, name + "_panel", (x, y, z), (depth, width, height), material=material, group=2)
    _box(parent, name + "_handle", (x + sign * (depth + 0.018), y, z), (0.018, width * 0.18, 0.012),
         material="lab_metal", group=2)


def _add_cabinet_run(parent: ET.Element, prefix: str, x: float, y: float, x_half: float, y_half: float,
                     *, top: float, front_axis: str = "y", drawers: int = 5) -> None:
    """Add a broad collidable cabinet and lightweight visual drawer fronts."""

    body_h = top - 0.06
    _box(parent, prefix + "_body", (x, y, body_h / 2.0), (x_half, y_half, body_h / 2.0),
         material="lab_cabinet", collision=True)
    _box(parent, prefix + "_counter", (x, y, top - 0.03), (x_half + 0.025, y_half + 0.025, 0.03),
         material="lab_counter", collision=True)
    # Fronts are separate so seams and handles remain legible in a render.
    if front_axis == "y":
        front = y - y_half - 0.012
        positions = [-x_half * 0.78, -x_half * 0.39, 0.0, x_half * 0.39, x_half * 0.78]
        for index, offset in enumerate(positions[:drawers]):
            _add_drawer_front(parent, f"{prefix}_drawer_{index}", x + offset, front, 0.42, x_half * 0.17)
            _add_drawer_front(parent, f"{prefix}_drawer_low_{index}", x + offset, front, 0.68, x_half * 0.17, height=0.10)
    elif front_axis in ("plus_x", "minus_x"):
        sign = 1.0 if front_axis == "plus_x" else -1.0
        front = x + sign * (x_half + 0.012)
        positions = [-y_half * 0.82, -y_half * 0.41, 0.0, y_half * 0.41, y_half * 0.82]
        for index, offset in enumerate(positions[:drawers]):
            _add_side_drawer_front(parent, f"{prefix}_drawer_{index}", front, y + offset, 0.42,
                                   0.018, y_half * 0.17, side=front_axis)
            _add_side_drawer_front(parent, f"{prefix}_drawer_low_{index}", front, y + offset, 0.68,
                                   0.018, y_half * 0.17, side=front_axis, height=0.10)
    else:
        raise ValueError(f"Unsupported cabinet front axis: {front_axis}")


def _add_bottle(parent: ET.Element, prefix: str, x: float, y: float, z: float, *, color: str = "lab_liquid_blue",
                height: float = 0.22, radius: float = 0.045) -> None:
    _cylinder(parent, prefix + "_body", (x, y, z + height * 0.42), radius, height * 0.42, material=color, group=3)
    _cylinder(parent, prefix + "_neck", (x, y, z + height * 0.86), radius * 0.58, height * 0.12, material="lab_glass", group=3)
    _cylinder(parent, prefix + "_cap", (x, y, z + height * 1.01), radius * 0.66, height * 0.045, material="lab_cap", group=3)


def _add_monitor(parent: ET.Element, prefix: str, x: float, y: float, z: float, *, facing: str = "x",
                 width: float = 0.52, height: float = 0.34) -> None:
    """Add a slim black display, white bezel, stand and keyboard."""

    if facing == "x":
        # The screen normal is +x.  This is used on the left bench.
        _box(parent, prefix + "_bezel", (x, y, z), (0.035, width / 2.0 + 0.035, height / 2.0 + 0.035), material="lab_monitor_frame", collision=True, group=3)
        _box(parent, prefix + "_screen", (x + 0.038, y, z), (0.012, width / 2.0, height / 2.0), material="lab_screen", group=3)
        _cylinder(parent, prefix + "_stem", (x, y, z - height / 2.0 - 0.10), 0.028, 0.10, material="lab_metal", group=3, euler=(0.0, 90.0, 0.0))
        _box(parent, prefix + "_foot", (x, y, z - height / 2.0 - 0.21), (0.18, 0.12, 0.018), material="lab_monitor_frame", group=3)
        _box(parent, prefix + "_keyboard", (x + 0.02, y + 0.34, z - height / 2.0 - 0.235), (0.18, 0.10, 0.012), material="lab_keyboard", group=3)
    elif facing == "minus_x":
        _box(parent, prefix + "_bezel", (x, y, z), (0.035, width / 2.0 + 0.035, height / 2.0 + 0.035), material="lab_monitor_frame", collision=True, group=3)
        _box(parent, prefix + "_screen", (x - 0.038, y, z), (0.012, width / 2.0, height / 2.0), material="lab_screen", group=3)
        _cylinder(parent, prefix + "_stem", (x, y, z - height / 2.0 - 0.10), 0.028, 0.10, material="lab_metal", group=3, euler=(0.0, 90.0, 0.0))
        _box(parent, prefix + "_foot", (x, y, z - height / 2.0 - 0.21), (0.18, 0.12, 0.018), material="lab_monitor_frame", group=3)
        _box(parent, prefix + "_keyboard", (x - 0.02, y + 0.34, z - height / 2.0 - 0.235), (0.18, 0.10, 0.012), material="lab_keyboard", group=3)
    else:
        # Screens on the rear worktop face toward the front (-y).
        _box(parent, prefix + "_bezel", (x, y, z), (width / 2.0 + 0.035, 0.035, height / 2.0 + 0.035), material="lab_monitor_frame", collision=True, group=3)
        _box(parent, prefix + "_screen", (x, y - 0.038, z), (width / 2.0, 0.012, height / 2.0), material="lab_screen", group=3)
        _cylinder(parent, prefix + "_stem", (x, y, z - height / 2.0 - 0.10), 0.028, 0.10, material="lab_metal", group=3)
        _box(parent, prefix + "_foot", (x, y - 0.02, z - height / 2.0 - 0.21), (0.12, 0.18, 0.018), material="lab_monitor_frame", group=3)
        _box(parent, prefix + "_keyboard", (x + 0.34, y - 0.02, z - height / 2.0 - 0.235), (0.10, 0.18, 0.012), material="lab_keyboard", group=3)


def _add_microscope(parent: ET.Element, prefix: str, x: float, y: float, top: float, *, scale: float = 1.0,
                    yaw: float = 0.0) -> None:
    """A recognizable binocular microscope built from simple primitives."""

    s = scale
    # One transparent proxy gives navigation/planning a conservative obstacle
    # volume without turning every eyepiece and focus knob into a collider.
    _box(parent, prefix + "_collision", (x, y, top + 0.40 * s), (0.24 * s, 0.19 * s, 0.43 * s),
         rgba="0 0 0 0", collision=True, group=3)
    _box(parent, prefix + "_base", (x, y, top + 0.035 * s), (0.22 * s, 0.15 * s, 0.035 * s), material="lab_metal_light", group=3)
    _cylinder(parent, prefix + "_column", (x - 0.12 * s, y, top + 0.24 * s), 0.045 * s, 0.20 * s, material="lab_metal", group=3)
    _box(parent, prefix + "_stage", (x, y, top + 0.43 * s), (0.18 * s, 0.13 * s, 0.025 * s), material="lab_metal_light", group=3)
    _box(parent, prefix + "_stage_clip_a", (x + 0.12 * s, y - 0.02 * s, top + 0.47 * s), (0.025 * s, 0.012 * s, 0.012 * s), material="lab_metal", group=3)
    _box(parent, prefix + "_stage_clip_b", (x - 0.12 * s, y + 0.02 * s, top + 0.47 * s), (0.025 * s, 0.012 * s, 0.012 * s), material="lab_metal", group=3)
    _capsule(parent, prefix + "_arm", (x - 0.10 * s, y, top + 0.36 * s), (x + 0.03 * s, y, top + 0.72 * s), 0.06 * s, material="lab_metal", group=3)
    _box(parent, prefix + "_head", (x + 0.10 * s, y, top + 0.72 * s), (0.18 * s, 0.13 * s, 0.085 * s), material="lab_metal", euler=(0.0, -9.0, yaw), group=3)
    _cylinder(parent, prefix + "_ocular_left", (x + 0.16 * s, y - 0.075 * s, top + 0.82 * s), 0.042 * s, 0.10 * s, material="lab_black", euler=(90.0, 0.0, 0.0), group=3)
    _cylinder(parent, prefix + "_ocular_right", (x + 0.16 * s, y + 0.075 * s, top + 0.82 * s), 0.042 * s, 0.10 * s, material="lab_black", euler=(90.0, 0.0, 0.0), group=3)
    _cylinder(parent, prefix + "_objective", (x + 0.09 * s, y, top + 0.57 * s), 0.042 * s, 0.12 * s, material="lab_lens", group=3)
    _cylinder(parent, prefix + "_focus", (x - 0.02 * s, y - 0.16 * s, top + 0.57 * s), 0.035 * s, 0.022 * s, material="lab_metal_dark", euler=(90.0, 0.0, 0.0), group=3)


def _add_centrifuge(parent: ET.Element, prefix: str, x: float, y: float, top: float) -> None:
    _cylinder(parent, prefix + "_body", (x, y, top + 0.11), 0.25, 0.11, material="lab_appliance", collision=True, group=3)
    _cylinder(parent, prefix + "_lid", (x, y, top + 0.235), 0.205, 0.018, material="lab_lid", group=3)
    _cylinder(parent, prefix + "_hub", (x, y, top + 0.258), 0.04, 0.012, material="lab_metal_dark", group=3)
    for index, angle in enumerate((0.0, 60.0, 120.0, 180.0, 240.0, 300.0)):
        rad = math.radians(angle)
        _cylinder(parent, f"{prefix}_well_{index}", (x + 0.115 * math.cos(rad), y + 0.115 * math.sin(rad), top + 0.275), 0.026, 0.014, material="lab_well", group=3)
    _box(parent, prefix + "_control", (x + 0.24, y, top + 0.11), (0.018, 0.09, 0.075), material="lab_appliance", group=3)
    _sphere(parent, prefix + "_button", (x + 0.26, y - 0.04, top + 0.16), 0.016, material="lab_led_blue", group=3)


def _add_pipette_rack(parent: ET.Element, prefix: str, x: float, y: float, top: float) -> None:
    _box(parent, prefix + "_base", (x, y, top + 0.04), (0.18, 0.12, 0.04), material="lab_metal_light", collision=True, group=3)
    for index, dx in enumerate((-0.11, -0.055, 0.0, 0.055, 0.11)):
        _capsule(parent, f"{prefix}_pipette_{index}", (x + dx, y, top + 0.07), (x + dx, y, top + 0.44 + 0.02 * (index % 2)), 0.014,
                 material="lab_pipette", group=3)
        _sphere(parent, f"{prefix}_tip_{index}", (x + dx, y, top + 0.47 + 0.02 * (index % 2)), 0.022, material="lab_pipette_tip", group=3)


def _add_stool(parent: ET.Element, prefix: str, x: float, y: float) -> None:
    """Static gray office stool with compact seat/back obstacle proxies."""

    _cylinder(parent, prefix + "_seat", (x, y, 0.68), 0.31, 0.065, material="lab_stool_gray", collision=True, group=3)
    _box(parent, prefix + "_back", (x - 0.15, y, 0.91), (0.065, 0.24, 0.20), material="lab_stool_gray", collision=True, group=3)
    _cylinder(parent, prefix + "_stem", (x, y, 0.36), 0.045, 0.25, material="lab_metal_dark", group=3)
    _cylinder(parent, prefix + "_hub", (x, y, 0.12), 0.11, 0.035, material="lab_metal_dark", group=3)
    for index, angle in enumerate((0.0, 72.0, 144.0, 216.0, 288.0)):
        rad = math.radians(angle)
        ex, ey = x + 0.34 * math.cos(rad), y + 0.34 * math.sin(rad)
        _capsule(parent, f"{prefix}_leg_{index}", (x, y, 0.14), (ex, ey, 0.09), 0.018, material="lab_metal_dark", group=3)
        _sphere(parent, f"{prefix}_caster_{index}", (ex, ey, 0.075), 0.045, material="lab_caster", group=3)


def _add_shelf(parent: ET.Element, prefix: str, x: float, y: float, z: float, x_half: float, *, depth: float = 0.34) -> None:
    _box(parent, prefix + "_shelf", (x, y, z), (x_half, depth, 0.035), material="lab_shelf", group=2)
    _box(parent, prefix + "_left_support", (x - x_half + 0.05, y + 0.02, z - 0.26), (0.035, 0.035, 0.26), material="lab_shelf", group=2)
    _box(parent, prefix + "_right_support", (x + x_half - 0.05, y + 0.02, z - 0.26), (0.035, 0.035, 0.26), material="lab_shelf", group=2)


def _add_wall_panels(parent: ET.Element) -> None:
    # Broad collidable room shell. The left wall is left open between sill and
    # header, so the blue skyline can actually be seen through the windows.
    _box(parent, "lab_floor_slab", (0.0, 0.0, -0.035), (3.4, 4.3, 0.035), material="lab_floor", collision=True)
    _box(parent, "lab_back_wall", (0.0, 4.27, 1.625), (3.4, 0.05, 1.625), material="lab_wall", collision=True)
    _box(parent, "lab_right_wall", (3.35, 0.0, 1.625), (0.05, 4.3, 1.625), material="lab_wall", collision=True)
    _box(parent, "lab_left_window_sill", (-3.35, 0.0, 0.52), (0.05, 4.3, 0.52), material="lab_wall", collision=True)
    _box(parent, "lab_left_window_header", (-3.35, 0.0, 3.15), (0.05, 4.3, 0.10), material="lab_wall", collision=True)
    # A blue-gray exterior plane and stepped building silhouettes make the
    # windows read as a city view without relying on an external image texture.
    # The sky plane is farther outside than the building silhouettes.  With
    # the camera inside the room (looking toward -x), this keeps the buildings
    # in front of the sky instead of letting one solid blue plane hide them.
    _box(parent, "lab_window_sky", (-3.70, 0.0, 2.06), (0.012, 4.12, 1.04), material="lab_sky", group=1)
    buildings = (
        (-3.49, -3.25, 0.56, 1.30, 0.82),
        (-3.50, -2.15, 0.45, 1.75, 0.95),
        (-3.51, -1.10, 0.62, 1.10, 1.16),
        (-3.50, 0.15, 0.44, 1.95, 0.70),
        (-3.50, 1.22, 0.56, 1.42, 1.02),
        (-3.49, 2.35, 0.78, 1.88, 1.24),
        (-3.49, 3.45, 0.40, 1.00, 0.88),
    )
    for index, (bx, by, half_y, height, half_z) in enumerate(buildings):
        _box(parent, f"lab_skyline_building_{index}", (bx, by, 0.52 + half_z), (0.025, half_y, half_z),
              material="lab_city_blue", group=1)
        _box(parent, f"lab_skyline_roof_{index}", (bx - 0.03, by, 0.52 + 2.0 * half_z), (0.012, half_y * 0.72, 0.018), material="lab_city_light", group=1)
    # Window panes, mullions and transoms.
    bay_centers = (-3.50, -1.75, 0.0, 1.75, 3.50)
    for index, center in enumerate(bay_centers):
        _box(parent, f"lab_window_glass_{index}", (-3.385, center, 2.05), (0.012, 0.83, 1.03), material="lab_glass_window", group=1)
    for index, center in enumerate((-4.15, -2.625, -0.875, 0.875, 2.625, 4.15)):
        _box(parent, f"lab_window_mullion_{index}", (-3.33, center, 2.05), (0.045, 0.035, 1.07), material="lab_window_frame", group=2)
    _box(parent, "lab_window_transom", (-3.33, 0.0, 1.05), (0.045, 4.15, 0.035), material="lab_window_frame", group=2)
    _box(parent, "lab_window_top_rail", (-3.33, 0.0, 3.08), (0.045, 4.15, 0.035), material="lab_window_frame", group=2)
    # Floor tile joints give the polished floor scale in overhead and wide views.
    for index, x in enumerate((-3.0, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4, 3.0)):
        _box(parent, f"lab_floor_joint_x_{index}", (x, 0.0, 0.002), (0.008, 4.25, 0.002), material="lab_floor_joint", group=1)
    for index, y in enumerate((-3.9, -3.3, -2.7, -2.1, -1.5, -0.9, -0.3, 0.3, 0.9, 1.5, 2.1, 2.7, 3.3, 3.9)):
        _box(parent, f"lab_floor_joint_y_{index}", (0.0, y, 0.002), (3.35, 0.008, 0.002), material="lab_floor_joint", group=1)


def _add_ceiling(parent: ET.Element) -> None:
    # Group 5 is hidden by the normal interior render pass; a top-down pass can
    # explicitly enable it when a complete roof view is desired.
    _box(parent, "lab_ceiling_slab", (0.0, 0.0, 3.22), (3.4, 4.3, 0.03), material="lab_ceiling", group=5)
    for index, x in enumerate((-3.0, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4, 3.0)):
        _box(parent, f"lab_ceiling_grid_x_{index}", (x, 0.0, 3.175), (0.018, 4.24, 0.018), material="lab_ceiling_grid", group=5)
    for index, y in enumerate((-3.9, -3.3, -2.7, -2.1, -1.5, -0.9, -0.3, 0.3, 0.9, 1.5, 2.1, 2.7, 3.3, 3.9)):
        _box(parent, f"lab_ceiling_grid_y_{index}", (0.0, y, 3.175), (3.35, 0.018, 0.018), material="lab_ceiling_grid", group=5)
    light_positions = ((-2.35, -2.85), (-0.75, -2.85), (0.85, -2.85), (2.45, -2.85), (-1.55, -0.95), (0.05, -0.95), (1.65, -0.95), (-2.35, 1.05), (-0.75, 1.05), (0.85, 1.05), (2.45, 1.05), (-1.55, 2.95), (0.05, 2.95), (1.65, 2.95))
    for index, (x, y) in enumerate(light_positions):
        _box(parent, f"lab_ceiling_panel_{index}", (x, y, 3.145), (0.36, 0.24, 0.012), material="lab_ceiling_light", group=5)


def _add_benches(parent: ET.Element) -> None:
    # Left run, with the inner edge at x=-2.08 and top at z=.90.
    _add_cabinet_run(parent, "lab_left_bench", -2.60, -0.15, 0.52, 2.95, top=0.90, front_axis="plus_x", drawers=5)
    # Right run is split so the front workstation has the requested lower top
    # and an unobstructed rectangle for manipulation.
    _add_cabinet_run(parent, "lab_right_front", 2.60, -2.25, 0.60, 0.85, top=0.90, front_axis="minus_x", drawers=4)
    _add_cabinet_run(parent, "lab_right_workstation", 2.60, -0.80, 0.60, 0.60, top=0.80, front_axis="minus_x", drawers=3)
    _add_cabinet_run(parent, "lab_right_rear", 2.60, 1.30, 0.60, 1.50, top=0.90, front_axis="minus_x", drawers=4)

    # Cabinet divisions, toe kicks and handles on the inward-facing sides.
    for index, y in enumerate((-2.82, -2.22, -1.62, -1.02, -0.42, 0.18, 0.78, 1.38, 1.98, 2.58)):
        _box(parent, f"lab_left_inner_seam_{index}", (-2.065, y, 0.44), (0.016, 0.012, 0.35), material="lab_cabinet_shadow", group=2)
    for index, y in enumerate((-2.86, -2.30, -1.74, -1.18, -0.62, -0.06, 0.50, 1.06, 1.62, 2.18, 2.62)):
        if -1.40 < y < -0.20:
            continue
        _box(parent, f"lab_right_inner_seam_{index}", (1.985, y, 0.43), (0.016, 0.012, 0.34), material="lab_cabinet_shadow", group=2)
    _box(parent, "lab_left_toe_kick", (-2.60, -0.15, 0.07), (0.45, 2.75, 0.07), material="lab_toe_kick", group=2)
    _box(parent, "lab_right_front_toe_kick", (2.60, -2.25, 0.07), (0.53, 0.78, 0.07), material="lab_toe_kick", group=2)
    _box(parent, "lab_right_rear_toe_kick", (2.60, 1.30, 0.07), (0.53, 1.42, 0.07), material="lab_toe_kick", group=2)


def _add_rear_cabinetry(parent: ET.Element) -> None:
    _add_cabinet_run(parent, "lab_back_bench", 0.0, 3.65, 2.82, 0.45, top=0.90, front_axis="y", drawers=5)
    _box(parent, "lab_back_splash", (0.0, 4.08, 1.42), (2.80, 0.035, 0.54), material="lab_splash", group=1)
    _add_shelf(parent, "lab_back_shelf_low", 0.0, 3.70, 1.52, 2.58, depth=0.30)
    _add_shelf(parent, "lab_back_shelf_high", 0.0, 3.88, 2.20, 2.58, depth=0.26)
    _box(parent, "lab_back_shelf_left_panel", (-2.63, 3.65, 1.92), (0.06, 0.35, 0.82), material="lab_shelf", group=2)
    _box(parent, "lab_back_shelf_right_panel", (2.63, 3.65, 1.92), (0.06, 0.35, 0.82), material="lab_shelf", group=2)
    # Symmetric reagent rows on the two shelves.
    bottle_colors = ("lab_liquid_blue", "lab_liquid_amber", "lab_liquid_clear", "lab_liquid_blue")
    for index, x in enumerate((-2.25, -1.88, -1.51, -1.14, -0.77, 0.82, 1.19, 1.56, 1.93, 2.30)):
        _add_bottle(parent, f"lab_back_bottle_low_{index}", x, 3.55, 1.56, color=bottle_colors[index % len(bottle_colors)], height=0.27 + 0.04 * (index % 2), radius=0.048)
    for index, x in enumerate((-2.32, -1.95, -1.58, -0.35, 0.02, 0.39, 1.62, 1.99, 2.36)):
        _add_bottle(parent, f"lab_back_bottle_high_{index}", x, 3.76, 2.24, color=bottle_colors[(index + 1) % len(bottle_colors)], height=0.23 + 0.03 * (index % 3), radius=0.042)
    _add_monitor(parent, "lab_back_monitor", -0.35, 3.18, 1.38, facing="y", width=0.62, height=0.40)
    _add_microscope(parent, "lab_back_microscope", 0.82, 3.32, 0.90, scale=0.90, yaw=180.0)
    _add_pipette_rack(parent, "lab_back_pipettes", 1.80, 3.33, 0.90)


def _add_lab_equipment(parent: ET.Element) -> None:
    # Left-side instrumentation, deliberately kept off the central aisle.
    _add_monitor(parent, "lab_left_monitor_a", -2.14, -2.45, 1.43, facing="x", width=0.54, height=0.36)
    _add_monitor(parent, "lab_left_monitor_b", -2.14, 0.72, 1.43, facing="x", width=0.48, height=0.32)
    _add_microscope(parent, "lab_left_microscope_a", -2.50, -1.42, 0.90, scale=1.00, yaw=0.0)
    _add_microscope(parent, "lab_left_microscope_b", -2.50, 1.65, 0.90, scale=0.92, yaw=180.0)
    _add_centrifuge(parent, "lab_left_centrifuge", -2.68, -2.88, 0.90)
    _add_pipette_rack(parent, "lab_left_pipettes", -2.38, 2.26, 0.90)
    for index, (x, y, color, height) in enumerate((
        (-2.84, -1.95, "lab_liquid_blue", 0.26),
        (-2.66, -1.95, "lab_liquid_amber", 0.22),
        (-2.48, -1.95, "lab_liquid_clear", 0.30),
        (-2.30, -1.95, "lab_liquid_blue", 0.24),
        (-2.77, 2.45, "lab_liquid_amber", 0.21),
        (-2.58, 2.45, "lab_liquid_blue", 0.27),
    )):
        _add_bottle(parent, f"lab_left_bottle_{index}", x, y, 0.90, color=color, height=height, radius=0.045)
    _box(parent, "lab_left_tray", (-2.55, -2.12, 0.935), (0.36, 0.16, 0.018), material="lab_tray", group=3)
    _box(parent, "lab_left_scale", (-2.55, 0.20, 0.98), (0.17, 0.14, 0.07), material="lab_appliance", group=3)
    _box(parent, "lab_left_scale_display", (-2.55, 0.05, 1.08), (0.10, 0.018, 0.032), material="lab_led_blue", group=3)

    # The right rear counter gets useful visual clutter while the lower front
    # workstation remains a clean place to put dynamic task objects.
    _add_monitor(parent, "lab_right_monitor", 2.16, 1.48, 1.43, facing="minus_x", width=0.55, height=0.35)
    _add_microscope(parent, "lab_right_microscope", 2.58, 2.16, 0.90, scale=0.92, yaw=180.0)
    _add_centrifuge(parent, "lab_right_centrifuge", 2.78, 0.34, 0.90)
    _add_pipette_rack(parent, "lab_right_pipettes", 2.35, 2.60, 0.90)
    for index, (x, y, color, height) in enumerate((
        (2.88, 1.05, "lab_liquid_blue", 0.24),
        (2.70, 1.05, "lab_liquid_amber", 0.28),
        (2.52, 1.05, "lab_liquid_clear", 0.22),
        (2.88, 2.72, "lab_liquid_blue", 0.27),
        (2.70, 2.72, "lab_liquid_amber", 0.22),
    )):
        _add_bottle(parent, f"lab_right_bottle_{index}", x, y, 0.90, color=color, height=height, radius=0.045)


def _add_sites_and_cameras(parent: ET.Element) -> None:
    # Exact site names are part of the task interface used by manipulation code.
    _box(parent, "lab_placement_target_mat", (2.14, -0.42, 0.805), (0.145, 0.145, 0.006), material="lab_target", group=4)
    _site(parent, "work_surface", (2.13, -0.80, 0.805), (0.11, 0.11, 0.008), rgba="0.15 0.72 0.86 0.95")
    _site(parent, "placement_target", (2.14, -0.42, 0.813), (0.14, 0.14, 0.008), rgba="1.0 0.28 0.10 0.95")
    # Both wide cameras live just inside the open entrance.  The previous
    # versions were outside the x=3.35 wall, so their view was a blank wall.
    _add_camera(parent, "overview", (0.65, -4.05, 2.65), (0.0, 0.80, 1.10), fovy=65.0)
    _add_camera(parent, "reference", (0.30, -4.10, 2.20), (0.0, 0.80, 0.95), fovy=65.0)
    _add_camera(parent, "workstation", (1.02, -3.00, 2.35), (2.28, -0.75, 0.82), fovy=44.0)
    _add_camera(parent, "top", (0.0, -0.25, 7.85), (0.0, 0.0, 0.0), fovy=58.0)


def _add_lights(parent: ET.Element) -> None:
    for index, position in enumerate(((-2.2, -2.4, 3.05), (0.0, -0.4, 3.10), (2.1, 1.8, 3.05))):
        ET.SubElement(parent, "light", {
            "name": f"lab_area_light_{index}",
            "pos": _v(position),
            "dir": "0 0 -1",
            "diffuse": "0.78 0.84 0.92",
            "specular": "0.26 0.30 0.36",
            "attenuation": "0.4 0.04 0.01",
            "cutoff": "70",
            "exponent": "2",
            "castshadow": "true",
        })
    ET.SubElement(parent, "light", {
        "name": "lab_window_fill",
        "pos": "-3.0 -0.2 2.4",
        "dir": "1 0 -0.15",
        "diffuse": "0.46 0.64 0.84",
        "specular": "0.12 0.15 0.20",
        "directional": "true",
        "castshadow": "false",
    })


def _add_visual(root: ET.Element) -> None:
    """Add the bright neutral world lighting shared by all scene renders."""

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update({
        "ambient": "0.52 0.56 0.62",
        "diffuse": "0.72 0.76 0.82",
        "specular": "0.22 0.24 0.28",
    })


def _add_assets(root: ET.Element) -> None:
    asset = _asset(root)
    _texture(asset, "lab_floor_checker", "0.78 0.81 0.83", "0.91 0.92 0.92")
    _texture(asset, "lab_glass_tint", "0.50 0.72 0.84", "0.62 0.82 0.90", builtin="flat")
    ET.SubElement(asset, "texture", {
        "name": "lab_skybox",
        "type": "skybox",
        "builtin": "gradient",
        "width": "512",
        "height": "512",
        "rgb1": "0.91 0.94 0.97",
        "rgb2": "0.57 0.70 0.82",
    })
    _material(asset, "lab_floor", "0.88 0.90 0.91 1", texture="lab_floor_checker", reflectance=0.12, shininess=0.4, specular=0.38, roughness=0.42)
    _material(asset, "lab_floor_joint", "0.53 0.57 0.59 1", reflectance=0.05, shininess=0.2, specular=0.2, roughness=0.68)
    _material(asset, "lab_wall", "0.95 0.96 0.965 1", reflectance=0.06, shininess=0.25, specular=0.25, roughness=0.6)
    _material(asset, "lab_ceiling", "0.91 0.93 0.94 1", reflectance=0.04, shininess=0.2, specular=0.25, roughness=0.65)
    _material(asset, "lab_ceiling_grid", "0.67 0.71 0.73 1", reflectance=0.05, shininess=0.25, specular=0.25, roughness=0.5)
    _material(asset, "lab_ceiling_light", "0.97 0.99 1.0 1", emission=0.72, reflectance=0.08, shininess=0.42, specular=0.35, roughness=0.32)
    _material(asset, "lab_cabinet", "0.94 0.95 0.95 1", reflectance=0.08, shininess=0.28, specular=0.32, roughness=0.46)
    _material(asset, "lab_counter", "0.985 0.988 0.985 1", reflectance=0.14, shininess=0.50, specular=0.48, roughness=0.28)
    _material(asset, "lab_cabinet_shadow", "0.75 0.78 0.79 1", reflectance=0.04, shininess=0.15, specular=0.15, roughness=0.75)
    _material(asset, "lab_toe_kick", "0.38 0.43 0.45 1", reflectance=0.10, shininess=0.24, specular=0.28, roughness=0.55)
    _material(asset, "lab_splash", "0.86 0.89 0.90 1", reflectance=0.08, shininess=0.32, specular=0.28, roughness=0.48)
    _material(asset, "lab_shelf", "0.90 0.92 0.92 1", reflectance=0.08, shininess=0.28, specular=0.28, roughness=0.5)
    _material(asset, "lab_window_frame", "0.78 0.82 0.84 1", reflectance=0.14, shininess=0.45, specular=0.42, metallic=0.15, roughness=0.33)
    _material(asset, "lab_glass_window", "0.50 0.75 0.88 0.24", texture="lab_glass_tint", reflectance=0.18, shininess=0.72, specular=0.82, roughness=0.12)
    _material(asset, "lab_sky", "0.48 0.68 0.82 1", reflectance=0.02, shininess=0.15, specular=0.10, roughness=0.84)
    _material(asset, "lab_city_blue", "0.30 0.47 0.62 1", reflectance=0.04, shininess=0.20, specular=0.17, roughness=0.72)
    _material(asset, "lab_city_light", "0.56 0.70 0.78 1", reflectance=0.05, shininess=0.22, specular=0.20, roughness=0.62)
    _material(asset, "lab_metal", "0.42 0.48 0.51 1", reflectance=0.18, shininess=0.64, specular=0.66, metallic=0.65, roughness=0.22)
    _material(asset, "lab_metal_light", "0.72 0.77 0.78 1", reflectance=0.18, shininess=0.52, specular=0.58, metallic=0.40, roughness=0.28)
    _material(asset, "lab_metal_dark", "0.16 0.20 0.22 1", reflectance=0.22, shininess=0.56, specular=0.52, metallic=0.55, roughness=0.25)
    _material(asset, "lab_appliance", "0.84 0.87 0.87 1", reflectance=0.12, shininess=0.45, specular=0.42, roughness=0.34)
    _material(asset, "lab_lid", "0.38 0.54 0.60 1", reflectance=0.20, shininess=0.58, specular=0.60, roughness=0.22)
    _material(asset, "lab_well", "0.08 0.12 0.14 1", reflectance=0.05, shininess=0.25, specular=0.2, roughness=0.60)
    _material(asset, "lab_glass", "0.68 0.82 0.85 0.60", reflectance=0.20, shininess=0.72, specular=0.76, roughness=0.12)
    _material(asset, "lab_cap", "0.15 0.28 0.33 1", reflectance=0.08, shininess=0.34, specular=0.34, roughness=0.44)
    _material(asset, "lab_liquid_blue", "0.12 0.54 0.70 0.92", reflectance=0.10, shininess=0.46, specular=0.46, roughness=0.24)
    _material(asset, "lab_liquid_amber", "0.76 0.46 0.12 0.92", reflectance=0.10, shininess=0.42, specular=0.44, roughness=0.28)
    _material(asset, "lab_liquid_clear", "0.74 0.88 0.88 0.72", reflectance=0.16, shininess=0.62, specular=0.65, roughness=0.18)
    _material(asset, "lab_monitor_frame", "0.27 0.31 0.33 1", reflectance=0.12, shininess=0.44, specular=0.45, roughness=0.32)
    _material(asset, "lab_screen", "0.025 0.065 0.085 1", reflectance=0.10, shininess=0.66, specular=0.72, roughness=0.16)
    _material(asset, "lab_keyboard", "0.68 0.72 0.73 1", reflectance=0.08, shininess=0.3, specular=0.3, roughness=0.45)
    _material(asset, "lab_black", "0.025 0.03 0.032 1", reflectance=0.08, shininess=0.28, specular=0.3, roughness=0.5)
    _material(asset, "lab_lens", "0.05 0.24 0.32 1", reflectance=0.20, shininess=0.72, specular=0.78, roughness=0.10)
    _material(asset, "lab_pipette", "0.95 0.96 0.94 1", reflectance=0.08, shininess=0.40, specular=0.40, roughness=0.34)
    _material(asset, "lab_pipette_tip", "0.28 0.63 0.70 0.9", reflectance=0.12, shininess=0.48, specular=0.46, roughness=0.24)
    _material(asset, "lab_tray", "0.54 0.61 0.63 1", reflectance=0.14, shininess=0.48, specular=0.48, metallic=0.2, roughness=0.30)
    _material(asset, "lab_led_blue", "0.08 0.54 0.95 1", emission=0.35, reflectance=0.08, shininess=0.55, specular=0.5, roughness=0.22)
    _material(asset, "lab_stool_gray", "0.31 0.36 0.39 1", reflectance=0.08, shininess=0.24, specular=0.24, roughness=0.64)
    _material(asset, "lab_caster", "0.08 0.10 0.11 1", reflectance=0.05, shininess=0.24, specular=0.22, roughness=0.62)
    _material(asset, "lab_target", "0.96 0.20 0.08 0.86", emission=0.08, reflectance=0.08, shininess=0.32, specular=0.30, roughness=0.44)


def add_lab(root: ET.Element) -> ET.Element:
    """Append the laboratory's assets and worldbody to an XML root.

    ``root`` may be a ``mujocoinclude`` fragment or a full ``mujoco`` model.
    Existing asset/worldbody sections are reused.  All generated geometry names
    carry the ``lab_`` prefix; the two public manipulation sites and four camera
    names are intentionally stable API names.
    """

    _add_assets(root)
    _add_visual(root)
    worldbody = _worldbody(root)
    _add_wall_panels(worldbody)
    _add_ceiling(worldbody)
    _add_benches(worldbody)
    _add_rear_cabinetry(worldbody)
    _add_lab_equipment(worldbody)
    # Chairs sit alongside the runs and leave the robot's front work rectangle clear.
    for index, (x, y) in enumerate(((1.48, -2.38), (1.48, 0.55), (1.48, 2.10), (-1.55, -2.70), (-1.55, 2.15))):
        _add_stool(worldbody, f"lab_stool_{index}", x, y)
    _add_sites_and_cameras(worldbody)
    _add_lights(worldbody)
    return root


def describe_layout() -> Mapping[str, object]:
    """Return the stable layout values useful to controllers and tests."""

    return {
        "room_bounds": {"x": (-3.4, 3.4), "y": (-4.3, 4.3), "z": (0.0, 3.25)},
        "window_wall": "x=-3.4",
        "right_bench": {"x": (2.0, 3.2), "y": (-3.1, 2.8)},
        "workstation": {"top_z": 0.80, "x": (2.0, 2.55), "y": (-1.4, -0.25)},
        "sites": {"work_surface": (2.13, -0.80, 0.805), "placement_target": (2.14, -0.42, 0.813)},
    }


def build(output_path: Path | str | None = None) -> Path:
    """Generate ``assets/lab_room.xml`` and return its path."""

    script_dir = Path(__file__).resolve().parent
    default_path = script_dir.parent / "assets" / "lab_room.xml"
    path = Path(output_path) if output_path is not None else default_path
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("mujocoinclude")
    add_lab(root)
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def main() -> None:
    path = build()
    print(path)


if __name__ == "__main__":
    main()
