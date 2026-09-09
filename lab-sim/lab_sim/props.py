"""Collidable task objects, a sliding drawer, spring button and task markers."""
import xml.etree.ElementTree as ET


def add(parent, tag, **kw):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in kw.items()})


def box(parent, name, pos, size, material="lab_counter", **kw):
    return add(parent, "geom", name=name, type="box", pos=pos, size=size, material=material, **kw)


def add_props(root):
    world = root.find("worldbody")
    for name, pos, color in (("sample_bottle", "2.10 -.92 .875", "lab_liquid_blue"),
                             ("sample_box", "2.10 -1.23 .839", "lab_cabinet")):
        body = add(world, "body", name=name, pos=pos)
        add(body, "freejoint", name=f"{name}_free")
        if name == "sample_bottle":
            add(body, "geom", name="sample_bottle_collision", type="cylinder", size=".026 .073", material=color,
                mass=".12", friction="1 .01 .001", condim="4")
            add(body, "geom", name="sample_bottle_cap", type="cylinder", pos="0 0 .079", size=".025 .008",
                material="lab_cap", mass=".01", contype="0", conaffinity="0")
            add(body, "geom", name="sample_bottle_label", type="cylinder", size=".0263 .019", material="lab_cabinet",
                mass=".001", contype="0", conaffinity="0")
        else:
            box(body, "sample_box_collision", "0 0 0", ".045 .065 .038", mass=".09", friction="1 .01 .001")
            box(body, "sample_box_lid", "0 0 .039", ".046 .066 .006", material="lab_lid", mass=".01", contype="0", conaffinity="0")
        add(body, "site", name=f"{name}_grasp", pos="0 0 .065" if name == "sample_bottle" else "0 0 .03", size=".005", rgba="0 0 0 0")
    # Tray is a shallow open container with a physical bottom and four sides.
    tray = add(world, "body", name="sample_tray", pos="2.14 -.42 .808")
    box(tray, "tray_bottom", "0 0 0", ".14 .14 .008", "lab_tray")
    for name, pos, size in (("left", "-.146 0 .018", ".006 .152 .025"), ("right", ".146 0 .018", ".006 .152 .025"),
                             ("front", "0 -.146 .018", ".14 .006 .025"), ("back", "0 .146 .018", ".14 .006 .025")):
        box(tray, f"tray_{name}", pos, size, "lab_tray")
    # Replace the solid work cabinet with a shell, leaving real clearance for a drawer.
    for g in list(world):
        if g.tag == "geom" and (g.get("name") == "lab_right_workstation_body" or g.get("name", "").startswith("lab_right_workstation_drawer")):
            world.remove(g)
    box(world, "work_cabinet_bottom", "2.6 -.8 .08", ".60 .60 .08", "lab_cabinet")
    box(world, "work_cabinet_back", "3.17 -.8 .45", ".03 .60 .29", "lab_cabinet")
    for i, y in enumerate((-1.38, -.22)):
        box(world, f"work_cabinet_side_{i}", f"2.6 {y} .45", ".60 .02 .29", "lab_cabinet")
    drawer = add(world, "body", name="lab_drawer", pos="2.59 -.80 .63")
    add(drawer, "joint", name="drawer_slide", type="slide", axis="-1 0 0", range="0 .32", damping="12", frictionloss="2", armature=".02")
    box(drawer, "drawer_bottom", "0 0 -.055", ".52 .48 .015", "lab_cabinet", mass="1")
    box(drawer, "drawer_front", "-.55 0 .01", ".025 .54 .10", "lab_cabinet", mass=".6")
    for i, y in enumerate((-.485, .485)):
        box(drawer, f"drawer_side_{i}", f"0 {y} .01", ".52 .012 .08", "lab_cabinet", mass=".15")
    box(drawer, "drawer_rear", ".52 0 .01", ".012 .48 .08", "lab_cabinet", mass=".15")
    add(drawer, "geom", name="drawer_handle", type="capsule", fromto="-.61 -.12 .01 -.61 .12 .01", size=".012", material="lab_metal", mass=".08")
    add(drawer, "site", name="drawer_handle_site", pos="-.625 0 .01", size=".008", rgba="0 0 0 0")
    # Benchtop analyzer with a physical spring-loaded button facing the aisle.
    machine = add(world, "body", name="analyzer", pos="2.33 -1.10 .96")
    box(machine, "analyzer_housing", "0 0 0", ".13 .09 .16", "lab_appliance")
    box(machine, "analyzer_screen", "-.132 0 .05", ".003 .058 .058", "lab_screen", contype="0", conaffinity="0")
    button = add(machine, "body", name="instrument_button", pos="-.165 0 -.065")
    add(button, "joint", name="button_slide", type="slide", axis="1 0 0", range="0 .014", stiffness="180", damping="2", armature=".001", solreflimit=".002 1")
    add(button, "geom", name="button_collision", type="cylinder", size=".021 .012", quat=".7071068 0 .7071068 0", material="lab_led_blue", mass=".025")
    add(button, "site", name="button_site", pos="-.014 0 0", size=".012", rgba="0 0 0 0")
    actuator = root.find("actuator")
    if actuator is None:
        actuator = add(root, "actuator")
    # These are explicit assistance actuators. The environment never applies
    # them during external-policy control; physical hand contact also works.
    add(actuator, "motor", name="drawer_assist", joint="drawer_slide", ctrllimited="true", ctrlrange="-45 45")
    add(actuator, "motor", name="button_assist", joint="button_slide", ctrllimited="true", ctrlrange="0 8")
    sensor = root.find("sensor")
    if sensor is None:
        sensor = add(root, "sensor")
    add(sensor, "jointpos", name="drawer_position", joint="drawer_slide")
    add(sensor, "jointpos", name="button_position", joint="button_slide")
    for i, (x, y) in enumerate(((0, -2.8), (-.6, -.8), (0, 2.5), (.45, .7))):
        add(world, "site", name=f"inspection_{i}", pos=f"{x} {y} .006", type="cylinder", size=".14 .003", rgba=".15 .58 .64 .25", group="4")
