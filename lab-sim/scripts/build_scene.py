#!/usr/bin/env python3
"""Merge the portable lab and G1 assets into directly loadable MJCF scenes."""
from pathlib import Path
import copy
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build():
    source = ROOT / "assets" / "lab_room.xml"
    if not source.exists():
        raise FileNotFoundError(f"Generate laboratory geometry first: {source}")
    lab = ET.parse(source).getroot()
    lab.tag = "mujoco"
    lab.set("model", "reference_laboratory")
    robot = ET.parse(ROOT / "assets/g1/robot.xml").getroot()
    (ROOT / "scenes").mkdir(exist_ok=True)
    # Every output is self contained apart from relative meshes/textures.
    compiler = lab.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(lab, "compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    compiler.set("meshdir", "../assets/g1/meshes")
    compiler.set("texturedir", "../assets/textures")
    option = lab.find("option")
    if option is None:
        option = ET.SubElement(lab, "option")
    option.attrib.update(timestep="0.002", integrator="implicitfast", iterations="80", cone="elliptic", impratio="5")
    visual = lab.find("visual")
    if visual is None:
        visual = ET.SubElement(lab, "visual")
    glob = visual.find("global")
    if glob is None:
        glob = ET.SubElement(visual, "global")
    glob.attrib.update(offwidth="1920", offheight="1200")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.attrib.update(ambient=".48 .48 .48", diffuse=".40 .40 .40", specular=".06 .06 .06")
    for light in lab.findall(".//worldbody/light"):
        light.set("diffuse", ".30 .30 .30" if light.get("directional") == "1" else ".24 .24 .24")
        light.set("specular", ".06 .06 .06")
        light.set("attenuation", "1 0 0")
    for mat in lab.findall(".//material"):
        if mat.get("name") == "lab_sky":
            mat.set("emission", ".6")
        elif mat.get("name") in ("lab_city_blue", "lab_city_light", "lab_ceiling"):
            mat.set("emission", ".15")
    from lab_sim.props import add_props
    add_props(lab)
    ET.indent(lab, space="  ")
    ET.ElementTree(lab).write(ROOT / "scenes/lab_empty.xml", encoding="unicode")
    lab.set("model", "reference_laboratory_unitree_g1")
    assets = lab.find("asset")
    if assets is None:
        assets = ET.SubElement(lab, "asset")
    for asset in robot.findall("asset"):
        for mesh in asset.findall("mesh"):
            assets.append(copy.deepcopy(mesh))
    world = lab.find("worldbody")
    pelvis = copy.deepcopy(robot.find(".//body[@name='pelvis']"))
    pelvis.set("pos", "1.72 -0.8 0.79")
    for side in ("left", "right"):
        wrist = pelvis.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(wrist, "site", name=f"{side}_grasp", pos=f"0.1115 {'0.003' if side=='left' else '-0.003'} 0", size="0.007", rgba="0 0 0 0", group="4")
        ET.SubElement(wrist, "camera", name=f"{side}_wrist", pos=".09 0 .04", xyaxes="0 -1 0 0 0 1", fovy="75")
    torso = pelvis.find(".//body[@name='torso_link']")
    # Forward and down 20 degrees so a G1-height head sees the table and hands.
    ET.SubElement(torso, "camera", name="head", pos=".12 0 .30", xyaxes="0 -1 0 .342 0 .940", fovy="75")
    for site in pelvis.iter("site"):
        if site.get("name", "").endswith("palm_site") or site.get("name", "").startswith("imu_"):
            site.set("rgba", "0 0 0 0")
    for joint in pelvis.iter("joint"):
        if joint.get("type") != "free":
            joint.set("damping", ".05" if "hand_" in joint.get("name", "") else ".1")
            joint.set("armature", ".002" if "hand_" in joint.get("name", "") else ".01")
    # Insert robot first so qpos[0:7] is always the floating base.
    world.insert(0, pelvis)
    ET.SubElement(world, "body", name="base_anchor", mocap="true", pos="1.72 -.8 .79")
    for tag in ("contact", "actuator", "sensor"):
        for section in robot.findall(tag):
            target = lab.find(tag)
            if target is None:
                target = ET.SubElement(lab, tag)
            target.extend(copy.deepcopy(list(section)))
    equality = lab.find("equality")
    if equality is None:
        equality = ET.SubElement(lab, "equality")
    ET.SubElement(equality, "weld", name="base_support", body1="base_anchor", body2="pelvis", active="false", solref=".01 1")
    for name in ("sample_bottle", "sample_box"):
        if world.find(f".//body[@name='{name}']") is not None:
            ET.SubElement(equality, "weld", name=f"grasp_{name}", body1="right_wrist_yaw_link", body2=name, active="false", solref=".008 1")
    ET.indent(lab, space="  ")
    output = ROOT / "scenes/lab_g1.xml"
    ET.ElementTree(lab).write(output, encoding="unicode")
    print(output)


if __name__ == "__main__":
    build()
