"""URDF with two UR7e arms (a_ and b_), each carrying the InfinyToolkit grasper."""
import argparse
import glob
import os
import subprocess
import xml.etree.ElementTree as ET

import scene


def ur_urdf(prefix, ur_type="ur7e"):
    """One arm from the official ur_description xacro."""
    from ament_index_python.packages import get_package_share_directory
    xacro_file = os.path.join(get_package_share_directory("ur_description"), "urdf", "ur.urdf.xacro")
    result = subprocess.run(["xacro", xacro_file, f"ur_type:={ur_type}", f"name:=ur_{prefix.strip('_')}",
                             f"tf_prefix:={prefix}"], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"xacro failed for {ur_type}:\n{result.stderr}")
    return result.stdout


def grasper_meshes():
    """package:// paths of the grasper's shaft and jaw meshes, or None if not installed."""
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("ur7e_grasper_description")
    except Exception:
        return None
    files = [f for f in glob.glob(os.path.join(share, "**", "*"), recursive=True)
             if f.lower().endswith((".stl", ".dae", ".obj"))]

    def find(*names):
        for f in files:
            if any(name in os.path.basename(f).lower() for name in names):
                return "package://ur7e_grasper_description/" + os.path.relpath(f, share)
        return None
    meshes = dict(shaft=find("shaft"), up=find("jaws_up", "jaw_up", "jawup"),
                  down=find("jaws_down", "jaw_down", "jawdown"))
    return meshes if all(meshes.values()) else None


def add_visual(link, geometry, xyz="0 0 0", rpy="0 0 0", rgba="0.6 0.6 0.65 1"):
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz=xyz, rpy=rpy)
    ET.SubElement(visual, "geometry").append(geometry)
    ET.SubElement(ET.SubElement(visual, "material", name=""), "color", rgba=rgba)


def add_grasper(robot, prefix, meshes):
    mount = ET.SubElement(robot, "joint", name=f"{prefix}grasper_mount", type="fixed")
    ET.SubElement(mount, "parent", link=f"{prefix}tool0")
    ET.SubElement(mount, "child", link=f"{prefix}grasper")
    ET.SubElement(mount, "origin", xyz=" ".join(map(str, scene.GRASPER_MOUNT_XYZ)),
                  rpy=" ".join(map(str, scene.GRASPER_MOUNT_RPY)))
    shaft = ET.SubElement(robot, "link", name=f"{prefix}grasper")
    if meshes:
        add_visual(shaft, ET.Element("mesh", filename=meshes["shaft"], scale="0.001 0.001 0.001"))
    else:
        add_visual(shaft, ET.Element("cylinder", radius=str(scene.SHAFT_RADIUS), length=str(scene.HINGE_OFFSET)),
                   xyz=f"0 {scene.HINGE_OFFSET / 2} 0", rpy="-1.5708 0 0")
    for side, sign in (("up", 1), ("down", -1)):
        hinge = ET.SubElement(robot, "joint", name=f"{prefix}jaw_{side}_joint", type="revolute")
        ET.SubElement(hinge, "parent", link=f"{prefix}grasper")
        ET.SubElement(hinge, "child", link=f"{prefix}jaw_{side}")
        ET.SubElement(hinge, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(hinge, "axis", xyz=f"{sign} 0 0")
        ET.SubElement(hinge, "limit", lower="-0.1", upper="1.0", effort="1", velocity="3")
        jaw = ET.SubElement(robot, "link", name=f"{prefix}jaw_{side}")
        if meshes:
            add_visual(jaw, ET.Element("mesh", filename=meshes[side], scale="0.001 0.001 0.001"))
        else:
            add_visual(jaw, ET.Element("box", size=f"0.004 {scene.JAW_LENGTH} 0.003"),
                       xyz=f"0 {-scene.JAW_LENGTH / 2} {sign * 0.002}")


def build(base_a=scene.BASE_A, yaw_a=scene.YAW_A, base_b=scene.BASE_B, yaw_b=scene.YAW_B,
          arm_urdf=ur_urdf, meshes=None):
    """URDF text for both arms; arm_urdf(prefix) gives one arm (the official model by default)."""
    robot = ET.Element("robot", name="dual_ur7e_grasper")
    ET.SubElement(robot, "link", name="world")
    materials = set()
    for prefix, base, yaw in (("a_", base_a, yaw_a), ("b_", base_b, yaw_b)):
        placed = False
        for el in list(ET.fromstring(arm_urdf(prefix))):
            if el.tag == "link" and el.get("name") == "world":
                continue
            if el.tag in ("ros2_control", "gazebo", "transmission"):
                continue
            if el.tag == "material":
                if el.get("name") in materials:
                    continue
                materials.add(el.get("name"))
            if el.tag == "joint" and el.find("parent").get("link") == "world":
                el.set("name", f"{prefix}world_joint")
                origin = el.find("origin")
                if origin is None:
                    origin = ET.SubElement(el, "origin")
                origin.set("xyz", " ".join(f"{v:.6f}" for v in base))
                origin.set("rpy", f"0 0 {yaw:.6f}")
                placed = True
            robot.append(el)
        if not placed:
            raise SystemExit(f"arm {prefix}: no joint from 'world' in the UR model")
        add_grasper(robot, prefix, meshes)
    return ET.tostring(robot, encoding="unicode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dual_ur7e_grasper.urdf")
    parser.add_argument("--ur-type", default="ur7e")
    parser.add_argument("--no-meshes", action="store_true",
                        help="draw the graspers as simple shapes (a cylinder and two jaws) instead of the "
                             "InfinyToolkit meshes, e.g. when RViz cannot load them")
    args = parser.parse_args()
    meshes = None if args.no_meshes else grasper_meshes()
    print("grasper meshes:", meshes if meshes else "not found - using simple shapes")
    open(args.out, "w").write(build(arm_urdf=lambda p: ur_urdf(p, args.ur_type), meshes=meshes))
    print("wrote", args.out)
