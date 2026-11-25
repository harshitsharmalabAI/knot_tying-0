"""Fallback two-arm URDF from the UR7e DH table, used when ur_description is missing."""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "knot_ros"))
import scene  # noqa: E402

DH = [(0.0, 0.1625, math.pi / 2), (-0.425, 0.0, 0.0), (-0.3922, 0.0, 0.0),
      (0.0, 0.1333, math.pi / 2), (0.0, 0.0997, -math.pi / 2), (0.0, 0.0996, 0.0)]   # (a, d, alpha)


def arm(prefix, base, yaw):
    out = [f'<link name="{prefix}base_link"/>',
           f'<joint name="{prefix}world_joint" type="fixed"><parent link="world"/><child link="{prefix}base_link"/>'
           f'<origin xyz="{base[0]} {base[1]} {base[2]}" rpy="0 0 {yaw}"/></joint>']
    parent = f"{prefix}base_link"
    for i, (a, d, alpha) in enumerate(DH, start=1):
        rot, link = f"{prefix}link{i}_rot", (f"{prefix}tool0" if i == 6 else f"{prefix}link{i}")
        out.append(f'<link name="{rot}"/><link name="{link}"/>')
        out.append(f'<joint name="{prefix}joint{i}" type="revolute"><parent link="{parent}"/><child link="{rot}"/>'
                   f'<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>'
                   f'<limit lower="-6.2832" upper="6.2832" effort="1" velocity="3.14"/></joint>')
        out.append(f'<joint name="{prefix}dh{i}" type="fixed"><parent link="{rot}"/><child link="{link}"/>'
                   f'<origin xyz="{a} 0 {d}" rpy="{alpha} 0 0"/></joint>')
        parent = link
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="knot_ros/dual_ur7e_dh.urdf")
    args = parser.parse_args()
    body = arm("a_", scene.BASE_A, scene.YAW_A) + "\n" + arm("b_", scene.BASE_B, scene.YAW_B)
    open(args.out, "w").write(f'<robot name="dual_ur7e_dh">\n<link name="world"/>\n{body}\n</robot>\n')
    print("wrote", args.out)
