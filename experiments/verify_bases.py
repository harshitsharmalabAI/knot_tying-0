"""Runs the best bases from base_placement.py through plan_motion.py's checks."""
import csv
import os
import subprocess
import sys

import common
from common import scene

out = common.out_dir("base_placement")
with open(os.path.join(out, "best_pairs.csv")) as f:
    best = next(r for r in csv.DictReader(f) if r["ok"] == "True")
urdf = common.load_urdf()
cases = {"current": ((scene.BASE_A[0], scene.BASE_A[1]), (scene.BASE_B[0], scene.BASE_B[1])),
         "best": ((float(best["a_x"]), float(best["a_y"])), (float(best["b_x"]), float(best["b_y"])))}
for name, ((ax, ay), (bx, by)) in cases.items():
    moved = common.move_base(common.move_base(urdf, "a_", (ax, ay, 0.0), scene.YAW_A), "b_", (bx, by, 0.0), scene.YAW_B)
    path = os.path.join(out, f"dual_ur7e_{name}.urdf")
    open(path, "w").write(moved)
    result = subprocess.run([sys.executable, os.path.join(common.KNOT_ROS, "plan_motion.py"), common.BASELINE_FRAMES,
                             "--urdf", path, "--out", os.path.join(out, f"knot_motion_{name}.npz")],
                            capture_output=True, text=True, cwd=common.KNOT_ROS)
    text = f"bases A ({ax:+.2f}, {ay:+.2f}), B ({bx:+.2f}, {by:+.2f})\n" + result.stdout + result.stderr
    open(os.path.join(out, f"verify_{name}_bases.txt"), "w").write(text)
    print(f"== {name}\n{text}")
