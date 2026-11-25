#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
SRC="$(cd "${1:-..}" && pwd)"
[ -f "$SRC/tension_knot.py" ] || { echo "no tension_knot.py in $SRC - pass the folder holding tension_knot.py"; exit 1; }

echo "== copying project files from $SRC"
cp "$SRC/frictional_contact.py" "$SRC/wound_tissue.py" "$SRC/knot_check.py" .
mkdir -p knot_ros
cp "$SRC/knot_ros/kin.py" "$SRC/knot_ros/scene.py" "$SRC/knot_ros/plan_motion.py" "$SRC/knot_ros/dual_arm.py" knot_ros/

echo "== two-arm URDF"
if [ -f "$SRC/knot_ros/dual_ur7e_grasper.urdf" ]; then
    cp "$SRC/knot_ros/dual_ur7e_grasper.urdf" knot_ros/ && echo "copied dual_ur7e_grasper.urdf"
elif (cd knot_ros && python3 dual_arm.py >/dev/null 2>&1); then
    echo "built dual_ur7e_grasper.urdf from ur_description"
else
    echo "WARNING: could not copy or build the official URDF (source ROS 2 and the grasper workspace first)."
    echo "         Using the DH-table fallback instead; the IK results will differ slightly."
    python3 ur_dh_urdf.py --out knot_ros/dual_ur7e_grasper.urdf
fi

echo "== environment"
python3 -c "import elastica, numba, numpy, matplotlib; print('numba', numba.__version__, '| numpy', numpy.__version__)"
python3 -c "import scipy; print('scipy', scipy.__version__)" 2>/dev/null || echo "scipy not found: significance tests will be skipped (optional: pip install scipy)"

if [ -n "$SKIP_CHECK" ]; then echo "setup done (check skipped)"; exit 0; fi
echo "== regression check: original vs experiment copy, settle stage only (about 1-2 min)"
CHECK=$(mktemp -d)
mkdir -p "$CHECK/orig" "$CHECK/exp"
cp "$SRC/tension_knot.py" "$CHECK/orig/"
( cd "$CHECK/orig" && PYTHONPATH="$OLDPWD" python3 tension_knot.py --stop-after 0 > log.txt 2>&1 )
( cd "$CHECK/exp" && python3 "$OLDPWD/tension_knot_exp.py" --stop-after 0 > log.txt 2>&1 )
python3 - "$CHECK" <<'PY'
import sys, numpy as np
d = sys.argv[1]
a, b = np.load(f"{d}/orig/tension_knot_stages.npy"), np.load(f"{d}/exp/tension_knot_stages.npy")
diff = float(np.abs(a - b).max())
print(f"max difference {diff:.3g} m ->", "IDENTICAL, good" if diff == 0 else "DIFFERENT - the experiment copy no longer matches tension_knot.py")
sys.exit(0 if diff == 0 else 1)
PY
rm -rf "$CHECK"
echo "setup done"
