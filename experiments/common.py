"""Shared helpers for the course experiments: paths, URDF, IK targets, IK solvers."""
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KNOT_ROS = os.path.join(HERE, "knot_ros")
sys.path.insert(0, KNOT_ROS)
sys.path.insert(0, HERE)

import scene  # noqa: E402
from kin import Chain  # noqa: E402

RESULTS = os.path.join(HERE, "results")
BASELINE_FRAMES = os.path.join(RESULTS, "runs", "baseline", "tension_knot_frames.npz")
DEFAULT_URDF = os.path.join(KNOT_ROS, "dual_ur7e_grasper.urdf")
A_AXIS_SIM = np.array([-1.0, 0.0, 0.0])     # as in plan_motion.py
POS_TOL, ROT_TOL = scene.IK_POS_TOL, scene.IK_ROT_TOL


def out_dir(*parts):
    path = os.path.join(RESULTS, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def load_urdf(path=DEFAULT_URDF):
    if not os.path.exists(path):
        raise SystemExit(f"no URDF at {path}: run setup.sh (it copies or builds dual_ur7e_grasper.urdf)")
    return open(path).read()


def move_base(urdf_xml, arm, xyz, yaw):
    """The same URDF with one arm's base (world joint) moved."""
    robot = ET.fromstring(urdf_xml)
    for joint in robot.findall("joint"):
        if joint.get("name") == f"{arm}world_joint":
            origin = joint.find("origin")
            origin.set("xyz", " ".join(f"{v:.6f}" for v in xyz))
            origin.set("rpy", f"0 0 {yaw:.6f}")
            return ET.tostring(robot, encoding="unicode")
    raise SystemExit(f"no {arm}world_joint in the URDF")


def chains(urdf_xml):
    return {arm: Chain(urdf_xml, "world", f"{arm}tool0") for arm in ("a_", "b_")}


def tool_pose(hinge, axis):
    """tool0 pose that puts the jaw hinge at `hinge` with the shaft along `axis` (as plan_motion.py)."""
    z = axis / np.linalg.norm(axis)
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = hinge - scene.HINGE_OFFSET * z
    return T


def frame_targets(frames_path=BASELINE_FRAMES, every=1):
    """tool0 targets for both arms from a recorded run, as plan_motion.py builds them."""
    if not os.path.exists(frames_path):
        raise SystemExit(f"no frames file at {frames_path}: run the baseline first (run_batch.py --phase 1)")
    data = np.load(frames_path)
    pick = np.arange(0, len(data["t"]), max(1, every))
    t, stage = data["t"][pick], data["stage"][pick]
    targets = {"a_": [], "b_": []}
    a_last = None
    a_all = data["a_pos"][pick]
    first_ok = a_all[np.isfinite(a_all).all(1)][0]
    for i in pick:
        targets["b_"].append(tool_pose(scene.to_world(data["b_tip"][i]), scene.direction_to_world(data["b_u"][i])))
        if np.all(np.isfinite(data["a_pos"][i])):
            a_last = scene.to_world(data["a_pos"][i])
        hinge = a_last if a_last is not None else scene.to_world(first_ok)
        targets["a_"].append(tool_pose(hinge, scene.direction_to_world(A_AXIS_SIM)))
    labels = [str(s) for s in data["labels"]]
    return t, stage, labels, targets, float(data["frame_dt"]) * max(1, every), float(data["b_r"])


def solve(chain, target, start, method="dls", lam=0.02, iters=200, tol=1e-5, max_step=0.3):
    """One IK solve from `start` with method newton, transpose or dls."""
    q = np.array(start, dtype=float)
    used = iters
    for k in range(iters):
        J, T = chain.jacobian(q)
        e = chain.pose_error(T, target)
        if np.linalg.norm(e[:3]) < tol and np.linalg.norm(e[3:]) < tol:
            used = k
            break
        if method == "newton":
            try:
                dq = np.linalg.solve(J, e)
            except np.linalg.LinAlgError:
                used = iters
                break
        elif method == "transpose":
            g = J.T @ e
            Jg = J @ g
            dq = (e @ Jg) / max(Jg @ Jg, 1e-12) * g
        else:
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(6), e)
        size = np.linalg.norm(dq)
        if not np.isfinite(size):
            used = iters
            break
        if size > max_step:
            dq *= max_step / size
        q = np.clip(q + dq, chain.lower, chain.upper)
    e = chain.pose_error(chain.fk(q), target)
    pos_err, rot_err = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
    return q, used, pos_err, rot_err, (pos_err <= POS_TOL and rot_err <= ROT_TOL)


def solve_from_scratch(chain, target, tries=300, seed=0):
    """IK from many random starts; keep the elbow-up answer (as plan_motion.py)."""
    rng = np.random.default_rng(seed)
    lo, hi = np.maximum(chain.lower, -np.pi), np.minimum(chain.upper, np.pi)
    best = None
    for start in lo + (hi - lo) * rng.random((tries, chain.n)):
        q, _, pos_err, rot_err, ok = solve(chain, target, start, iters=150)
        if ok:
            elbow_height = chain.frames(q)[2][2][2]
            score = elbow_height - 0.01 * np.abs(q).sum()
            if best is None or score > best[0]:
                best = (score, q)
    return None if best is None else best[1]


def manipulability(chain, q):
    """Yoshikawa's index sqrt(det(J J^T)) = |det J| for 6 joints, and the smallest singular value."""
    J, _ = chain.jacobian(q)
    s = np.linalg.svd(J, compute_uv=False)
    return float(np.prod(s)), float(s[-1])


def track(chain, targets, method="dls", lam=0.02, q0=None, iters=200, tol=1e-5):
    """Solve a whole target sequence with warm starts. Returns dict of arrays."""
    if q0 is None:
        q0 = solve_from_scratch(chain, targets[0])
        if q0 is None:
            raise SystemExit("no IK solution for the first pose")
    q, out = np.array(q0), dict(q=[], iters=[], pos=[], rot=[], ok=[], ms=[])
    for target in targets:
        start = time.perf_counter()
        q, used, pos_err, rot_err, ok = solve(chain, target, q, method=method, lam=lam, iters=iters, tol=tol)
        out["ms"].append(1e3 * (time.perf_counter() - start))
        for key, value in (("q", q.copy()), ("iters", used), ("pos", pos_err), ("rot", rot_err), ("ok", ok)):
            out[key].append(value)
    return {k: np.array(v) for k, v in out.items()}


def stage_spans(t, stage):
    """(start, end, stage) for shading plots."""
    spans, start = [], 0
    for i in range(1, len(stage) + 1):
        if i == len(stage) or stage[i] != stage[start]:
            spans.append((t[start], t[i - 1], int(stage[start])))
            start = i
    return spans
