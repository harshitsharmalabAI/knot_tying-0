"""Joint motion for both arms from a knot run; saved only if all checks pass."""
import argparse
import os
import sys

import numpy as np

import scene
from kin import Chain

JOINT_SPEED_LIMIT = np.pi
A_AXIS_SIM = np.array([-1.0, 0.0, 0.0])


def tool_pose(hinge, axis):
    """tool0 pose with the jaw hinge at `hinge` and the shaft along `axis`."""
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


def solve_from_scratch(chain, target, tries=300):
    """IK from many random starts; keep the elbow-up answer (links stay high)."""
    rng = np.random.default_rng(0)
    lo, hi = np.maximum(chain.lower, -np.pi), np.minimum(chain.upper, np.pi)
    best = None
    for start in lo + (hi - lo) * rng.random((tries, chain.n)):
        q, pos_err, rot_err = chain.ik(target, start, iters=150)
        if pos_err < scene.IK_POS_TOL and rot_err < scene.IK_ROT_TOL:
            elbow_height = chain.frames(q)[2][2][2]
            score = elbow_height - 0.01 * np.abs(q).sum()
            if best is None or score > best[0]:
                best = (score, q)
    return None if best is None else best[1]


def capsules(chain, q, instrument_radius):
    """Capsules (start, end, radius, link, is_instrument) for the arm links and instrument."""
    poses = chain.frames(q)[0]
    points = [T[:3, 3] for T in poses]
    names = [j["child"] for j in chain.joints]
    out = []
    for k in range(1, len(points)):
        if np.linalg.norm(points[k] - points[k - 1]) > 1e-4:
            out.append((points[k - 1], points[k], scene.LINK_RADIUS, names[k], False))
    tool = poses[-1]
    out.append((tool[:3, 3], tool[:3, 3] + (scene.HINGE_OFFSET + scene.JAW_LENGTH) * tool[:3, 2],
                instrument_radius, "instrument", True))
    return out


def segment_distance(p0, p1, q0, q1):
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(r)
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p0 + d1 * s) - (q0 + d2 * t)))


def box_distance(p, box):
    lo, hi = box
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, 0), p - hi)))


def pedestal_box():
    return (np.array([scene.PEDESTAL_X[0], scene.PEDESTAL_Y[0], 0.0]),
            np.array([scene.PEDESTAL_X[1], scene.PEDESTAL_Y[1], scene.PEDESTAL_HEIGHT]))


def report(ok, text):
    print(("PASS " if ok else "FAIL ") + text)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", nargs="?", default="tension_knot_frames.npz")
    parser.add_argument("--urdf", default="dual_ur7e_grasper.urdf")
    parser.add_argument("--out", default="knot_motion.npz")
    parser.add_argument("--every", type=int, default=1, help="use every n-th frame")
    args = parser.parse_args()

    if not os.path.exists(args.urdf):
        import dual_arm
        meshes = dual_arm.grasper_meshes()
        print("grasper meshes:", meshes if meshes else "not found - using simple shapes")
        open(args.urdf, "w").write(dual_arm.build(meshes=meshes))
        print("wrote", args.urdf)
    urdf = open(args.urdf).read()
    chains = {arm: Chain(urdf, "world", f"{arm}tool0") for arm in ("a_", "b_")}

    data = np.load(args.frames)
    picked = np.arange(0, len(data["t"]), max(1, args.every))
    times, stages = data["t"][picked], data["stage"][picked]
    b_tips, b_axes, a_positions, gripping = data["b_tip"][picked], data["b_u"][picked], data["a_pos"][picked], data["grip"][picked]
    labels = [str(s) for s in data["labels"]]
    n_frames = len(picked)
    b_radius = float(data["b_r"])
    print(f"{n_frames} frames, t = {times[0]:.2f} .. {times[-1]:.2f} s")

    hinges = {"a_": [], "b_": []}
    axes = {"a_": [], "b_": []}
    a_last = None
    for i in range(n_frames):
        hinges["b_"].append(scene.to_world(b_tips[i]))
        axes["b_"].append(scene.direction_to_world(b_axes[i]))
        if np.all(np.isfinite(a_positions[i])):
            a_last = scene.to_world(a_positions[i])
        hinges["a_"].append(a_last if a_last is not None
                            else scene.to_world(a_positions[np.isfinite(a_positions).all(1)][0]))
        axes["a_"].append(scene.direction_to_world(A_AXIS_SIM))
    a_holding = np.all(np.isfinite(a_positions), axis=1)

    ok = True
    joints = {}
    for arm, chain in chains.items():
        name = arm[0].upper()
        path, worst_pos, worst_rot = [], 0.0, 0.0
        q = None
        for i in range(n_frames):
            target = tool_pose(np.asarray(hinges[arm][i]), np.asarray(axes[arm][i]))
            if q is None:
                q = solve_from_scratch(chain, target)
                if q is None:
                    print(f"FAIL arm {name}: no IK solution for the first pose - "
                          f"out of reach from its base; move scene.BASE_{name}")
                    sys.exit(1)
                pos_err, rot_err = 0.0, 0.0
            else:
                q, pos_err, rot_err = chain.ik(target, q)
                if pos_err > scene.IK_POS_TOL or rot_err > scene.IK_ROT_TOL:
                    fresh = solve_from_scratch(chain, target)
                    if fresh is not None:
                        q, pos_err, rot_err = fresh, 0.0, 0.0
            worst_pos, worst_rot = max(worst_pos, pos_err), max(worst_rot, rot_err)
            path.append(q.copy())
        joints[arm] = np.array(path)
        ok &= report(worst_pos <= scene.IK_POS_TOL and worst_rot <= scene.IK_ROT_TOL,
                     f"arm {name}: IK reaches every pose (worst {worst_pos*1000:.2f} mm, "
                     f"{np.degrees(worst_rot):.2f} deg)")

    dt = float(data["frame_dt"]) * max(1, args.every)
    for arm in chains:
        steps = np.abs(np.diff(joints[arm], axis=0))
        biggest, at = steps.max(), int(np.unravel_index(steps.argmax(), steps.shape)[0])
        speed = biggest / dt
        ok &= report(speed <= JOINT_SPEED_LIMIT,
                     f"arm {arm[0].upper()}: fastest joint {np.degrees(speed):.0f} deg/s "
                     f"(limit {np.degrees(JOINT_SPEED_LIMIT):.0f}), largest step {np.degrees(biggest):.1f} deg "
                     f"at t = {times[at+1]:.2f} s ({labels[stages[at+1]]})")

    pedestal = pedestal_box()
    closest = (np.inf, None, None, None)
    lowest = (np.inf, None, None)
    near_pedestal = (np.inf, None, None)
    for i in range(n_frames):
        caps_a = capsules(chains["a_"], joints["a_"][i], scene.SHAFT_RADIUS)
        caps_b = capsules(chains["b_"], joints["b_"][i], b_radius)
        for c1 in caps_a:
            for c2 in caps_b:
                d = segment_distance(c1[0], c1[1], c2[0], c2[1]) - c1[2] - c2[2]
                if d < closest[0]:
                    closest = (d, i, c1[3], c2[3])
        for arm, caps in (("A", caps_a), ("B", caps_b)):
            for k, (p0, p1, r, link, is_instrument) in enumerate(caps):
                if k == 0 or is_instrument:
                    continue
                bottom = min(p0[2], p1[2]) - r
                if bottom < lowest[0]:
                    lowest = (bottom, i, f"{arm}:{link}")
                d = min(box_distance(p0 + (p1 - p0) * s, pedestal) for s in np.linspace(0, 1, 9)) - r
                if d < near_pedestal[0]:
                    near_pedestal = (d, i, f"{arm}:{link}")
    ok &= report(closest[0] >= scene.CLEARANCE,
                 f"arms never closer than {closest[0]*100:.1f} cm (limit {scene.CLEARANCE*100:.0f} cm; "
                 f"closest A:{closest[2]} - B:{closest[3]} at t = {times[closest[1]]:.2f} s, "
                 f"{labels[stages[closest[1]]]})")
    ok &= report(lowest[0] >= 0.0, f"links stay above the table (lowest {lowest[0]*100:.1f} cm, "
                                   f"{lowest[2]} at t = {times[lowest[1]]:.2f} s)")
    ok &= report(near_pedestal[0] >= 0.0, f"links clear the tissue pedestal (closest {near_pedestal[0]*100:.1f} cm, "
                                          f"{near_pedestal[2]} at t = {times[near_pedestal[1]]:.2f} s)")
    if not ok:
        print("\nnot saved: fix the failing check first (base placement in scene.py)")
        sys.exit(1)

    jaws = {}
    for arm, wanted in (("a_", np.where(a_holding, 0.0, scene.JAW_OPEN)), ("b_", np.where(gripping, 0.0, scene.JAW_OPEN))):
        jaw = np.empty(n_frames)
        jaw[0] = wanted[0]
        for i in range(1, n_frames):
            jaw[i] = jaw[i - 1] + np.clip(wanted[i] - jaw[i - 1], -scene.JAW_SPEED * dt, scene.JAW_SPEED * dt)
        jaws[arm] = jaw
    names, columns = [], []
    for arm, chain in chains.items():
        names += chain.names + [f"{arm}jaw_up_joint", f"{arm}jaw_down_joint"]
        columns += [joints[arm], jaws[arm][:, None], jaws[arm][:, None]]
    thread = np.stack([scene.to_world(p.astype(float)) for p in data["pos"][picked]]).astype(np.float32)
    np.savez_compressed(
        args.out, t=times, stage=stages, labels=np.array(labels), frame_dt=dt,
        joint_names=np.array(names), q=np.hstack(columns), thread=thread, hand_g=data["hand_g"][picked],
        bite=data["bite"], n=int(data["n"]), r_rod=float(data["r_rod"]),
        wound_iL=int(data["wound_iL"]), wound_iR=int(data["wound_iR"]), edge_bite=float(data["edge_bite"]),
        b_tip=np.array(hinges["b_"]), b_u=np.array(axes["b_"]), b_r=b_radius, b_len=float(data["b_len"]),
        grip=gripping, wound=data["wound"][picked], pedestal_lo=pedestal[0], pedestal_hi=pedestal[1],
        min_clearance=closest[0])
    print(f"\nall checks passed - wrote {args.out} ({n_frames} frames, {len(names)} joints)")


if __name__ == "__main__":
    main()
