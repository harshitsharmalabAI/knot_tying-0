"""E6: grid search over arm base positions, scored by the smallest |det J| along the knot."""
import argparse
import csv
import itertools
from multiprocessing import Pool

import numpy as np

import common
from common import scene

GRID_B_X = np.round(np.arange(-1.00, -0.699, 0.05), 3)
GRID_A_X = np.round(np.arange(0.80, 1.101, 0.05), 3)
GRID_Y = np.round(np.arange(-0.20, 0.201, 0.10), 3)
SPEED_LIMIT = np.pi


def capsules(chain, q, instrument_radius):
    poses = chain.frames(q)[0]
    points = [T[:3, 3] for T in poses]
    out = []
    for k in range(1, len(points)):
        if np.linalg.norm(points[k] - points[k - 1]) > 1e-4:
            out.append((points[k - 1], points[k], scene.LINK_RADIUS, False))
    tool = poses[-1]
    out.append((tool[:3, 3], tool[:3, 3] + (scene.HINGE_OFFSET + scene.JAW_LENGTH) * tool[:3, 2],
                instrument_radius, True))
    return out


def segment_distance(p0, p1, q0, q1):
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
    t = (b * s + f) / e
    if t < 0:
        t, s = 0.0, np.clip(-c / a, 0, 1)
    elif t > 1:
        t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p0 + d1 * s) - (q0 + d2 * t)))


def pedestal_box():
    return (np.array([scene.PEDESTAL_X[0], scene.PEDESTAL_Y[0], 0.0]),
            np.array([scene.PEDESTAL_X[1], scene.PEDESTAL_Y[1], scene.PEDESTAL_HEIGHT]))


def evaluate(job):
    arm, x, y, urdf_xml, frames_path, every = job
    yaw = scene.YAW_A if arm == "a_" else scene.YAW_B
    chain = common.chains(common.move_base(urdf_xml, arm, (x, y, 0.0), yaw))[arm]
    t, stage, labels, targets, frame_dt, b_r = common.frame_targets(frames_path, every)
    result = dict(arm=arm[0].upper(), x=x, y=y, feasible=False, ik_ok_pct=0.0, min_manip=0.0,
                  max_joint_speed_dps=np.nan, lowest_link_cm=np.nan, pedestal_gap_cm=np.nan)
    q0 = common.solve_from_scratch(chain, targets[arm][0], tries=200)
    if q0 is None:
        return result, None
    r = common.track(chain, targets[arm], q0=q0)
    result["ik_ok_pct"] = 100.0 * r["ok"].mean()
    speed = np.abs(np.diff(r["q"], axis=0)).max() / frame_dt
    manip = np.array([common.manipulability(chain, q)[0] for q in r["q"]])
    lo, hi = pedestal_box()
    lowest, near = np.inf, np.inf
    for q in r["q"][::2]:
        for k, (p0, p1, rad, is_tool) in enumerate(capsules(chain, q, 0.005)):
            if k == 0 or is_tool:
                continue
            lowest = min(lowest, min(p0[2], p1[2]) - rad)
            for s in np.linspace(0, 1, 5):
                p = p0 + (p1 - p0) * s
                near = min(near, float(np.linalg.norm(np.maximum(np.maximum(lo - p, 0), p - hi))) - rad)
    result.update(min_manip=float(manip.min()), max_joint_speed_dps=float(np.degrees(speed)),
                  lowest_link_cm=100 * lowest, pedestal_gap_cm=100 * near)
    result["feasible"] = bool(r["ok"].all() and speed <= SPEED_LIMIT and lowest >= 0 and near >= 0)
    return result, r["q"]


def arm_gap(urdf_xml, base_a, base_b, qa, qb, b_r):
    urdf = common.move_base(common.move_base(urdf_xml, "a_", base_a, scene.YAW_A), "b_", base_b, scene.YAW_B)
    ch = common.chains(urdf)
    gap = np.inf
    for i in range(0, len(qa), 2):
        for c1 in capsules(ch["a_"], qa[i], scene.SHAFT_RADIUS):
            for c2 in capsules(ch["b_"], qb[i], b_r):
                gap = min(gap, segment_distance(c1[0], c1[1], c2[0], c2[1]) - c1[2] - c2[2])
    return gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default=common.BASELINE_FRAMES)
    parser.add_argument("--urdf", default=common.DEFAULT_URDF)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--every", type=int, default=2, help="use every n-th frame")
    args = parser.parse_args()
    out = common.out_dir("base_placement")
    urdf_xml = common.load_urdf(args.urdf)
    jobs = [("b_", x, y, urdf_xml, args.frames, args.every) for x, y in itertools.product(GRID_B_X, GRID_Y)]
    jobs += [("a_", x, y, urdf_xml, args.frames, args.every) for x, y in itertools.product(GRID_A_X, GRID_Y)]
    current = {("b_", round(float(scene.BASE_B[0]), 3), round(float(scene.BASE_B[1]), 3)),
               ("a_", round(float(scene.BASE_A[0]), 3), round(float(scene.BASE_A[1]), 3))}
    with Pool(args.jobs) as pool:
        results = pool.map(evaluate, jobs)
    rows, paths = [], {}
    for job, (row, q) in zip(jobs, results):
        row["current"] = (job[0], job[1], job[2]) in current
        rows.append(row)
        paths[(job[0], job[1], job[2])] = q
        print(f"arm {row['arm']} base ({row['x']:+.2f}, {row['y']:+.2f}): feasible {row['feasible']!s:5s} "
              f"min |det J| {row['min_manip']:.4f}  speed {row['max_joint_speed_dps']:.0f} deg/s", flush=True)
    with open(f"{out}/base_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()} for row in rows)
    _, _, _, _, _, b_r = common.frame_targets(args.frames, args.every)
    best = {}
    for arm in ("a_", "b_"):
        ok = sorted((r for r in rows if r["arm"] == arm[0].upper() and r["feasible"]), key=lambda r: -r["min_manip"])
        best[arm] = ok[:4]
    pairs = []
    for ra, rb in itertools.product(best["a_"], best["b_"]):
        qa, qb = paths[("a_", ra["x"], ra["y"])], paths[("b_", rb["x"], rb["y"])]
        gap = arm_gap(urdf_xml, (ra["x"], ra["y"], 0.0), (rb["x"], rb["y"], 0.0), qa, qb, b_r)
        pairs.append(dict(a_x=ra["x"], a_y=ra["y"], b_x=rb["x"], b_y=rb["y"], score=min(ra["min_manip"], rb["min_manip"]),
                          arm_gap_cm=100 * gap, ok=gap >= scene.CLEARANCE))
    cur_a = next(r for r in rows if r["arm"] == "A" and r["current"])
    cur_b = next(r for r in rows if r["arm"] == "B" and r["current"])
    pairs.sort(key=lambda p: (not p["ok"], -p["score"]))
    with open(f"{out}/best_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0]) if pairs else ["none"])
        w.writeheader()
        w.writerows(pairs)
    print(f"\ncurrent bases: A min |det J| {cur_a['min_manip']:.4f}, B {cur_b['min_manip']:.4f}")
    if pairs:
        p = pairs[0]
        print(f"best pair: A ({p['a_x']:+.2f}, {p['a_y']:+.2f}), B ({p['b_x']:+.2f}, {p['b_y']:+.2f}), "
              f"score {p['score']:.4f}, arm gap {p['arm_gap_cm']:.1f} cm, ok {p['ok']}")
    print(f"wrote {out}/base_grid.csv and best_pairs.csv")


if __name__ == "__main__":
    main()
