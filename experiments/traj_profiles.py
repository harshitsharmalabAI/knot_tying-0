"""E2: joint speed, acceleration and jerk for different time scalings of the instrument paths."""
import argparse
import csv

import numpy as np

import common
import tension_knot_exp as tk

PROFILES = [("linear", False), ("cosine", False), ("cubic", False), ("quintic", False),
            ("cosine", True), ("quintic", True)]


def name_of(profile, ease_wraps):
    return profile + (" + eased wraps" if ease_wraps else "") + (" (ours)" if (profile, ease_wraps) == ("cosine", False) else "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", default=common.DEFAULT_URDF)
    parser.add_argument("--dt", type=float, default=0.004)
    args = parser.parse_args()
    out = common.out_dir("traj_profiles")
    chains = common.chains(common.load_urdf(args.urdf))
    pos, _, _ = tk.build_thread()
    a_start = pos[:, tk.GRIP_NODE].copy()
    rows, series = [], {}
    q_first = {}
    for profile, ease_wraps in PROFILES:
        tk.PROFILE, tk.EASE_WRAPS = profile, ease_wraps
        plan = tk.Plan(tk.stages(a_start), a_start)
        ts = np.arange(0.0, plan.ends[-1], args.dt)
        hinges = {"a_": np.array([common.scene.to_world(plan.a_pos(t)) for t in ts]),
                  "b_": np.array([common.scene.to_world(plan.b_tip(t)) for t in ts])}
        axes = {"a_": common.scene.direction_to_world(common.A_AXIS_SIM),
                "b_": common.scene.direction_to_world(tk.B_AXIS)}
        label = name_of(profile, ease_wraps)
        series[f"{label}|t"] = ts
        for arm, chain in chains.items():
            targets = [common.tool_pose(h, axes[arm]) for h in hinges[arm]]
            if arm not in q_first:
                q_first[arm] = common.solve_from_scratch(chain, targets[0])
            r = common.track(chain, targets, q0=q_first[arm], iters=100, tol=1e-9)
            q = np.unwrap(r["q"], axis=0)
            qd = np.gradient(q, args.dt, axis=0)
            qdd = np.gradient(qd, args.dt, axis=0)
            qddd = np.gradient(qdd, args.dt, axis=0)
            v = np.gradient(hinges[arm], args.dt, axis=0)
            acc = np.gradient(v, args.dt, axis=0)
            rows.append(dict(profile=label, arm=arm[0].upper(), ik_ok_pct=100 * r["ok"].mean(),
                             peak_joint_speed_dps=np.degrees(np.abs(qd).max()),
                             peak_joint_acc_dps2=np.degrees(np.abs(qdd).max()),
                             peak_joint_jerk_dps3=np.degrees(np.abs(qddd).max()),
                             rms_joint_jerk_dps3=np.degrees(np.sqrt((qddd ** 2).sum(1).mean())),
                             peak_tool_speed_mps=np.linalg.norm(v, axis=1).max(),
                             peak_tool_acc_mps2=np.linalg.norm(acc, axis=1).max(),
                             max_speed_jump_mps=np.linalg.norm(np.diff(v, axis=0), axis=1).max()))
            series[f"{label}|{arm}qd"] = qd.astype(np.float32)
            series[f"{label}|{arm}qdd"] = qdd.astype(np.float32)
            series[f"{label}|{arm}speed"] = np.linalg.norm(v, axis=1).astype(np.float32)
            print(f"{label:28s} arm {arm[0].upper()}: joint speed {rows[-1]['peak_joint_speed_dps']:7.1f} deg/s, "
                  f"acc {rows[-1]['peak_joint_acc_dps2']:9.1f} deg/s2, jerk {rows[-1]['peak_joint_jerk_dps3']:11.0f} deg/s3, "
                  f"IK ok {rows[-1]['ik_ok_pct']:.0f}%", flush=True)
    with open(f"{out}/traj_profiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in rows)
    np.savez_compressed(f"{out}/traj_series.npz", stage_ends=plan.ends, **series)
    print(f"wrote {out}/traj_profiles.csv and traj_series.npz")


if __name__ == "__main__":
    main()
