"""E1: Newton, Jacobian transpose and DLS IK on every frame of the baseline knot."""
import argparse
import csv

import numpy as np

import common

METHODS = [("newton", "Newton (J^-1)", 0.0), ("transpose", "Jacobian transpose", 0.0),
           ("dls", "DLS lambda=0.005", 0.005), ("dls", "DLS lambda=0.02 (ours)", 0.02),
           ("dls", "DLS lambda=0.1", 0.1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default=common.BASELINE_FRAMES)
    parser.add_argument("--urdf", default=common.DEFAULT_URDF)
    parser.add_argument("--every", type=int, default=1)
    args = parser.parse_args()
    out = common.out_dir("ik_compare")
    t, stage, labels, targets, frame_dt, _ = common.frame_targets(args.frames, args.every)
    chains = common.chains(common.load_urdf(args.urdf))
    rows, series = [], {"t": t, "stage": stage}
    for arm, chain in chains.items():
        name = arm[0].upper()
        q0 = common.solve_from_scratch(chain, targets[arm][0])
        if q0 is None:
            raise SystemExit(f"arm {name}: no IK solution for the first pose")
        for method, label, lam in METHODS:
            r = common.track(chain, targets[arm], method=method, lam=lam, q0=q0)
            steps = np.degrees(np.abs(np.diff(r["q"], axis=0)).max(axis=1))
            rows.append(dict(arm=name, method=label, frames=len(t),
                             success_pct=100.0 * r["ok"].mean(),
                             iters_mean=r["iters"].mean(), iters_std=r["iters"].std(ddof=1),
                             iters_p95=np.percentile(r["iters"], 95),
                             pos_err_mm_mean=1e3 * r["pos"].mean(), pos_err_mm_max=1e3 * r["pos"].max(),
                             rot_err_deg_max=np.degrees(r["rot"].max()),
                             max_joint_step_deg=steps.max(), max_joint_speed_dps=steps.max() / frame_dt,
                             ms_per_frame=r["ms"].mean()))
            series[f"{arm}{label}_iters"] = r["iters"]
            series[f"{arm}{label}_pos"] = r["pos"]
            print(f"arm {name} {label:24s} success {rows[-1]['success_pct']:5.1f}%  "
                  f"iters {rows[-1]['iters_mean']:6.1f} +- {rows[-1]['iters_std']:5.1f}  "
                  f"max err {rows[-1]['pos_err_mm_max']:7.3f} mm  max step {rows[-1]['max_joint_step_deg']:6.2f} deg",
                  flush=True)
            if label.endswith("(ours)"):
                w = np.array([common.manipulability(chain, q) for q in r["q"]])
                series[f"{arm}manip"], series[f"{arm}sigma_min"] = w[:, 0], w[:, 1]
                series[f"{arm}q"] = r["q"]
    with open(f"{out}/ik_compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in rows)
    np.savez_compressed(f"{out}/ik_series.npz", labels=np.array(labels), frame_dt=frame_dt, **series)

    chain = chains["b_"]
    base = chain.frames(np.zeros(chain.n))[2][1]          # shoulder-lift joint origin
    start = targets["b_"][0]
    out_dir_vec = start[:3, 3] - base
    out_dir_vec[2] = 0.0
    out_dir_vec /= np.linalg.norm(out_dir_vec)
    reach = np.linspace(0.0, 0.40, 161)
    q_start = common.solve_from_scratch(chain, start)
    stress = {"extra_m": reach, "flange_dist_m": []}
    for method, label, lam in METHODS + [("newton", "Newton, no step limit", 0.0)]:
        max_step = np.inf if label.endswith("no step limit") else 0.3
        q, errs, jumps = q_start.copy(), [], []
        for d in reach:
            target = start.copy()
            target[:3, 3] = start[:3, 3] + d * out_dir_vec
            q_new, _, pos_err, _, _ = common.solve(chain, target, q, method=method, lam=lam, max_step=max_step)
            jumps.append(np.degrees(np.abs(q_new - q).max()))
            errs.append(pos_err)
            q = q_new
        stress[f"{label}_err"], stress[f"{label}_jump"] = np.array(errs), np.array(jumps)
    stress["flange_dist_m"] = np.array([np.linalg.norm(start[:3, 3] + d * out_dir_vec - base) for d in reach])
    np.savez_compressed(f"{out}/ik_stress.npz", **stress)
    print(f"wrote {out}/ik_compare.csv, ik_series.npz, ik_stress.npz")


if __name__ == "__main__":
    main()
