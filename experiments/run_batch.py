"""Runs the physics experiments, one folder per run under results/runs/."""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "results", "runs")
SIM = os.path.join(HERE, "tension_knot_exp.py")
CHECKPOINT = os.path.join(RUNS, "baseline", "tension_checkpoint_after_throw.npz")
SEEDS = range(10)
PI_GAINS = ["--kp", "0.01", "--ki", "0.01"]


def variation(seed):
    """Random variation for the robustness runs (the same for both controllers)."""
    rng = np.random.default_rng(1000 + seed)
    return dict(mu=round(float(rng.uniform(0.25, 0.35)), 4), orbit=round(float(rng.uniform(0.05, 0.07)), 4),
                noise=0.003, seed=seed)


def jobs(phase):
    if phase == 1:
        return [("baseline", ["--t-cinch", "15"])]
    if phase == 3:
        out = [("release_mu0.3", ["--t-cinch", "15", "--release"]),
               ("release_mu0.1", ["--t-cinch", "15", "--release", "--mu-thread", "0.1"]),
               ("release_mu0", ["--t-cinch", "15", "--release", "--mu-thread", "0"]),
               ("break_grip10N", ["--t-cinch", "15", "--grip-fmax", "10"]),
               ("break_wraps_at_tip", ["--t-cinch", "15", "--wrap-shift", "0.07"]),
               ("break_noise10mm", ["--t-cinch", "15", "--thread-noise", "0.01", "--seed", "0"]),
               ("break_orbit4.8cm", ["--t-cinch", "15", "--orbit-radius", "0.048"])]
        for seed in SEEDS:
            v = variation(seed)
            vary = ["--mu-thread", str(v["mu"]), "--orbit-radius", str(v["orbit"]),
                    "--thread-noise", str(v["noise"]), "--seed", str(v["seed"])]
            out.append((f"rob_pi2_s{seed}", vary + ["--t-cinch", "12", "--controller", "pi", "--kp", "0.02", "--ki", "0.02"]))
        return out
    resume = ["--resume", CHECKPOINT]
    out = [("ctrl_deadband_T15", resume + ["--t-cinch", "15"]),
           ("ctrl_deadband_T12", resume + ["--t-cinch", "12"])]
    for kp in ("0.005", "0.01", "0.02"):
        out.append((f"ctrl_p_kp{kp}", resume + ["--t-cinch", "12", "--controller", "p", "--kp", kp]))
    for kp, ki in (("0.01", "0.005"), ("0.01", "0.01"), ("0.02", "0.02")):
        out.append((f"ctrl_pi_kp{kp}_ki{ki}", resume + ["--t-cinch", "12", "--controller", "pi", "--kp", kp, "--ki", ki]))
    for seed in SEEDS:
        v = variation(seed)
        vary = ["--mu-thread", str(v["mu"]), "--orbit-radius", str(v["orbit"]),
                "--thread-noise", str(v["noise"]), "--seed", str(v["seed"])]
        out.append((f"rob_deadband_s{seed}", vary + ["--t-cinch", "15"]))
        out.append((f"rob_pi_s{seed}", vary + ["--t-cinch", "12", "--controller", "pi"] + PI_GAINS))
    out += [("fail_no_friction", ["--t-cinch", "15", "--mu-thread", "0"]),
            ("fail_wraps_near_tip", ["--t-cinch", "15", "--wrap-shift", "0.04"]),
            ("fail_no_filter", ["--t-cinch", "15", "--no-filter"]),
            ("prof_linear", ["--t-cinch", "15", "--profile", "linear"]),
            ("prof_quintic", ["--t-cinch", "15", "--profile", "quintic"]),
            ("prof_cosine_eased_wraps", ["--t-cinch", "15", "--ease-wraps"])]
    return out


def run_one(name, extra, extra_global):
    folder = os.path.join(RUNS, name)
    if os.path.exists(os.path.join(folder, "run_summary.json")):
        return name, "skipped (done)", 0.0
    os.makedirs(folder, exist_ok=True)
    env = dict(os.environ, NUMBA_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    start = time.time()
    with open(os.path.join(folder, "log.txt"), "w") as log:
        code = subprocess.call([sys.executable, SIM] + extra + extra_global, cwd=folder, stdout=log,
                               stderr=subprocess.STDOUT, env=env)
    status = "ok" if code == 0 else f"exit {code} (see {folder}/log.txt)"
    return name, status, time.time() - start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3])
    parser.add_argument("--jobs", type=int, default=max(1, min(14, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--only", default="", help="run only jobs whose name starts with this")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--time-scale", default=None, help=argparse.SUPPRESS)   # smoke tests only
    args = parser.parse_args()
    if args.list:
        for phase in (1, 2, 3):
            for name, extra in jobs(phase):
                print(f"phase {phase}  {name:28s} {' '.join(extra)}")
        return
    if args.phase is None:
        raise SystemExit("give --phase 1 or --phase 2 (or --list)")
    if args.phase in (2, 3) and not os.path.exists(os.path.join(RUNS, "baseline", "run_summary.json")):
        raise SystemExit("run phase 1 (the baseline) first")
    if args.phase == 2 and not os.path.exists(CHECKPOINT):
        raise SystemExit(f"phase 2 needs {CHECKPOINT}: run phase 1 first")
    todo = [(n, e) for n, e in jobs(args.phase) if n.startswith(args.only)]
    extra_global = ["--time-scale", args.time_scale] if args.time_scale else []
    os.makedirs(RUNS, exist_ok=True)
    print(f"phase {args.phase}: {len(todo)} runs, {args.jobs} at a time", flush=True)
    start = time.time()
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(run_one, n, e, extra_global) for n, e in todo]
        for future in as_completed(futures):
            name, status, secs = future.result()
            print(f"  {name:28s} {status:30s} {secs/60:6.1f} min   (elapsed {(time.time()-start)/60:.0f} min)", flush=True)
    with open(os.path.join(RUNS, f"variations.json"), "w") as f:
        json.dump({f"s{s}": variation(s) for s in SEEDS}, f, indent=1)
    print(f"phase {args.phase} done in {(time.time()-start)/60:.0f} min")


if __name__ == "__main__":
    main()
