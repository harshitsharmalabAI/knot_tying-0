# Course experiments: knot tying with two UR7e arms (AR-523)

New experiments for the final presentation. They test the classical-robotics parts of
the project: Jacobian IK, trajectory time scaling, tension feedback control, base placement,
plus robustness and failure cases of the PyElastica knot. The knot code itself lives in the repo
root; setup.sh copies the shared files into this folder.

## Run it

```bash
conda activate harshit_env
cd experiments

# only needed if knot_ros/ has no dual_ur7e_grasper.urdf yet (builds it from ur_description):
source /opt/ros/jazzy/setup.bash
PKG=$(find ~ -type d -path "*/install/ur7e_grasper_description" 2>/dev/null | head -1); source "$(dirname "$PKG")/setup.bash"

bash setup.sh                                   # copies the shared files, gets the URDF, checks the copy is IDENTICAL (1-2 min)
nohup bash run_all.sh > run_all.log 2>&1 &      # batch 1, about 2-2.5 h on a 16-core EPYC
tail -f run_all.log                             # Ctrl+C stops watching, not the run
```

It ends with `all done`; everything is in `results/` (`results/RESULTS.md`, `results/figures/`,
`results/tables/`), and `make_figures.py` also packs the small files into `results_summary.zip`.

Check progress at any time:
```bash
ls results/runs/*/run_summary.json | wc -l      # finished physics runs (35 in batch 1)
```
If the machine restarts or you stop it, run `nohup bash run_all.sh > run_all.log 2>&1 &` again:
finished runs are skipped.

## What runs

| Step | Script | What it answers | Time |
|---|---|---|---|
| Baseline | `run_batch.py --phase 1` | the published knot (deadband control, T* = 15 N, holds about 12 N); its frames and checkpoints feed everything else | ~20 min |
| E3a | `controller_model.py` | deadband vs P vs PI on a spring model of one strand (classical control, analytic tau, wn, zeta) | seconds |
| E1 | `ik_compare.py` | Newton J^-1 vs Jacobian transpose vs damped least squares (lambda 0.005 / 0.02 / 0.1) on every recorded frame; manipulability along the knot; singularity stress test | minutes |
| E2 | `traj_profiles.py` | linear vs cosine vs cubic vs quintic time scaling, with and without eased wraps: joint speed, acceleration, jerk | minutes |
| E6 | `base_placement.py` | grid search of both arm bases, score = smallest manipulability along the knot | 5-15 min |
| E3 | `run_batch.py --phase 2` (`ctrl_*`) | the same controllers in the full simulation, from the after-throw checkpoint | 8 runs |
| E4 | (`rob_*`) | 10 random variations (thread mu 0.25-0.35, wrap radius 5-7 cm, 3 mm thread offset) x {deadband, PI}: success rate, mean ± std, 95% CI, paired test | 20 runs |
| E5 | (`fail_*`) | failure cases: no thread friction, wraps near B's tip, no Laplace filter | 3 runs |
| E2b | (`prof_*`) | full knots with linear, quintic and eased-wrap timing | 3 runs |
| Figures | `make_figures.py` | figures/, tables/, RESULTS.md, results_summary.zip | 1 min |

## Running one piece again

```bash
conda activate harshit_env
cd experiments
python3 run_batch.py --list                                  # all run names
rm -rf results/runs/rob_pi_s3 && python3 run_batch.py --phase 2 --only rob_pi_s3
python3 make_figures.py
```
A failed run leaves its reason in `results/runs/<name>/log.txt`. An "UNSTABLE" or failed knot in
`fail_*` runs is expected (that is the point of those runs).

## Files

- `tension_knot_exp.py` - copy of tension_knot.py with extra options (`--controller`, `--profile`,
  `--ease-wraps`, `--orbit-radius`, `--wrap-shift`, `--thread-noise`, `--seed`, `--no-filter`); with no
  extra options it is identical to the original (setup.sh checks this). Writes `run_summary.json` and
  `tension_log.csv` (tensions every 5 ms).
- `common.py` - URDF, IK targets (as plan_motion.py), IK solvers, manipulability.
- `ur_dh_urdf.py` - fallback two-arm URDF from the DH table, used only if the official URDF is missing.

## Second batch (run_all2.sh)

```bash
conda activate harshit_env
cd experiments
JOBS=14 nohup bash run_all2.sh > run_all2.log 2>&1 &
tail -f run_all2.log
```
About 70-80 min. Adds:
- `release_mu0.3`, `release_mu0.1`, `release_mu0`: tie, hold, then both instruments let go for 3 s.
- `break_grip10N` (B's jaws capped at 10 N), `break_wraps_at_tip` (wraps moved 7 cm toward B's tip),
  `break_noise10mm` (1 cm thread offset), `break_orbit4.8cm` (wraps closer to B).
- `rob_pi2_s0..9`: the 10 robustness variations again with PI kp = ki = 0.02.
- `verify_bases.py`: the best E6 bases through plan_motion.py's 7 checks.
Then `make_figures.py` rebuilds everything and `results_summary.zip`.
