# knot_tying-0: two-instrument surgical knot tying (PyElastica + ROS 2, two UR7e arms)

Code for the AR-523 (Robot Manipulators) project presentation *Knot Tying with Robotic Manipulators*.
Two instruments tie one surgeon's double throw around a stitch on a soft wound, cinch it and hold it
at 12 N. The thread is a PyElastica Cosserat rod (5 mm radius, E = 1 MPa) with thread-on-thread
Coulomb friction added by us; in ROS 2 the instruments are carried by two UR7e arms with the
InfinyToolkit grasper.

the knot simulation, the ROS 2 two-arm pipeline, and the course experiments E0-E6.

| Experiment | Files |
|---|---|
| Thread model: stretch, bend, twist, Laplace filter (Theory) | `tension_knot.py` (PyElastica rod setup) |
| Contact and thread-on-thread friction (μ = 0.3 thread, 0.2 on B's shaft) | `frictional_contact.py`, `tension_knot.py` |
| Soft wound around the stitch | `wound_tissue.py`, `test_wound_tissue.py` (9 tests) |
| Nine-stage instrument tie, dead-band tension rule, cinch and hold | `tension_knot.py` |
| Knot checks: winding, Alexander determinant per strand | `knot_check.py` (`--strands`) |
| Movie of a run | `animate_knot.py` |
| UR7e kinematics from the URDF, two-arm scene | `knot_ros/kin.py`, `knot_ros/dual_arm.py`, `knot_ros/scene.py` |
| Motion planning: waypoints, IK on every frame, 7 safety checks | `knot_ros/plan_motion.py` |
| Replay and live (PyElastica in the loop) in RViz | `knot_ros/knot_replay.py`, `knot_ros/knot_live.py`, launch files, `knot.rviz` |
| E1 IK methods (Newton, Jacobian transpose, damped least squares) | `experiments/ik_compare.py`, `experiments/common.py` |
| E2 time scaling (linear, cosine, cubic, quintic, eased wraps) | `experiments/traj_profiles.py` |
| E3 tension control (dead-band, P, PI; spring model and full sim) | `experiments/controller_model.py`, `experiments/tension_knot_exp.py --controller` |
| E4 robustness, E5 ablations and breaking points | `experiments/run_batch.py` (runs `tension_knot_exp.py` with variations) |
| E6 base placement (manipulability grid) | `experiments/base_placement.py`, `experiments/verify_bases.py` |
| Figures, tables, statistics (Wilson intervals, paired tests) | `experiments/make_figures.py` |

`experiments/tension_knot_exp.py` is `tension_knot.py` with extra options (`--controller`,
`--profile`, `--ease-wraps`, `--orbit-radius`, `--wrap-shift`, `--thread-noise`, `--seed`,
`--no-filter`); with default options it gives exactly the same result (`experiments/setup.sh` checks this).

## Run

Ubuntu 24.04, ROS 2 Jazzy, conda env `harshit_env` (Python 3.12, PyElastica 1.0, numba, numpy,
matplotlib; scipy for the significance tests); ffmpeg for MP4.

The knot (about 20 min; holds at 12 N):
```bash
conda activate harshit_env
python3 tension_knot.py --t-cinch 15
python3 knot_check.py tension_knot_stages.npy --first-stage 0 --strands
python3 animate_knot.py tension_knot_frames.npz --out tension_knot.mp4
python3 test_wound_tissue.py
```

The two arms in RViz:
```bash
conda activate harshit_env
source /opt/ros/jazzy/setup.bash
PKG=$(find ~ -type d -path "*/install/ur7e_grasper_description" 2>/dev/null | head -1); source "$(dirname "$PKG")/setup.bash"
cd knot_ros
python3 dual_arm.py                                   # builds dual_ur7e_grasper.urdf (--no-meshes for simple shapes)
python3 plan_motion.py ../tension_knot_frames.npz     # IK on every frame + 7 checks -> knot_motion.npz
ros2 launch ./knot_replay.launch.py                   # [speed:=0.5] [thread_width:=0.003]
ros2 launch ./knot_live.launch.py                     # live, about 55x slower than real time
```

The experiments (batch 1 about 2-2.5 h, batch 2 about 75 min on a 16-core EPYC):
```bash
conda activate harshit_env
cd experiments
bash setup.sh
nohup bash run_all.sh > run_all.log 2>&1 &
JOBS=14 nohup bash run_all2.sh > run_all2.log 2>&1 &   # after batch 1 has finished
```
Details in [`experiments/README.md`](experiments/README.md).

## Main results 

- 52 / 52 physics runs tied and held the knot across 13 configurations (tested range only).
- Baseline: strand A 12.10 N, tail 12.03 N, arms never closer than 4.5 cm, winding −2.37.
- E1: Newton and damped least squares about 1 iteration / 1 ms per frame; Jacobian transpose 24-43× slower.
- E2: eased wraps cut arm A's peak acceleration 29× and jerk 51×.
- E3: P-control error × k_p constant within 3.5 % (error 17.8 → 8.7 → 4.3 %); PI k_p = k_i = 0.02 holds 12 N within 0.7 %.
- E6: better bases raise the smallest |det J| along the knot 3× (0.023 → 0.068).

