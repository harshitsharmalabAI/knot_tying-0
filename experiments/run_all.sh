#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
JOBS=${JOBS:-14}
stamp() { echo; echo "== $(date '+%H:%M') $*"; }
stamp "phase 1: baseline knot (about 20 min)";                python3 run_batch.py --phase 1
stamp "E3a: controller on the spring model (seconds)";        python3 controller_model.py
stamp "E1: IK methods (a few minutes)";                       python3 ik_compare.py
stamp "E2: time-scaling profiles (a few minutes)";            python3 traj_profiles.py
stamp "E6: base placement grid (5-15 min)";                   python3 base_placement.py --jobs "$JOBS"
stamp "phase 2: 34 physics runs, $JOBS at a time (1-1.5 h)";  python3 run_batch.py --phase 2 --jobs "$JOBS"
stamp "figures, tables and results_summary.zip";              python3 make_figures.py
stamp "all done"
