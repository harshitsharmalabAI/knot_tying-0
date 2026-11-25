#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
JOBS=${JOBS:-14}
stamp() { echo; echo "== $(date '+%H:%M') $*"; }
stamp "best bases through the 7-check planner (seconds)";   python3 verify_bases.py
stamp "phase 3: 17 physics runs, $JOBS at a time (70-80 min)"; python3 run_batch.py --phase 3 --jobs "$JOBS"
stamp "figures, tables and results_summary.zip";             python3 make_figures.py
stamp "all done"
