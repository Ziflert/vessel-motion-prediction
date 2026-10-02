#!/usr/bin/env bash
# A3.1 stage 2: repeat the matched learning curve at H=10 and H=30 on the
# common transas_core target panel (12k record), same protocol as stage 1.
# K={2,4,8}, seeds={42,43,44} -> 18 runs; H=20 reuses stage-1 runs.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DATA=data/raw/your_data.csv
LOG=results/a31_learning_curve/batch_12k_core_h1030.log
mkdir -p results/a31_learning_curve

for H in 10 30; do
  for K in 2 4 8; do
    for SEED in 42 43 44; do
      echo "=== START A3.1-S2 12k core H=$H K=$K SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
      "$PY" -X utf8 run_training.py \
        --data-path "$DATA" \
        --profile transas_core \
        --prediction-horizon "$H" \
        --train-segments "$K" \
        --subset-seed "$SEED" \
        --seed "$SEED" \
        --notes "A3.1 12k-core H=$H K=$K seed=$SEED" >> "$LOG" 2>&1
      RC=$?
      echo "=== DONE A3.1-S2 12k core H=$H K=$K SEED=$SEED rc=$RC $(date +%H:%M:%S) ===" >> "$LOG"
      if [ "$RC" -ne 0 ]; then
        echo "A3.1-S2 failed: H=$H K=$K seed=$SEED rc=$RC" >&2
        exit "$RC"
      fi
    done
  done
done
echo "=== A3.1-S2 12k CORE H10/H30 BATCH FINISHED $(date) ===" >> "$LOG"
