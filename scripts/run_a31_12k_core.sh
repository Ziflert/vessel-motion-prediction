#!/usr/bin/env bash
# A3.1: matched learning-curve runs on your_data_minimal using transas_core.
# Same target panel/protocol as E1T 71k; H=20, K={2,4,8}, seeds={42,43,44}.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
DATA=data/raw/your_data.csv
LOG=results/a31_learning_curve/batch_12k_core.log
mkdir -p results/a31_learning_curve

for K in 2 4 8; do
  for SEED in 42 43 44; do
    echo "=== START A3.1 12k transas_core K=$K SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
    "$PY" -X utf8 run_training.py \
      --data-path "$DATA" \
      --profile transas_core \
      --train-segments "$K" \
      --subset-seed "$SEED" \
      --seed "$SEED" \
      --notes "A3.1 12k-core K=$K seed=$SEED" >> "$LOG" 2>&1
    RC=$?
    echo "=== DONE A3.1 12k transas_core K=$K SEED=$SEED rc=$RC $(date +%H:%M:%S) ===" >> "$LOG"
    if [ "$RC" -ne 0 ]; then
      echo "A3.1 failed: K=$K seed=$SEED rc=$RC" >&2
      exit "$RC"
    fi
  done
done
echo "=== A3.1 12k CORE BATCH FINISHED $(date) ===" >> "$LOG"
