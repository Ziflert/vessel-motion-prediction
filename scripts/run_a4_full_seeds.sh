#!/usr/bin/env bash
# A4 step 1: E1T FULL сиды. Текущий датасет (CYCLE_TAG пуст): дозаполняются
# сиды 43/44 (seed 42 существует). Новый датасет (CYCLE_TAG=v2...): все 3 сида.
set -u
cd "$(dirname "$0")/.."
source scripts/cycle_config.sh
PY=.venv/Scripts/python.exe
OUTD="results/a31_learning_cycle${CYCLE_TAG:+_$CYCLE_TAG}"
LOG=results/real_e1prime/batch_full_seeds${CYCLE_TAG:+_$CYCLE_TAG}.log
mkdir -p "$OUTD" results/real_e1prime
SEEDS_FULL="42 43 44"
[ -z "$CYCLE_TAG" ] && SEEDS_FULL="43 44"
for SEED in $SEEDS_FULL; do
  echo "=== RUN E1TFULL SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
  "$PY" -X utf8 run_training.py \
    --data-path "$DS_71K" \
    --profile transas_core \
    --seed "$SEED" \
    --notes "E1T transas-real FULL${CYCLE_TAG:+ $CYCLE_TAG} seed=$SEED" >> "$LOG" 2>&1
  echo "=== DONE E1TFULL SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
done
echo "=== E1TFULL SEEDS BATCH FINISHED $(date) ===" >> "$LOG"
