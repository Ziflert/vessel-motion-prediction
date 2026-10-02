#!/usr/bin/env bash
# Этап C (combined): реал+реал, два режима в одном train, фиксированный test.
#   C12: train = 12k(K сегментов) + 71k(extra), test = test 12k
#   C71: train = 71k(K сегментов) + 12k(extra), test = test 71k
#   C12FULL/C71FULL: полные объёмы обоих файлов, 3 сида.
# H=20 (сравнение с базами E1T / A3.1 этап 1 на тех же test-панелях).
# Вопрос: помогает или мешает чужой режим волнения на «своём» test-режиме?
set -u
cd "$(dirname "$0")/.."
source scripts/cycle_config.sh
PY=.venv/Scripts/python.exe
LOG=results/a31_learning_cycle${CYCLE_TAG:+_$CYCLE_TAG}/batch_combined.log
mkdir -p "results/a31_learning_cycle${CYCLE_TAG:+_$CYCLE_TAG}"
SUF="${CYCLE_TAG:+ $CYCLE_TAG}"

for SEED in $SEEDS; do
  for K in $KS_SMALL; do
    echo "=== START C12 K=$K SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
    "$PY" -X utf8 run_training.py \
      --data-path "$DS_12K" --extra-train-csv "$DS_71K" \
      --profile transas_core \
      --train-segments "$K" --subset-seed "$SEED" --seed "$SEED" \
      --notes "A3.1${SUF} C12 12k+71k K=$K seed=$SEED" >> "$LOG" 2>&1
    echo "=== DONE C12 K=$K SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG" && [ $? -eq 0 ] || true
  done
  echo "=== START C12FULL SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
  "$PY" -X utf8 run_training.py \
    --data-path "$DS_12K" --extra-train-csv "$DS_71K" \
    --profile transas_core \
    --seed "$SEED" \
    --notes "A3.1${SUF} C12FULL 12k+71k seed=$SEED" >> "$LOG" 2>&1
  echo "=== DONE C12FULL SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
done

for SEED in $SEEDS; do
  for K in $KS_BIG; do
    echo "=== START C71 K=$K SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
    "$PY" -X utf8 run_training.py \
      --data-path "$DS_71K" --extra-train-csv "$DS_12K" \
      --profile transas_core \
      --train-segments "$K" --subset-seed "$SEED" --seed "$SEED" \
      --notes "A3.1${SUF} C71 71k+12k K=$K seed=$SEED" >> "$LOG" 2>&1
    echo "=== DONE C71 K=$K SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
  done
  echo "=== START C71FULL SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
  "$PY" -X utf8 run_training.py \
    --data-path "$DS_71K" --extra-train-csv "$DS_12K" \
    --profile transas_core \
    --seed "$SEED" \
    --notes "A3.1${SUF} C71FULL 71k+12k seed=$SEED" >> "$LOG" 2>&1
  echo "=== DONE C71FULL SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
done
echo "=== COMBINED BATCH FINISHED $(date) ===" >> "$LOG"
