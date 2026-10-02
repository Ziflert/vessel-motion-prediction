#!/usr/bin/env bash
# E1' learning curve на реальных записях Transas (w-5/6/7 merged, 71000 строк).
# Профиль transas_core (без вырожденного Pitch). Train = K случайных чанков
# (subset-seed = training seed), val/test фиксированы на полном датасете.
# Сиды {42,43,44} — mean±std. Запуск: bash scripts/run_e1prime_batch.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=results/real_e1prime/batch.log
mkdir -p results/real_e1prime

for K in 2 4 8 16; do
  for SEED in 42 43 44; do
    echo "=== RUN K=$K SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
    "$PY" -X utf8 run_training.py \
      --data-path data/converted/real_w5w6w7_merged.csv \
      --profile transas_core \
      --train-segments "$K" --subset-seed "$SEED" \
      --seed "$SEED" \
      --notes "E1T transas-real K=$K seed=$SEED" >> "$LOG" 2>&1
    echo "=== DONE K=$K SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
  done
done
echo "=== BATCH FINISHED $(date) ===" >> "$LOG"
