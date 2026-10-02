#!/usr/bin/env bash
# E2'/E3' — проверка аугментации режимной синтетикой на реальных записях Transas.
# E2T: train ТОЛЬКО синтетика (переносимость генератора).
# E3T: реал K чанков + 30k строк синтетики (extra-train, скалеры по реалу).
# База сравнения — E1T (реал-only). Запуск: bash scripts/run_e23prime_batch.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=results/real_e1prime/batch.log
SYN=data/converted/synthetic_regime.csv
mkdir -p results/real_e1prime

# --- E2T: только синтетика ---
for SEED in 42 43 44; do
  echo "=== RUN E2T SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
  "$PY" -X utf8 run_training.py \
    --data-path data/converted/real_w5w6w7_merged.csv \
    --profile transas_core \
    --train-synthetic-csv "$SYN" \
    --seed "$SEED" \
    --notes "E2T transas syn-only seed=$SEED" >> "$LOG" 2>&1
  echo "=== DONE E2T SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
done

# --- E3T: реал (K чанков) + 30000 синтетики ---
for K in 2 8; do
  for SEED in 42 43 44; do
    echo "=== RUN E3T K=$K SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
    "$PY" -X utf8 run_training.py \
      --data-path data/converted/real_w5w6w7_merged.csv \
      --profile transas_core \
      --train-segments "$K" --subset-seed "$SEED" \
      --extra-train-csv "$SYN" --extra-train-rows 30000 \
      --seed "$SEED" \
      --notes "E3T transas K=$K +syn30k seed=$SEED" >> "$LOG" 2>&1
    echo "=== DONE E3T K=$K SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
  done
done
echo "=== BATCH E23 FINISHED $(date) ===" >> "$LOG"
