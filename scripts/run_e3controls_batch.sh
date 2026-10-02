#!/usr/bin/env bash
# Контрольные серии E-цикла (после E1T/E2T/E3T):
#   E3TBAD  — негативный контроль: аугментация синтетикой, откалиброванной на
#             ЧУЖОМ домене (your_data). Ожидание: хуже E1T K=2 (1.110).
#   E3TFULL — полный реал (49700) + вся синтетика (70997): вредит ли аугментация
#             при большом реале. Ожидание: паритет с E1T FULL (0.488).
#   E3TDOSE — дозовая кривая при K=2: дозы 5k/10k/70k (30k уже есть в E3T).
# Запуск: bash scripts/run_e3controls_batch.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
LOG=results/e3_controls/batch.log
BAD=data/converted/synthetic_bad_calib.csv
SYN=data/converted/synthetic_regime.csv
mkdir -p results/e3_controls

# --- stage 0: плохая синтетика (если ещё нет) ---
if [ ! -f "$BAD" ]; then
  echo "=== GEN BAD SYNTH $(date +%H:%M:%S) ===" >> "$LOG"
  "$PY" -X utf8 scripts/make_bad_synthetic.py >> "$LOG" 2>&1
fi

# --- stage 1: негативный контроль ---
for SEED in 42 43 44; do
  echo "=== RUN E3TBAD K=2 SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
  "$PY" -X utf8 run_training.py \
    --data-path data/converted/real_w5w6w7_merged.csv \
    --profile transas_core \
    --train-segments 2 --subset-seed "$SEED" \
    --extra-train-csv "$BAD" --extra-train-rows 30000 \
    --seed "$SEED" \
    --notes "E3TBAD K=2 +bad-syn30k seed=$SEED" >> "$LOG" 2>&1
  echo "=== DONE E3TBAD K=2 SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
done

# --- stage 2: полный реал + вся синтетика ---
for SEED in 42 43 44; do
  echo "=== RUN E3TFULL SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
  "$PY" -X utf8 run_training.py \
    --data-path data/converted/real_w5w6w7_merged.csv \
    --profile transas_core \
    --extra-train-csv "$SYN" \
    --seed "$SEED" \
    --notes "E3TFULL real-full +syn71k seed=$SEED" >> "$LOG" 2>&1
  echo "=== DONE E3TFULL SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
done

# --- stage 3: дозовая кривая K=2 ---
for DOSE in 5000 10000 70000; do
  for SEED in 42 43 44; do
    echo "=== RUN E3TDOSE dose=$DOSE K=2 SEED=$SEED $(date +%H:%M:%S) ===" >> "$LOG"
    "$PY" -X utf8 run_training.py \
      --data-path data/converted/real_w5w6w7_merged.csv \
      --profile transas_core \
      --train-segments 2 --subset-seed "$SEED" \
      --extra-train-csv "$SYN" --extra-train-rows "$DOSE" \
      --seed "$SEED" \
      --notes "E3TDOSE K=2 dose=$DOSE seed=$SEED" >> "$LOG" 2>&1
    echo "=== DONE E3TDOSE dose=$DOSE SEED=$SEED rc=$? $(date +%H:%M:%S) ===" >> "$LOG"
  done
done
echo "=== BATCH E3-CONTROLS FINISHED $(date) ===" >> "$LOG"
