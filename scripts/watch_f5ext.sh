#!/bin/bash
# Сторож Ф5-ext: по метке FINISHED запускает отчёт по горизонтам F5
cd "D:/Temprary/LSTM-Vessel_Motion_Prediction"
LOG=results/horizons_f4/f5_ext_batch.log
echo "=== watcher started $(date) ==="
for i in $(seq 1 240); do
  if grep -q "F5-EXT BATCH FINISHED" "$LOG" 2>/dev/null; then
    echo "=== batch finished detected $(date) ==="
    .venv/Scripts/python.exe -X utf8 scripts/horizon_report.py --prefix F5 \
      && echo "=== summary OK $(date) ===" \
      || echo "=== summary FAILED rc=$? $(date) ==="
    exit 0
  fi
  sleep 60
done
echo "=== watcher timeout (4h) $(date) ==="
