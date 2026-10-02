#!/bin/bash
cd "D:/Temprary/LSTM-Vessel_Motion_Prediction"
LOG=results/horizons_f4/f5v2_batch.log
echo "=== watcher v2 started $(date) ==="
for i in $(seq 1 300); do
  if grep -q "F5V2 BATCH FINISHED" "$LOG" 2>/dev/null; then
    echo "=== batch finished detected $(date) ==="
    .venv/Scripts/python.exe -X utf8 scripts/horizon_report.py --prefix F5 \
      && echo "=== summary OK $(date) ===" || echo "=== summary FAILED $(date) ==="
    exit 0
  fi
  sleep 60
done
echo "=== watcher timeout (5h) $(date) ==="
