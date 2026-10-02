#!/usr/bin/env bash
# Сторож батча E3-контролей: ждёт метки BATCH E3-CONTROLS FINISHED в
# results/e3_controls/batch.log (или таймаут), затем запускает сводку
# scripts/summarize_e3_controls.py -> results/e3_controls/report.txt
# Запуск (detached): powershell Start-Process bash -ArgumentList 'scripts/watch_e3controls.sh'
set -u
cd "$(dirname "$0")/.."
LOG=results/e3_controls/batch.log
WLOG=results/e3_controls/watcher.log
TIMEOUT_S=21600   # 6 ч страховочный таймаут
POLL_S=120

echo "=== watcher started $(date '+%F %T') ===" >> "$WLOG"

start=$(date +%s)
until grep -q "BATCH E3-CONTROLS FINISHED" "$LOG" 2>/dev/null; do
  now=$(date +%s)
  if [ $((now - start)) -ge "$TIMEOUT_S" ]; then
    echo "=== watcher TIMEOUT $(date '+%F %T'): метка не найдена за ${TIMEOUT_S}s ===" >> "$WLOG"
    exit 1
  fi
  sleep "$POLL_S"
done

echo "=== batch finished detected $(date '+%F %T') ===" >> "$WLOG"
sleep 10  # дать последнему manifest.json дописаться

if .venv/Scripts/python.exe -X utf8 scripts/summarize_e3_controls.py >> "$WLOG" 2>&1; then
  echo "=== summary OK $(date '+%F %T') -> results/e3_controls/report.txt ===" >> "$WLOG"
else
  echo "=== summary FAILED rc=$? $(date '+%F %T') ===" >> "$WLOG"
  exit 2
fi
