#!/usr/bin/env bash
# =============================================================
# МАСТЕР-ЦИКЛ: полный исследовательский цикл для пары датасетов.
# Использование при НОВОМ датасете:
#   1. отредактировать scripts/cycle_config.sh (пути DS_12K/DS_71K, CYCLE_TAG=v2)
#   2. bash scripts/run_full_cycle.sh            # весь цикл
#      bash scripts/run_full_cycle.sh combined   # только один этап
# Этапы (skip, если results/.../<tag>/<stage>.done существует):
#   small12   — 12k learning curve H={10,30} (18 прогонов)
#   big71     — 71k learning curve H={10,30} (24 прогона)
#   combined  — C12/C71 реал+реал, H=20 (42 прогона)
#   fullseeds — E1T FULL сиды 43/44 (2 прогона; при новом датасете — все 3)
#   a4        — CLI-playback 3 сидов FULL + сводка
#   reports   — все сводки (быстро, только чтение манифестов)
# Результаты: results/a31_learning_cycle[_<tag>]/ + results/online/.
# =============================================================
set -u
cd "$(dirname "$0")/.."
source scripts/cycle_config.sh
STAGE="${1:-all}"
OUTD="results/a31_learning_cycle${CYCLE_TAG:+_$CYCLE_TAG}"
mkdir -p "$OUTD"

run_stage() {  # run_stage <name> <marker> <command...>
  local name="$1" marker="$2"; shift 2
  if [ -f "$OUTD/$marker" ]; then echo "[skip] $name ($marker exists)"; return 0; fi
  echo "[run ] $name"
  "$@" || { echo "[FAIL] $name — см. лог этапа"; return 1; }
  touch "$OUTD/$marker"
}

# --- этапы обучения (только обучающие; полные батчи см. в самих скриптах) ---
if [ "$STAGE" = "all" ] || [ "$STAGE" = "small12" ]; then
  run_stage small12 small12.done bash scripts/run_a31_stage2_horizons.sh
fi
if [ "$STAGE" = "all" ] || [ "$STAGE" = "big71" ]; then
  run_stage big71 big71.done bash scripts/run_a31_stage3_71k_horizons.sh
fi
if [ "$STAGE" = "all" ] || [ "$STAGE" = "combined" ]; then
  run_stage combined combined.done bash scripts/run_cycle_combined.sh
fi
if [ "$STAGE" = "all" ] || [ "$STAGE" = "fullseeds" ]; then
  run_stage fullseeds fullseeds.done bash scripts/run_a4_full_seeds.sh
fi

# --- A4 playback (3 FULL-сида, окно из cycle_config) ---
if [ "$STAGE" = "all" ] || [ "$STAGE" = "a4" ]; then
  echo "[run ] a4 playback"
  .venv/Scripts/python.exe -X utf8 -c "
import registry as reg
runs = [m['run_id'] for it in reg.list_runs()
        if (m := it.get('manifest')) and (m.get('hypothesis') or '').startswith('E1T transas-real FULL')]
print('\n'.join(runs))" > "$OUTD/full_runs.txt"
  while read -r RUN; do
    echo "  playback $RUN"
    .venv/Scripts/python.exe -X utf8 -m online.playback \
      --model "$RUN" --csv "$DS_71K" \
      --start-row "$A4_START_ROW" --limit "$A4_LIMIT" --speed max \
      > "$OUTD/playback_${RUN}.log" 2>&1
  done < "$OUTD/full_runs.txt"
  touch "$OUTD/a4.done"
fi

# --- отчёты (только чтение манифестов, всегда перезапуск) ---
echo "[run ] reports"
PYUTF8=.venv/Scripts/python.exe
"$PYUTF8" -X utf8 scripts/a31_stage2_horizon_report.py > "$OUTD/report_stage23.txt" 2>&1
"$PYUTF8" -X utf8 scripts/a31_stage4_powerlaw.py     > "$OUTD/report_stage4.txt"  2>&1
echo "[done] цикл завершён; отчёты: $OUTD/report_stage23.txt, report_stage4.txt"
