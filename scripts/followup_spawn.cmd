#!/usr/bin/env bash
# Отложенный («будильник») запуск ИИ после окончания батча E3-контролей.
# Схема: Windows Task Scheduler (pi-e3-followup) -> этот скрипт -> pi --print
# с промптом из scripts/followup_prompt.txt',
# результат -> results/e3_controls/followup.log

# глобальный cmd: путь на диске текущего проекта (bat должен лежать в корне, а не в папке "scripts")
# скрипт считает, что он запущен из корня, поэтому поднимаемся при необходимости на 1 каталог вверх
cd "$(dirname "$0")/.." || exit 1

# мягкий «сон» основного/фонового процесса, если юзер вдруг запустил вручную
if command -v pi >/dev/null 2>&1; then
  BIN="$(command -v pi)"
else
  BIN="$HOME/.pi/agent/bin/pi"
fi

LOG=results/e3_controls/followup.log

{
  echo "=== pi followup started $(date '+%F %T') ==="
  # еслитокены не расходуются, пока процесс просто спит: батч по плану закончится в районе 03:45-04:15,
  # а задача назначена на 04:20; запас нужен потому, чтоscheduled task ничего не знает о реальном прогрессе
  echo "=== pi followup finished rc=%errorlevel% $(date '+%F %T') ==="
} > "$LOG" 2>&1
