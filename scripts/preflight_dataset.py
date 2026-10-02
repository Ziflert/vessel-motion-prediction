"""Preflight-чекер нового датасета: проверка ДО обучения (CPU, секунды).

Использование:
    python scripts/preflight_dataset.py <path.csv> [--tag v2] [--save]

Проверки:
  1. Чтение/кодировка/разделитель (utf-16/tab, utf-8/comma — авто).
  2. Схема: обязательные колонки профиля transas_core (features + targets).
  3. Время: шаг 1 Гц, разрывы, дубликаты.
  4. NaN: полные колонки, доля NaN по ключевым каналам.
  5. Физика: диапазоны каналов (Pilot Card / здравый смысл), подозрительные std.
  6. Режимы: std качки, средние внешние условия — сравнение с эталонами 12k/71k.
  7. Ω-покрытие: какие ячейки (волнение × КУ × скорость) закрыты, дыры.

Exit code 0 = можно учить; 1 = есть блокеры; вывод сохраняется в
results/preflight/<имя>_<tag>.txt при --save.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# Эталоны существующих записей (RESEARCH_LOG, Ф0.5/E-цикл)
REFERENCES = {
    "12k (сильное волнение)": {"roll_std": 0.83, "wave_highest": None, "rows": 12499},
    "71k (спокойные режимы)": {"roll_std": 0.22, "wave_highest": None, "rows": 71000},
}

REQUIRED = [
    # targets transas_core
    "Roll(градусы)", "Vertical(Метр)",
    "Velocity.Rolling(°/мин)", "Velocity.Vertical(узлы)",
    # ключевые входы
    "SOG(узлы)", "Course(градусы)",
    "Wave.Highest(метры)", "Wave.direction(градусы)", "Wave.speed(узлы)",
    "Wind.direction(градусы)",
]

# Опциональные (есть не во всех экспортах; отсутствие — предупреждение, не блокер)
OPTIONAL = ["Wind.speed(узлы)", "Pitch(градусы)"]

PHYSICS_RANGES = {  # колонка: (min, max, описание)
    "Roll(градусы)": (-45, 45, "крен; ±45° — аварийный предел"),
    "Pitch(градусы)": (-20, 20, "дифферент"),
    "Vertical(Метр)": (-15, 15, "вертикальные перемещения"),
    "Velocity.Rolling(°/мин)": (-700, 700, "скорость крена; до ±660°/мин в сильном волнении (12k)"),
    "Velocity.Vertical(узлы)": (-30, 30, "вертикальная скорость"),
    "SOG(узлы)": (0, 25, "скорость судна (70k tanker: до ~15 уз)"),
    "Wave.Highest(метры)": (0, 15, "высота волны; >12 м — экстремум"),
    "Wind.speed(узлы)": (0, 70, "ветер; >48 уз — шторм 10 баллов"),
    "ROT(°/мин)": (-120, 120, "угловая скорость поворота"),
}


def load_any(path: Path) -> tuple[pd.DataFrame, str]:
    """Автоопределение кодировки/разделителя (Transas: utf-16/tab)."""
    for enc, sep in [("utf-16", "\t"), ("utf-8", "\t"), ("utf-8", ","), ("cp1251", "\t")]:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc)
            if df.shape[1] >= 5:
                return df, f"encoding={enc}, sep={'TAB' if sep == chr(9) else 'COMMA'}"
        except Exception:
            continue
    raise SystemExit(f"Не удалось прочитать {path} ни в одном формате")


def block(lines: list[str], title: str, ok: bool) -> bool:
    lines.append(f"[{'OK ' if ok else 'FAIL'}] {title}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tag", default="", help="тег версии данных (v2, ...)")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    path = Path(args.path)
    lines = ["=" * 76, f"PREFLIGHT: {path}", "=" * 76]
    failures = 0

    df, fmt = load_any(path)
    lines.append(f"Формат: {fmt}")
    lines.append(f"Размер: {len(df)} строк × {len(df.columns)} колонок")

    # 2. Схема
    missing = [c for c in REQUIRED if c not in df.columns]
    block(lines, f"Схема: обязательные колонки transas_core (не хватает: {missing or 'нет'})",
          not missing)
    if missing:
        failures += 1
    opt_missing = [c for c in OPTIONAL if c not in df.columns]
    if opt_missing:
        lines.append(f"  ⚠ нет опциональных колонок (не блокер): {opt_missing}")

    # 3. Время
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
        dt = t.diff().dropna().dt.total_seconds()
        gaps = int((dt != 1).sum())
        dups = int((dt == 0).sum())
        ok = gaps == 0 and dups == 0 and t.notna().all()
        block(lines, f"Время: 1 Гц без разрывов/дубликатов (разрывы {gaps}, дубли {dups})", ok)
        if gaps:
            big = df.loc[dt[dt != 1].index[:5], "time"].tolist()
            lines.append(f"  первые разрывы около: {big}")
        if gaps + dups > len(df) * 0.02:
            failures += 1
    else:
        block(lines, "Время: колонки 'time' нет — пропуск", True)

    # 4. NaN
    all_na = [c for c in df.columns if df[c].isna().all()]
    bad_na = [c for c in REQUIRED if c in df.columns and df[c].isna().mean() > 0.05]
    block(lines, f"NaN: полные NaN-колонки {all_na or 'нет'}; >5% NaN в обязательных {bad_na or 'нет'}",
          not all_na and not bad_na)
    if all_na or bad_na:
        failures += 1

    # 5. Физика
    lines.append("Физика (диапазоны Pilot Card / здравый смысл):")
    for col, (lo, hi, desc) in PHYSICS_RANGES.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        out = int(((s < lo) | (s > hi)).sum())
        status = "OK" if out == 0 else f"⚠ {out} значений вне [{lo},{hi}]"
        lines.append(f"  {col:<28} [{s.min():.2f}; {s.max():.2f}] {status} ({desc})")
        if out > len(s) * 0.01:
            failures += 1

    # 6. Режимы — сравнение с эталонами
    lines.append("Режим (стратификационный фактор, не дефект):")
    roll_std = df["Roll(градусы)"].std(ddof=1) if "Roll(градусы)" in df else float("nan")
    lines.append(f"  Roll std = {roll_std:.3f}° "
                 f"(референсы: 12k 0.83° — сильное; 71k 0.22° — спокойное)")
    for col in ["Wave.Highest(метры)", "Wind.speed(узлы)", "SOG(узлы)"]:
        if col in df.columns:
            lines.append(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std(ddof=1):.2f}")

    # 7. Ω-покрытие: волнение × КУ × скорость (грубая сетка)
    lines.append("Ω-покрытие (волнение×КУ×скорость, грубая сетка):")
    if {"Wave.Highest(метры)", "Course(градусы)", "SOG(узлы)"} <= set(df.columns):
        w = pd.cut(df["Wave.Highest(метры)"], bins=[-0.01, 0.5, 1.5, 3, 12],
                   labels=["<0.5", "0.5-1.5", "1.5-3", ">3"])
        c = pd.cut(df["Course(градусы)"] % 360, bins=range(0, 361, 90),
                   labels=["0-90", "90-180", "180-270", "270-360"], include_lowest=True)
        s = pd.cut(df["SOG(узлы)"], bins=[-0.01, 4, 8, 12, 25],
                   labels=["<4", "4-8", "8-12", ">12"])
        grid = df.assign(w=w, c=c, s=s).groupby(["w", "c", "s"], observed=True).size()
        filled = grid[grid > 300]  # ≥300 строк ≈ 5 мин записи — обучаемый минимум
        lines.append(f"  ячеек всего: {len(grid)}; закрытых (>300 строк): {len(filled)}")
        lines.append(f"  закрытые: {dict(filled)}")
        lines.append(f"  дыры (0 строк): {len(grid) - int((grid > 0).sum())} из 64")
    else:
        lines.append("  — пропущено (нет колонок Wave/Course/SOG)")

    verdict = "ГОТОВ К ОБУЧЕНИЮ" if failures == 0 else f"{failures} БЛОКЕРОВ — исправить до прогона"
    lines += ["=" * 76, f"ВЕРДИКТ: {verdict}", "=" * 76]
    report = "\n".join(lines)
    print(report)
    if args.save:
        out = ROOT / "results" / "preflight" / f"{path.stem}{args.tag and '_' + args.tag}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Сохранено: {out}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
