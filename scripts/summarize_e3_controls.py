# -*- coding: utf-8 -*-
"""Сводка серий E-цикла (E1T/E2T/E3T/E3TBAD/E3TFULL/E3TDOSE) по манифестам.

Читает manifests из models_archive/, группирует по серии (из поля hypothesis),
считает mean±std (physical overall MAE/R2, per-target MAE, skill vs persistence),
пишет results/e3_controls/report.txt.

Запуск: .venv/Scripts/python.exe -X utf8 scripts/summarize_e3_controls.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE = os.path.join(ROOT, "models_archive")
OUT = os.path.join(ROOT, "results", "e3_controls", "report.txt")

TARGETS = [
    "Roll(градусы)",
    "Vertical(Метр)",
    "Velocity.Rolling(°/мин)",
    "Velocity.Vertical(узлы)",
]
TARGET_SHORT = {
    "Roll(градусы)": "Roll, град",
    "Vertical(Метр)": "Vert, м",
    "Velocity.Rolling(°/мин)": "Vel.Roll, °/мин",
    "Velocity.Vertical(узлы)": "Vel.Vert, уз",
}

# порядок серий в отчёте (что не найдено — выводится в конце с пометкой)
SERIES_ORDER = [
    "E1T K=2",
    "E1T K=4",
    "E1T K=8",
    "E1T K=16",
    "E1T FULL",
    "E2T syn-only",
    "E3T K=2 +syn30k",
    "E3T K=8 +syn30k",
    "E3TBAD K=2 +bad-syn30k",
    "E3TFULL real-full +syn71k",
    "E3TDOSE K=2 dose=5000",
    "E3TDOSE K=2 dose=10000",
    "E3TDOSE K=2 dose=70000",
]


def series_of(hypothesis: str):
    """Извлечь имя серии из поля hypothesis."""
    h = hypothesis.strip()
    if h.startswith("smoke"):
        return None
    m = re.match(r"E1T\s+transas-real\s+K=(\d+)\s+seed", h)
    if m:
        return f"E1T K={m.group(1)}"
    if h.startswith("E1T transas-real FULL"):
        return "E1T FULL"
    if h.startswith("E2T"):
        return "E2T syn-only"
    m = re.match(r"E3T\s+transas\s+K=(\d+)\s+\+syn30k", h)
    if m:
        return f"E3T K={m.group(1)} +syn30k"
    if h.startswith("E3TBAD"):
        return h.rsplit(" seed", 1)[0].strip()
    if h.startswith("E3TFULL"):
        return h.rsplit(" seed", 1)[0].strip()
    m = re.match(r"E3TDOSE\s+(K=\d+\s+dose=\d+)", h)
    if m:
        return f"E3TDOSE {m.group(1)}"
    # fallback: убрать seed
    return h.rsplit(" seed", 1)[0].strip()


def load_runs():
    runs = []
    for path in sorted(glob.glob(os.path.join(ARCHIVE, "*", "manifest.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! skip {path}: {e}")
            continue
        s = series_of(m.get("hypothesis", ""))
        if s is None:
            continue
        runs.append({"series": s, "run_id": m["run_id"], "res": m.get("results", {})})
    return runs


def agg(values):
    a = np.asarray(values, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def fmt(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def main():
    runs = load_runs()
    by_series = defaultdict(list)
    for r in runs:
        by_series[r["series"]].append(r)

    lines = []
    lines.append("=" * 100)
    lines.append("СВОДКА СЕРИЙ E-ЦИКЛА (по manifest.json из models_archive)")
    lines.append("=" * 100)

    known = set(SERIES_ORDER)
    unknown = [s for s in by_series if s not in known]
    if unknown:
        lines.append(f"(серии вне E-цикла пропущены: {', '.join(sorted(unknown))})")
    all_series = [s for s in SERIES_ORDER if s in by_series]

    # --- Таблица 1: overall ---
    lines.append("")
    lines.append("1) Physical overall (test): MAE / RMSE / R2, mean±std по seed'ам")
    lines.append("-" * 100)
    hdr = f"{'Серия':<28}{'n':>3} {'MAE':>16} {'RMSE':>16} {'R2':>16} {'Skill pers':>10}"
    lines.append(hdr)
    for s in all_series:
        rr = by_series[s]
        valid = [r for r in rr if r["res"].get("physical", {}).get("overall")]
        if not valid:
            lines.append(f"{s:<28}{len(rr):>3}  (нет physical overall)")
            continue
        mae_m, mae_s = agg([r["res"]["physical"]["overall"]["mae"] for r in valid])
        rm_m, rm_s = agg([r["res"]["physical"]["overall"]["rmse"] for r in valid])
        r2_m, r2_s = agg([r["res"]["physical"]["overall"]["r2"] for r in valid])
        skills = [
            r["res"]["skill_vs_persistence"]["overall"]
            for r in valid
            if r["res"].get("skill_vs_persistence", {}).get("overall") is not None
        ]
        sk = f"{np.mean(skills):+.3f}" if skills else "—"
        lines.append(
            f"{s:<28}{len(valid):>3} {fmt(mae_m, mae_s):>16} {fmt(rm_m, rm_s):>16} "
            f"{fmt(r2_m, r2_s):>16} {sk:>10}"
        )

    # --- Таблица 2: per-target MAE ---
    lines.append("")
    lines.append("2) Per-target MAE (physical), mean±std")
    lines.append("-" * 100)
    hdr = f"{'Серия':<28}{'n':>3}" + "".join(
        f" {TARGET_SHORT[t]:>18}" for t in TARGETS
    )
    lines.append(hdr)
    for s in all_series:
        rr = by_series[s]
        valid = [
            r
            for r in rr
            if r["res"].get("physical", {}).get("per_target", {}).get(TARGETS[0])
        ]
        if not valid:
            continue
        cells = []
        for t in TARGETS:
            vals = [r["res"]["physical"]["per_target"][t]["mae"] for r in valid]
            cells.append(f"{fmt(*agg(vals)):>18}")
        lines.append(f"{s:<28}{len(valid):>3}" + "".join(cells))

    # --- Таблица 3: per-target skill vs persistence ---
    lines.append("")
    lines.append("3) Per-target skill vs persistence (>0 лучше бейзлайна), mean по seed'ам")
    lines.append("-" * 100)
    hdr = f"{'Серия':<28}{'n':>3}" + "".join(
        f" {TARGET_SHORT[t]:>18}" for t in TARGETS
    )
    lines.append(hdr)
    for s in all_series:
        rr = by_series[s]
        cells = []
        ok = False
        for t in TARGETS:
            vals = [
                r["res"]["skill_vs_persistence"]["per_target"][t]
                for r in rr
                if t in r["res"].get("skill_vs_persistence", {}).get("per_target", {})
            ]
            if vals:
                ok = True
                cells.append(f"{np.mean(vals):>+18.3f}")
            else:
                cells.append(f"{'—':>18}")
        if ok:
            lines.append(f"{s:<28}{len(rr):>3}" + "".join(cells))

    # --- Список прогонов ---
    lines.append("")
    lines.append("4) Прогоны по сериям")
    lines.append("-" * 100)
    for s in all_series:
        lines.append(f"  {s}:")
        for r in sorted(by_series[s], key=lambda x: x["run_id"]):
            res = r["res"]
            mae = res.get("physical", {}).get("overall", {}).get("mae")
            mae_s = f"MAE={mae:.4f}" if mae is not None else "MAE=—"
            lines.append(f"    {r['run_id']}  {mae_s}")

    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
