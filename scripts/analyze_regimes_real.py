"""EDA по режимам объединённой реальной записи Transas (w-5/6/7, 2026-10-01).

Сегментация: ступени Wave.Highest (дизайн упражнения — дискретные уровни) +
внутри уровня — по файлу-источнику (записи склеены). На выходе: per-regime
статистика качки/управления/среды + карта покрытия Ω-lite + периоды качки (Welch).

Использование:
  .venv\\Scripts\\python.exe -X utf8 scripts/analyze_regimes_real.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_real_data import welch_psd

MERGED = PROJECT_ROOT / "data/converted/real_w5w6w7_merged.csv"
OUT_DIR = PROJECT_ROOT / "results/real_w5w7_eda"

MOTION = ["Roll(градусы)", "Pitch(градусы)", "Vertical(Метр)",
          "Velocity.Rolling(°/мин)", "Velocity.Vertical(узлы)"]
ENV = ["Wave.Highest(метры)", "Wind.speed(узлы)", "Swell(метры)",
       "Current.speed(узлы)"]
CTRL = ["Rudder Order(градусы)", "Rudder State(градусы)", "RPM(Обороты в минуту)"]


def peak_period(x, fs=1.0):
    f, p = welch_psd(x, fs=fs)  # welch_psd возвращает (freqs, psd)
    f, p = np.asarray(f), np.asarray(p)
    band = (f >= 1 / 60) & (f <= 0.5)
    if not band.any() or p[band].sum() == 0:
        return float("nan")
    return float(1.0 / f[band][np.argmax(p[band])])


def load_parts():
    from scripts.convert_transas_csv import read_transas
    parts = []
    for i in (5, 6, 7):
        p = PROJECT_ROOT / f"data/converted/2026-10-01_w-{i}_converted.csv"
        df = pd.read_csv(p, sep="\t", encoding="utf-16")
        df["src"] = f"w-{i}"
        # Wind.speed есть только в сыром экспорте (в модельную схему не входит) —
        # подтягиваем по позиции для аннотации режимов
        raw_p = PROJECT_ROOT / f"data/raw/2026-10-01_w-{i}.csv"
        if raw_p.exists():
            raw, _ = read_transas(raw_p)
            wind_col = [c for c in raw.columns if "скорость истинного ветра" in c.lower()]
            if wind_col:
                df["Wind.speed(узлы)"] = pd.to_numeric(
                    raw[wind_col[0]].iloc[:len(df)], errors="coerce").values
        parts.append(df)
    return parts


def main():
    parts = load_parts()
    df = pd.concat(parts, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    say = lines.append

    say("=" * 78)
    say("EDA: объединённая реальная запись Transas w-5/6/7 (2026-10-01)")
    say(f"всего строк: {len(df)} ({len(df)/3600:.2f} ч), колонок: {len(df.columns)}")
    say("=" * 78)

    # --- 1. Режимы = (файл, уровень Wave.Highest) ---
    df["regime"] = df["src"] + " / W.H=" + df["Wave.Highest(метры)"].map(lambda v: f"{v:.1f}")
    regimes = [g for _, g in df.groupby("regime", sort=False)]
    # слить подряд идущие одинаковые (после конкатенации группы могут чередоваться
    # только на стыках файлов — groupby(sort=False) уже сгруппировал по имени,
    # но подряд идущие одинаковые уровни внутри файла важны):
    run_id = (df["regime"] != df["regime"].shift()).cumsum()
    df["run"] = run_id

    say("\n--- РЕЖИМЫ (непрерывные участки: файл × уровень Wave.Highest) ---")
    rows = []
    for rid, g in df.groupby("run"):
        n = len(g)
        if n < 120:  # короче 2 мин — не режим, пограничный шум
            continue
        roll, vert = g["Roll(градусы)"], g["Vertical(Метр)"]
        pitch = g["Pitch(градусы)"]
        row = {
            "режим": g["regime"].iloc[0],
            "строк": n,
            "мин": round(n / 60, 1),
            "Roll std": round(roll.std(), 3),
            "Roll max| |": round(roll.abs().max(), 2),
            "T_roll с": round(peak_period(roll.values), 1),
            "Vert std": round(vert.std(), 3),
            "T_vert с": round(peak_period(vert.values), 1),
            "Pitch std": round(pitch.std(), 4),
            "Rudder std": round(g["Rudder State(градусы)"].std(), 2),
            "RPM mean": round(g["RPM(Обороты в минуту)"].mean(), 1),
            "SOG mean": round(g["SOG(узлы)"].mean(), 2),
            "Wind max": round(g["Wind.speed(узлы)"].max(), 1),
            "Course med": round(g["Course(градусы)"].median(), 0),
        }
        rows.append(row)
    reg = pd.DataFrame(rows)
    say(reg.to_string(index=False))

    # --- 2. Глобальные периоды и сравнение с your_data ---
    say("\n--- Периоды качки (Welch, медиана по файлам) vs эталон your_data ---")
    ref = pd.read_csv(PROJECT_ROOT / "data/raw/your_data.csv",
                      sep="\t", encoding="utf-16")
    for col in ["Roll(градусы)", "Pitch(градусы)", "Vertical(Метр)"]:
        new_p = np.nanmedian([peak_period(p[col].values) for p in parts])
        ref_p = peak_period(ref[col].values)
        say(f"  {col:22s}: w5-7={new_p:5.1f} с   your_data={ref_p:5.1f} с")

    # --- 3. Замороженные/вырожденные каналы ---
    say("\n--- Каналы-константы (вырожденные) в merged ---")
    for c in df.columns:
        if c in ("src", "regime", "run", "Wind.speed(узлы)"):
            continue
        s = df[c]
        if s.nunique() <= 2:
            say(f"  {c}: уникальных={s.nunique()} значения={sorted(s.unique())[:3]}")

    # --- 4. Покрытие Ω-lite: (ветер) × (волна) ---
    say("\n--- Покрытие Ω-lite (строк на пару: уровень ветра × уровень волны) ---")
    cov = df.groupby([df["Wind.speed(узлы)"].round(0),
                      df["Wave.Highest(метры)"].round(1)]).size()
    say(cov.to_string())

    # --- 5. Сигнальность целей (для выбора профиля E1) ---
    say("\n--- Сигнальность потенциальных целей (std по merged) ---")
    for c in MOTION + ["ROT(°/мин)", "SOG(узлы)"]:
        if c in df.columns:
            say(f"  {c:28s}: std={df[c].std():.4f}  max|{df[c].abs().max():.3f}|")

    report = "\n".join(lines) + "\n"
    (OUT_DIR / "report.txt").write_text(report, encoding="utf-8")
    reg.to_csv(OUT_DIR / "regimes.csv", index=False, encoding="utf-8")
    print(report)
    print(f"\nСохранено: {OUT_DIR / 'report.txt'} и regimes.csv")


if __name__ == "__main__":
    main()
