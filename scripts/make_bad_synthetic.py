"""Негативный контроль (IDEAS B4 / статья): синтетика, откалиброванная на ЧУЖОМ
домене — ваш_data.csv (другой режим: roll std 3.3°, Wave.dir=180, живой Pitch).
По инварианту L-6 такая синтетика в аугментации должна УХУДШАТЬ результат
относительно E1T (реал-only) — измеряем контраст «хорошая vs плохая калибровка».

Выход: data/converted/synthetic_bad_calib.csv (схема your_data; колонки,
нужные transas_core, присутствуют).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from run_training import load_data
from scripts.generate_synthetic_data import generate

OUT = PROJECT_ROOT / "data/converted/synthetic_bad_calib.csv"

if __name__ == "__main__":
    real = load_data(PROJECT_ROOT / "data/raw/your_data.csv")  # калибровка на ЧУЖОЙ записи
    syn = generate(n_rows=71000, seed=7, real_df=real, psd_chunks=None)
    syn.to_csv(OUT, sep="\t", index=False, encoding="utf-16")
    print(f"Сохранено: {OUT} — {syn.shape[0]} строк, {syn.shape[1]} колонок")
    print("Калибровка: your_data.csv (чужой домен: roll std %.2f, Wave.dir %.0f)"
          % (real["Roll(градусы)"].std(), real["Wave.direction(градусы)"].mean()))
