"""Проверка конвертированных файлов: схема, время, NaN, обучаемость по профилям."""
import sys
from pathlib import Path

sys.path.insert(0, ".")
import pandas as pd

from config.config import Config
from data.dataset import VesselDataset
from data.features import engineer_features_dataframe

FILES = [
    "data/converted/2026-10-01_w-5_converted.csv",
    "data/converted/2026-10-01_w-6_converted.csv",
    "data/converted/2026-10-01_w-7_converted.csv",
    "data/converted/test_converted.csv",
]

PROFILES = ["full_prediction", "motion_prediction", "motion_core_prediction",
            "rot_prediction", "speed_prediction", "minimal_prediction"]

print("=== 1. Схема, время, NaN ===")
for path in FILES:
    df = pd.read_csv(path, sep="\t", encoding="utf-16")
    t = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
    dt = t.diff().dropna().dt.total_seconds()
    time_ok = bool((dt == 1.0).all()) if len(dt) else True
    na_cols = [c for c in df.columns if df[c].isna().all()]
    print(f"  {Path(path).name}: {len(df)} строк, {len(df.columns)} колонок, "
          f"шаг 1с={time_ok}, полностью-NaN колонки: {na_cols or '-'}")

print("\n=== 2. Обучаемость по профилям (VesselDataset) ===")
df = pd.read_csv(FILES[0], sep="\t", encoding="utf-16")
for prof in PROFILES:
    try:
        config = Config(profile=prof, verbose=False)
        df2 = engineer_features_dataframe(
            df, cyclic=config.cyclic_encoding,
            relative_wave_angle=config.relative_wave_angle,
            relative_wind_angle=config.relative_wind_angle)
        ds = VesselDataset([df2], config, fit_scalers=True)
        nan = int(pd.isna(ds.feature_data).sum()) + int(pd.isna(ds.target_data).sum())
        x, y = ds[0]
        status = "OK" if nan == 0 else f"NaN={nan} (!)"
        print(f"  {prof}: {status} — features {ds.feature_data.shape[1]}, "
              f"targets {ds.target_data.shape[1]}, окно x{tuple(x.shape)} y{tuple(y.shape)}")
    except Exception as e:
        print(f"  {prof}: FAIL — {str(e)[:120]}")
