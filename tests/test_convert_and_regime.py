"""Тесты конвертера Transas-экспортов и режимного генератора (IDEAS B4).

Покрывают:
  - определение кодировки (UTF-16 BOM);
  - разбор сырого экспорта (шапка метаданных, канал 'time', пустая колонка);
  - маппинг каналов (правила CHANNEL_RULES, нормализация префикса судна);
  - вывод производных колонок (Velocity.Vertical из позиции, Wave.speed из Swell);
  - фолбэк SOG=STW при нулевом SOG и живом STW;
  - фильтрацию колонок (union профилей / конкретный профиль / --keep-all);
  - сегментацию режимов по ступеням Wave.Highest и калибровку;
  - гладкость сшивки режимов (smooth_steps) и связку «среда→качка»;
  - обучаемость выхода генератора схемой профиля transas_core.

Запуск:  pytest tests/test_convert_and_regime.py   (или .venv/Scripts/python.exe tests/...)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.convert_transas_csv import (
    CHANNEL_RULES, detect_encoding, map_columns, normalize_channel, read_transas,
)
from scripts.generate_regime_synthetic import (
    OUT_COLUMNS, generate, segment_regimes, smooth_steps,
)


# ---------------------------------------------------------------------------
# Фикстуры: минимальный синтетический экспорт в формате Transas
# ---------------------------------------------------------------------------

def make_transas_file(tmp_path, n=300, ship="СС 1", wave_steps=(1.0, 4.0)):
    """Сырой экспорт: UTF-16, шапка 10 строк, каналы, ступени Wave.Highest."""
    rng = np.random.default_rng(0)
    t = [f"{i // 3600:02d}:{(i // 60) % 60:02d}:{i % 60:02d}" for i in range(n)]
    rows = {"time": t}
    rows[f"{ship}.Сенсоры.Гировертикаль.1.Угол дифферента(градусы)"] = rng.normal(0, 0.5, n)
    rows[f"{ship}.Сенсоры.Гировертикаль.1.Угол крена(градусы)"] = 2 * np.sin(np.arange(n) * 2 * np.pi / 8)
    rows[f"{ship}.Ship Model.Kinematics.Position.Vertical(метры)"] = np.sin(np.arange(n) * 2 * np.pi / 8)
    rows[f"{ship}.Движение.Угловая скорость бортовой качки(°/мин)"] = rng.normal(0, 1, n)
    rows[f"{ship}.Движение.Угловая скорость килевой качки(°/мин)"] = rng.normal(0, 1, n)
    rows[f"{ship}.Сенсоры.Гирокомпас.1.Курс(градусы)"] = 150.0 + rng.normal(0, 1, n)
    rows[f"{ship}.Сенсоры.Гирокомпас.1.Угловая скорость поворота(°/мин)"] = rng.normal(0, 5, n)
    rows[f"{ship}.Сенсоры.Лаг.1.Скорость относительно воды(узлы)"] = 5.0 + rng.normal(0, 0.2, n)
    rows[f"{ship}.Сенсоры.Лаг.1.Скорость относительно грунта(узлы)"] = np.zeros(n)  # мёртвый SOG
    rows[f"{ship}.Ship Model.Machinery.Steering.Central.Rudder Order(градусы)"] = rng.normal(0, 15, n)
    rows[f"{ship}.Ship Model.Machinery.Steering.Central.Rudder State(градусы)"] = rng.normal(0, 15, n)
    rows[f"{ship}.Ship Model.Machinery.Propulsion.Central.Engine.RPM(Обороты в минуту)"] = 58 + rng.normal(0, 1, n)
    wh = np.concatenate([np.full(n // 2, wave_steps[0]), np.full(n - n // 2, wave_steps[1])])
    rows[f"{ship}.Окружающая среда.Наиболее значимая высота волны(метры)"] = wh
    rows[f"{ship}.Окружающая среда.Мгновенная амплитуда волны(метры)"] = rng.normal(0, 0.3, n)
    rows[f"{ship}.Окружающая среда.Направление волны(градусы)"] = np.zeros(n)
    rows[f"{ship}.Окружающая среда.Направление истинного ветра(градусы)"] = np.zeros(n)
    rows[f"{ship}.Окружающая среда.Скорость истинного ветра(узлы)"] = 20.0 + rng.normal(0, 0.5, n)
    rows[f"{ship}.Окружающая среда.Высота зыби(метры)"] = np.zeros(n)
    rows[f"{ship}.Окружающая среда.Направление зыби(градусы)"] = np.zeros(n)
    rows[f"{ship}.Окружающая среда.Направление течения(градусы)"] = np.zeros(n)
    rows[f"{ship}.Окружающая среда.Скорость течения(узлы)"] = np.zeros(n)
    df = pd.DataFrame(rows)

    path = tmp_path / "export.csv"
    with open(path, "w", encoding="utf-16") as f:
        f.write("Current time:\t16:00:00\nStudent:\t\nArea:\tOpen Sea\n"
                "Exercise:\tExercise1\nShip:\tСС 1\nComments:\t\n"
                "Exercise start:\t12:00:00\nStart:\t12:00:00\nFinish:\t13:00:00\n"
                "Time step:\t00:00:01\n")
        f.write("\t".join(df.columns) + "\t\n")
        for _, r in df.iterrows():
            f.write("\t".join(str(v) for v in r.values) + "\t\n")
    return path, df


# ---------------------------------------------------------------------------
# Конвертер
# ---------------------------------------------------------------------------

def test_detect_encoding_utf16(tmp_path):
    path, _ = make_transas_file(tmp_path)
    assert detect_encoding(path) == "utf-16"


def test_read_transas_skips_header_and_empty_col(tmp_path):
    path, ref = make_transas_file(tmp_path)
    df, meta = read_transas(path)
    assert len(df) == len(ref)
    assert df.columns[0] == "time"
    assert "Ship" in meta and meta["Ship"] == "СС 1"
    assert all(not c.startswith("Unnamed") for c in df.columns)


def test_normalize_channel_strips_ship_prefix():
    assert normalize_channel("СС 1.Сенсоры.Гировертикаль.1.Угол крена(градусы)") \
        == "сенсоры.гировертикаль.1.угол крена"


def test_map_columns_all_known_channels(tmp_path):
    path, _ = make_transas_file(tmp_path)
    df, _ = read_transas(path)
    mapping, unmapped = map_columns(df)
    assert unmapped == []
    assert mapping["time"] == "time"
    # ключевые каналы нашли свои модельные имена
    targets = set(mapping.values())
    assert {"Pitch(градусы)", "Roll(градусы)", "Vertical(Метр)", "SOG(узлы)",
            "RPM(Обороты в минуту)", "Wave.Highest(метры)"} <= targets


def test_map_rules_unique_targets():
    targets = [t for _p, t in CHANNEL_RULES]
    assert len(targets) == len(set(targets)), "дубли целевых колонок в CHANNEL_RULES"


def test_derivatives_and_sog_fallback(tmp_path):
    from scripts.convert_transas_csv import convert_file
    path, _ = make_transas_file(tmp_path)
    out = convert_file(path, tmp_path / "out", profile=None, keep_all=False)
    df = pd.read_csv(out, sep="\t", encoding="utf-16")
    # Velocity.Vertical выведен из позиции и коррелирует с ней
    v = np.gradient(df["Vertical(Метр)"].values) * 1.94384
    assert np.corrcoef(df["Velocity.Vertical(узлы)"], v)[0, 1] > 0.9
    # Wave.speed = 7.102 * Swell
    assert np.allclose(df["Wave.speed(узлы)"], 7.102 * df["Swell(метры)"])
    # SOG был 0 при живом STW -> подставлен STW
    stw_ref = 5.0 + pd.Series(np.random.default_rng(0).normal(0, 0.2, len(df)))
    assert df["SOG(узлы)"].std() > 0.1
    # нет NaN-заглушек и нет колонок вне needed
    assert not df.isna().any().any()
    assert "Long(градусы)" not in df.columns


def test_filter_keeps_profile_columns(tmp_path):
    from config.config import Config
    from scripts.convert_transas_csv import convert_file
    path, _ = make_transas_file(tmp_path)
    out = convert_file(path, tmp_path / "out", profile="transas_core")
    df = pd.read_csv(out, sep="\t", encoding="utf-16")
    feats, targs, _w = Config.get_profile_config("transas_core")
    for c in list(feats) + list(targs):
        assert c in df.columns, f"{c} отсутствует при --profile transas_core"


def test_drop_time_and_align_chunks(tmp_path):
    from scripts.convert_transas_csv import convert_file
    path, _ = make_transas_file(tmp_path, n=1250)
    out = convert_file(path, tmp_path / "out", drop_time=True, align_chunks=1000)
    df = pd.read_csv(out, sep="\t", encoding="utf-16")
    assert "time" not in df.columns
    assert len(df) == 1000


# ---------------------------------------------------------------------------
# Режимный генератор
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_merged():
    p = ROOT / "data/converted/real_w5w6w7_merged.csv"
    if not p.exists():
        pytest.skip("merged-файл реальных записей отсутствует")
    return pd.read_csv(p, sep="\t", encoding="utf-16")


def test_segment_regimes_finds_wave_steps(real_merged):
    regimes = segment_regimes(real_merged)
    levels = [round(p["wave_h"], 1) for p in regimes]
    assert levels == [0.3, 2.2, 3.5, 4.7]
    # режимы монотонны во времени и не перекрываются (щели между ними
    # допустимы — переходные строки Wave.Highest короче MIN_REGIME_ROWS)
    for p1, p2 in zip(regimes[:-1], regimes[1:]):
        assert p1["b"] <= p2["a"]
    assert all(p["n"] >= 600 for p in regimes)


def test_smooth_steps_matches_regime_levels(real_merged):
    regimes = segment_regimes(real_merged)
    n = sum(p["n"] for p in regimes)
    curve = smooth_steps(regimes, n, "wave_h")
    # в середине каждого режима значение == уровню режима
    for p in regimes:
        mid = (p["a"] + p["b"]) // 2
        assert abs(curve[mid] - p["wave_h"]) < 1e-6


def test_generate_schema_and_no_nan(real_merged):
    syn = generate(real_merged, seed=7)
    assert list(syn.columns) == OUT_COLUMNS
    assert not syn.isna().any().any()
    assert len(syn) == sum(p["n"] for p in segment_regimes(real_merged))


def test_generate_regime_coupling(real_merged):
    """Связка «среда→качка»: std Roll по режимам отслеживает уровни волны."""
    syn = generate(real_merged, seed=7)
    regimes = segment_regimes(real_merged)
    syn_stds = [syn["Roll(градусы)"].iloc[p["a"]:p["b"]].std() for p in regimes]
    real_stds = [p["Roll(градусы)"] for p in regimes]
    # порядок совпадает и разброс воспроизведён (не сплющено в константу)
    assert np.argsort(syn_stds).tolist() == np.argsort(real_stds).tolist()
    assert max(syn_stds) / min(syn_stds) > 1.3


def test_generate_matches_transas_core_profile(real_merged):
    from config.config import Config
    from data.features import engineer_features_dataframe
    syn = generate(real_merged, seed=7)
    config = Config(profile="transas_core", verbose=False)
    df2 = engineer_features_dataframe(
        syn, cyclic=config.cyclic_encoding,
        relative_wave_angle=config.relative_wave_angle,
        relative_wind_angle=config.relative_wind_angle)
    missing = [c for c in list(config.feature_columns) + list(config.target_columns)
               if c not in df2.columns]
    assert missing == []
