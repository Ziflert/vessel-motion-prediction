"""Конвертер «сырых» экспортов Transas (Navi-Trainer) в схему пайплайна LSTM.

Формат входа (см. data/raw/test.csv, 2026-10-01_w-5.csv, ...):
  - кодировка UTF-16 (LE, с BOM), разделитель — таб;
  - шапка из 10 строк метаданных (Current time / Student / Area / ... / Time step);
  - строка заголовков каналов с префиксом судна ('СС 1.Сенсоры....');
  - далее данные: колонка 'time' (HH:MM:SS) + числовые значения, возможна
    пустая замыкающая колонка.

Формат выхода — «raw-схема train» (как data/raw/your_data.csv):
  UTF-16 TSV, первая колонка 'time', модельные имена колонок
  (Pitch(градусы), Roll(градусы), ...). Такая схема читается и обучением
  (run_training.py), и онлайн-просмотром (online/playback.py через
  online/sources.load_raw_csv).

Использование:
  .venv\\Scripts\\python.exe scripts/convert_transas_csv.py data/raw/test.csv
  .venv\\Scripts\\python.exe scripts/convert_transas_csv.py data/raw/*.csv --out data/converted
  .venv\\Scripts\\python.exe scripts/convert_transas_csv.py data/raw/test.csv --list-columns

Файлы, уже имеющие модельную схему (your_data.csv), пропускаются без изменений.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Запуск из любого каталога: делаем доступным импорт config.* из корня проекта
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Маппинг каналов Transas -> модельные колонки.
# Каждое правило: (regex по нормализованному имени канала, целевая колонка).
# Нормализация: нижний регистр, без префикса судна ('СС 1.' и т.п.),
# единицы в скобках сохраняются для проверки, но в матчинге не участвуют.
# ---------------------------------------------------------------------------

CHANNEL_RULES: list[tuple[str, str]] = [
    # --- качка (сенсоры + кинематика) ---
    (r"угол дифферента", "Pitch(градусы)"),
    (r"угол крена", "Roll(градусы)"),
    (r"position\.vertical|вертикаль.*позиция|кинематика.*вертикал", "Vertical(Метр)"),
    (r"угловая скорость бортовой качки", "Velocity.Rolling(°/мин)"),
    (r"угловая скорость килевой качки", "Velocity.Pitching(°/мин)"),
    (r"acceleration\.vertical|вертикальное ускорение", "Acceleration.Vertical()"),
    # --- курс / скорости ---
    (r"гирокомпас.*курс(?!\s*скорост)|(?<!скорость )\bкурс\b", "Course(градусы)"),
    (r"угловая скорость поворота", "ROT(°/мин)"),
    (r"скорость относительно воды", "STW(узлы)"),
    (r"скорость относительно грунта", "SOG(узлы)"),
    # --- управление ---
    (r"rudder order", "Rudder Order(градусы)"),
    (r"rudder state", "Rudder State(градусы)"),
    (r"engine\.rpm|обороты", "RPM(Обороты в минуту)"),
    # --- окружающая среда ---
    (r"наиболее значимая высота волны", "Wave.Highest(метры)"),
    (r"мгновенная амплитуда волны", "Wave.current(метры)"),
    (r"направление волны", "Wave.direction(градусы)"),
    (r"направление истинного ветра", "Wind.direction(градусы)"),
    (r"скорость истинного ветра", "Wind.speed(узлы)"),
    (r"высота зыби", "Swell(метры)"),
    (r"направление зыби", "Swell.direction(градусы)"),
    (r"направление течения", "Current.direction(градусы)"),
    (r"скорость течения", "Current.speed(узлы)"),
    (r"направление кажущегося волнения", "Apparent.Wave.direction(градусы)"),
    (r"направление кажущегося ветра", "Apparent.Wind.direction(градусы)"),
    (r"скорость кажущегося ветра", "Apparent.Wind.speed(узлы)"),
    # --- волновые силы/моменты (дрейф) ---
    (r"drift\.force lateral", "Force Lateral(тс)"),
    (r"drift\.force longitudinal", "Force Longitudinal(тс)"),
    (r"drift\.force summary", "Force Summary(тс)"),
    (r"drift\.moment pitching", "Moment Pitching(тс*м)"),
    (r"drift\.moment rolling", "Moment Rolling(тс*м)"),
    (r"drift\.moment yawing", "Moment Yawing(тс*м)"),
]

MPS_TO_KNOTS = 1.94384  # 1 м/с = 1.94384 узла

# Все профили обучения (config.Config). Колонки, не входящие ни в один
# профиль (features/targets), по умолчанию УДАЛЯЮТСЯ — в файле остаются
# только данные, реально потребляемые моделью.
ALL_PROFILES = ["full_prediction", "motion_prediction", "motion_core_prediction",
                "rot_prediction", "speed_prediction", "minimal_prediction"]


def needed_columns(profile: str | None = None) -> set[str]:
    """Колонки, нужные пайплайну: union features+targets (по всем профилям
    или по одному, если profile задан)."""
    from config.config import Config
    names = [profile] if profile else ALL_PROFILES
    keep: set[str] = set()
    for name in names:
        feats, targs, _w = Config.get_profile_config(name)
        keep |= set(feats) | set(targs)
    return keep

MPS_TO_KNOTS = 1.94384  # 1 м/с = 1.94384 узла

# Как выводить отсутствующие модельные колонки: (целевая колонка, функция от df).
DERIVED_RULES = {
    # Velocity.Vertical = вертикальная скорость (м/с) -> узлы.
    # В your_data corr с d(Vertical)/dt = 0.986.
    "Velocity.Vertical(узлы)": lambda df: np.gradient(
        df["Vertical(Метр)"].interpolate(limit_direction="both").values
    ) * MPS_TO_KNOTS,
    # Velocity.Yawing = гирокомпасный ROT (°/мин). В your_data corr = 0.992.
    "Velocity.Yawing(°/мин)": lambda df: (
        df["ROT(°/мин)"] if "ROT(°/мин)" in df.columns else np.nan
    ),
    # Wave.speed = фазовая скорость зыби. В your_data линейна по Swell:
    # Wave.speed ≈ 7.102 * Swell (R² = 0.998). При Swell = 0 → 0 — корректно.
    "Wave.speed(узлы)": lambda df: 7.102 * df["Swell(метры)"].fillna(0.0),
}


def detect_encoding(path: Path) -> str:
    with open(path, "rb") as f:
        head = f.read(4)
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    return "utf-8"


def read_transas(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Читает сырой экспорт Transas. Возвращает (data, metadata)."""
    enc = detect_encoding(path)
    text = path.read_text(encoding=enc, errors="replace")
    lines = text.splitlines()

    # Ищем строку заголовков каналов: первая строка, начинающаяся с 'time'
    header_idx = None
    for i, line in enumerate(lines):
        first = line.split("\t", 1)[0].strip().lower()
        if first == "time":
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"{path.name}: не найдена строка заголовков (колонка 'time')")

    metadata: dict[str, str] = {}
    for line in lines[:header_idx]:
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[0].strip().endswith(":"):
            metadata[parts[0].strip().rstrip(":")] = parts[1].strip()

    # Парсим шапку+данные с одного заголовка (pandas сам выбросит пустую колонку)
    body = "\n".join([lines[header_idx]] + lines[header_idx + 1:])
    df = pd.read_csv(io.StringIO(body), sep="\t", dtype=str)
    # Убираем пустые/безымянные колонки (замыкающий таб в каждой строке)
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    return df, metadata


def normalize_channel(name: str) -> str:
    """Убирает префикс судна ('СС 1.', 'CC 1.' и т.п.), приводит к нижнему регистру."""
    name = re.sub(r"^[^\s]+\s+\d+\.", "", name.strip())  # 'СС 1.Сенсоры...' -> 'Сенсоры...'
    name = re.sub(r"\([^)]*\)$", "", name.strip())        # хвостовые единицы '(градусы)'
    return name.lower().strip()


def map_columns(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    """Возвращает (маппинг исходная->целевая, список непомеченных исходных колонок)."""
    mapping: dict[str, str] = {}
    used_targets: set[str] = set()
    unmapped: list[str] = []
    for col in df.columns:
        if col == "time":
            mapping[col] = "time"
            continue
        norm = normalize_channel(col)
        target = None
        for pattern, tgt in CHANNEL_RULES:
            if re.search(pattern, norm) and tgt not in used_targets:
                target = tgt
                break
        if target:
            mapping[col] = target
            used_targets.add(target)
        else:
            unmapped.append(col)
    return mapping, unmapped


def convert_file(path: Path, out_dir: Path, keep_unmapped: bool = False,
                 profile: str | None = None, keep_all: bool = False,
                 drop_time: bool = False, align_chunks: int = 0) -> Path | None:
    enc = detect_encoding(path)
    # Файл уже в модельной схеме? (первая колонка 'time' с первой строки)
    first_line = path.read_text(encoding=enc, errors="replace").splitlines()[0]
    already_model = first_line.lower().startswith("time\tpitch(градусы)")

    if already_model:
        # Уже конвертированный файл: читаем как есть, далее — общая ветка
        # (вывод правил и фильтрация применятся повторно без изменений данных).
        df = pd.read_csv(path, sep="\t", encoding=enc, dtype=str)
        metadata, unmapped = {}, []
        out = pd.DataFrame()
        out["time"] = df["time"].str.strip()
        for col in df.columns:
            if col != "time":
                out[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        df, metadata = read_transas(path)
        mapping, unmapped = map_columns(df)

        out = pd.DataFrame()
        out["time"] = df["time"].str.strip()
        for src, tgt in mapping.items():
            if src == "time":
                continue
            out[tgt] = pd.to_numeric(df[src], errors="coerce")

    # Дубликаты имён целевых колонок (два датчика крена и т.п.) — берём первую
    out = out.loc[:, ~out.columns.duplicated()]

    # Отсутствующие в экспорте модельные колонки выводятся из имеющихся
    # (см. DERIVED_RULES). Никаких NaN-заглушек не создаётся.
    for col, rule in DERIVED_RULES.items():
        if col not in out.columns or out[col].isna().all():
            out[col] = rule(out)
            print(f"       выведена колонка {col}")

    # Артефакт экспорта Transas: лаг грунтовой скорости не сконфигурирован,
    # SOG весь ноль, при том что STW живой. Физика: SOG = STW + вектор течения;
    # при нулевом течении SOG = STW. Подставляем только если SOG весь ноль.
    if ("SOG(узлы)" in out.columns and "STW(узлы)" in out.columns
            and (out["SOG(узлы)"].fillna(0) == 0).all()
            and (out["STW(узлы)"].fillna(0) != 0).any()):
        current = out["Current.speed(узлы)"].fillna(0) if "Current.speed(узлы)" in out.columns else 0
        out["SOG(узлы)"] = out["STW(узлы)"] + current
        print("       ВНИМАНИЕ: SOG был константный 0 при живом STW — "
              "подставлено SOG = STW + Current.speed")

    if keep_unmapped and unmapped:
        for col in unmapped:
            out[f"RAW.{col}"] = pd.to_numeric(df[col], errors="coerce")

    # Фильтрация: оставляем только колонки, реально нужные пайплайну.
    # (DERIVED_RULES и SOG-фолбэк уже отработали, их источники больше не нужны.)
    n_before = len(out.columns)
    if not keep_all:
        keep = {"time"} | needed_columns(profile)
        dropped = [c for c in out.columns if c not in keep]
        out = out[[c for c in out.columns if c in keep or c == "time"]]
        print(f"       удалено лишних колонок: {len(dropped)}"
              f" (профиль: {profile or 'union всех профилей'})")

    # Порядок колонок: time, качка, скорости/курс, управление, среда, силы
    priority = [
        "time", "Pitch(градусы)", "Roll(градусы)", "Vertical(Метр)",
        "Velocity.Pitching(°/мин)", "Velocity.Rolling(°/мин)", "Velocity.Vertical(узлы)",
        "Velocity.Yawing(°/мин)", "ROT(°/мин)", "Course(градусы)",
        "STW(узлы)", "SOG(узлы)",
        "Rudder Order(градусы)", "Rudder State(градусы)", "RPM(Обороты в минуту)",
        "Swell(метры)", "Swell.direction(градусы)", "Wave.Highest(метры)",
        "Wave.current(метры)", "Wave.direction(градусы)", "Wave.speed(узлы)",
        "Wind.direction(градусы)", "Wind.speed(узлы)",
        "Current.direction(градусы)", "Current.speed(узлы)",
        "Apparent.Wave.direction(градусы)", "Apparent.Wind.direction(градусы)",
        "Apparent.Wind.speed(узлы)", "Acceleration.Vertical()",
        "Force Lateral(тс)", "Force Longitudinal(тс)", "Force Summary(тс)",
        "Moment Pitching(тс*м)", "Moment Rolling(тс*м)", "Moment Yawing(тс*м)",
    ]
    ordered = [c for c in priority if c in out.columns]
    ordered += [c for c in out.columns if c not in ordered]
    out = out[ordered]

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem[:-len("_converted")] if already_model and path.stem.endswith("_converted") else path.stem
    out_path = out_dir / f"{stem}_converted.csv"

    n_motion_na = int(out[[c for c in ["Pitch(градусы)", "Roll(градусы)", "Vertical(Метр)"] if c in out.columns]].isna().sum().sum())

    # Выравнивание длины под кратность чанку сплиттера (run_training.split_segments
    # режет df по chunk_size строк): без кратности склейка нескольких файлов
    # даёт чанки, пересекающие стык двух записей. Хвост отбрасывается явно.
    trimmed = 0
    if align_chunks and align_chunks > 1:
        trimmed = len(out) % align_chunks
        if trimmed:
            out = out.iloc[:-trimmed]

    if drop_time and "time" in out.columns:
        out = out.drop(columns=["time"])

    out.to_csv(out_path, sep="\t", index=False, encoding="utf-16")

    print(f"  OK   {path.name} -> {out_path.name}: {len(out)} строк, {len(out.columns)} колонок"
          f" (было {n_before} до фильтра; непомеченных каналов: {len(unmapped)}, NaN в качке: {n_motion_na}"
          + (f", отброшен хвост {trimmed} строк для кратности {align_chunks}" if trimmed else "") + ")")
    if unmapped:
        print(f"       непомеченные: {unmapped}")
    return out_path


def list_columns(path: Path) -> None:
    df, metadata = read_transas(path)
    mapping, unmapped = map_columns(df)
    print(f"\n{path.name}: {len(df)} строк данных, {len(df.columns)} колонок")
    print("  Метаданные:", {k: v for k, v in metadata.items() if v})
    for col in df.columns:
        tgt = mapping.get(col)
        mark = f"-> {tgt}" if tgt else "-> (не распознан)"
        print(f"  {col}  {mark}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("inputs", nargs="+", help="входные файлы (экспорт Transas, UTF-16 TSV)")
    p.add_argument("--out", default="data/converted", help="каталог для результатов")
    p.add_argument("--keep-unmapped", action="store_true",
                   help="сохранить нераспознанные каналы как 'RAW.<имя>'")
    p.add_argument("--profile", default=None,
                   help="оставить только колонки конкретного профиля "
                        "(по умолчанию union всех профилей)")
    p.add_argument("--keep-all", action="store_true",
                   help="не удалять лишние колонки (полная схема)")
    p.add_argument("--drop-time", action="store_true",
                   help="удалить колонку 'time' (пайплайн всё равно её отбрасывает "
                        "при загрузке; полезна только для отладки/сопоставления с журналом)")
    p.add_argument("--align-chunks", type=int, default=0, metavar="N",
                   help="отбросить хвост так, чтобы длина файла была кратна N "
                        "(безопасная склейка нескольких записей в один файл; "
                        "чанк сплиттера = 1000)")
    p.add_argument("--list-columns", action="store_true",
                   help="только показать распознавание каналов, без конвертации")
    args = p.parse_args(argv)

    for pattern in args.inputs:
        for path in sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?") else [Path(pattern)]:
            if not path.exists():
                print(f"  MISSING {path}")
                continue
            try:
                if args.list_columns:
                    list_columns(path)
                else:
                    convert_file(path, Path(args.out), keep_unmapped=args.keep_unmapped,
                                 profile=args.profile, keep_all=args.keep_all,
                                 drop_time=args.drop_time, align_chunks=args.align_chunks)
            except Exception as e:  # noqa: BLE001 — идём дальше по списку файлов
                print(f"  ERROR  {path.name}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
