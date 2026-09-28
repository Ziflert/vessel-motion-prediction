"""
Инженерия признаков для исследования качки.

Ключевая идея: навигационные и метео углы (0..360) — ЦИКЛИЧЕСКИЕ величины.
Для нейросети 0° и 359° — противоположные числа, что уничтожает информацию.
Поэтому все направленческие углы кодируются парой sin/cos, а вместо абсолютных
углов добавляется ОТНОСИТЕЛЬНЫЙ курс волны/ветра (угол встречи), который
физически определяет режим качки (попутная / встречная / лаговая волна).

ВАЖНО: функции чистые — не мутируют вход. Применяются ОДИНАКОВО при обучении
(run_training) и при inference (run_inference); набор применённых преобразований
фиксируется в manifest.json версии модели.

Не циклические величины в градусах НЕ кодируются как циклические:
- Rudder Order/State — линейные углы отклонения руля (±35°);
- Long/Lat — координаты.
"""

import numpy as np
import pandas as pd

# Базовые имена колонок, являющихся циклическими углами (0..360)
CYCLIC_ANGLE_BASES = (
    'Wave.direction',
    'Swell.direction',
    'Wind.direction',
    'Current.direction',
    'Course',
)


def circular_diff(a, b):
    """Круговая разность a - b, приведённая к диапазону [-180, 180)."""
    return (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _is_cyclic(col: str) -> bool:
    return any(col == f'{base}(градусы)' or col.startswith(base + '(') for base in CYCLIC_ANGLE_BASES)


def engineer_feature_columns(feature_columns, cyclic: bool = True, relative_wave_angle: bool = True,
                             relative_wind_angle: bool = True):
    """
    Преобразует список ИСХОДНЫХ имён признаков в список итоговых
    (после инженерии). Зеркально повторяет то, что делает
    engineer_features_dataframe с DataFrame.
    """
    out = []
    has_wave_dir = any(c.startswith('Wave.direction') for c in feature_columns)
    has_course = any(c.startswith('Course') for c in feature_columns)

    for col in feature_columns:
        if cyclic and _is_cyclic(col):
            base = col.rsplit('(', 1)[0]  # 'Wave.direction(градусы)' -> 'Wave.direction'
            out.append(f'{base}(sin)')
            out.append(f'{base}(cos)')
        else:
            out.append(col)

    if relative_wave_angle and has_wave_dir and has_course:
        for suffix in ('(sin)', '(cos)'):
            name = f'Rel.Wave.angle{suffix}'
            if name not in out:
                out.append(name)

    if relative_wind_angle and any(c.startswith('Wind.direction') for c in feature_columns) and has_course:
        # Относительный угол ветра (Wind.direction - Course): важен КУСОВОЙ УГОЛ,
        # под которым ветер наваливается на корпус, а не абсолютное направление
        for suffix in ('(sin)', '(cos)'):
            name = f'Rel.Wind.angle{suffix}'
            if name not in out:
                out.append(name)

    # Убираем дубликаты с сохранением порядка
    seen = set()
    result = []
    for c in out:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def engineer_features_dataframe(df: pd.DataFrame, cyclic: bool = True,
                                relative_wave_angle: bool = True,
                                relative_wind_angle: bool = True) -> pd.DataFrame:
    """
    Применяет инженерию к DataFrame:
    1. Циклические углы (0..360) -> пара sin/cos (радианы), исходная колонка удаляется.
    2. Добавляет относительный угол встречи волны: круговая разность
       Wave.direction - Course, кодируется sin/cos (физика: режим качки
       определяется углом встречи с волной, а не абсолютными направлениями).

    Колонки-цели (качка, ROT, SOG) не трогаются.
    """
    df = df.copy()

    wave_dir_col = wave_dir_base = None
    course_col = course_base = None
    wind_dir_col = wind_dir_base = None
    for col in df.columns:
        if col.startswith('Wave.direction'):
            wave_dir_col, wave_dir_base = col, col.rsplit('(', 1)[0]
        elif col.startswith('Wind.direction'):
            wind_dir_col, wind_dir_base = col, col.rsplit('(', 1)[0]
        elif col.startswith('Course'):
            course_col, course_base = col, col.rsplit('(', 1)[0]

    if cyclic:
        for col in list(df.columns):
            if _is_cyclic(col):
                base = col.rsplit('(', 1)[0]
                rad = np.deg2rad(df[col].astype(np.float64).values)
                df[f'{base}(sin)'] = np.sin(rad)
                df[f'{base}(cos)'] = np.cos(rad)
                df.drop(columns=[col], inplace=True)

    if relative_wave_angle and wave_dir_base is not None and course_base is not None:
        # Используем исходные (ещё не закодированные) значения, если они есть;
        # иначе восстанавливаем угол из пары sin/cos
        if wave_dir_col in df.columns and course_col in df.columns:
            rel = circular_diff(df[wave_dir_col].values, df[course_col].values)
        else:
            wd = _angle_from_components(df, wave_dir_base)
            cs = _angle_from_components(df, course_base)
            rel = circular_diff(wd, cs)
        rel_rad = np.deg2rad(rel)
        df['Rel.Wave.angle(sin)'] = np.sin(rel_rad)
        df['Rel.Wave.angle(cos)'] = np.cos(rel_rad)

    # Относительный угол ветра (Wind.direction - Course) — КУСОВОЙ УГОЛ ветра
    # (используем исходные значения, захваченные ДО циклической кодировки; если они
    # уже удалены — восстанавливаем угол из пары sin/cos)
    if relative_wind_angle and wind_dir_base is not None and course_base is not None:
        if wind_dir_col in df.columns and course_col in df.columns:
            relw = circular_diff(df[wind_dir_col].values, df[course_col].values)
        else:
            wd = _angle_from_components(df, wind_dir_base)
            cs = _angle_from_components(df, course_base)
            relw = circular_diff(wd, cs)
        relw_rad = np.deg2rad(relw)
        df['Rel.Wind.angle(sin)'] = np.sin(relw_rad)
        df['Rel.Wind.angle(cos)'] = np.cos(relw_rad)

    return df


def _angle_from_components(df: pd.DataFrame, base_col_prefix: str) -> np.ndarray:
    """Восстанавливает угол в градусах из пары sin/cos колонок (atan2)."""
    sin_col = next((c for c in df.columns if c.startswith(base_col_prefix) and c.endswith('(sin)')), None)
    cos_col = next((c for c in df.columns if c.startswith(base_col_prefix) and c.endswith('(cos)')), None)
    if sin_col is None or cos_col is None:
        raise ValueError(f'Cannot recover angle for {base_col_prefix}: sin/cos columns not found')
    return np.rad2deg(np.arctan2(df[sin_col].values, df[cos_col].values)) % 360.0
