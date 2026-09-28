"""
Создание минимального дата-сета (вопрос заказчика 2026-09-28): «зачем кормить нейросети
значения, которые не нужны или не изменяются».

Из полной записи (39 колонок) убирается всё, что не несёт информации для прогноза качки:
координаты, дубликаты, вычисленные эффекты (моменты/силы — следствие, а не причина),
константы-каналы, не используемые в признаках. Остаётся НЕОБХОДИМЫЙ МИНИМУМ (17 колонок):
качка (6) + навигация (SOG/ROT/Course) + управление (Rudder Order/State/RPM) +
внешние условия (Wave.Highest, Wave.speed, Wind.Force Summary) + основы КУ
(Wave.direction/Wind.direction — константы, нужны для расчёта курсовых углов).

Направления ветра/волнения в нейросеть НЕ подаются как абсолютные значения —
инженерия признаков (data/features.py) считает КУСОВОЙ УГОЛ (Wave.direction − Course,
Wind.direction − Course) и кодирует sin/cos (relative_wave_angle/relative_wind_angle).

Обоснование — `docs/MINIMAL_DATASET.md` (статистика колонок + литература).
Формат: TSV/UTF-16 (как у исходной записи — load_data читает без изменений).

Запуск: .venv/Scripts/python.exe scripts/make_minimal_dataset.py
"""

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

SRC = 'data/raw/your_data.csv'
OUT = 'data/raw/your_data_minimal.csv'

# НЕОБХОДИМЫЙ МИНИМУМ (17 колонок)
KEEP = [
    # === КАЧКА (цели) ===
    'Pitch(градусы)',
    'Roll(градусы)',
    'Vertical(Метр)',
    'Velocity.Pitching(°/мин)',
    'Velocity.Rolling(°/мин)',
    'Velocity.Vertical(узлы)',
    # === НАВИГАЦИЯ ===
    'SOG(узлы)',          # цель + скорость
    'ROT(°/мин)',         # цель + контур «руль→ROT→курс»
    'Course(градусы)',    # основа КУ волнения/ветра
    # === УПРАВЛЕНИЕ (главная зависимость) ===
    'Rudder Order(градусы)',   # заданная перекладка (команда)
    'Rudder State(градусы)',   # фактическая перекладка
    'RPM(Обороты в минуту)',   # нагрузка двигателя
    # === ВНЕШНИЕ УСЛОВИЯ ===
    'Wave.Highest(метры)',     # тяжесть состояния (h_s, сценарный индекс L-3)
    'Wave.speed(узлы)',        # прокси периода волны (частота встречи)
    'Wind.Force Summary(тс)',  # прокси интенсивности ветра (канала Wind.speed нет)
    'Wave.direction(градусы)', # основа КУ волнения (константа — для расчёта)
    'Wind.direction(градусы)', # основа КУ ветра (константа — для расчёта)
]

# Что убирается (обоснование — docs/MINIMAL_DATASET.md)
REMOVE = [
    'Velocity.Yawing(°/мин)',       # дубликат ROT в производной форме
    'Swell(метры)',                 # сценарный параметр, corr(Roll)≈0
    'Wave.current(метры)',          # неоднозначный канал, corr(Roll)≈−0.04
    'Swell.direction(градусы)',     # константа, не используется в признаках
    'Current.direction(градусы)',   # константа (течение не менялось)
    'Current.speed(узлы)',          # константа (нулевое течение)
    'Long(градусы)',                # координаты (судно в одном районе)
    'Lat(градусы)',                 # координаты
    'STW(узлы)',                    # дубликат SOG
    'Moment Yawing(тс*м)', 'Moment Rolling(тс*м)', 'Moment Pitching(тс*м)',
    'Force Vertical(тс)', 'Force Summary(тс)', 'Force Longitudinal(тс)', 'Force Lateral(тс)',
    'Wind.Moment Yawing(тс*м)', 'Wind.Moment Rolling(тс*м)', 'Wind.Moment Pitching(тс*м)',
    'Wind.Force Vertical(тс)', 'Wind.Force Longitudinal(тс)', 'Wind.Force Lateral(тс)',
    # Wind.Force Summary ОСТАВЛЕН — единственный прокси интенсивности ветра
]


def main():
    df = pd.read_csv(SRC, sep='\t', encoding='utf-16')
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        print(f'!! Отсутствуют колонки: {missing}')
        return

    removed = [c for c in df.columns if c not in KEEP and c != 'time']
    print(f'Исходная запись: {len(df)} строк, {len(df.columns)} колонок')
    print(f'Оставляем: {len(KEEP)} колонок (необходимый минимум)')
    print(f'Убираем:   {len(removed)} колонок:')
    for c in removed:
        print(f'  - {c}')

    out = df[KEEP]
    out.to_csv(OUT, sep='\t', encoding='utf-16', index=False)
    print(f'\n✓ Минимальный дата-сет сохранён: {OUT} ({len(out)} строк, {len(out.columns)} колонок)')

    # верификация: инженерия признаков работает на минимуме
    from data.features import engineer_features_dataframe
    eng = engineer_features_dataframe(out)
    print(f'✓ После инженерии: {len(eng.columns)} переменных (вход модели)')
    print(f'  КУ-каналы: {[c for c in eng.columns if "Rel." in c]}')


if __name__ == '__main__':
    main()
