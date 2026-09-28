"""
Анализ дизайна записи заказчика (уточнение 2026-09-28): ветер/волнение заданы КОНСТАНТАМИ,
происходит постоянное изменение курса (манёвры) → КУ (курсовой угол) волнения/ветра меняется
ЧЕРЕЗ КУРС СУДНА. Погода применяется ОТНОСИТЕЛЬНО судна. Затем погодные условия менялись
ступенчато, снова манёвры, и т.д.

Проверка: (1) КУ волнения/ветра действительно меняются в записи; (2) амплитуда качки
зависит от КУ внутри погодного состояния; (3) амплитуда качки зависит от тяжести состояния
(Wave.Highest); (4) ROT связан с манёврами (скоростью изменения курса).
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

df = pd.read_csv('data/raw/your_data.csv', sep='\t', encoding='utf-16')
act = df.iloc[4000:].reset_index(drop=True)  # активный регион (исследование)

# --- КУ (курсовой угол) волнения и ветра относительно судна ---
wave_dir = act['Wave.direction(градусы)'].values
wind_dir = act['Wind.direction(градусы)'].values
course = act['Course(градусы)'].values

def circular_diff(a, b):
    return (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64) + 180.0) % 360.0 - 180.0

ku_wave = circular_diff(wave_dir, course)   # КУ волнения [-180, 180)
ku_wind = circular_diff(wind_dir, course)   # КУ ветра

print('=' * 78)
print('1. КУ (относительный угол) в записи — меняется ли через курс судна')
print('=' * 78)
print(f'Wave.direction (задан константой): std={act["Wave.direction(градусы)"].std():.3f} '
      f'уник={act["Wave.direction(градусы)"].nunique()}')
print(f'Wind.direction (задан константой): std={act["Wind.direction(градусы)"].std():.3f} '
      f'уник={act["Wind.direction(градусы)"].nunique()}')
print(f'Course (менялся манёврами):        std={act["Course(градусы)"].std():.1f} '
      f'уник={act["Course(градусы)"].nunique()}')
print(f'КУ волнения = волна − курс:  std={np.std(ku_wave):.1f}°  диапазон '
      f'[{np.min(ku_wave):.0f}; {np.max(ku_wave):.0f}]')
print(f'КУ ветра = ветер − курс:     std={np.std(ku_wind):.1f}°  диапазон '
      f'[{np.min(ku_wind):.0f}; {np.max(ku_wind):.0f}]')
# распределение КУ по секторам (лаговая/встречная/попутная)
sectors = {'встречная (±0-30°)': (np.abs(ku_wave) < 30).mean(),
           'лаговая (60-120°)': ((np.abs(ku_wave) > 60) & (np.abs(ku_wave) < 120)).mean(),
           'кормовая (150-180°)': (np.abs(ku_wave) > 150).mean()}
for k, v in sectors.items():
    print(f'  доля времени КУ {k}: {v * 100:.0f} %')

print()
print('=' * 78)
print('2. Погодные состояния (ступени Wave.Highest) × амплитуда качки × КУ')
print('=' * 78)
# ступени погодных условий: группируем по Wave.Highest (сценарный параметр, менялся ступенчато)
act['wh_round'] = act['Wave.Highest(метры)'].round(1)
states = []
for wh, g in act.groupby('wh_round'):
    if len(g) < 30:  # слишком короткие переходные куски
        continue
    ku = circular_diff(g['Wave.direction(градусы)'].values, g['Course(градусы)'].values)
    states.append({
        'h_s': wh, 'n_rows': len(g),
        'roll_std': g['Roll(градусы)'].std(), 'roll_absmax': g['Roll(градусы)'].abs().max(),
        'pitch_std': g['Pitch(градусы)'].std(),
        'vert_std': g['Vertical(Метр)'].std(),
        'ku_std': np.std(ku), 'ku_mean_abs': np.mean(np.abs(ku)),
        'ku_lag_share': ((np.abs(ku) > 60) & (np.abs(ku) < 120)).mean(),
        'sog_mean': g['SOG(узлы)'].mean(), 'sog_std': g['SOG(узлы)'].std(),
        'rot_std': g['ROT(°/мин)'].std(),
        'wind_force': g['Wind.Force Summary(тс)'].mean(),
        'course_std': g['Course(градусы)'].std(),
    })
st = pd.DataFrame(states)
print(f'Погодных состояний (ступеней): {len(st)}')
print(st[['h_s', 'n_rows', 'roll_std', 'pitch_std', 'vert_std', 'ku_std', 'ku_lag_share',
          'sog_mean', 'rot_std', 'wind_force']].to_string(index=False,
          float_format=lambda x: f'{x:.2f}'))

print()
print('--- Корреляции по состояниям (амплитуда качки vs условия) ---')
print(f'corr(std Roll, h_s):            {st["roll_std"].corr(st["h_s"]):+.3f}')
print(f'corr(std Pitch, h_s):           {st["pitch_std"].corr(st["h_s"]):+.3f}')
print(f'corr(std Vertical, h_s):        {st["vert_std"].corr(st["h_s"]):+.3f}')
print(f'corr(std Roll, разброс КУ):     {st["roll_std"].corr(st["ku_std"]):+.3f}')
print(f'corr(std Roll, доля лагового КУ): {st["roll_std"].corr(st["ku_lag_share"]):+.3f}')
print(f'corr(std Roll, std Course):     {st["roll_std"].corr(st["course_std"]):+.3f}')
print(f'corr(std ROT, std Course):      {st["rot_std"].corr(st["course_std"]):+.3f}  (манёвры)')
print(f'corr(SOG_mean, h_s):            {st["sog_mean"].corr(st["h_s"]):+.3f}')

print()
print('=' * 78)
print('3. Мгновенные корреляции с КУ (sin/cos, весь активный регион)')
print('=' * 78)
ku_wave_s = pd.Series(np.sin(np.radians(ku_wave)), index=act.index)
ku_wave_c = pd.Series(np.cos(np.radians(ku_wave)), index=act.index)
print(f'corr(Roll, sin(КУ волны)):  {act["Roll(градусы)"].corr(ku_wave_s):+.3f}')
print(f'corr(Roll, cos(КУ волны)):  {act["Roll(градусы)"].corr(ku_wave_c):+.3f}')
print(f'corr(|Roll|, |sin(КУ)|):    {act["Roll(градусы)"].abs().corr(ku_wave_s.abs()):+.3f}')
print(f'corr(Pitch, cos(КУ волны)): {act["Pitch(градусы)"].corr(ku_wave_c):+.3f}')

print()
print('=' * 78)
print('4. Скорость судна / ветер / ROT — сводка по состояниям')
print('=' * 78)
print(f'SOG: медиана {act["SOG(узлы)"].median():.1f} уз, диапазон [{act["SOG(узлы)"].min():.1f}; {act["SOG(узлы)"].max():.1f}]')
print(f'Wind.Force Summary: медиана {act["Wind.Force Summary(тс)"].median():.1f} тс, '
      f'std {act["Wind.Force Summary(тс)"].std():.1f}')
print(f'corr(Wave.Highest, Wind.Force Summary) = {act["Wave.Highest(метры)"].corr(act["Wind.Force Summary(тс)"]):+.3f}'
      '  (согласованность изменения погодных условий)')
print(f'ROT: std {act["ROT(°/мин)"].std():.1f} °/мин; '
      f'corr(ROT, скорость изменения курса): ', end='')
dcourse = np.diff(act['Course(градусы)'].values)
dcourse = (dcourse + 180) % 360 - 180
rot_mid = act['ROT(°/мин)'].values[1:]
print(f'{np.corrcoef(rot_mid, dcourse)[0, 1]:+.3f}')
