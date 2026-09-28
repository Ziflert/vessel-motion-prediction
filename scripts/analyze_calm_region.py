"""
Проверка «мёртвого» региона (строки 0–4000) — уточнение заказчика 2026-09-28:

заказчик: в строках 0–4000 реальные отклики судна на перекладку руля (манёвренные
характеристики), влияние внешних факторов минимально. Разве правильно их отбрасывать?
Спокойная вода — большая часть морского перехода; модель, обученная только на шторме,
не видит спокойную воду → в онлайне ей будет казаться, что она сломалась.

Проверка: (1) какая часть строк 0–4000 действительно без сигнала (стоянка);
(2) есть ли там манёвры (перекладки руля → ROT → курс); (3) gain «руль→ROT» на спокойной
воде vs шторм (базовая манёвренная характеристика судна).
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

df = pd.read_csv('data/raw/your_data.csv', sep='\t', encoding='utf-16')
calm = df.iloc[0:4000].reset_index(drop=True)   # «мёртвый» регион
storm = df.iloc[4000:].reset_index(drop=True)   # активный регион

print('=' * 78)
print('1. Сегменты региона 0–4000: где сигнал есть, где нет')
print('=' * 78)
rows = []
for i in range(0, 4000, 500):
    g = calm.iloc[i:i + 500]
    if len(g) == 0:
        continue
    rows.append({
        'строки': f'{i}-{i + len(g) - 1}',
        'std_Pitch': g['Pitch(градусы)'].std(), 'std_Roll': g['Roll(градусы)'].std(),
        'std_Vert': g['Vertical(Метр)'].std(),
        'std_Rudder': g['Rudder State(градусы)'].std(), 'rudder_max': g['Rudder State(градусы)'].abs().max(),
        'std_ROT': g['ROT(°/мин)'].std(), 'std_SOG': g['SOG(узлы)'].std(),
        'std_Course': g['Course(градусы)'].std(),
        'wh': g['Wave.Highest(метры)'].mean(),
        'wind_f': g['Wind.Force Summary(тс)'].mean(),
    })
t = pd.DataFrame(rows)
print(t.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
print()
print('Сравнение с активным регионом (4000+):')
print(f'  std_Rudder: {storm["Rudder State(градусы)"].std():.1f}   std_ROT: {storm["ROT(°/мин)"].std():.1f}   '
      f'std_Roll: {storm["Roll(градусы)"].std():.1f}')

print()
print('=' * 78)
print('2. Манёвры в «мёртвом» регионе: перекладки руля → ROT → курс')
print('=' * 78)
rs = calm['Rudder State(градусы)'].values
rot = calm['ROT(°/мин)'].values
course = calm['Course(градусы)'].values
dcourse = np.diff(course, prepend=course[0])
dcourse = (dcourse + 180) % 360 - 180
print(f'Эпизодов с |руль| > 10° (≥ 10 с): ', end='')
mask = np.abs(rs) > 10.0
n_ep = 0
i = 0
n = len(calm)
while i < n:
    if mask[i]:
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= 10:
            n_ep += 1
        i = j
    else:
        i += 1
print(n_ep)
print(f'corr(ROT, скорость изменения курса): {np.corrcoef(rot[1:], dcourse[1:] * 60)[0, 1]:+.3f}')
print(f'corr(Rudder State, Rudder Order):    {np.corrcoef(rs[1:], calm["Rudder Order(градусы)"].values[1:])[0, 1]:+.3f}')
gain = np.polyfit(rs[1:], rot[1:], 1)[0]
print(f'GAIN «руль→ROT» на спокойной воде:   ROT ≈ {gain:.3f} °/мин на 1° руля')
gain_storm = np.polyfit(storm['Rudder State(градусы)'].values, storm['ROT(°/мин)'].values, 1)[0]
print(f'GAIN в активном регионе (шторм):     ROT ≈ {gain_storm:.3f} °/мин на 1° руля')

print()
print('=' * 78)
print('3. Wave.Highest / Wind.Force по всему региону (границы состояний)')
print('=' * 78)
print(f'Wave.Highest: строки 0-4000: {calm["Wave.Highest(метры)"].min():.2f}-{calm["Wave.Highest(метры)"].max():.2f} м '
      f'(уник. {calm["Wave.Highest(метры)"].nunique()})')
print(f'Wave.Highest: строки 4000+:  {storm["Wave.Highest(метры)"].min():.2f}-{storm["Wave.Highest(метры)"].max():.2f} м')
print(f'Wind.Force Summary: 0-4000: медиана {calm["Wind.Force Summary(тс)"].median():.1f} тс; '
      f'4000+: медиана {storm["Wind.Force Summary(тс)"].median():.1f} тс')
print(f'SOG: спокойная вода: медиана {calm["SOG(узлы)"].median():.1f} уз; '
      f'шторм: медиана {storm["SOG(узлы)"].median():.1f} уз')
print(f'Доля строк 0-4000 с качкой std>0 (посегментно): '
      f'{(t["std_Roll"] > 0.1).mean() * 100:.0f} %')
