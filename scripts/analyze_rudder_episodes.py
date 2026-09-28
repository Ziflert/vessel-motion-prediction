"""
Анализ по манёвренным эпизодам (уточнение заказчика 2026-09-28, часть 2):

Эпизод = непрерывный участок с перекладкой руля |Rudder State| > 10°. Для каждого эпизода:
средний |руль|, средний ROT, знак поворота, знак КУ волны (волна справа/слева), тяжесть
погодного состояния (Wave.Highest), амплитуда качки. Затем сравнение:
  (а) тот же |руль| → ROT в штиле/среднем/сильном шторме (gain «руль → ROT» по погоде);
  (б) тот же |руль| → ROT при повороте по направлению волны vs против (асимметрия);
  (в) RPM vs SOG по состояниям: скорость падала при растущей нагрузке → сопротивление.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

df = pd.read_csv('data/raw/your_data.csv', sep='\t', encoding='utf-16')
act = df.iloc[4000:].reset_index(drop=True)  # активный регион (исследование)

rs = act['Rudder State(градусы)'].values
rot = act['ROT(°/мин)'].values
sog = act['SOG(узлы)'].values
rpm = act['RPM(Обороты в минуту)'].values
roll = act['Roll(градусы)'].values
course = act['Course(градусы)'].values
wave_dir = act['Wave.direction(градусы)'].values
wh = act['Wave.Highest(метры)'].values

ku_wave = (wave_dir - course + 180.0) % 360.0 - 180.0  # >0: волна с правого борта

# --- эпизоды: |Rudder| > 10° ---
mask = np.abs(rs) > 10.0
episodes = []
i = 0
n = len(act)
while i < n:
    if mask[i]:
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= 10:  # эпизод ≥ 10 с
            r_mean = np.mean(rs[i:j])
            ku_mean = np.mean(np.abs(ku_wave[i:j]))
            ku_sign = np.sign(np.mean(ku_wave[i:j]))
            episodes.append({
                'start': i, 'dur': j - i,
                'rudder_abs': np.abs(r_mean),
                'rudder_sign': np.sign(r_mean),       # +1 вправо, -1 влево
                'rot_mean': np.mean(rot[i:j]),
                'rot_abs': np.mean(np.abs(rot[i:j])),
                'rot_std': np.std(rot[i:j]),
                'roll_std': np.std(roll[i:j]),
                'ku_mean_abs': ku_mean, 'ku_sign': ku_sign,
                'wh': np.median(wh[i:j]),
                'sog_mean': np.mean(sog[i:j]), 'rpm_mean': np.mean(rpm[i:j]),
            })
        i = j
    else:
        i += 1

ep = pd.DataFrame(episodes)
print('=' * 78)
print(f'1. Манёвренных эпизодов (|руль| > 10°, ≥ 10 с): {len(ep)}')
print('=' * 78)
print(f'|руль|: медиана {ep["rudder_abs"].median():.1f}°, диапазон [{ep["rudder_abs"].min():.1f}; {ep["rudder_abs"].max():.1f}]')
print(f'ROT в эпизодах: медиана |ROT| {ep["rot_abs"].median():.1f} °/мин')
print(f'corr(|ROT|, |руль|) по эпизодам: {ep["rot_abs"].corr(ep["rudder_abs"]):+.3f}')

# группы по тяжести погоды и направлению поворота относительно волны
ep['severity'] = pd.cut(ep['wh'], bins=[0, 6, 9, 20], labels=['умеренная (h<6)', 'сильная (6-9)', 'очень сильная (>9)'])
ep['turn_vs_wave'] = np.where(ep['rudder_sign'] > 0,
                              np.where(ep['ku_sign'] > 0, 'вправо, волна справа (по волне)',
                                       'вправо, волна слева (против)'),
                              np.where(ep['ku_sign'] < 0, 'влево, волна слева (по волне)',
                                       'влево, волна справа (против)'))

print()
print('=' * 78)
print('2. Тот же |руль| → ROT по тяжести погодного состояния (gain «руль → ROT»)')
print('=' * 78)
print('Сопоставимые эпизоды: |руль| 10-25°, продолжительность ≥ 20 с')
sel = ep[(ep['rudder_abs'] >= 10) & (ep['rudder_abs'] <= 25) & (ep['dur'] >= 20)]
tbl = sel.groupby('severity', observed=True).agg(
    n_ep=('start', 'count'), rudder=('rudder_abs', 'mean'),
    rot_abs=('rot_abs', 'mean'), rot_std=('rot_std', 'mean'),
    roll_std=('roll_std', 'mean'), sog=('sog_mean', 'mean'), rpm=('rpm_mean', 'mean'),
)
tbl['gain'] = tbl['rot_abs'] / tbl['rudder']
print(tbl.to_string(float_format=lambda x: f'{x:.2f}'))

print()
print('=' * 78)
print('3. Асимметрия: тот же |руль|, поворот ПО направлению волны vs ПРОТИВ')
print('=' * 78)
tbl2 = sel.groupby(['severity', 'turn_vs_wave'], observed=True).agg(
    n_ep=('start', 'count'), rudder=('rudder_abs', 'mean'),
    rot_abs=('rot_abs', 'mean'), rot_std=('rot_std', 'mean'), roll_std=('roll_std', 'mean'),
)
print(tbl2.to_string(float_format=lambda x: f'{x:.2f}'))

print()
print('=' * 78)
print('4. RPM / сопротивление: скорость падала при растущей нагрузке двигателя?')
print('=' * 78)
tbl3 = act.groupby(act['Wave.Highest(метры)'].round(1)).agg(
    rpm=('RPM(Обороты в минуту)', 'mean'), sog=('SOG(узлы)', 'mean'),
    roll_std=('Roll(градусы)', 'std'), n=('time', 'count') if 'time' in act.columns else ('Roll(градусы)', 'count'),
)
tbl3 = tbl3[tbl3['n'] >= 60]
print(tbl3.to_string(float_format=lambda x: f'{x:.2f}'))
print()
print('Вывод: если RPM растёт, а SOG падает → скорость теряется из-за сопротивления')
print('(встречное/поперечное сопротивление волн и ветра), а не из-за сброса нагрузки.')
