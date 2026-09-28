"""
Анализ контура «руль → ROT» по погодным состояниям (уточнение заказчика 2026-09-28):

1. Перекладка руля — главная зависимость: положили руль 15° вправо → судно должно начать
   изменение курса вправо, ROT растёт. Если при том же руле ROT растёт медленнее/быстрее
   обычного — это волнение/ветер воздействуют на судно.
   → считаем GAIN (наклон ROT ~ Rudder) по каждому погодному состоянию и сравниваем.
2. В штиль судно слушается руля идеально; в плохую погоду волна «захватывает» судно
   и ROT сильно прыгает — это НЕ шумы, а реальное поведение судна (симулятор ≈ действительность).
3. ROT в сильный шторм различался в зависимости от направления поворота относительно волны.
4. Скорость: учитывать RPM/нагрузку двигателя — в шторм скорость падала из-за встречного/
   поперечного сопротивления; проверить corr(SOG, RPM) и corr(RPM, h_s).
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd

df = pd.read_csv('data/raw/your_data.csv', sep='\t', encoding='utf-16')
act = df.iloc[4000:].reset_index(drop=True)  # активный регион (исследование)

course = act['Course(градусы)'].values
rot = act['ROT(°/мин)'].values
rudder_order = act['Rudder Order(градусы)'].values
rudder_state = act['Rudder State(градусы)'].values
rpm = act['RPM(Обороты в минуту)'].values
sog = act['SOG(узлы)'].values
wave_dir = act['Wave.direction(градусы)'].values

# скорость изменения курса (°/с → в минуту, для сопоставления с ROT)
dcourse = np.diff(course, prepend=course[0])
dcourse = (dcourse + 180.0) % 360.0 - 180.0 * (np.arange(len(course)) > 0)
dcourse_min = dcourse * 60.0  # °/мин
# КУ волны (знак: волна с правого/левого борта относительно курса)
ku_wave = (wave_dir - course + 180.0) % 360.0 - 180.0

print('=' * 78)
print('1. Контур «руль → ROT» глобально (весь активный регион)')
print('=' * 78)
print(f'corr(ROT, Rudder State):   {np.corrcoef(rot[1:], rudder_state[1:])[0, 1]:+.3f}')
print(f'corr(ROT, Rudder Order):   {np.corrcoef(rot[1:], rudder_order[1:])[0, 1]:+.3f}')
print(f'corr(ROT, dCourse/мин):    {np.corrcoef(rot[1:], dcourse_min[1:])[0, 1]:+.3f}')
print(f'corr(Rudder State, Rudder Order): '
      f'{np.corrcoef(rudder_state[1:], rudder_order[1:])[0, 1]:+.3f}  (исполнение руля)')
# gain: ROT на 1° перекладки (линейная регрессия ROT ~ Rudder State)
gain_all = np.polyfit(rudder_state[1:], rot[1:], 1)[0]
print(f'GAIN глобальный: ROT ≈ {gain_all:.3f} °/мин на 1° перекладки руля')

print()
print('=' * 78)
print('2. GAIN «руль → ROT» по погодным состояниям (Wave.Highest, ступени)')
print('=' * 78)
act['wh_round'] = act['Wave.Highest(метры)'].round(1)
rows = []
for wh, g in act.groupby('wh_round'):
    if len(g) < 60:
        continue
    rs = g['Rudder State(градусы)'].values
    rt = g['ROT(°/мин)'].values
    ku = (g['Wave.direction(градусы)'].values - g['Course(градусы)'].values + 180.0) % 360.0 - 180.0
    # gain: наклон ROT ~ Rudder State (если разброс руля достаточен)
    gain = np.polyfit(rs, rt, 1)[0] if np.std(rs) > 2 else np.nan
    # асимметрия по направлению поворота относительно волны:
    # правый поворот (руль > +15°) при волне с правого борта (КУ > 0) = «по направлению волны»
    right_turn_wave_right = (rs > 15) & (ku > 30)   # поворот вправо, волна справа
    right_turn_wave_left = (rs > 15) & (ku < -30)   # поворот вправо, волна слева
    rows.append({
        'h_s': wh, 'n': len(g),
        'rot_std': np.std(rt), 'roll_std': g['Roll(градусы)'].std(),
        'gain': gain,
        'rot_right_waveR': rt[right_turn_wave_right].mean() if right_turn_wave_right.sum() > 20 else np.nan,
        'rot_right_waveL': rt[right_turn_wave_left].mean() if right_turn_wave_left.sum() > 20 else np.nan,
        'asym': (rt[right_turn_wave_right].mean() - rt[right_turn_wave_left].mean())
                if (right_turn_wave_right.sum() > 20 and right_turn_wave_left.sum() > 20) else np.nan,
        'rpm_mean': g['RPM(Обороты в минуту)'].mean(),
        'sog_mean': g['SOG(узлы)'].mean(),
        'wind_force': g['Wind.Force Summary(тс)'].mean(),
    })
st = pd.DataFrame(rows)
print(st[['h_s', 'n', 'rot_std', 'roll_std', 'gain', 'rot_right_waveR', 'rot_right_waveL',
          'asym', 'rpm_mean', 'sog_mean', 'wind_force']].to_string(index=False,
          float_format=lambda x: f'{x:.3f}'))

print()
print('--- Сопоставление gain / ROT-вариация / RPM с тяжестью состояния ---')
valid_gain = st.dropna(subset=['gain'])
print(f'corr(gain, h_s):            {valid_gain["gain"].corr(valid_gain["h_s"]):+.3f}')
print(f'corr(rot_std, h_s):         {st["rot_std"].corr(st["h_s"]):+.3f}')
print(f'corr(rpm_mean, h_s):        {st["rpm_mean"].corr(st["h_s"]):+.3f}  '
      '(нагрузка двигателя vs шторм)')
print(f'corr(sog_mean, rpm_mean):   {st["sog_mean"].corr(st["rpm_mean"]):+.3f}  '
      '(скорость vs RPM)')
print(f'corr(sog_mean, h_s):        {st["sog_mean"].corr(st["h_s"]):+.3f}  '
      '(скорость падает со штормом)')
print(f'corr(asym, h_s):            {st["asym"].corr(st["h_s"]):+.3f}  '
      '(асимметрия ROT по направлению поворота vs шторм)')

print()
print('=' * 78)
print('3. Глобальные корреляции RPM / сопротивления')
print('=' * 78)
print(f'corr(RPM, h_s):                '
      f'{act["Wave.Highest(метры)"].corr(act["RPM(Обороты в минуту)"]):+.3f}')
print(f'corr(RPM, SOG):                '
      f'{act["RPM(Обороты в минуту)"].corr(act["SOG(узлы)"]):+.3f}')
print(f'corr(SOG, Wind.Force Summary): '
      f'{act["Wind.Force Summary(тс)"].corr(act["SOG(узлы)"]):+.3f}')
print(f'corr(|ROT|, |sin(КУ волны)|):  '
      f'{pd.Series(np.abs(rot)).corr(pd.Series(np.abs(np.sin(np.radians(ku_wave))), index=act.index)):+.3f}')
