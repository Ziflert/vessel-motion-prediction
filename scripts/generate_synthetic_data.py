"""
Генератор синтетических судовых данных (для исследования «объём выборки vs качество»).

Философия: НЕ имитация сложной гидродинамики, а статистический портрет реальной записи
(scripts/analyze_real_data.py): те же спектры (периоды качки), те же масштабы шумов и
приращений, те же медленные процессы среды, та же структура зависимостей (руль→ROT,
общий фактор для SOG/RPM), те же константы записи (Wave.direction=180 и т.п.).

Каналы:
  - Roll/Pitch/Vertical: затухающие стохастические осцилляторы (AR(2)) с медленно
    меняющейся амплитудой (OU-огибающая) + второй гармоники для ширины спектра;
  - Velocity.*: численные производные позиций ×60 (+ шум) — как в реальной записи;
  - Rudder Order: телеграфный процесс (перекладки), State: инерционное следование;
  - ROT: инерционный отклик на руль (лаг ~40 с) + шум; Course: интеграл ROT;
  - Среда (Wave.Highest/Swell/Wave.speed/т.д.): медленные OU-процессы;
    Wave.direction и Wind.direction — КОНСТАНТЫ (как в реальной записи!);
  - SOG/RPM: общий медленный фактор + свои шумы (корреляция ~0.24);
  - Long/Lat: интегрирование скорости по курсу; Moments/Forces: OU-заполнители.

Использование:
  python scripts/generate_synthetic_data.py --rows 30000 --seed 7
  python scripts/generate_synthetic_data.py --rows 30000 --validate
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from run_training import load_data

RAW_COLUMNS = [
    'Pitch(градусы)', 'Roll(градусы)', 'Vertical(Метр)',
    'Velocity.Pitching(°/мин)', 'Velocity.Rolling(°/мин)', 'Velocity.Vertical(узлы)',
    'Velocity.Yawing(°/мин)',
    'Swell(метры)', 'Wave.Highest(метры)', 'Wave.current(метры)',
    'Wave.direction(градусы)', 'Swell.direction(градусы)', 'Wind.direction(градусы)',
    'Current.direction(градусы)', 'Wave.speed(узлы)', 'Current.speed(узлы)',
    'Rudder Order(градусы)', 'Rudder State(градусы)', 'RPM(Обороты в минуту)',
    'Long(градусы)', 'Lat(градусы)', 'STW(узлы)', 'SOG(узлы)', 'ROT(°/мин)',
    'Course(градусы)',
    'Moment Yawing(тс*м)', 'Moment Rolling(тс*м)', 'Moment Pitching(тс*м)',
    'Force Vertical(тс)', 'Force Summary(тс)', 'Force Longitudinal(тс)', 'Force Lateral(тс)',
    'Wind.Moment Yawing(тс*м)', 'Wind.Moment Rolling(тс*м)', 'Wind.Moment Pitching(тс*м)',
    'Wind.Force Vertical(тс)', 'Wind.Force Summary(тс)', 'Wind.Force Longitudinal(тс)',
    'Wind.Force Lateral(тс)',
]


def ou_process(n, mean, std, tau_s, rng, clip=None):
    """Процесс Орнштейна–Уленбека: корреляционное время tau_s, стационарное std."""
    theta = 1.0 / max(tau_s, 1.0)
    sigma = std * np.sqrt(2 * theta)
    x = np.empty(n)
    x[0] = mean
    noise = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = x[i - 1] + theta * (mean - x[i - 1]) + sigma * np.sqrt(1.0) * noise[i]
    if clip:
        x = np.clip(x, clip[0], clip[1])
    return x


def step_process(n, real_values, mean_hold_s, rng):
    """Кусочно-ЛИНЕЙНЫЙ процесс: узлы раз в ~mean_hold_s с, между ними линейные рампы.
    Значения узлов — БУТСТРАП из эмпирического распределения реальной колонки:
    предельное распределение совпадает с реальным по построению, приращения за 1 с
    малые (без скачков) — имитация «замороженных» судовых метрик."""
    real_values = np.asarray(real_values, float)
    n_nodes = max(2, int(np.ceil(n / max(mean_hold_s, 1))) + 1)
    node_vals = rng.choice(real_values, n_nodes, replace=True)
    node_idx = np.linspace(0, n - 1, n_nodes)
    return np.interp(np.arange(n), node_idx, node_vals)


def ar2_oscillator(n, f0_hz, zeta, rng, secondary=None):
    """AR(2) резонансный осциллятор: пик спектра на f0 (Гц), коэффициент демпфирования zeta.

    secondary: (f2, доля амплитуды) — вторая компонента для ширины спектра.
    Возвращает нормированный (std=1) ряд.
    """
    def gen(f0, z):
        omega0 = 2 * np.pi * f0
        r = np.exp(-z * omega0)          # fs = 1 Гц
        psi = omega0 * np.sqrt(1 - z * z)
        phi1, phi2 = 2 * r * np.cos(psi), -(r ** 2)
        x = np.zeros(n)
        eps = rng.normal(0, 1, n)
        for i in range(2, n):
            x[i] = phi1 * x[i - 1] + phi2 * x[i - 2] + eps[i]
        return x

    x = gen(f0_hz, zeta)
    if secondary is not None:
        f2, share = secondary
        x = x + share * gen(f2, zeta * 1.5)
    return x / x.std()


def slow_envelope(n, rng, tau_s=120.0, std=0.35, lo=0.35, hi=1.9):
    """Медленная модуляция амплитуды качки (порывы волновых групп)."""
    return np.clip(ou_process(n, 1.0, std, tau_s, rng), lo, hi)


def telegraph_rudder_order(n, rng, mean_hold_s=64.0, step_std=15.0, clip=35.0):
    """Перекладки руля: интервалы ~экспоненциальные, величина — случайный шаг ±35."""
    order = np.zeros(n)
    cur = 0.0
    i = 0
    while i < n:
        hold = max(1, int(rng.exponential(mean_hold_s)))
        # 15% перекладок — возврат около нуля (как в реальной записи)
        if rng.random() < 0.15:
            cur = rng.normal(0, 2)
        else:
            cur = float(np.clip(cur + rng.normal(0, step_std), -clip, clip))
        order[i:i + hold] = cur
        i += hold
    return order


def peak_period(x, fmin=1 / 60, fmax=0.5):
    from scripts.analyze_real_data import welch_psd
    f, p = welch_psd(x)
    mask = (f >= fmin) & (f <= fmax)
    i = np.argmax(p[mask])
    fp = f[mask][i]
    return 1 / fp if fp > 0 else float('nan')


def generate(n_rows: int, seed: int = 7, real_df: pd.DataFrame = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_rows

    # --- Параметры из реальных данных ---
    r = real_df
    std_of = lambda c: float(np.std(r[c].values.astype(float)))
    mean_of = lambda c: float(np.mean(r[c].values.astype(float)))

    T_roll = peak_period(r['Roll(градусы)'].values)          # ~11.6 с
    T_pitch = peak_period(r['Pitch(градусы)'].values)        # ~9.0 с
    T_vert = peak_period(r['Vertical(Метр)'].values)         # ~10.4 с

    # --- Качка: осцилляторы с медленной огибающей ---
    def motion(T_peak, std_target, sec_share=0.35):
        osc = ar2_oscillator(n, 1 / T_peak, 0.06, rng,
                             secondary=(1.13 / T_peak, sec_share))
        env = slow_envelope(n, rng)
        x = osc * env
        return x * (std_target / x.std())

    roll = motion(T_roll, std_of('Roll(градусы)'))
    pitch = motion(T_pitch, std_of('Pitch(градусы)'))
    vert = motion(T_vert, std_of('Vertical(Метр)'))

    # --- Производные (как в записи: скорости — дифференциалы позиций) ---
    vel_pitch = np.gradient(pitch) * 60 * (std_of('Velocity.Pitching(°/мин)') /
                                           (np.gradient(pitch).std() * 60))
    vel_roll = np.gradient(roll) * 60 * (std_of('Velocity.Rolling(°/мин)') /
                                         (np.gradient(roll).std() * 60))
    vel_vert = np.gradient(vert) * 60 * (std_of('Velocity.Vertical(узлы)') /
                                         (np.gradient(vert).std() * 60))
    vel_yaw = np.gradient(np.zeros(n))  # заполняется ниже после ROT

    # --- Среда: почти «замороженные» каналы, узлы — бутстрап из реального распределения ---
    wave_h = step_process(n, r['Wave.Highest(метры)'].values, 400, rng)
    swell = step_process(n, r['Swell(метры)'].values, 400, rng)
    wave_current = step_process(n, r['Wave.current(метры)'].values, 400, rng)
    wave_speed = step_process(n, r['Wave.speed(узлы)'].values, 400, rng)

    # Константы записи (важно: в реальной записи direction-каналы НЕ меняются)
    wave_dir = np.full(n, mean_of('Wave.direction(градусы)'))
    swell_dir = np.full(n, mean_of('Swell.direction(градусы)'))
    wind_dir = np.full(n, mean_of('Wind.direction(градусы)'))
    cur_dir = np.full(n, mean_of('Current.direction(градусы)'))
    cur_speed = ou_process(n, mean_of('Current.speed(узлы)'), std_of('Current.speed(узлы)'),
                           600, rng)

    # --- Управление и навигация ---
    rudder_order = telegraph_rudder_order(n, rng)
    rudder_state = np.zeros(n)
    alpha = 1 / 8.0                      # инерция привода ~8 с (в реальной записи медленнее)
    for i in range(1, n):
        rudder_state[i] = rudder_state[i - 1] + alpha * (rudder_order[i - 1] - rudder_state[i - 1])
    rudder_state += rng.normal(0, 0.15, n)

    # ROT: инерционный отклик на руль (лаг ~40 с) + независимая рысканная качка
    rot = np.zeros(n)
    beta = 1 / 40.0
    gamma = 0.35
    for i in range(1, n):
        rot[i] = rot[i - 1] * (1 - beta) + beta * gamma * rudder_state[i - 1] * 40 / 40 \
                 + rng.normal(0, 0.4)
    rot = rot - rot.mean()
    rot = rot * (0.75 * std_of('ROT(°/мин)') / rot.std())
    yaw_osc = ar2_oscillator(n, 1 / 40.0, 0.12, rng)        # рыскание на волнении
    rot = rot + yaw_osc * (0.45 * std_of('ROT(°/мин)'))
    rot = rot * (std_of('ROT(°/мин)') / rot.std())

    course = 200.0 + np.cumsum(rot) / 60.0               # ROT °/мин → °/с
    course = course % 360.0

    # SOG: медленный процесс; RPM: точная корреляция с SOG (как в записи ~0.2)
    sog = ou_process(n, mean_of('SOG(узлы)'), std_of('SOG(узлы)'), 1800, rng)
    sog = np.clip(sog, 2.6, 9.3)
    rho = 0.20
    sog_z = (sog - sog.mean()) / sog.std()
    rpm_z = rho * sog_z + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)
    rpm = rpm_z * std_of('RPM(Обороты в минуту)') + mean_of('RPM(Обороты в минуту)')
    rpm = np.clip(rpm, 0, 91)
    vel_yaw = ou_process(n, mean_of('Velocity.Yawing(°/мин)'), std_of('Velocity.Yawing(°/мин)'),
                         5, rng)

    stw = sog + rng.normal(0, 0.2, n)                    # STW ≈ SOG (без течения в записи)

    # Координаты: интегрирование скорости по курсу (заполнитель)
    nm_per_s = sog / 3600.0
    lat = 49.7 + np.cumsum(nm_per_s * np.cos(np.deg2rad(course))) / 60.0
    lon = -9.25 + np.cumsum(nm_per_s * np.sin(np.deg2rad(course))) / 60.0

    # Моменты и силы: OU-заполнители с правильными масштабами
    fillers = {}
    for c in ['Moment Yawing(тс*м)', 'Moment Rolling(тс*м)', 'Moment Pitching(тс*м)',
              'Force Vertical(тс)', 'Force Summary(тс)', 'Force Longitudinal(тс)',
              'Force Lateral(тс)', 'Wind.Moment Yawing(тс*м)', 'Wind.Moment Rolling(тс*м)',
              'Wind.Moment Pitching(тс*м)', 'Wind.Force Vertical(тс)', 'Wind.Force Summary(тс)',
              'Wind.Force Longitudinal(тс)', 'Wind.Force Lateral(тс)']:
        fillers[c] = ou_process(n, mean_of(c), std_of(c), 30, rng)

    df = pd.DataFrame({
        'Pitch(градусы)': pitch, 'Roll(градусы)': roll, 'Vertical(Метр)': vert,
        'Velocity.Pitching(°/мин)': vel_pitch, 'Velocity.Rolling(°/мин)': vel_roll,
        'Velocity.Vertical(узлы)': vel_vert, 'Velocity.Yawing(°/мин)': vel_yaw,
        'Swell(метры)': swell, 'Wave.Highest(метры)': wave_h,
        'Wave.current(метры)': wave_current, 'Wave.direction(градусы)': wave_dir,
        'Swell.direction(градусы)': swell_dir, 'Wind.direction(градусы)': wind_dir,
        'Current.direction(градусы)': cur_dir, 'Wave.speed(узлы)': wave_speed,
        'Current.speed(узлы)': cur_speed, 'Rudder Order(градусы)': rudder_order,
        'Rudder State(градусы)': rudder_state, 'RPM(Обороты в минуту)': rpm,
        'Long(градусы)': lon, 'Lat(градусы)': lat, 'STW(узлы)': stw, 'SOG(узлы)': sog,
        'ROT(°/мин)': rot, 'Course(градусы)': course, **fillers,
    })
    return df[RAW_COLUMNS]


def validate(real: pd.DataFrame, syn: pd.DataFrame):
    """Сравнение портретов реальной и синтетической записи."""
    lines = ['=' * 76, 'ВАЛИДАЦИЯ ГЕНЕРАТОРА: реальные vs синтетические', '=' * 76, '']
    rows = []
    cols = ['Pitch(градусы)', 'Roll(градусы)', 'Vertical(Метр)', 'ROT(°/мин)',
            'SOG(узлы)', 'Rudder State(градусы)', 'Wave.Highest(метры)']
    for c in cols:
        a = real[c].values.astype(float)
        b = syn[c].values.astype(float)
        da, db = np.std(np.diff(a)), np.std(np.diff(b))
        rows.append({'column': c, 'mean_real': a.mean(), 'mean_syn': b.mean(),
                     'std_real': a.std(), 'std_syn': b.std(),
                     'std_ratio': b.std() / a.std() if a.std() > 0 else np.nan,
                     'T_peak_real': peak_period(a), 'T_peak_syn': peak_period(b),
                     'd1s_real': da, 'd1s_syn': db})
        q = rows[-1]
        lines.append(f"{c:28s} std: {q['std_real']:8.3f} -> {q['std_syn']:8.3f} "
                     f"(x{q['std_ratio']:.2f}) | T: {q['T_peak_real']:5.1f}s -> "
                     f"{q['T_peak_syn']:5.1f}s | Δ1s: {da:.3f} -> {db:.3f}")

    # Структурные зависимости
    cc_real = np.corrcoef(real['ROT(°/мин)'].values.astype(float)[40:],
                          real['Rudder State(градусы)'].values.astype(float)[:-40])[0, 1]
    cc_syn = np.corrcoef(syn['ROT(°/мин)'].values.astype(float)[40:],
                         syn['Rudder State(градусы)'].values.astype(float)[:-40])[0, 1]
    lines.append(f"corr(ROT, Rudder State, lag 40s): real {cc_real:.3f} vs syn {cc_syn:.3f}")
    cc2r = np.corrcoef(real['SOG(узлы)'].values.astype(float),
                       real['RPM(Обороты в минуту)'].values.astype(float))[0, 1]
    cc2s = np.corrcoef(syn['SOG(узлы)'].values.astype(float),
                       syn['RPM(Обороты в минуту)'].values.astype(float))[0, 1]
    lines.append(f"corr(SOG, RPM):                real {cc2r:.3f} vs syn {cc2s:.3f}")

    # Режимы волнения
    wh_r = real['Wave.Highest(метры)'].values.astype(float)
    wh_s = syn['Wave.Highest(метры)'].values.astype(float)
    lines.append(f"доля severe (>2.5 м):          real {np.mean(wh_r > 2.5) * 100:.1f}% "
                 f"vs syn {np.mean(wh_s > 2.5) * 100:.1f}%")
    report = '\n'.join(lines)
    print(report)
    return report


def main():
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        try:
            _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    parser = argparse.ArgumentParser(description='Synthetic vessel data generator')
    parser.add_argument('--rows', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--out', type=str, default='data/raw/synthetic_data.csv')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()

    real = load_data(PROJECT_ROOT / 'data' / 'raw' / 'your_data.csv')
    syn = generate(args.rows, seed=args.seed, real_df=real)

    out = PROJECT_ROOT / args.out
    syn.to_csv(out, sep='\t', index=False, encoding='utf-8')
    print(f'✓ Синтетика сохранена: {out} ({len(syn)} строк × {len(syn.columns)} колонок)')

    if args.validate:
        report = validate(real, syn)
        (out.parent / 'synthetic_validation.txt').write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
