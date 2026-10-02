"""Режимно-зависимый генератор синтетики с автокалибровкой по чанкам (IDEAS B4).

Отличие от scripts/generate_synthetic_data.py (глобальная калибровка по всей записи):
1. Автосегментация реальной записи на режимы: смены уровня Wave.Highest
   (дизайн упражнения — дискретные ступени) + склейка коротких кусков.
2. Автокалибровка параметров на каждый режим: std/период качки (Welch на куске),
   уровни среды, активность руля, средние SOG/RPM.
3. Связка «среда→качка»: амплитуда качки модулируется гладкой кривой целевых std,
   привязанных к режимам среды (без независимого блуждания — нет ложных комбинаций
   «штиль + качка 6°»).
4. Непрерывность: один непрерывный AR(2)-осциллятор на весь ряд, амплитудная
   модуляция гладкая (Hann-сглаживание границ режимов) — щелчков на стыках нет.
5. Валидация по режимам + корреляция «локальный std качки ↔ Wave.Highest».

Выход — схема профиля transas_core (совместима с конвертированными записями):
16 колонок, UTF-16 TSV.

Использование:
  .venv\\Scripts\\python.exe -X utf8 scripts/generate_regime_synthetic.py
  .venv\\Scripts\\python.exe -X utf8 scripts/generate_regime_synthetic.py --rows 30000 --seed 7
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_real_data import welch_psd
from scripts.generate_synthetic_data import (
    ar2_oscillator, ou_process, telegraph_rudder_order, peak_period,
)

DEFAULT_REAL = PROJECT_ROOT / "data/converted/real_w5w6w7_merged.csv"

OUT_COLUMNS = [
    "Roll(градусы)", "Vertical(Метр)",
    "Velocity.Rolling(°/мин)", "Velocity.Vertical(узлы)",
    "ROT(°/мин)", "SOG(узлы)",
    "Rudder Order(градусы)", "Rudder State(градусы)", "RPM(Обороты в минуту)",
    "Swell(метры)", "Wave.Highest(метры)", "Wave.speed(узлы)",
    "Wave.direction(градусы)", "Wind.direction(градусы)",
    "Current.direction(градусы)", "Current.speed(узлы)",
]

MOTION_POS = ["Roll(градусы)", "Vertical(Метр)"]          # позиции
MOTION_VEL = ["Velocity.Rolling(°/мин)", "Velocity.Vertical(узлы)"]
MPS_TO_KNOTS = 1.94384
MIN_REGIME_ROWS = 600          # короче 10 мин — не режим
TRANSITION_S = 180             # ширина сглаживания границ режимов, с


# ---------------------------------------------------------------------------
# Сегментация и калибровка
# ---------------------------------------------------------------------------

def segment_regimes(df: pd.DataFrame) -> list[dict]:
    """Режимы = непрерывные участки постоянного Wave.Highest (>= MIN_REGIME_ROWS)."""
    wh = df["Wave.Highest(метры)"].values
    change = np.flatnonzero(np.diff(wh) != 0) + 1
    bounds = np.concatenate([[0], change, [len(df)]])
    regimes = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < MIN_REGIME_ROWS:
            continue
        g = df.iloc[a:b]
        regimes.append(calibrate_regime(g, a, b))
    return regimes


def calibrate_regime(g: pd.DataFrame, a: int, b: int) -> dict:
    """Автокалибровка параметров генератора на одном режиме."""
    p = {"a": a, "b": b, "n": b - a}
    p["wave_h"] = float(np.median(g["Wave.Highest(метры)"]))
    p["wind"] = float(np.median(g["Wind.speed(узлы)"])) if "Wind.speed(узлы)" in g else 0.0
    for col in MOTION_POS + ["ROT(°/мин)", "SOG(узлы)",
                             "Rudder Order(градусы)", "Rudder State(градусы)",
                             "RPM(Обороты в минуту)"]:
        p[col] = float(g[col].std())
        p[col + " mean"] = float(g[col].mean())
    # периоды качки по режиму (Welch на куске)
    for col, key in [("Roll(градусы)", "T_roll"), ("Vertical(Метр)", "T_vert")]:
        t = peak_period(g[col].values)
        p[key] = float(t) if np.isfinite(t) else float("nan")
    # характерное время удержания руля: медиана длины пробега одинакового знака
    ro = g["Rudder Order(градусы)"].values
    sign = np.sign(ro)
    sign = sign[sign != 0]
    if len(sign) > 10:
        changes = np.flatnonzero(np.diff(sign) != 0)
        runs = np.diff(np.concatenate([[-1], changes, [len(sign) - 1]]))
        p["rudder_hold"] = float(np.median(runs)) if len(runs) else 64.0
    else:
        p["rudder_hold"] = 64.0
    return p


def smooth_steps(regimes: list[dict], n: int, key: str, default=0.0) -> np.ndarray:
    """Ступенчатая кривая значений по режимам, сглаженная Hann-окном на стыках."""
    raw = np.full(n, default, float)
    for p in regimes:
        raw[p["a"]:p["b"]] = p[key]
    k = np.hanning(TRANSITION_S * 2 + 1)
    k /= k.sum()
    pad = np.pad(raw, TRANSITION_S, mode="edge")
    return np.convolve(pad, k, mode="same")[TRANSITION_S:TRANSITION_S + n]


# ---------------------------------------------------------------------------
# Генерация
# ---------------------------------------------------------------------------

def generate(real: pd.DataFrame, seed: int = 7, rows: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regimes = segment_regimes(real)
    n = rows or sum(p["n"] for p in regimes)
    # масштабирование длительностей режимов пропорционально реальным
    total_real = sum(p["n"] for p in regimes)
    scale = n / total_real
    acc = 0.0
    for i, p in enumerate(regimes):
        a = int(round(acc))
        b = n if i == len(regimes) - 1 else int(round(acc + p["n"] * scale))
        p["a"], p["b"] = a, max(b, a + 1)
        acc += p["n"] * scale

    # --- среда: уровни по режимам (ступени дизайна упражнения) ---
    wave_h = smooth_steps(regimes, n, "wave_h")
    wind = smooth_steps(regimes, n, "wind")
    swell = np.zeros(n)                                   # в реальной записи 0
    wave_speed = 7.102 * swell                            # эмпирика your_data (R²=0.998)
    wave_dir = np.zeros(n)                                # константы записи (дизайн)
    wind_dir = np.zeros(n)
    cur_dir = np.zeros(n)
    cur_speed = np.zeros(n)

    # --- качка: непрерывный осциллятор + амплитудная модуляция по режимам ---
    # период: энерговзвешенный по энергетическим режимам; выбросы (пик в шуме
    # малоэнергетического режима, T>15 с) исключаем
    Ts, ws = [], []
    for p in regimes:
        t = p["T_roll"]
        if np.isfinite(t) and 2.0 <= t <= 15.0 and p["Roll(градусы)"] > 0.05:
            Ts.append(t)
            ws.append(p["Roll(градусы)"] ** 2)
    T_roll = float(np.average(Ts, weights=ws)) if Ts else 8.0
    Ts_v, ws_v = [], []
    for p in regimes:
        t = p["T_vert"]
        if np.isfinite(t) and 2.0 <= t <= 15.0 and p["Vertical(Метр)"] > 0.02:
            Ts_v.append(t)
            ws_v.append(p["Vertical(Метр)"] ** 2)
    T_vert = float(np.average(Ts_v, weights=ws_v)) if Ts_v else 8.0

    def motion_channel(pos_col, T):
        osc = ar2_oscillator(n, 1.0 / T, 0.06, rng, secondary=(1.13 / T, 0.35))
        osc /= osc.std()
        target_std = smooth_steps(regimes, n, pos_col)
        env = ou_process(n, 1.0, 0.15, 180, rng)          # локальная вариативность
        env = np.clip(env, 0.5, 2.0)
        return osc * target_std * env

    roll = motion_channel("Roll(градусы)", T_roll)
    vert = motion_channel("Vertical(Метр)", T_vert)

    # --- скорости = дифференциалы позиций (как в записи), масштаб по реальности ---
    vel_roll = np.gradient(roll) * 60.0
    vel_roll *= real["Velocity.Rolling(°/мин)"].std() / (vel_roll.std() + 1e-12)
    vel_vert = np.gradient(vert) * MPS_TO_KNOTS
    vel_vert *= real["Velocity.Vertical(узлы)"].std() / (vel_vert.std() + 1e-12)

    # --- управление: телеграф с режимной активностью, инерционный привод ---
    order_std = smooth_steps(regimes, n, "Rudder Order(градусы)")
    hold = float(np.mean([p["rudder_hold"] for p in regimes]))
    order = telegraph_rudder_order(n, rng, mean_hold_s=max(hold, 5.0), step_std=1.0)
    # телеграф даёт фиксированный разброс — приводим к режимной цели амплитуды:
    # нормируем и умножаем на сглаженную кривую std
    order = order / (order.std() + 1e-12) * order_std
    state = np.zeros(n)
    alpha = 1 / 8.0
    for i in range(1, n):
        state[i] = state[i - 1] + alpha * (order[i - 1] - state[i - 1])
    state += rng.normal(0, 0.15, n)

    # --- ROT: инерционный отклик на руль + рысканная качка, режимная амплитуда ---
    rot = np.zeros(n)
    beta = 1 / 40.0
    for i in range(1, n):
        rot[i] = rot[i - 1] * (1 - beta) + beta * 0.35 * state[i - 1] + rng.normal(0, 0.4)
    yaw = ar2_oscillator(n, 1 / 40.0, 0.12, rng)
    yaw /= yaw.std()
    rot_std = smooth_steps(regimes, n, "ROT(°/мин)")
    rot = rot / (rot.std() + 1e-12) * 0.75 * rot_std + yaw * 0.45 * rot_std
    rot = rot * (real["ROT(°/мин)"].std() / (rot.std() + 1e-12))

    # --- курс: интеграл ROT (нужен синусоидальный маневровый дрейф) ---
    course = (200.0 + np.cumsum(rot) / 60.0) % 360.0

    # --- SOG/RPM: режимные средние + короткомасштабная вариативность ---
    # Медленная компонента SOG = режимное среднее (ступень); OU-остаток
    # высокочастотно детрендируем, иначе длинный tau уплывает от цели на 1+ уз
    # внутри короткого режима.
    sog_mean = smooth_steps(regimes, n, "SOG(узлы) mean")
    sog_std = smooth_steps(regimes, n, "SOG(узлы)")
    sog_raw = ou_process(n, 0.0, 1.0, 600, rng)
    slow = pd.Series(sog_raw).rolling(1200, center=True, min_periods=300).mean().bfill().ffill().values
    sog = sog_mean + (sog_raw - slow) * sog_std
    sog = np.clip(sog, 2.0, 9.5)
    rho = float(np.corrcoef(real["SOG(узлы)"], real["RPM(Обороты в минуту)"])[0, 1])
    rpm_mean = smooth_steps(regimes, n, "RPM(Обороты в минуту) mean")
    rpm_std = smooth_steps(regimes, n, "RPM(Обороты в минуту)")
    sog_z = (sog - sog.mean()) / (sog.std() + 1e-12)
    rpm = (rho * sog_z + np.sqrt(1 - rho ** 2) * rng.normal(0, 1, n)) * rpm_std + rpm_mean

    df = pd.DataFrame({
        "Roll(градусы)": roll, "Vertical(Метр)": vert,
        "Velocity.Rolling(°/мин)": vel_roll, "Velocity.Vertical(узлы)": vel_vert,
        "ROT(°/мин)": rot, "SOG(узлы)": sog,
        "Rudder Order(градусы)": order, "Rudder State(градусы)": state,
        "RPM(Обороты в минуту)": rpm,
        "Swell(метры)": swell, "Wave.Highest(метры)": wave_h,
        "Wave.speed(узлы)": wave_speed, "Wave.direction(градусы)": wave_dir,
        "Wind.direction(градусы)": wind_dir, "Current.direction(градусы)": cur_dir,
        "Current.speed(узлы)": cur_speed,
    })
    return df[OUT_COLUMNS]


# ---------------------------------------------------------------------------
# Валидация
# ---------------------------------------------------------------------------

def local_std(x: np.ndarray, win: int = 120) -> np.ndarray:
    return pd.Series(x).rolling(win, center=True, min_periods=win // 2).std().values


def validate(real: pd.DataFrame, syn: pd.DataFrame) -> str:
    lines = []
    say = lines.append
    say("=" * 78)
    say("ВАЛИДАЦИЯ режимного генератора: реальные w-5/6/7 (merged) vs синтетика")
    say("=" * 78)

    say("\n--- По режимам (режим = уровень Wave.Highest, как в реальной записи) ---")
    hdr = (f"{'режим':>14} {'n':>7} | {'Roll std':>9} {'T_roll':>7} | {'Vert std':>9} "
           f"{'T_vert':>7} | {'ROT std':>8} {'SOG mean':>9} {'RPM mean':>9}")
    say(hdr + "\n" + "-" * len(hdr))
    regimes = segment_regimes(real)
    for p in regimes:
        a, b = p["a"], p["b"]
        g, s = real.iloc[a:b], syn.iloc[a:b]
        say(f"{p['wave_h']:>10.1f} м {p['n']:>7} | "
            f"{g['Roll(градусы)'].std():>7.3f}/{s['Roll(градусы)'].std():<7.3f}"
            f" {peak_period(g['Roll(градусы)'].values):>6.1f}/{peak_period(s['Roll(градусы)'].values):<6.1f} | "
            f"{g['Vertical(Метр)'].std():>7.3f}/{s['Vertical(Метр)'].std():<7.3f}"
            f" {peak_period(g['Vertical(Метр)'].values):>6.1f}/{peak_period(s['Vertical(Метр)'].values):<6.1f} | "
            f"{g['ROT(°/мин)'].std():>7.2f}/{s['ROT(°/мин)'].std():<7.2f} "
            f"{g['SOG(узлы)'].mean():>8.2f}/{s['SOG(узлы)'].mean():<8.2f} "
            f"{g['RPM(Обороты в минуту)'].mean():>8.1f}/{s['RPM(Обороты в минуту)'].mean():<8.1f}")
    say("   (формат: реал/синт)")

    say("\n--- Связка «среда→качка»: corr(локальный std Roll, Wave.Highest) ---")
    for name, d in [("реал", real), ("синт", syn)]:
        ls = local_std(d["Roll(градусы)"].values)
        m = ~np.isnan(ls)
        c = np.corrcoef(ls[m], d["Wave.Highest(метры)"].values[m])[0, 1]
        say(f"  {name}: r = {c:+.3f}")

    say("\n--- Глобальные спектры (периоды, с) ---")
    for col in MOTION_POS:
        tr, ts = peak_period(real[col].values), peak_period(syn[col].values)
        say(f"  {col:22s}: реал {tr:5.1f} | синт {ts:5.1f}")

    say("\n--- Управление/навигация (глобально) ---")
    for col in ["Rudder Order(градусы)", "Rudder State(градусы)", "ROT(°/мин)",
                "SOG(узлы)", "RPM(Обороты в минуту)"]:
        say(f"  {col:26s}: std реал {real[col].std():8.3f} | синт {syn[col].std():8.3f}")

    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real", default=str(DEFAULT_REAL))
    ap.add_argument("--out", default="data/converted/synthetic_regime.csv")
    ap.add_argument("--rows", type=int, default=None,
                    help="длина синтетики (по умолчанию — зеркало реальной записи)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report", default="results/regime_generator_validation.txt")
    args = ap.parse_args(argv)

    real = pd.read_csv(args.real, sep="\t", encoding="utf-16")
    syn = generate(real, seed=args.seed, rows=args.rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    syn.to_csv(out, sep="\t", index=False, encoding="utf-16")
    print(f"Синтетика: {out} — {syn.shape[0]} строк, {syn.shape[1]} колонок")

    report = validate(real, syn)
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(report, encoding="utf-8")
    print(report)
    print(f"Отчёт: {rp}")


if __name__ == "__main__":
    main()
