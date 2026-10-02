"""G2/N10 — метрика покрытия Ω: локальная плотность train-условий рядом с test-окном.

Гипотеза (RESEARCH_LOG §7.2, IDEAS G2): «полнота базы в похожих условиях»
(плотность train в окрестности текущего условия Ω) предсказывает ожидаемую ошибку
прогноза. Этот скрипт — зачаток: считает плотность per-test-window и раскладывает
ошибку по квартилям плотности. Связь с фактической ошибкой прогонов оценивается
позже, когда манифесты прогонов будут содержать per-window разрез (сейчас —
общая агрегатная картина по датасету).

Пространство условий Ω (стратификационные оси CONCEPT §5):
  Wave.Highest(м) | Wave.direction - Course (КУ волны) | SOG(уз)
Нормировка: min-max по объединению train+test; метрика — расстояние до k-го
ближайшего train-окна (меньше = плотнее покрытие). CPU, минуты.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

WINDOW = 120  # длина окна модели (seq), строк


def load_any(path: Path) -> pd.DataFrame:
    for enc, sep in [("utf-16", "\t"), ("utf-8", "\t"), ("utf-8", ",")]:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc)
            if df.shape[1] >= 5:
                return df
        except Exception:
            continue
    raise SystemExit(f"Не удалось прочитать {path}")


def omega_axes(df: pd.DataFrame) -> np.ndarray:
    """Нормированное пространство условий: [волнение, КУ(sin,cos), скорость]."""
    wave = df["Wave.Highest(метры)"].to_numpy(float)
    rel = np.deg2rad(df["Wave.direction(градусы)"].to_numpy(float)
                     - df["Course(градусы)"].to_numpy(float))
    sog = df["SOG(узлы)"].to_numpy(float)

    def mm(x: np.ndarray) -> np.ndarray:
        lo, hi = np.nanmin(x), np.nanmax(x)
        return (x - lo) / max(hi - lo, 1e-9)

    return np.column_stack([mm(wave), mm(sog), np.sin(rel), np.cos(rel)])


def window_density(train_df: pd.DataFrame, test_df: pd.DataFrame, k: int = 20) -> np.ndarray:
    """Для каждого test-окна (центр каждые WINDOW строк): расстояние до k-го
    ближайшего train-центра. Возвращает массив расстояний (меньше = плотнее)."""
    tr = omega_axes(train_df)
    te = omega_axes(test_df)
    tr_c = tr[::WINDOW]          # центры окон train
    te_c = te[WINDOW // 2::WINDOW]  # центры окон test
    # расстояния |te| × |tr| — датасеты ≤ 100k строк, окна ~ сотни: ок для CPU
    d = np.sqrt(((te_c[:, None, :] - tr_c[None, :, :]) ** 2).sum(-1))
    d_sorted = np.sort(d, axis=1)
    k_eff = min(k, d_sorted.shape[1])
    return d_sorted[:, k_eff - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("train_csv")
    ap.add_argument("test_csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    train = load_any(Path(args.train_csv))
    test = load_any(Path(args.test_csv))
    dens = window_density(train, test)

    lines = [
        "=" * 76,
        f"Ω-ПЛОТНОСТЬ: покрытие train-условий рядом с test-окнами (k=20-й сосед)",
        "=" * 76,
        f"train: {args.train_csv} ({len(train)} строк); test: {args.test_csv} ({len(test)} строк)",
        f"test-окон: {len(dens)}; оси Ω: волнение, скорость, КУ волны (sin/cos)",
        "",
        f"расстояние до 20-го соседа: median={np.median(dens):.4f}, "
        f"p25={np.percentile(dens, 25):.4f}, p75={np.percentile(dens, 75):.4f}, "
        f"max={dens.max():.4f}",
        "",
        "Квартили плотности (Q1 = самые плотно покрытые окна):",
    ]
    q = pd.qcut(dens, 4, labels=["Q1(плотно)", "Q2", "Q3", "Q4(редко)"])
    for lab in ["Q1(плотно)", "Q2", "Q3", "Q4(редко)"]:
        m = np.asarray(q == lab)
        lines.append(f"  {lab:<12} окон={int(m.sum()):>4}  "
                     f"диапазон d=[{dens[m].min():.4f}; {dens[m].max():.4f}]")

    lines += [
        "",
        "Следующий шаг (когда появится per-window разрез ошибки):",
        "  corr(плотность, MAE окна) — ожидание: отрицательная (плотнее база →",
        "  меньше ошибка). Валидировать на прогонах E1T/C12/C71 с разной dens.",
        "Интерпретация: d — в нормированных единицах Ω; абсолютные значения",
        "сравнимы только между окнами одной записи (одна нормировка).",
        "=" * 76,
    ]
    report = "\n".join(lines)
    print(report)
    if args.save:
        out = ROOT / "results" / "omega_density" / (
            f"{Path(args.test_csv).stem}_vs_{Path(args.train_csv).stem}{args.tag and '_' + args.tag}.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        np.save(out.with_suffix(".npy"), dens)
        print(f"Сохранено: {out} (+ .npy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
