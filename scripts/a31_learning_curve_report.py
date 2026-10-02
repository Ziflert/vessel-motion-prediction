"""A3.1: matched learning-curve comparison for 12k and 71k records.

Uses manifests only. Compares the common transas_core target panel at H=20:
12k A3.1 runs (K=2,4,8) versus 71k E1T runs (K=2,4,8,16,FULL).
Reports mean +/- SD over seeds, per-target MAE, MAE normalized by fixed test
standard deviation, Spearman trends, and an exploratory log-volume slope.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress

import registry as reg

OUT = ROOT / "results" / "a31_learning_curve"
TARGETS = [
    "Roll(градусы)",
    "Vertical(Метр)",
    "Velocity.Rolling(°/мин)",
    "Velocity.Vertical(узлы)",
]


def load_test_std(manifest: dict) -> dict[str, float]:
    """Compute test target SD from manifest test ranges and source CSV."""
    data = manifest["data"]
    path = ROOT / Path(data["path"])
    if not path.exists():
        path = ROOT / Path(data["path"].replace("\\", "/"))
    df = pd.read_csv(path, sep="\t", encoding="utf-16")
    ranges = data.get("test_row_ranges") or []
    if not ranges:
        return {t: np.nan for t in TARGETS}
    idx = np.concatenate([np.arange(int(a), int(b)) for a, b in ranges])
    test = df.iloc[idx]
    return {t: float(test[t].std(ddof=1)) for t in TARGETS}


def collect() -> pd.DataFrame:
    rows = []
    for item in reg.list_runs():
        m = item.get("manifest")
        if not m:
            continue
        notes = m.get("hypothesis") or ""
        if notes.startswith("A3.1 12k-core"):
            dataset = "12k"
        elif notes.startswith("E1T transas-real"):
            dataset = "71k"
        else:
            continue
        if m.get("config", {}).get("prediction_horizon") not in (None, 20):
            continue
        result = m.get("results") or {}
        phys = result.get("physical") or {}
        overall = phys.get("overall") or {}
        per_target = phys.get("per_target") or {}
        if not overall:
            continue
        data = m.get("data") or {}
        std_test = load_test_std(m)
        rows.append({
            "dataset": dataset,
            "run_id": m["run_id"],
            "seed": (m.get("training") or {}).get("seed"),
            "n_train": int(data.get("rows_train", 0)),
            "mae": float(overall["mae"]),
            "r2": float(overall["r2"]),
            "skill": float((result.get("skill_vs_persistence") or {}).get("overall")),
            "per_target_mae": {t: float(per_target[t]["mae"]) for t in TARGETS},
            "test_std": std_test,
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (dataset, n), g in df.groupby(["dataset", "n_train"], sort=True):
        per_target = {t: np.array([x[t] for x in g.per_target_mae]) for t in TARGETS}
        per_std = {t: np.array([x[t] for x in g.test_std]) for t in TARGETS}
        norm = np.array([
            np.mean([per_target[t][i] / per_std[t][i] for t in TARGETS])
            for i in range(len(g))
        ])
        row = {
            "dataset": dataset, "n_train": int(n), "n_seeds": len(g),
            "mae_mean": g.mae.mean(), "mae_sd": g.mae.std(ddof=1) if len(g) > 1 else 0.0,
            "r2_mean": g.r2.mean(), "r2_sd": g.r2.std(ddof=1) if len(g) > 1 else 0.0,
            "skill_mean": g.skill.mean(), "skill_sd": g.skill.std(ddof=1) if len(g) > 1 else 0.0,
            "mae_over_sigma_mean": norm.mean(),
            "mae_over_sigma_sd": norm.std(ddof=1) if len(norm) > 1 else 0.0,
        }
        for t in TARGETS:
            short = t.split("(")[0].replace("Velocity.", "Vel_")
            vals = per_target[t]
            row[f"mae_{short}_mean"] = vals.mean()
            row[f"mae_{short}_sd"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            row[f"mae_sigma_{short}_mean"] = np.mean(vals / per_std[t])
        out.append(row)
    return pd.DataFrame(out).sort_values(["dataset", "n_train"])


def trend_table(agg: pd.DataFrame) -> list[str]:
    lines = []
    for dataset, g in agg.groupby("dataset"):
        x = np.log2(g.n_train.to_numpy(float))
        lines.append(f"{dataset}: {len(g)} aggregate volume points")
        for col, label in [("skill_mean", "skill"), ("mae_over_sigma_mean", "MAE/sigma")]:
            y = g[col].to_numpy(float)
            rho, p = spearmanr(g.n_train, y)
            fit = linregress(x, y)
            # 3–5 aggregate volume points and shared source records do not support a
            # reliable frequentist p-value; report direction/effect scale only.
            lines.append(
                f"  {label}: Spearman rho={rho:+.3f}; "
                f"slope per doubling={fit.slope:+.4f}, exploratory R2={fit.rvalue**2:.3f}; "
                "p-value not reported (too few non-independent aggregate points)"
            )
    return lines


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    raw = collect()
    if raw.empty:
        raise SystemExit("No A3.1/E1T manifests found")
    raw.to_json(OUT / "a31_matched_runs.json", orient="records", force_ascii=False, indent=2)
    agg = aggregate(raw)
    agg.to_csv(OUT / "a31_matched_overall.csv", index=False)

    lines = [
        "=" * 88,
        "A3.1 — MATCHED LEARNING CURVE: ОБЪЁМ ДАННЫХ → КАЧЕСТВО ВНУТРИ ОБЛАСТИ ОБУЧЕНИЯ",
        "=" * 88,
        "Система: одно судно, один симулятор, одна категория загрузки; H=20, transas_core.",
        "Оценка: тестовые сегменты фиксированы внутри каждого датасета; mean±SD по сидами.",
        "12k = your_data.csv, A3.1; 71k = real_w5w6w7_merged.csv, E1T.",
        "Общая панель целей: Roll, Vertical, Velocity.Rolling, Velocity.Vertical.",
        "MAE/sigma = macro-average per-target MAE, делённая на SD соответствующей test-цели.",
        "Важно: test-панели 12k и 71k различаются по режимам; это стратификационный фактор.",
        "Поэтому междатасетное сравнение — описательное, не причинное.",
        "",
        "--- aggregate table ---",
        "dataset  n_train  seeds  MAE mean±SD       R2 mean±SD       skill mean±SD     MAE/sigma mean±SD",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"{r.dataset:>7} {int(r.n_train):>8} {int(r.n_seeds):>6}  "
            f"{r.mae_mean:.3f}±{r.mae_sd:.3f}   "
            f"{r.r2_mean:+.3f}±{r.r2_sd:.3f}   "
            f"{r.skill_mean:+.3f}±{r.skill_sd:.3f}   "
            f"{r.mae_over_sigma_mean:.3f}±{r.mae_over_sigma_sd:.3f}"
        )
    lines += ["", "--- trend diagnostics (aggregate points; exploratory) ---"]
    lines += trend_table(agg)
    lines += [
        "", "--- interpretation boundary ---",
        "Сейчас доступна описательная статистика: 12k имеет 3 объёма × 3 сида,",
        "71k — 4 объёма × 3 сида плюс FULL с одним сидом. Этого достаточно для оценки",
        "направления и приблизительного насыщения, но недостаточно для сильного вывода",
        "об эффекте объёма отдельно от режима. Для подтверждения нужны одинаковые split,",
        "общие цели, одинаковые H={10,20,30}, режимная стратификация и дополнительные повторы.",
        "Не считать отдельные окна внутри одной записи независимыми репликами: единица",
        "репликации здесь — seed/выбор train-сегментов, а не отдельная строка временного ряда.",
        "=" * 88,
    ]
    (OUT / "a31_matched_report.txt").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.alpha": .3,
                         "figure.dpi": 200, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    colors = {"12k": "#0072B2", "71k": "#D55E00"}
    labels = {"12k": "12k: A3.1", "71k": "71k: E1T"}
    for dataset, g in agg.groupby("dataset"):
        style = dict(color=colors[dataset], marker="o", lw=1.6, capsize=3)
        axes[0].errorbar(g.n_train, g.skill_mean, yerr=g.skill_sd, label=labels[dataset], **style)
        axes[1].errorbar(g.n_train, g.mae_over_sigma_mean, yerr=g.mae_over_sigma_sd,
                          label=labels[dataset], **style)
    for ax, ylabel in [(axes[0], "Skill vs persistence"), (axes[1], "MAE / test SD (macro over 4 targets)")]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Train volume, rows (log₂ scale)")
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[0].axhline(0, color="0.4", ls=":", lw=1)
    fig.suptitle("A3.1: quality vs training volume within covered regimes (H=20)")
    fig.savefig(OUT / "a31_learning_curve_matched.png")
    fig.savefig(OUT / "a31_learning_curve_matched.pdf")
    print("\n".join(lines))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
