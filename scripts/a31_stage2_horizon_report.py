"""A3.1 stage 2: matched learning curve across horizons H={10,20,30}.

Uses manifests only. All runs share the common transas_core target panel on the
12k record (your_data.csv): H=20 from stage 1, H=10 and H=30 from stage 2.
Question: does the quality(volume) effect persist across horizons?
Reports mean +/- SD over seeds, per-horizon trends, and slope comparison.
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
from a31_learning_curve_report import TARGETS, load_test_std

OUT = ROOT / "results" / "a31_learning_curve"
HORIZONS = [10, 20, 30]


def collect() -> pd.DataFrame:
    rows = []
    for item in reg.list_runs():
        m = item.get("manifest")
        if not m:
            continue
        notes = m.get("hypothesis") or ""
        if notes.startswith("A3.1 12k-core"):
            dataset = "12k"
        elif notes.startswith("A3.1 71k-core") or notes.startswith("E1T transas-real"):
            dataset = "71k"
        else:
            continue
        h = (m.get("model") or {}).get("prediction_horizon")
        if h not in HORIZONS:
            continue
        result = m.get("results") or {}
        phys = result.get("physical") or {}
        overall = phys.get("overall") or {}
        per_target = phys.get("per_target") or {}
        if not overall:
            continue
        std_test = load_test_std(m)
        rows.append({
            "dataset": dataset,
            "horizon": int(h),
            "run_id": m["run_id"],
            "seed": (m.get("training") or {}).get("seed"),
            "n_train": int((m.get("data") or {}).get("rows_train", 0)),
            "mae": float(overall["mae"]),
            "r2": float(overall["r2"]),
            "skill": float((result.get("skill_vs_persistence") or {}).get("overall")),
            "per_target_mae": {t: float(per_target[t]["mae"]) for t in TARGETS},
            "test_std": std_test,
        })
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (dataset, h, n), g in df.groupby(["dataset", "horizon", "n_train"], sort=True):
        per_target = {t: np.array([x[t] for x in g.per_target_mae]) for t in TARGETS}
        per_std = {t: np.array([x[t] for x in g.test_std]) for t in TARGETS}
        norm = np.array([
            np.mean([per_target[t][i] / per_std[t][i] for t in TARGETS])
            for i in range(len(g))
        ])
        row = {
            "dataset": dataset,
            "horizon": int(h), "n_train": int(n), "n_seeds": len(g),
            "mae_mean": g.mae.mean(), "mae_sd": g.mae.std(ddof=1) if len(g) > 1 else 0.0,
            "r2_mean": g.r2.mean(), "r2_sd": g.r2.std(ddof=1) if len(g) > 1 else 0.0,
            "skill_mean": g.skill.mean(), "skill_sd": g.skill.std(ddof=1) if len(g) > 1 else 0.0,
            "mae_over_sigma_mean": norm.mean(),
            "mae_over_sigma_sd": norm.std(ddof=1) if len(norm) > 1 else 0.0,
        }
        out.append(row)
    return pd.DataFrame(out).sort_values(["dataset", "horizon", "n_train"])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    raw = collect()
    if raw.empty:
        raise SystemExit("No A3.1 12k-core manifests found")
    raw.to_json(OUT / "a31_stage2_runs.json", orient="records", force_ascii=False, indent=2)
    agg = aggregate(raw)
    agg.to_csv(OUT / "a31_stage2_overall.csv", index=False)

    lines = [
        "=" * 88,
        "A3.1 STAGE 2+3 — LEARNING CURVE ACROSS HORIZONS H={10,20,30}, 12k AND 71k",
        "=" * 88,
        "Common transas_core target panel: Roll, Vertical, Vel.Rolling, Vel.Vertical.",
        "12k (your_data.csv): H=20 stage 1, H=10/30 stage 2, K={2,4,8}.",
        "71k (real_w5w6w7_merged.csv): H=20 E1T, H=10/30 stage 3, K={2,4,8,16}.",
        "Question: does the quality(volume) effect persist across horizons and regimes?",
        "",
        "--- aggregate table (per dataset x horizon) ---",
        "DS   H    n_train  seeds  MAE mean±SD       R2 mean±SD       skill mean±SD     MAE/sigma mean±SD",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"{r.dataset:>4} {int(r.horizon):>3} {int(r.n_train):>8} {int(r.n_seeds):>6}  "
            f"{r.mae_mean:.3f}±{r.mae_sd:.3f}   "
            f"{r.r2_mean:+.3f}±{r.r2_sd:.3f}   "
            f"{r.skill_mean:+.3f}±{r.skill_sd:.3f}   "
            f"{r.mae_over_sigma_mean:.3f}±{r.mae_over_sigma_sd:.3f}"
        )

    lines += ["", "--- trend diagnostics per dataset x horizon (aggregate points; exploratory) ---"]
    trends = {}
    for (dataset, h), g in agg.groupby(["dataset", "horizon"]):
        x = np.log2(g.n_train.to_numpy(float))
        trends[(dataset, int(h))] = {}
        lines.append(f"{dataset} H={h}: {len(g)} aggregate volume points")
        for col, label in [("skill_mean", "skill"), ("mae_over_sigma_mean", "MAE/sigma")]:
            y = g[col].to_numpy(float)
            rho, _ = spearmanr(g.n_train, y)
            fit = linregress(x, y)
            trends[(dataset, int(h))][label] = fit.slope
            lines.append(
                f"  {label}: Spearman rho={rho:+.3f}; slope per doubling={fit.slope:+.4f}, "
                f"exploratory R2={fit.rvalue**2:.3f}"
            )

    lines += ["", "--- horizon comparison of volume slopes ---"]
    for label in ("skill", "MAE/sigma"):
        for dataset in sorted(agg.dataset.unique()):
            slopes = {h: v[label] for (ds, h), v in trends.items() if ds == dataset}
            desc = "; ".join(f"H={h}: {val:+.4f}/doubling" for h, val in sorted(slopes.items()))
            sign_consistent = len({np.sign(v) for v in slopes.values()}) == 1
            lines.append(f"  {dataset} {label}: {desc} -> sign-consistent: {sign_consistent}")

    lines += [
        "", "--- interpretation boundary ---",
        "Описательная статистика: 12k — 3 объёма × 3 сида на горизонт; 71k — 4 объёма × 3 сида.",
        "Достаточно для проверки направления и знакового согласования наклонов;",
        "недостаточно для точного сравнения величин наклонов и причинного вывода.",
        "Единица репликации — seed/выбор train-сегментов, не отдельные строки ряда.",
        "Абсолютный MAE между датасетами/горизонтами несопоставим напрямую; основные",
        "межгрупповые показатели — skill vs persistence и MAE/sigma. Test-панели 12k и 71k",
        "относятся к разным режимам волнения — режимы считать стратификационным фактором.",
        "=" * 88,
    ]
    report = "\n".join(lines)
    (OUT / "a31_stage2_report.txt").write_text(report, encoding="utf-8")
    print(report)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.alpha": .3,
                         "figure.dpi": 200, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    colors = {10: "#0072B2", 20: "#009E73", 30: "#D55E00"}
    lstyles = {"12k": "-", "71k": "--"}
    for (dataset, h), g in agg.groupby(["dataset", "horizon"]):
        style = dict(color=colors[int(h)], linestyle=lstyles[dataset], marker="o",
                     lw=1.6, capsize=3)
        axes[0].errorbar(g.n_train, g.skill_mean, yerr=g.skill_sd,
                         label=f"{dataset} H={h}", **style)
        axes[1].errorbar(g.n_train, g.mae_over_sigma_mean, yerr=g.mae_over_sigma_sd,
                         label=f"{dataset} H={h}", **style)
    for ax, ylabel in [(axes[0], "Skill vs persistence"),
                       (axes[1], "MAE / test SD (macro over 4 targets)")]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Train volume, rows (log₂ scale)")
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[0].axhline(0, color="0.4", ls=":", lw=1)
    fig.suptitle("A3.1: quality vs volume across horizons, 12k vs 71k (common panel)")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"a31_stage2_horizons.{ext}")


if __name__ == "__main__":
    main()
