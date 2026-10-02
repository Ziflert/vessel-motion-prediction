"""A3.1 stage 4: power-law quality(n_train) with seed bootstrap CIs.

For each dataset x horizon, fit skill = c * n^b (linear regression in log-log
space) on aggregate seed means. Confidence intervals come from a bootstrap that
resamples seeds WITHIN each volume point (unit of replication = seed / segment
choice, not individual rows — no pseudo-replication). Exploratory only: 3 seeds
per volume point bound the resolution of the CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from a31_stage2_horizon_report import collect, OUT

RNG = np.random.default_rng(20261002)
N_BOOT = 2000


def fit_loglog(x_rows: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (slope_per_doubling, intercept at n=1) of log2-log2 fit."""
    lx = np.log2(x_rows)
    ly = np.log2(y)
    b, logc = np.polyfit(lx, ly, 1)
    return float(b), float(logc)


def powerlaw_fit(df: pd.DataFrame, metric: str) -> dict:
    """Point fit + seed bootstrap CI for one dataset x horizon group.

    metric: column name with per-run values ('skill', 'mae_over_sigma').
    """
    groups = []
    for n, g in df.groupby("n_train", sort=True):
        vals = g[metric].to_numpy(float)
        if metric == "mae_over_sigma":
            vals = g["mae_over_sigma_raw"].to_numpy(float) if "mae_over_sigma_raw" in g else vals
        groups.append((int(n), vals))
    groups.sort(key=lambda t: t[0])
    ns = np.array([n for n, _ in groups], float)
    means = np.array([v.mean() for _, v in groups], float)

    # skill must be positive for log-log; shift is not applied (all points > 0 here)
    if (means <= 0).any():
        return {"ok": False, "ns": ns.tolist(), "means": means.tolist()}

    b, logc = fit_loglog(ns, means)
    pred = lambda n, bb, lc: 2 ** (lc + bb * np.log2(n))  # noqa: E731

    boot_b, boot_c5000 = [], []
    n_seeds = min(len(v) for _, v in groups)
    for _ in range(N_BOOT):
        bs = []
        for _, v in groups:
            idx = RNG.integers(0, len(v), len(v))
            bs.append(v[idx].mean())
        bs = np.array(bs)
        if (bs <= 0).any():
            continue
        bb, llc = fit_loglog(ns, bs)
        boot_b.append(bb)
        boot_c5000.append(pred(5000, bb, llc))
    q = lambda arr, p: float(np.percentile(arr, p)) if arr else float("nan")  # noqa: E731
    return {
        "ok": True,
        "ns": ns.tolist(),
        "means": means.tolist(),
        "exponent_b": b,
        "coef_c": pred(1.0, b, logc),
        "skill_at_5000": pred(5000, b, logc),
        "b_ci95": [q(boot_b, 2.5), q(boot_b, 97.5)],
        "q5000_ci95": [q(boot_c5000, 2.5), q(boot_c5000, 97.5)],
        "n_boot_valid": len(boot_b),
        "n_seeds_per_point": n_seeds,
    }


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute per-run MAE/sigma (raw) so bootstrap works at run level."""
    rows = []
    for _, r in df.iterrows():
        per_target = r.per_target_mae
        std_test = r.test_std
        norm = float(np.mean([per_target[t] / std_test[t] for t in per_target]))
        rows.append({**r.to_dict(), "mae_over_sigma": norm})
    return pd.DataFrame(rows)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raw = prepare(collect())
    results = []
    lines = [
        "=" * 88,
        "A3.1 STAGE 4 — POWER-LAW quality(n) WITH SEED-BOOTSTRAP CIs",
        "=" * 88,
        "Fit: metric = c * n^b (log-log OLS on seed means), metric ∈ {skill, MAE/σ}.",
        "CI: bootstrap resampling of seeds WITHIN each volume point (2000 draws,",
        "percentile 95%); unit of replication = seed/segment choice, not rows.",
        "Exploratory: only 3 seeds per point — CI width reflects that limit.",
        "",
        "group                 metric      b [95% CI]          quality(5000) [95% CI]",
    ]
    for (dataset, horizon), g in raw.groupby(["dataset", "horizon"]):
        for metric, label in [("skill", "skill"), ("mae_over_sigma", "MAE/sigma")]:
            res = powerlaw_fit(g, metric)
            key = f"{dataset} H={horizon}"
            if not res["ok"]:
                lines.append(f"{key:<20} {label:<10}  — fit skipped (non-positive values)")
                continue
            results.append({"dataset": dataset, "horizon": int(horizon),
                            "metric": label, **{k: v for k, v in res.items()
                                                if k not in ("ns", "means")}})
            lines.append(
                f"{key:<20} {label:<10} "
                f"{res['exponent_b']:+.3f} [{res['b_ci95'][0]:+.3f}, {res['b_ci95'][1]:+.3f}]   "
                f"{res['skill_at_5000']:.3f} [{res['q5000_ci95'][0]:.3f}, {res['q5000_ci95'][1]:.3f}]"
            )

    lines += [
        "",
        "--- reading the exponent b ---",
        "b < 0.1  : слабый эффект объёма (почти плато)",
        "0.1–0.3  : умеренный степенной рост (типичный диапазон learning curves)",
        "b > 0.3  : сильный эффект (модель сильно недообучена на малых объёмах)",
        "Сравнение b между группами — только описательное (число сидов мало).",
        "",
        "--- interpretation boundary ---",
        "Степенной закон — эмпирическая аппроксимация на 3–4 точках объёма,",
        "не проверка механизма. CI построены бутстрепом по сидам внутри точек объёма",
        "и занижены относительно полной популяции условий; p-values не приводятся.",
        "Точки объёма на одной записи — не независимы (общий test, общие сегменты).",
        "=" * 88,
    ]
    report = "\n".join(lines)
    (OUT / "a31_stage4_powerlaw_report.txt").write_text(report, encoding="utf-8")
    pd.DataFrame(results).to_csv(OUT / "a31_stage4_powerlaw.csv", index=False)
    print(report)


if __name__ == "__main__":
    main()
