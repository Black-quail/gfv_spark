from __future__ import annotations

"""
Write pooled PR and reliability plots for the final reported models.
"""

import os
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==================== CONFIG ====================
OOF_DIR = os.getenv("GFV_SPARK_OOF_DIR", os.path.join("path", "ensemble"))
OUT_DIR = os.path.join(OOF_DIR, "plots_best_models_pooled")
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_YEARS: Tuple[int, ...] = (2022, 2023, 2024)
BINS = 10

FILES: Dict[str, str] = {
    "E3": "ensemble_E3_oof.parquet",
    "XGB": "xgb_logit_metro_oof.parquet",
    "TCN": "oof_tcn_metroEA.parquet",
    "L2 GLM": "ml_logit_l2_metro_oof.parquet",
}
# ================================================


def _require_cols(df: pd.DataFrame, cols: Iterable[str]) -> None:
    """Check required columns."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")



def load_oof(path: str, years: Tuple[int, ...] = REPORT_YEARS) -> pd.DataFrame:
    """Load one OOF file and keep report years only."""
    df = pd.read_parquet(path)
    _require_cols(df, ["y_true", "prob", "weight", "year_snap"])

    df = df.loc[df["year_snap"].astype(int).isin(set(years))].copy()
    df["prob"] = df["prob"].astype(float)
    df["weight"] = df["weight"].astype(float)
    df["y_true"] = df["y_true"].astype(int)
    return df



def weighted_prevalence(y: np.ndarray, w: np.ndarray) -> float:
    """Exposure-weighted prevalence."""
    return float(np.sum(w * y) / np.sum(w))



def weighted_pr_curve(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build a weighted PR curve and AUPRC."""
    order = np.argsort(-p, kind="mergesort")
    y = y[order]
    w = w[order]

    tp_w = np.cumsum(w * (y == 1))
    fp_w = np.cumsum(w * (y == 0))
    p_w = np.sum(w * (y == 1))

    precision = tp_w / np.maximum(tp_w + fp_w, 1e-12)
    recall = tp_w / np.maximum(p_w, 1e-12)

    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))

    dr = np.diff(recall)
    auprc = float(np.sum(dr * (precision[1:] + precision[:-1]) * 0.5))
    return recall, precision, auprc



def reliability_bins(
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    *,
    bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Build equal-width reliability bins and pooled ECE."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(np.minimum(p, 1.0 - 1e-12), edges, right=False) - 1
    idx = np.clip(idx, 0, bins - 1)

    conf = np.full(bins, np.nan, dtype=float)
    acc = np.full(bins, np.nan, dtype=float)
    mass = np.zeros(bins, dtype=float)

    for b in range(bins):
        mask = idx == b
        if not np.any(mask):
            continue
        wb = w[mask]
        mass[b] = float(np.sum(wb))
        conf[b] = float(np.sum(wb * p[mask]) / mass[b])
        acc[b] = float(np.sum(wb * y[mask]) / mass[b])

    total_mass = float(np.sum(mass))
    ece = float(np.nansum((mass / max(total_mass, 1e-12)) * np.abs(acc - conf)))
    return conf, acc, mass, ece



def plot_pooled_pr(dfs: Dict[str, pd.DataFrame], *, out_dir: str) -> None:
    """Write the pooled PR plot."""
    fig = plt.figure()
    ax = plt.gca()

    for label, df in dfs.items():
        y = df["y_true"].to_numpy(dtype=int)
        p = df["prob"].to_numpy(dtype=float)
        w = df["weight"].to_numpy(dtype=float)
        recall, precision, auprc = weighted_pr_curve(y, p, w)
        ax.plot(recall, precision, label=f"{label} (AUPRC={auprc:.4f})")

    pi = weighted_prevalence(
        dfs["E3"]["y_true"].to_numpy(dtype=int),
        dfs["E3"]["weight"].to_numpy(dtype=float),
    )
    ax.axhline(pi, linestyle="--", linewidth=1.2, label=f"Prevalence π={pi:.4f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Pooled Precision–Recall (OOF, 2022–2024)")
    fig.tight_layout()

    fig.savefig(os.path.join(out_dir, "pr_pooled_E3_XGB_TCN_L2.png"), dpi=220)
    fig.savefig(os.path.join(out_dir, "pr_pooled_E3_XGB_TCN_L2.pdf"))
    plt.close(fig)



def plot_pooled_reliability(dfs: Dict[str, pd.DataFrame], *, out_dir: str, bins: int) -> Dict[str, float]:
    """Write the pooled reliability plot and return pooled ECE."""
    fig = plt.figure()
    ax = plt.gca()
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, label="Ideal")

    ece_out: Dict[str, float] = {}
    for label, df in dfs.items():
        y = df["y_true"].to_numpy(dtype=int)
        p = df["prob"].to_numpy(dtype=float)
        w = df["weight"].to_numpy(dtype=float)
        conf, acc, mass, ece = reliability_bins(y, p, w, bins=bins)
        ece_out[label] = ece
        mask = mass > 0
        ax.plot(
            conf[mask],
            acc[mask],
            marker="o",
            markersize=4,
            linewidth=1.2,
            label=f"{label} (ECE={ece:.4f})",
        )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Pooled Reliability (OOF, 2022–2024, {bins} bins)")
    fig.tight_layout()

    fig.savefig(os.path.join(out_dir, "reliability_pooled_E3_XGB_TCN_L2.png"), dpi=220)
    fig.savefig(os.path.join(out_dir, "reliability_pooled_E3_XGB_TCN_L2.pdf"))
    plt.close(fig)
    return ece_out



def main() -> None:
    """Load final OOF files and write pooled plots."""
    dfs: Dict[str, pd.DataFrame] = {}
    for label, fname in FILES.items():
        path = os.path.join(OOF_DIR, fname)
        dfs[label] = load_oof(path, years=REPORT_YEARS)

    plot_pooled_pr(dfs, out_dir=OUT_DIR)
    ece_out = plot_pooled_reliability(dfs, out_dir=OUT_DIR, bins=BINS)

    print(f"[plot_models] wrote plots to: {OUT_DIR}")
    for label, value in ece_out.items():
        print(f"  {label}: pooled ECE={value:.6f}")


if __name__ == "__main__":
    main()
