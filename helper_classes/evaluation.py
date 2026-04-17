from __future__ import annotations

"""Shared evaluation helpers for model runs and OOF summaries."""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


# =================== PRINTING ====================
def print_run_config(
    model_name: str,
    weekly: pd.DataFrame,
    config: Dict,
    *,
    feature_cols: Sequence[str],
) -> None:
    """Print a compact run banner and weekly-panel summary."""
    required = ["metro", "weight"]
    missing = [c for c in required if c not in weekly.columns]
    if missing:
        raise KeyError(f"print_run_config: missing columns: {missing}")

    n_rows = int(len(weekly))
    metros = int(weekly["metro"].nunique(dropna=True))

    w = pd.to_numeric(weekly["weight"], errors="coerce").fillna(0.0).astype(float)
    obs = w > 0

    n_rows_obs = int(obs.sum())
    metros_obs = int(weekly.loc[obs, "metro"].nunique(dropna=True))

    years_all: List[int] = []
    if "year_snap" in weekly.columns:
        ys_all = pd.to_numeric(weekly["year_snap"], errors="coerce")
        years_all = sorted([int(x) for x in ys_all.dropna().unique()])

    n_obs_with_target = None
    if "p_tilde_lead1" in weekly.columns:
        tgt = pd.to_numeric(weekly["p_tilde_lead1"], errors="coerce")
        n_obs_with_target = int((obs & tgt.notna()).sum())

    label_prevalence = None
    n_obs_with_label = None
    if "flu_label" in weekly.columns:
        lab = pd.to_numeric(weekly["flu_label"], errors="coerce")
        keep = obs & lab.isin([0.0, 1.0])
        n_obs_with_label = int(keep.sum())
        if n_obs_with_label > 0:
            label_prevalence = float(lab.loc[keep].mean())

    print("===== SPARK_ILI: run configuration =====")
    print(f"MODEL={model_name}")

    for k in [
        "DATA_DIR",
        "OOF_DIR",
        "VAL_YEARS",
        "SEED",
        "Q_EVENT",
        "POST_START_YEAR",
        "POST_PANDEMIC_START",
        "SEQ_LEN_WEEKS",
        "MIN_OBS_WEEKS",
        "MIN_WEEKS_PER_METRO",
        "ECE_BINS",
    ]:
        if k in config:
            print(f"{k}={config[k]}")

    print(f"FEATURE_COLS ({len(feature_cols)}): " + ", ".join(map(str, feature_cols)))
    print("----- Weekly panel diagnostics -----")
    print(f"Weekly table: rows={n_rows:,}  metros={metros:,}")
    print(f"Observed (weight>0): rows={n_rows_obs:,}  metros={metros_obs:,}")

    if years_all:
        print("Fold key: year_snap = ISO year of TARGET week (target_week_start)")

    if n_obs_with_target is not None:
        print(f"Observed rows with finite target (p_tilde_lead1): {n_obs_with_target:,}")

    if n_obs_with_label is not None:
        print(f"Observed rows with binary label (flu_label): {n_obs_with_label:,}")
        if label_prevalence is not None:
            print(f"Label prevalence on observed+binary (unweighted): {label_prevalence:.4f}")

    print("=======================================")


# ==================== TABLE ====================
def format_metrics_table(
    rows: List[Dict[str, float]],
    *,
    index_key: str = "model",
    metric_order: Sequence[str] = ("PREV", "AUPRC", "ECE", "AUROC", "Brier", "ACC", "F1", "MCC"),
    digits: int = 4,
) -> str:
    """Format a compact metrics table."""
    df = pd.DataFrame(rows)
    if index_key not in df.columns:
        raise KeyError(f"format_metrics_table: missing {index_key!r}")

    required = [index_key, *[m for m in metric_order if m != index_key]]
    if "AUPRC" in metric_order:
        required += ["AUPRC_LO", "AUPRC_HI"]
    if "ECE" in metric_order:
        required += ["ECE_LO", "ECE_HI"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"format_metrics_table: missing columns: {missing}")

    for c in df.columns:
        if c != index_key:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    bad = {}
    for c in [x for x in required if x != index_key]:
        mask = ~np.isfinite(df[c].to_numpy(dtype=float))
        if bool(mask.any()):
            bad[c] = df.loc[mask, index_key].astype(str).tolist()[:10]
    if bad:
        raise ValueError(f"format_metrics_table: non-finite values: {bad}")

    def _fmt(x: float) -> str:
        return f"{float(x):.{int(digits)}f}"

    if "AUPRC" in metric_order:
        df["AUPRC_CI"] = df.apply(
            lambda r: f"{_fmt(r['AUPRC'])} [{_fmt(r['AUPRC_LO'])}, {_fmt(r['AUPRC_HI'])}]",
            axis=1,
        )
    if "ECE" in metric_order:
        df["ECE_CI"] = df.apply(
            lambda r: f"{_fmt(r['ECE'])} [{_fmt(r['ECE_LO'])}, {_fmt(r['ECE_HI'])}]",
            axis=1,
        )

    display_cols = [index_key]
    for m in metric_order:
        if m == "AUPRC":
            display_cols.append("AUPRC_CI")
        elif m == "ECE":
            display_cols.append("ECE_CI")
        else:
            display_cols.append(m)

    for c in ["N_USED", "W_SUM", "N_CLUSTERS", "N_BOOT_USED"]:
        if c in df.columns and c not in display_cols:
            display_cols.append(c)

    df_disp = df.loc[:, display_cols].copy()
    num_cols = df_disp.select_dtypes(include=["number"]).columns
    df_disp.loc[:, num_cols] = df_disp.loc[:, num_cols].round(int(digits))
    return df_disp.to_string(index=False)


# ==================== CV PAIRS ====================
def cv_pairs(
    years_obs: Iterable[int],
    *,
    min_train_years: int,
    val_years: Iterable[int],
) -> List[Tuple[List[int], int]]:
    """Build expanding-origin train/validation year pairs."""
    ys = sorted({int(y) for y in years_obs})
    vyears = sorted({int(y) for y in val_years})

    if not ys:
        raise ValueError("cv_pairs: empty years_obs")
    if not vyears:
        raise ValueError("cv_pairs: empty val_years")
    if int(min_train_years) <= 0:
        raise ValueError("cv_pairs: min_train_years must be >= 1")

    pairs: List[Tuple[List[int], int]] = []
    for vy in vyears:
        if vy not in ys:
            raise ValueError(f"cv_pairs: val_year {vy} not in observed years")

        train = [y for y in ys if y < vy]
        if len(train) < int(min_train_years):
            raise ValueError(f"cv_pairs: not enough training years for val_year={vy}")
        pairs.append((train, vy))

    return pairs


# ==================== CALIBRATION ====================
def ece(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    n_bins: int = 10,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Expected calibration error with fixed-width bins."""
    y = np.asarray(labels, dtype=float).ravel()
    p = np.asarray(probs, dtype=float).ravel()

    if y.shape[0] != p.shape[0]:
        raise ValueError("ece: length mismatch")
    if not np.all(np.isfinite(p)):
        raise ValueError("ece: probs must be finite")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("ece: probs must be in [0,1]")
    if not np.all(np.isfinite(y)):
        raise ValueError("ece: labels must be finite")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("ece: labels must be binary")

    y = y.astype(int, copy=False)

    if sample_weight is None:
        w = np.ones_like(p, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float).ravel()
        if w.shape[0] != p.shape[0]:
            raise ValueError("ece: weight length mismatch")
        if not np.all(np.isfinite(w)):
            raise ValueError("ece: weights must be finite")
        if np.any(w < 0.0):
            raise ValueError("ece: weights must be nonnegative")

    total_w = float(w.sum())
    if total_w <= 0.0:
        raise ValueError("ece: total weight must be > 0")

    nb = int(n_bins)
    if nb <= 0:
        raise ValueError("ece: n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, nb + 1)
    out = 0.0

    for i in range(nb):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == nb - 1 else (p >= lo) & (p < hi)
        if not np.any(mask):
            continue

        w_bin = float(w[mask].sum())
        if w_bin <= 0.0:
            continue

        conf = float(np.average(p[mask], weights=w[mask]))
        acc = float(np.average(y[mask], weights=w[mask]))
        out += (w_bin / total_w) * abs(acc - conf)

    return float(out)


# ==================== BOOTSTRAP ====================
@dataclass(frozen=True)
class BootstrapResult:
    mean: float
    lower: float
    upper: float
    n_boot: int
    ci_level: float
    seed: int
    n_used: int
    n_clusters: int
    n_boot_used: int


def bootstrap_ci(
    *,
    metro: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> BootstrapResult:
    """Percentile cluster bootstrap by metro."""
    nb = int(n_boot)
    if nb <= 0:
        raise ValueError("bootstrap_ci: n_boot must be >= 1")

    cl = float(ci_level)
    if not (0.0 < cl < 1.0):
        raise ValueError("bootstrap_ci: ci_level must be in (0,1)")

    sd = int(seed)
    m = np.asarray(metro).ravel()
    n = int(m.shape[0])
    if n == 0:
        return BootstrapResult(
            mean=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n_boot=nb,
            ci_level=cl,
            seed=sd,
            n_used=0,
            n_clusters=0,
            n_boot_used=0,
        )

    u = np.unique(m)
    G = int(u.size)
    if G <= 0:
        return BootstrapResult(
            mean=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
            n_boot=nb,
            ci_level=cl,
            seed=sd,
            n_used=n,
            n_clusters=0,
            n_boot_used=0,
        )

    idx_by_g = {g: np.flatnonzero(m == g) for g in u}
    rng = np.random.default_rng(sd)

    base_idx = np.arange(n, dtype=np.int64)
    base_stat = float(stat_fn(base_idx))

    boots = np.full(nb, np.nan, dtype=float)
    for b in range(nb):
        sampled = rng.choice(u, size=G, replace=True)
        parts = [idx_by_g[g] for g in sampled]
        idx = np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.int64)
        if idx.size == 0:
            continue
        boots[b] = float(stat_fn(idx))

    boots_f = boots[np.isfinite(boots)]
    if boots_f.size == 0:
        return BootstrapResult(
            mean=base_stat,
            lower=float("nan"),
            upper=float("nan"),
            n_boot=nb,
            ci_level=cl,
            seed=sd,
            n_used=n,
            n_clusters=G,
            n_boot_used=0,
        )

    alpha = 1.0 - cl
    lo, hi = np.quantile(boots_f, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapResult(
        mean=base_stat,
        lower=float(lo),
        upper=float(hi),
        n_boot=nb,
        ci_level=cl,
        seed=sd,
        n_used=n,
        n_clusters=G,
        n_boot_used=int(boots_f.size),
    )


# ==================== METRIC HELPERS ====================
def _require_binary_labels(y: np.ndarray, *, name: str = "y") -> np.ndarray:
    y = np.asarray(y).ravel()
    if y.size == 0:
        raise ValueError(f"{name}: empty")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name}: non-finite values")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError(f"{name}: labels must be binary")
    return y.astype(int, copy=False)


def _require_probs(p: np.ndarray, *, name: str = "p") -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    if p.size == 0:
        raise ValueError(f"{name}: empty")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name}: non-finite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError(f"{name}: values must be in [0,1]")
    return p


def _require_weights(w: np.ndarray, *, name: str = "w") -> np.ndarray:
    w = np.asarray(w, dtype=float).ravel()
    if w.size == 0:
        raise ValueError(f"{name}: empty")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name}: non-finite values")
    if np.any(w < 0.0):
        raise ValueError(f"{name}: negative weights")
    if float(w.sum()) <= 0.0:
        raise ValueError(f"{name}: total weight must be > 0")
    return w


def _require_both_classes(y01: np.ndarray, *, context: str) -> None:
    n1 = int(np.sum(y01 == 1))
    n0 = int(np.sum(y01 == 0))
    if n0 == 0 or n1 == 0:
        raise ValueError(f"{context}: requires both classes")


def auroc_strict(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    y01 = _require_binary_labels(y, name="y")
    p01 = _require_probs(p, name="probs")
    ww = _require_weights(w, name="weights")
    if y01.shape[0] != p01.shape[0] or y01.shape[0] != ww.shape[0]:
        raise ValueError("auroc: length mismatch")
    _require_both_classes(y01, context="AUROC")
    return float(roc_auc_score(y01, p01, sample_weight=ww))


def auprc_strict(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    y01 = _require_binary_labels(y, name="y")
    p01 = _require_probs(p, name="probs")
    ww = _require_weights(w, name="weights")
    if y01.shape[0] != p01.shape[0] or y01.shape[0] != ww.shape[0]:
        raise ValueError("auprc: length mismatch")
    _require_both_classes(y01, context="AUPRC")
    return float(average_precision_score(y01, p01, sample_weight=ww))


def mcc_strict(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    y01 = _require_binary_labels(y, name="y")
    pr = np.asarray(pred).ravel()
    if pr.size == 0:
        raise ValueError("pred: empty")
    if not np.all(np.isfinite(pr)):
        raise ValueError("pred: non-finite values")
    if not np.all(np.isin(pr, [0.0, 1.0])):
        raise ValueError("pred: predictions must be binary")
    pr01 = pr.astype(int, copy=False)

    ww = _require_weights(w, name="weights")
    if y01.shape[0] != pr01.shape[0] or y01.shape[0] != ww.shape[0]:
        raise ValueError("mcc: length mismatch")

    _require_both_classes(y01, context="MCC(y)")
    _require_both_classes(pr01, context="MCC(pred)")
    return float(matthews_corrcoef(y01, pr01, sample_weight=ww))


# ==================== METRICS ====================
def metric_dict(
    *,
    labels: np.ndarray,
    probs: np.ndarray,
    sample_weight: np.ndarray,
    threshold: float,
    ece_bins: int,
) -> Dict[str, float]:
    """Compute metrics on the eligible scored subset."""
    y = np.asarray(labels).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel()

    if y.shape[0] != p.shape[0] or y.shape[0] != w.shape[0]:
        raise ValueError("metric_dict: length mismatch")

    keep = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    if not np.any(keep):
        raise ValueError("metric_dict: no eligible scored rows")

    y = y[keep]
    p = p[keep]
    w = w[keep]

    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("metric_dict: labels must be binary")
    y = y.astype(int, copy=False)

    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("metric_dict: probs must be in [0,1]")
    if np.any(w < 0.0):
        raise ValueError("metric_dict: negative weights")

    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0.0:
        raise ValueError("metric_dict: total weight must be > 0")

    prev = float(np.average(y, weights=w))

    thr = float(threshold)
    if not np.isfinite(thr):
        raise ValueError("metric_dict: threshold must be finite")
    pred = (p >= thr).astype(int)

    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n0 == 0 or n1 == 0:
        raise ValueError("metric_dict: requires both classes")

    pn1 = int(np.sum(pred == 1))
    pn0 = int(np.sum(pred == 0))
    if pn0 == 0 or pn1 == 0:
        raise ValueError("metric_dict: MCC requires both predicted classes")

    return {
        "PREV": prev,
        "AUPRC": float(average_precision_score(y, p, sample_weight=w)),
        "AUROC": float(roc_auc_score(y, p, sample_weight=w)),
        "Brier": float(brier_score_loss(y, p, sample_weight=w)),
        "ECE": float(ece(y, p, n_bins=int(ece_bins), sample_weight=w)),
        "ACC": float(accuracy_score(y, pred, sample_weight=w)),
        "F1": float(f1_score(y, pred, zero_division=0, sample_weight=w)),
        "MCC": float(matthews_corrcoef(y, pred, sample_weight=w)),
        "N_USED": float(y.size),
        "W_SUM": w_sum,
    }


# ==================== FOLD EVALUATION ====================
def evaluate_binary_probs_by_fold(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metro: np.ndarray,
    folds: np.ndarray,
    weights: Optional[np.ndarray] = None,
    threshold_by_fold: Dict[int, float],
    ece_bins: int = 10,
    seed: int = 42,
    n_boot: int = 2000,
    ci_level: float = 0.95,
) -> Dict[str, float]:
    """Evaluate probabilities by fold and add pooled bootstrap CIs."""
    y = np.asarray(y_true).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    mtr = np.asarray(metro).ravel()
    f = np.asarray(folds, dtype=int).ravel()

    if y.shape[0] != p.shape[0] or y.shape[0] != f.shape[0] or y.shape[0] != mtr.shape[0]:
        raise ValueError("evaluate_binary_probs_by_fold: length mismatch")

    if weights is None:
        w = np.ones_like(p, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != y.shape[0]:
            raise ValueError("evaluate_binary_probs_by_fold: weight length mismatch")

    fold_ids = [int(x) for x in np.unique(f)]
    if not fold_ids:
        raise ValueError("evaluate_binary_probs_by_fold: no folds")

    missing_thr = [fid for fid in fold_ids if fid not in threshold_by_fold]
    if missing_thr:
        raise ValueError(f"evaluate_binary_probs_by_fold: missing thresholds for folds {missing_thr}")

    per_fold: List[Dict[str, float]] = []
    for fid in fold_ids:
        mask = f == fid
        thr = float(threshold_by_fold[fid])
        md = metric_dict(
            labels=y[mask],
            probs=p[mask],
            sample_weight=w[mask],
            threshold=thr,
            ece_bins=int(ece_bins),
        )
        md["fold"] = float(fid)
        per_fold.append(md)

    summary: Dict[str, float] = {}
    keys = [k for k in per_fold[0].keys() if k != "fold"]
    for k in keys:
        vals = np.asarray([d[k] for d in per_fold], dtype=float)
        summary[k] = float(np.mean(vals))
    summary["N_FOLDS"] = float(len(per_fold))

    base_ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(w) & (w > 0)
    if not np.any(base_ok):
        raise ValueError("evaluate_binary_probs_by_fold: no eligible rows for CI")

    y0 = y[base_ok]
    p0 = p[base_ok]
    w0 = w[base_ok]
    m0 = mtr[base_ok]
    f0 = f[base_ok]

    fold_ids0 = [int(x) for x in np.unique(f0)]
    if not fold_ids0:
        raise ValueError("evaluate_binary_probs_by_fold: no eligible folds for CI")

    metros_by_fold: Dict[int, np.ndarray] = {}
    idx_by_fold_metro: Dict[int, Dict[object, np.ndarray]] = {}
    for fid in fold_ids0:
        mf = f0 == fid
        metros = np.unique(m0[mf])
        metros_by_fold[fid] = metros
        idx_by_fold_metro[fid] = {g: np.flatnonzero(mf & (m0 == g)) for g in metros}

    def _fold_mean_metric(metric_name: str) -> float:
        vals: List[float] = []
        for fid in fold_ids0:
            mf = f0 == fid
            if not np.any(mf):
                continue
            thr = float(threshold_by_fold[int(fid)])
            md = metric_dict(
                labels=y0[mf],
                probs=p0[mf],
                sample_weight=w0[mf],
                threshold=thr,
                ece_bins=int(ece_bins),
            )
            vals.append(float(md[metric_name]))
        return float(np.mean(vals)) if vals else float("nan")

    def _bootstrap_fold_mean_metric(
        metric_name: str,
        *,
        n_boot: int,
        seed: int,
        ci_level: float,
    ) -> BootstrapResult:
        nb = int(n_boot)
        if nb <= 0:
            raise ValueError("evaluate_binary_probs_by_fold: n_boot must be >= 1")

        cl = float(ci_level)
        if not (0.0 < cl < 1.0):
            raise ValueError("evaluate_binary_probs_by_fold: ci_level must be in (0,1)")

        rng = np.random.default_rng(int(seed))
        point = float(_fold_mean_metric(metric_name))

        boots = np.full(nb, np.nan, dtype=float)
        for b in range(nb):
            fold_vals: List[float] = []
            for fid in fold_ids0:
                metros = metros_by_fold[fid]
                if metros.size == 0:
                    continue

                sampled = rng.choice(metros, size=metros.size, replace=True)
                parts = [idx_by_fold_metro[fid][g] for g in sampled]
                idx = np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=int)
                if idx.size == 0:
                    continue

                thr = float(threshold_by_fold[int(fid)])
                md = metric_dict(
                    labels=y0[idx],
                    probs=p0[idx],
                    sample_weight=w0[idx],
                    threshold=thr,
                    ece_bins=int(ece_bins),
                )
                fold_vals.append(float(md[metric_name]))

            if fold_vals:
                boots[b] = float(np.mean(np.asarray(fold_vals, dtype=float)))

        boots_f = boots[np.isfinite(boots)]
        if boots_f.size == 0:
            return BootstrapResult(
                mean=point,
                lower=float("nan"),
                upper=float("nan"),
                n_boot=nb,
                ci_level=cl,
                seed=int(seed),
                n_used=int(y0.size),
                n_clusters=int(np.unique(m0).size),
                n_boot_used=0,
            )

        alpha = 1.0 - cl
        lo, hi = np.quantile(boots_f, [alpha / 2.0, 1.0 - alpha / 2.0])
        return BootstrapResult(
            mean=point,
            lower=float(lo),
            upper=float(hi),
            n_boot=nb,
            ci_level=cl,
            seed=int(seed),
            n_used=int(y0.size),
            n_clusters=int(np.unique(m0).size),
            n_boot_used=int(boots_f.size),
        )

    br_auprc = _bootstrap_fold_mean_metric(
        "AUPRC",
        n_boot=int(n_boot),
        seed=int(seed) + 11,
        ci_level=float(ci_level),
    )
    br_ece = _bootstrap_fold_mean_metric(
        "ECE",
        n_boot=int(n_boot),
        seed=int(seed) + 17,
        ci_level=float(ci_level),
    )

    summary["AUPRC_LO"] = float(br_auprc.lower)
    summary["AUPRC_HI"] = float(br_auprc.upper)
    summary["ECE_LO"] = float(br_ece.lower)
    summary["ECE_HI"] = float(br_ece.upper)
    summary["CI_N_BOOT"] = float(br_auprc.n_boot)
    summary["CI_LEVEL"] = float(br_auprc.ci_level)
    summary["CI_SEED"] = float(br_auprc.seed)
    summary["CI_N_CLUSTERS"] = float(br_auprc.n_clusters)
    summary["CI_N_BOOT_USED_AUPRC"] = float(br_auprc.n_boot_used)
    summary["CI_N_BOOT_USED_ECE"] = float(br_ece.n_boot_used)

    return summary


__all__ = [
    "print_run_config",
    "format_metrics_table",
    "cv_pairs",
    "ece",
    "BootstrapResult",
    "bootstrap_ci",
    "metric_dict",
    "evaluate_binary_probs_by_fold",
]
