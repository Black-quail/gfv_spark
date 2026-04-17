from __future__ import annotations

"""Shared model-side helpers for features, OOF packing, and evaluation.

This module is a contract layer shared by model runners and the deep engine.
It intentionally performs a deferred import of ``ensemble_model`` in
``_validate_oof_table`` to avoid import-time circular dependency issues.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


FEATURE_COLS: List[str] = [
    "p_tilde",
    "p_tilde_lag_1",
    "ppt_log_sum",
    "ppt_any",
    "wet_days",
    "vacc_prop",
    "tavg_mean",
    "diurnal_range_mean",
    "gender_female_share",
    "age_19_64_share",
    "age_64_share",
]


def _validate_oof_table(
    df: pd.DataFrame,
    *,
    name: str,
    coerce_week_start: bool = False,
) -> pd.DataFrame:
    from model_scripts.ensemble_model import validate_oof_table

    return validate_oof_table(df, name=name, coerce_week_start=bool(coerce_week_start))


def feature_columns_present(df: pd.DataFrame, feature_cols: Sequence[str]) -> None:
    """Check that all requested feature columns are present."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")


def require_model_columns(
    df: pd.DataFrame,
    *,
    require_label: bool = True,
) -> None:
    """Check the canonical weekly-panel columns used by the models."""
    required = ["metro", "week_start", "target_week_start", "year_snap", "iso_week", "weight"]
    if require_label:
        required.append("flu_label")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def pick_feature_columns() -> List[str]:
    """Return the shared feature list."""
    return list(FEATURE_COLS)


def is_padding_row(df: pd.DataFrame) -> np.ndarray:
    """Identify padded rows."""
    n = pd.to_numeric(df["n_reports"], errors="coerce").fillna(0).to_numpy()
    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy()
    return (n <= 0) & (w <= 0.0)


def weight_rule(df: pd.DataFrame) -> None:
    """Check basic weight semantics."""
    w = pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("weight contains NaN or inf")
    if (w < 0).any():
        raise ValueError("weight must be nonnegative")

    pad = is_padding_row(df)
    if pad.any() and (np.abs(w[pad]) > 0.0).any():
        raise ValueError("Padding rows must have weight == 0")



def fit_impute_standardize(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    weight_col: str = "weight",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Fit median imputation and z-score scaling on positive-weight rows."""
    cols = list(feature_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns to fit preprocessing: {missing}")

    X = df.loc[:, cols].copy()
    for c in cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fit_mask = np.isfinite(w) & (w > 0)
    if not np.any(fit_mask):
        raise ValueError("No positive-weight rows available to fit preprocessing")

    X_fit = X.loc[fit_mask]
    med = X_fit.median(axis=0, skipna=True).reindex(cols)
    X_imp = X_fit.fillna(med)
    mu = X_imp.mean(axis=0).reindex(cols)
    sd = X_imp.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0).reindex(cols)

    return med, mu, sd



def apply_impute_standardize(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    med: pd.Series,
    mu: pd.Series,
    sd: pd.Series,
    *,
    n_reports_col: str = "n_reports",
    weight_col: str = "weight",
    preserve_padding_zeros: bool = True,
) -> pd.DataFrame:
    """Apply fitted imputation and standardization."""
    cols = list(feature_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for preprocessing apply: {missing}")

    out = df.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce").fillna(med[c])
        out[c] = (x - mu[c]) / sd[c]

    if preserve_padding_zeros and n_reports_col in out.columns and weight_col in out.columns:
        n = pd.to_numeric(out[n_reports_col], errors="coerce").fillna(0).to_numpy()
        w = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pad_mask = (n <= 0) & (w <= 0.0)
        if np.any(pad_mask):
            out.loc[pad_mask, cols] = 0.0

    return out



def pack_oof_from_weekly(
    origin_df: pd.DataFrame,
    prob: Sequence[float],
    *,
    fold: int,
    T0p9: float,
    tau: float,
    y_col: str = "flu_label",
    weight_col: str = "weight",
) -> pd.DataFrame:
    """Pack weekly-origin predictions into the canonical OOF set."""
    df = origin_df.copy()
    p = np.asarray(prob, dtype=float).ravel()
    if p.shape[0] != len(df):
        raise ValueError(f"pack_oof_from_weekly: length mismatch (p={p.shape[0]} vs df={len(df)})")

    req = ["metro", "week_start", "target_week_start", "year_snap", "iso_week", y_col, weight_col]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"pack_oof_from_weekly: missing required columns: {missing}")

    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if np.any(~np.isfinite(y)) or not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("pack_oof_from_weekly: y_true must be finite and binary")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("pack_oof_from_weekly: prob must be finite and in [0,1]")
    if np.any(w <= 0):
        raise ValueError("pack_oof_from_weekly: scored rows must have weight > 0")

    return pd.DataFrame(
        {
            "metro": df["metro"].to_numpy(),
            "week_start": df["week_start"].to_numpy(),
            "target_week_start": df["target_week_start"].to_numpy(),
            "year_snap": pd.to_numeric(df["year_snap"], errors="raise").to_numpy(dtype=int),
            "iso_week": pd.to_numeric(df["iso_week"], errors="raise").to_numpy(dtype=int),
            "fold": int(fold),
            "y_true": y.astype(int),
            "weight": w.astype(float),
            "prob": p.astype(float),
            "tau": float(tau),
            "T0p9": float(T0p9),
            "target_iso_year": pd.to_numeric(df["year_snap"], errors="raise").to_numpy(dtype=int),
            "target_iso_week": pd.to_numeric(df["iso_week"], errors="raise").to_numpy(dtype=int),
        }
    )



def write_validated_oof(oof_df: pd.DataFrame, *, path: str, name: str) -> pd.DataFrame:
    """Validate an OOF table and write it."""
    out = _validate_oof_table(oof_df, name=name)
    out.to_parquet(path, index=False)
    return out



def fold_prevalence_thresholds(
    train_oof: pd.DataFrame,
    *,
    fold_col: str = "fold",
    prob_col: str = "prob",
    weight_col: str = "weight",
    y_col: str = "y_true",
    tau_col: str = "tau",
    audit_tau: bool = True,
    tau_tol: float = 1e-6,
) -> Dict[int, float]:
    """Compute a prevalence-matching operating threshold by fold."""
    req = [fold_col, prob_col, weight_col, y_col]
    missing = [c for c in req if c not in train_oof.columns]
    if missing:
        raise ValueError(f"Rule threshold: missing columns: {missing}")

    out: Dict[int, float] = {}

    for fold_id, g in train_oof.groupby(fold_col, sort=True):
        p = pd.to_numeric(g[prob_col], errors="coerce").to_numpy(dtype=float)
        w = pd.to_numeric(g[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)

        m = np.isfinite(p) & np.isfinite(w) & (w > 0) & np.isfinite(y) & np.isin(y, [0.0, 1.0])
        if not np.any(m):
            raise ValueError(f"Rule threshold: fold={fold_id} has no eligible train rows")

        p = p[m]
        w = w[m]
        y = y[m].astype(int, copy=False)

        if np.any((p < 0.0) | (p > 1.0)):
            raise ValueError(f"Rule threshold: fold={fold_id} prob outside [0,1]")

        tau_hat = float(np.average(y, weights=w))
        if not np.isfinite(tau_hat) or not (0.0 < tau_hat < 1.0):
            raise ValueError(f"Rule threshold: fold={fold_id} has degenerate tau")

        if audit_tau and (tau_col in g.columns):
            tau_vals = pd.to_numeric(g[tau_col], errors="coerce").dropna().unique()
            if tau_vals.size == 1 and np.isfinite(float(tau_vals[0])):
                tau_stored = float(tau_vals[0])
                if abs(tau_stored - tau_hat) > float(tau_tol):
                    raise ValueError(f"Rule threshold: fold={fold_id} stored tau mismatch")

        q = float(1.0 - tau_hat)
        order = np.argsort(p, kind="mergesort")
        p_sorted = p[order]
        w_sorted = w[order]
        cw = np.cumsum(w_sorted)
        cutoff = q * float(cw[-1])
        idx = int(np.searchsorted(cw, cutoff, side="left"))
        idx = max(0, min(idx, p_sorted.size - 1))
        out[int(fold_id)] = float(p_sorted[idx])

    return out



def evaluate_from_oof(
    oof: pd.DataFrame,
    *,
    threshold_by_fold: Dict[int, float],
    ece_bins: int = 10,
) -> Dict[str, float]:
    """Evaluate an OOF table with the shared evaluation module."""
    req = ["metro", "fold", "y_true", "prob", "weight"]
    missing = [c for c in req if c not in oof.columns]
    if missing:
        raise ValueError(f"evaluate_from_oof: missing required columns: {missing}")

    from helper_classes import evaluation as ev

    return ev.evaluate_binary_probs_by_fold(
        y_true=pd.to_numeric(oof["y_true"], errors="raise").to_numpy(dtype=int),
        y_prob=pd.to_numeric(oof["prob"], errors="raise").to_numpy(dtype=float),
        metro=oof["metro"].to_numpy(),
        folds=pd.to_numeric(oof["fold"], errors="raise").to_numpy(dtype=int),
        weights=pd.to_numeric(oof["weight"], errors="raise").to_numpy(dtype=float),
        threshold_by_fold=threshold_by_fold,
        ece_bins=int(ece_bins),
    )



def _eligible_mask_for_fold_stats(df: pd.DataFrame) -> np.ndarray:
    """Rows eligible for fold summaries."""
    w = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(df["flu_label"], errors="coerce").to_numpy(dtype=float)
    return np.isfinite(w) & (w > 0) & np.isfinite(y) & np.isin(y, [0.0, 1.0])



def fold_diagnostics_row(
    *,
    fold: int,
    val_year: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    T0p9: float,
    tau: float,
    thr_B: Optional[float] = None,
) -> Dict[str, Any]:
    """Build one fold diagnostics row."""
    mt = _eligible_mask_for_fold_stats(train_df)
    mv = _eligible_mask_for_fold_stats(val_df)

    ytr = pd.to_numeric(train_df.loc[mt, "flu_label"], errors="raise").to_numpy(dtype=int)
    yva = pd.to_numeric(val_df.loc[mv, "flu_label"], errors="raise").to_numpy(dtype=int)
    wtr = pd.to_numeric(train_df.loc[mt, "weight"], errors="raise").to_numpy(dtype=float)

    n_tr = int(np.sum(mt))
    n_va = int(np.sum(mv))
    tau_re = float(np.average((ytr == 1).astype(float), weights=wtr)) if n_tr > 0 else np.nan

    ws_min = pd.to_datetime(val_df.loc[mv, "week_start"], errors="coerce").min() if n_va > 0 else pd.NaT
    ws_max = pd.to_datetime(val_df.loc[mv, "week_start"], errors="coerce").max() if n_va > 0 else pd.NaT
    tws_min = pd.to_datetime(val_df.loc[mv, "target_week_start"], errors="coerce").min() if n_va > 0 else pd.NaT
    tws_max = pd.to_datetime(val_df.loc[mv, "target_week_start"], errors="coerce").max() if n_va > 0 else pd.NaT

    return {
        "fold": int(fold),
        "val_year": int(val_year),
        "n_train": n_tr,
        "pos_train": int(np.sum(ytr == 1)),
        "tau_train": float(tau),
        "tau_train_recalc": float(tau_re),
        "metros_train": int(train_df.loc[mt, "metro"].nunique()) if n_tr > 0 else 0,
        "n_val": n_va,
        "pos_val": int(np.sum(yva == 1)),
        "metros_val": int(val_df.loc[mv, "metro"].nunique()) if n_va > 0 else 0,
        "val_origin_min": None if pd.isna(ws_min) else str(ws_min.date()),
        "val_origin_max": None if pd.isna(ws_max) else str(ws_max.date()),
        "val_target_min": None if pd.isna(tws_min) else str(tws_min.date()),
        "val_target_max": None if pd.isna(tws_max) else str(tws_max.date()),
        "T0p9": float(T0p9),
        "thr_B": np.nan if thr_B is None else float(thr_B),
    }



def print_fold_diagnostics(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    """Print and return the fold diagnostics table."""
    if not rows:
        raise ValueError("print_fold_diagnostics: no rows provided")

    df = pd.DataFrame(list(rows)).sort_values(["val_year", "fold"], kind="mergesort").reset_index(drop=True)
    cols = [
        "fold",
        "val_year",
        "n_train",
        "pos_train",
        "tau_train",
        "tau_train_recalc",
        "metros_train",
        "n_val",
        "pos_val",
        "metros_val",
        "val_origin_min",
        "val_origin_max",
        "val_target_min",
        "val_target_max",
        "T0p9",
        "thr_B",
    ]
    df = df.loc[:, [c for c in cols if c in df.columns]]

    print("\n[model_builder] Fold diagnostics")
    print(df.to_string(index=False))
    return df



def concat_and_validate_oof(
    oof_list: Sequence[pd.DataFrame],
    *,
    name: str,
    require_positive_weight: bool = True,
    enforce_val_fold_year_unique: bool = False,
    coerce_week_start: bool = True,
) -> pd.DataFrame:
    """Concatenate OOF parts and check the canonical OOF schema."""
    if not oof_list:
        raise ValueError(f"{name}: no OOF tables to concatenate")

    oof = pd.concat(list(oof_list), axis=0, ignore_index=True)
    req = [
        "metro",
        "week_start",
        "fold",
        "y_true",
        "weight",
        "prob",
        "tau",
        "T0p9",
        "target_week_start",
        "year_snap",
        "iso_week",
    ]
    missing = [c for c in req if c not in oof.columns]
    if missing:
        raise ValueError(f"{name}: OOF missing required columns: {missing}")

    oof = _validate_oof_table(oof, name=name, coerce_week_start=bool(coerce_week_start))

    w = pd.to_numeric(oof["weight"], errors="raise").to_numpy(dtype=float)
    if require_positive_weight and (np.any(w <= 0) or np.any(~np.isfinite(w))):
        raise ValueError(f"{name}: weight must be finite and > 0")

    if enforce_val_fold_year_unique:
        tmp = oof.loc[:, ["fold"]].copy()
        tmp["_ys_"] = pd.to_numeric(oof["year_snap"], errors="raise").to_numpy(dtype=int)
        nunq = tmp.groupby("fold", sort=True)["_ys_"].nunique()
        bad = nunq[nunq != 1]
        if len(bad) > 0:
            raise ValueError(f"{name}: folds with non-unique year_snap: {bad.to_dict()}")

    return oof


__all__ = [
    "FEATURE_COLS",
    "pick_feature_columns",
    "feature_columns_present",
    "require_model_columns",
    "is_padding_row",
    "weight_rule",
    "fit_impute_standardize",
    "apply_impute_standardize",
    "pack_oof_from_weekly",
    "write_validated_oof",
    "fold_prevalence_thresholds",
    "evaluate_from_oof",
    "fold_diagnostics_row",
    "print_fold_diagnostics",
    "concat_and_validate_oof",
]
