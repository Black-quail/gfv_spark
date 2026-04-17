from __future__ import annotations

"""Ensemble models for year-ahead metro-week ILI forecasting.

This runner consumes OOF artifacts from ml_models.py, nbeat.py, tcn.py, and
tft.py, and writes ensemble-level OOF outputs.
"""

import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from helper_classes import evaluation as ev
from helper_classes import model_contract as mc


OOF_DIR = os.getenv("GFV_SPARK_OOF_DIR", os.path.join("path", "ensemble"))

SEED = 30

KEY_COLS = ["metro", "week_start", "target_week_start"]

_REQUIRED_COLS = KEY_COLS + [
    "fold",
    "year_snap",
    "iso_week",
    "y_true",
    "prob",
    "weight",
    "tau",
    "T0p9",
]


def validate_oof_table(
    df: pd.DataFrame,
    *,
    name: str = "oof",
    coerce_week_start: bool = False,
) -> pd.DataFrame:
    """Check the canonical OOF schema."""
    out = df.copy()

    if "metro" not in out.columns and "cbsa" in out.columns:
        out = out.rename(columns={"cbsa": "metro"})

    if coerce_week_start:
        for c in ["week_start", "target_week_start"]:
            if c in out.columns:
                out[c] = pd.to_datetime(out[c], errors="raise").dt.normalize()

    missing = [c for c in _REQUIRED_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"[{name}] missing cols: {missing}")

    tmp = out[["fold", "tau", "T0p9"]].copy()
    tmp["fold"] = pd.to_numeric(tmp["fold"], errors="raise").astype(int)
    tmp["tau"] = pd.to_numeric(tmp["tau"], errors="raise").astype(float)
    tmp["T0p9"] = pd.to_numeric(tmp["T0p9"], errors="raise").astype(float)

    for col in ["tau", "T0p9"]:
        nunq = tmp.groupby("fold", sort=True)[col].nunique()
        bad = nunq[nunq != 1]
        if len(bad) > 0:
            raise ValueError(f"[{name}] non-unique {col} by fold: {bad.to_dict()}")

    return out


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _beta_features(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.column_stack([np.log(p), np.log1p(-p)])


def _mean_prob(meta: pd.DataFrame, members: Tuple[str, ...]) -> np.ndarray:
    if not members:
        raise ValueError("members is empty")

    s = np.zeros(len(meta), dtype=float)
    for name in members:
        col = f"prob_{name}"
        if col not in meta.columns:
            raise KeyError(f"Missing {col}")
        s += pd.to_numeric(meta[col], errors="raise").to_numpy(dtype=float)

    return s / float(len(members))


def _weighted_quantile(p: np.ndarray, w: np.ndarray, q: float) -> float:
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(p) & np.isfinite(w) & (w > 0)
    p = p[m]
    w = w[m]
    if p.size == 0:
        raise ValueError("weighted_quantile: no eligible rows")

    q = float(np.clip(q, 0.0, 1.0))
    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    w_sorted = w[order]
    cw = np.cumsum(w_sorted)
    cutoff = q * float(cw[-1])

    idx = int(np.searchsorted(cw, cutoff, side="left"))
    idx = max(0, min(idx, p_sorted.size - 1))
    return float(p_sorted[idx])


def _fit_prevalence_threshold(
    p_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
) -> Tuple[float, float]:
    p = np.asarray(p_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    w = np.asarray(w_train, dtype=float)

    m = np.isfinite(p) & np.isfinite(w) & (w > 0) & np.isfinite(y) & np.isin(y, [0.0, 1.0])
    if not np.any(m):
        raise ValueError("thr_prevalence_match: no eligible train rows")

    p = p[m]
    y = y[m].astype(int, copy=False)
    w = w[m]

    tau_hat = float(np.average(y, weights=w))
    if not np.isfinite(tau_hat) or not (0.0 < tau_hat < 1.0):
        raise ValueError(f"thr_prevalence_match: bad tau_hat={tau_hat}")

    thr = _weighted_quantile(p, w, q=1.0 - tau_hat)
    return thr, tau_hat


@dataclass(frozen=True)
class EnsembleConfig:
    stack_years: Tuple[int, ...] = (2022, 2023, 2024)
    report_years: Tuple[int, ...] = (2022, 2023, 2024)

    ml_logit_l2_metro_file: str = "ml_logit_l2_metro_oof.parquet"
    panel_logit_en_metro_file: str = "panel_logit_en_metro_oof.parquet"
    xgb_logit_metro_file: str = "xgb_logit_metro_oof.parquet"

    tft_file: str = "oof_tft_metroEA.parquet"
    nbeats_file: str = "oof_nbeats_metroEA.parquet"
    tcn_file: str = "oof_tcn_metroEA.parquet"

    e0_members: Tuple[str, ...] = ("xgb_logit_metro", "ml_logit_l2_metro", "tcn")
    e1_members: Tuple[str, ...] = (
        "ml_logit_l2_metro",
        "panel_logit_en_metro",
        "xgb_logit_metro",
        "tft",
        "nbeats",
        "tcn",
    )

    e_bal_deep_members: Tuple[str, ...] = ("tft", "nbeats", "tcn")
    e_bal_linear_members: Tuple[str, ...] = ("ml_logit_l2_metro", "panel_logit_en_metro")
    e_bal_tree_members: Tuple[str, ...] = ("xgb_logit_metro",)

    out_prefix: str = "ensemble"

    strict_weight: bool = False
    weight_rtol: float = 1e-6
    weight_atol: float = 1e-10


def _load_oof(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing OOF for {name}: {path}")
    df = pd.read_parquet(path)
    return validate_oof_table(df, name=name)


def _assert_same_y_weight(
    base: pd.DataFrame,
    other: pd.DataFrame,
    name: str,
    *,
    strict_weight: bool,
    weight_rtol: float,
    weight_atol: float,
) -> None:
    keys = list(KEY_COLS)

    a = base[keys + ["y_true", "weight"]].copy()
    b = other[keys + ["y_true", "weight"]].copy()
    m = a.merge(b, on=keys, how="inner", validate="one_to_one", suffixes=("_a", "_b"))

    if len(m) != len(base):
        raise ValueError(
            f"Alignment mismatch vs {name}: base={len(base)} matched={len(m)}"
        )

    ya = pd.to_numeric(m["y_true_a"], errors="raise").to_numpy(dtype=int)
    yb = pd.to_numeric(m["y_true_b"], errors="raise").to_numpy(dtype=int)
    if not np.array_equal(ya, yb):
        bad = np.where(ya != yb)[0][:10].tolist()
        raise ValueError(f"y_true mismatch vs {name}: {bad}")

    wa = pd.to_numeric(m["weight_a"], errors="raise").to_numpy(dtype=float)
    wb = pd.to_numeric(m["weight_b"], errors="raise").to_numpy(dtype=float)

    if not np.allclose(wa, wb, rtol=weight_rtol, atol=weight_atol, equal_nan=False):
        diff = np.abs(wa - wb)
        denom = np.maximum(np.abs(wa), 1e-12)
        bad = np.where(diff > (weight_atol + weight_rtol * denom))[0][:10].tolist()
        max_abs = float(np.max(diff))
        max_rel = float(np.max(diff / denom))
        msg = f"weight mismatch vs {name}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, idx={bad}"
        if strict_weight:
            raise ValueError(msg)
        print(f"[ensemble_model] WARNING: {msg}")


def _merge_oofs(
    oofs: Dict[str, pd.DataFrame],
    *,
    base_name: str,
    cfg: EnsembleConfig,
) -> pd.DataFrame:
    if base_name not in oofs:
        raise KeyError(f"Unknown base_name={base_name}")

    keys = list(KEY_COLS)
    base = oofs[base_name]

    keep_base = keys + ["fold", "year_snap", "iso_week", "y_true", "weight", "tau", "T0p9"]
    keep_base = [c for c in keep_base if c in base.columns]
    meta = base[keep_base].copy()

    for name, df in oofs.items():
        _assert_same_y_weight(
            base,
            df,
            name=name,
            strict_weight=cfg.strict_weight,
            weight_rtol=cfg.weight_rtol,
            weight_atol=cfg.weight_atol,
        )
        meta = meta.merge(
            df[keys + ["prob"]].rename(columns={"prob": f"prob_{name}"}),
            on=keys,
            how="inner",
            validate="one_to_one",
        )

    return meta


def _val_year_by_fold(meta: pd.DataFrame) -> Dict[int, int]:
    tmp = meta[["fold", "year_snap"]].copy()
    tmp["fold"] = pd.to_numeric(tmp["fold"], errors="raise").astype(int)
    tmp["year_snap"] = pd.to_numeric(tmp["year_snap"], errors="raise").astype(int)

    nunq = tmp.groupby("fold", sort=True)["year_snap"].nunique()
    bad = nunq[nunq != 1]
    if len(bad) > 0:
        raise ValueError(f"Non-unique year_snap by fold: {bad.to_dict()}")

    m = tmp.groupby("fold", sort=True)["year_snap"].first().to_dict()
    return {int(k): int(v) for k, v in m.items()}


def _base_cols(meta: pd.DataFrame) -> List[str]:
    cols = list(KEY_COLS) + ["fold", "year_snap", "iso_week", "y_true", "weight", "tau", "T0p9"]
    cols = [c for c in cols if c in meta.columns]

    out: List[str] = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out


def build_meta_table(
    *,
    oof_dir: str = OOF_DIR,
    cfg: EnsembleConfig = EnsembleConfig(),
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Load base-model OOF files and build the meta table."""
    paths = {
        "ml_logit_l2_metro": os.path.join(oof_dir, cfg.ml_logit_l2_metro_file),
        "panel_logit_en_metro": os.path.join(oof_dir, cfg.panel_logit_en_metro_file),
        "xgb_logit_metro": os.path.join(oof_dir, cfg.xgb_logit_metro_file),
        "tft": os.path.join(oof_dir, cfg.tft_file),
        "nbeats": os.path.join(oof_dir, cfg.nbeats_file),
        "tcn": os.path.join(oof_dir, cfg.tcn_file),
    }

    oofs = {name: _load_oof(path, name) for name, path in paths.items()}
    base_name = "ml_logit_l2_metro"
    meta = _merge_oofs(oofs, base_name=base_name, cfg=cfg)

    val_year_by_fold = _val_year_by_fold(meta)
    return meta, val_year_by_fold


def _run_time_safe_ensemble(
    *,
    model_name: str,
    folds: List[int],
    val_year_by_fold: Dict[int, int],
    year_snap_all: np.ndarray,
    fold_all: np.ndarray,
    y_all: np.ndarray,
    w_all: np.ndarray,
    fit_predict: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Fit each ensemble fold using prior years only."""
    n = len(fold_all)
    if len(year_snap_all) != n or len(y_all) != n or len(w_all) != n:
        raise ValueError("_run_time_safe_ensemble: length mismatch")

    p_all = np.full(n, np.nan, dtype=float)
    thrB: Dict[int, float] = {}

    for f in folds:
        f = int(f)
        if f not in val_year_by_fold:
            raise KeyError(f"Missing val_year for fold={f}")

        vy = int(val_year_by_fold[f])
        te = fold_all == f
        tr = year_snap_all < vy

        n_tr = int(np.sum(tr))
        n_te = int(np.sum(te))
        if n_te == 0:
            raise ValueError(f"Fold={f}: no rows in meta")
        if n_tr == 0:
            raise ValueError(f"{model_name} fold={f}: no prior-year rows before {vy}")

        p_tr_fit, p_te = fit_predict(tr, te)
        p_tr_fit = np.asarray(p_tr_fit, dtype=float)
        p_te = np.asarray(p_te, dtype=float)

        if p_tr_fit.shape[0] != n_tr:
            raise ValueError(f"{model_name} fold={f}: bad train length")
        if p_te.shape[0] != n_te:
            raise ValueError(f"{model_name} fold={f}: bad val length")

        p_all[te] = np.clip(p_te, 0.0, 1.0)

        thr, _ = _fit_prevalence_threshold(p_tr_fit, y_all[tr], w_all[tr])
        thrB[f] = float(thr)

    return p_all, thrB


def run_all(
    *,
    oof_dir: str = OOF_DIR,
    cfg: EnsembleConfig = EnsembleConfig(),
    write_oof: bool = True,
    ece_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Run all ensemble models and return metrics."""
    meta, val_year_by_fold = build_meta_table(oof_dir=oof_dir, cfg=cfg)

    min_stack_year = int(min(cfg.stack_years))
    ys = pd.to_numeric(meta["year_snap"], errors="raise").to_numpy(dtype=int)
    if (ys < min_stack_year).sum() == 0:
        raise ValueError(f"No pre-{min_stack_year} OOF rows in meta")

    all_folds = sorted(val_year_by_fold.keys(), key=lambda f: val_year_by_fold[int(f)])

    stack_years = set(int(y) for y in cfg.stack_years)
    report_years = set(int(y) for y in cfg.report_years)

    stack_folds = [int(f) for f in all_folds if int(val_year_by_fold[int(f)]) in stack_years]
    report_folds = [int(f) for f in all_folds if int(val_year_by_fold[int(f)]) in report_years]

    if len(stack_folds) == 0:
        raise ValueError(f"No folds for stack_years={sorted(stack_years)}")
    if len(report_folds) == 0:
        raise ValueError(f"No folds for report_years={sorted(report_years)}")

    y_all = meta["y_true"].to_numpy(int)
    w_all = meta["weight"].to_numpy(float)
    fold_all = meta["fold"].to_numpy(int)
    year_snap_all = pd.to_numeric(meta["year_snap"], errors="raise").to_numpy(dtype=int)

    results: Dict[str, Dict[str, float]] = {}
    report_mask = np.isin(fold_all, np.asarray(report_folds, dtype=int))

    score_e0 = _mean_prob(meta, cfg.e0_members)

    def _fit_predict_e0(tr: np.ndarray, te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z_tr = _safe_logit(score_e0[tr]).reshape(-1, 1)
        z_te = _safe_logit(score_e0[te]).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=2000, C=10.0, random_state=SEED)
        lr.fit(z_tr, y_all[tr], sample_weight=w_all[tr])
        return lr.predict_proba(z_tr)[:, 1], lr.predict_proba(z_te)[:, 1]

    p_e0_all, thrB_e0_all = _run_time_safe_ensemble(
        model_name="E0",
        folds=stack_folds,
        val_year_by_fold=val_year_by_fold,
        year_snap_all=year_snap_all,
        fold_all=fold_all,
        y_all=y_all,
        w_all=w_all,
        fit_predict=_fit_predict_e0,
    )

    e0_oof = meta.loc[report_mask, _base_cols(meta)].copy()
    e0_oof["prob"] = np.clip(p_e0_all[report_mask], 0.0, 1.0)
    e0_oof = validate_oof_table(e0_oof, name="E0")

    thrB_e0 = {f: thrB_e0_all[f] for f in report_folds}
    results["E0"] = mc.evaluate_from_oof(e0_oof, threshold_by_fold=thrB_e0, ece_bins=int(ece_bins))
    results["E0"]["model"] = "E0"
    if write_oof:
        mc.write_validated_oof(
            e0_oof,
            path=os.path.join(oof_dir, f"{cfg.out_prefix}_E0_oof.parquet"),
            name=f"{cfg.out_prefix}_E0_oof",
        )

    X_all = np.column_stack([_safe_logit(meta[f"prob_{name}"].to_numpy(float)) for name in cfg.e1_members])

    def _fit_predict_e1(tr: np.ndarray, te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000, random_state=SEED)
        lr.fit(X_all[tr], y_all[tr], sample_weight=w_all[tr])
        return lr.predict_proba(X_all[tr])[:, 1], lr.predict_proba(X_all[te])[:, 1]

    p_e1_all, thrB_e1_all = _run_time_safe_ensemble(
        model_name="E1",
        folds=stack_folds,
        val_year_by_fold=val_year_by_fold,
        year_snap_all=year_snap_all,
        fold_all=fold_all,
        y_all=y_all,
        w_all=w_all,
        fit_predict=_fit_predict_e1,
    )

    e1_oof = meta.loc[report_mask, _base_cols(meta)].copy()
    e1_oof["prob"] = np.clip(p_e1_all[report_mask], 0.0, 1.0)
    e1_oof = validate_oof_table(e1_oof, name="E1")

    thrB_e1 = {f: thrB_e1_all[f] for f in report_folds}
    results["E1"] = mc.evaluate_from_oof(e1_oof, threshold_by_fold=thrB_e1, ece_bins=int(ece_bins))
    results["E1"]["model"] = "E1"
    if write_oof:
        mc.write_validated_oof(
            e1_oof,
            path=os.path.join(oof_dir, f"{cfg.out_prefix}_E1_oof.parquet"),
            name=f"{cfg.out_prefix}_E1_oof",
        )

    def _fit_predict_e2(tr: np.ndarray, te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lr1 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000, random_state=SEED)
        lr1.fit(X_all[tr], y_all[tr], sample_weight=w_all[tr])
        p1_tr = lr1.predict_proba(X_all[tr])[:, 1]
        p1_te = lr1.predict_proba(X_all[te])[:, 1]

        Xb_tr = _beta_features(p1_tr)
        Xb_te = _beta_features(p1_te)
        lr2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=4000, random_state=SEED)
        lr2.fit(Xb_tr, y_all[tr], sample_weight=w_all[tr])
        return lr2.predict_proba(Xb_tr)[:, 1], lr2.predict_proba(Xb_te)[:, 1]

    p_e2_all, thrB_e2_all = _run_time_safe_ensemble(
        model_name="E2",
        folds=stack_folds,
        val_year_by_fold=val_year_by_fold,
        year_snap_all=year_snap_all,
        fold_all=fold_all,
        y_all=y_all,
        w_all=w_all,
        fit_predict=_fit_predict_e2,
    )

    e2_oof = meta.loc[report_mask, _base_cols(meta)].copy()
    e2_oof["prob"] = np.clip(p_e2_all[report_mask], 0.0, 1.0)
    e2_oof = validate_oof_table(e2_oof, name="E2")

    thrB_e2 = {f: thrB_e2_all[f] for f in report_folds}
    results["E2"] = mc.evaluate_from_oof(e2_oof, threshold_by_fold=thrB_e2, ece_bins=int(ece_bins))
    results["E2"]["model"] = "E2"
    if write_oof:
        mc.write_validated_oof(
            e2_oof,
            path=os.path.join(oof_dir, f"{cfg.out_prefix}_E2_oof.parquet"),
            name=f"{cfg.out_prefix}_E2_oof",
        )

    deep_mean = _mean_prob(meta, cfg.e_bal_deep_members)
    lin_mean = _mean_prob(meta, cfg.e_bal_linear_members)
    tree_mean = _mean_prob(meta, cfg.e_bal_tree_members)
    score_e3 = (deep_mean + lin_mean + tree_mean) / 3.0

    def _fit_predict_e3(tr: np.ndarray, te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        z_tr = _safe_logit(score_e3[tr]).reshape(-1, 1)
        z_te = _safe_logit(score_e3[te]).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs", max_iter=2000, C=10.0, random_state=SEED)
        lr.fit(z_tr, y_all[tr], sample_weight=w_all[tr])
        return lr.predict_proba(z_tr)[:, 1], lr.predict_proba(z_te)[:, 1]

    p_e3_all, thrB_e3_all = _run_time_safe_ensemble(
        model_name="E3",
        folds=stack_folds,
        val_year_by_fold=val_year_by_fold,
        year_snap_all=year_snap_all,
        fold_all=fold_all,
        y_all=y_all,
        w_all=w_all,
        fit_predict=_fit_predict_e3,
    )

    e3_oof = meta.loc[report_mask, _base_cols(meta)].copy()
    e3_oof["prob"] = np.clip(p_e3_all[report_mask], 0.0, 1.0)
    e3_oof = validate_oof_table(e3_oof, name="E3")

    thrB_e3 = {f: thrB_e3_all[f] for f in report_folds}
    results["E3"] = mc.evaluate_from_oof(e3_oof, threshold_by_fold=thrB_e3, ece_bins=int(ece_bins))
    results["E3"]["model"] = "E3"
    if write_oof:
        mc.write_validated_oof(
            e3_oof,
            path=os.path.join(oof_dir, f"{cfg.out_prefix}_E3_oof.parquet"),
            name=f"{cfg.out_prefix}_E3_oof",
        )

    return results


if __name__ == "__main__":
    out = run_all(write_oof=True)

    rows = [{"model": k, **v} for k, v in out.items()]
    if rows:
        print("\n[ ensemble_model ] OOF metrics")
        print(ev.format_metrics_table(rows, index_key="model"))
