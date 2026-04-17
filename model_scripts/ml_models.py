from __future__ import annotations

"""Run baseline linear and tree models on the weekly metro panel.

This runner consumes shared helpers from dataset, contract, evaluation, and
engine modules, and writes base-model OOF artifacts used by ensemble_model.py.
"""

import gc
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from helper_classes import dataset_builder as du
from helper_classes import evaluation as ev
from helper_classes.model_contract import (
    FEATURE_COLS,
    apply_impute_standardize,
    concat_and_validate_oof,
    evaluate_from_oof,
    feature_columns_present,
    fit_impute_standardize,
    fold_diagnostics_row,
    pack_oof_from_weekly,
    print_fold_diagnostics,
    require_model_columns,
    fold_prevalence_thresholds,
    weight_rule,
)
from helper_classes.model_engine import seed_all
from model_scripts.ensemble_model import validate_oof_table


# ==================== RUN CONFIG ====================
CONFIG: Dict[str, object] = {
    "SEED": 30,
    "DATA_DIR": os.getenv("GFV_SPARK_DATA_DIR", os.path.join("path", "model_input")),
    "OOF_DIR": os.getenv("GFV_SPARK_OOF_DIR", os.path.join("path", "ensemble")),
    # Keep 2021 so ensembles can train for 2022.
    "VAL_YEARS": (2021, 2022, 2023, 2024),
    "REPORT_YEARS": (2022, 2023, 2024),
    "Q_EVENT": 0.9, # Default = 0.90, sensitivity = 0.95
    "WEIGHT_MODE": "capped",   # Default = "capped, sensitivity = unweighted
    "WEIGHT_CAP_Q": 0.95,      # Default=0.95, sensitivity=0.90, ignored if unweighted
    "POST_PANDEMIC_START": 2022,
    "ECE_BINS": 10,
    "WRITE_OOF_PARQUET": True,
    "WRITE_TRAIN_PARQUET": True,
    "WRITE_TABLES": True,
    "VERBOSE": True,
    "MIN_METRO_WEEKS": 26,
    "MIN_OBS_WEEKS": 26,
    "L2_C": 1.0,
    "EN_L1_RATIO": 0.20,
    "EN_C": 0.20,
    "METRO_OHE_MIN_FREQ": 1,
    "XGB_N_ESTIMATORS": 600,
    "XGB_MAX_DEPTH": 4,
    "XGB_LEARNING_RATE": 0.03,
    "XGB_SUBSAMPLE": 0.8,
    "XGB_COLSAMPLE_BYTREE": 0.8,
    "XGB_REG_LAMBDA": 1.0,
    "XGB_REG_ALPHA": 0.5,
    "XGB_MIN_CHILD_WEIGHT": 5.0,
    "XGB_GAMMA": 0.0,
    "XGB_TREE_METHOD": "hist",
}
# ====================================================


def _fit_metro_ohe(
    train_metro: pd.Series,
    val_metro: pd.Series,
    *,
    min_freq: int = 1,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Fit train-only metro one-hot encoding."""
    base = {"handle_unknown": "ignore"}

    if int(min_freq) > 1:
        try:
            ohe = OneHotEncoder(
                **base,
                sparse_output=True,
                min_frequency=int(min_freq),
            )
        except TypeError:
            ohe = OneHotEncoder(**base, sparse=True)
    else:
        try:
            ohe = OneHotEncoder(**base, sparse_output=True)
        except TypeError:
            ohe = OneHotEncoder(**base, sparse=True)

    Xtr_m = ohe.fit_transform(train_metro.astype(str).to_frame())
    Xva_m = ohe.transform(val_metro.astype(str).to_frame())
    return Xtr_m.tocsr(), Xva_m.tocsr()


def _filter_min_obs(
    origin_df: pd.DataFrame,
    *,
    ctx_df: pd.DataFrame,
    min_obs_weeks: int,
    seq_len_weeks: int,
) -> pd.DataFrame:
    """Keep origins with enough observed history."""
    return du.filter_origins_min_obs(
        origin_df,
        ctx_df=ctx_df,
        min_observed_weeks=int(min_obs_weeks),
        seq_len_weeks=int(seq_len_weeks),
    )

def _apply_weights(
    weekly: pd.DataFrame,
    *,
    mode: str,
    cap_q: float,
) -> pd.DataFrame:
    """Weekly weights for capped or unweighted."""
    out = weekly.copy()

    n = pd.to_numeric(out["n_reports"], errors="raise").to_numpy(dtype=float)
    w = np.zeros_like(n, dtype=float)

    obs = np.isfinite(n) & (n > 0)
    mode = str(mode).lower().strip()

    if mode == "capped":
        cap_q = float(cap_q)
        if not (0.0 < cap_q <= 1.0):
            # Avoid accidental misuse
            raise ValueError(f"cap_q must be in (0, 1]; got {cap_q}")

        if np.any(obs):
            cap = float(np.quantile(n[obs], cap_q))
            n_cap = np.minimum(n[obs], cap)
            mu = float(np.mean(n_cap))
            if not np.isfinite(mu) or mu <= 0:
                raise ValueError("Bad mean after weight capping.")
            w[obs] = n_cap / mu

    elif mode == "unweighted":
        w[obs] = 1.0

    else:
        raise ValueError(
            f"WEIGHT_MODE={mode!r}. Expected 'capped' or 'unweighted'."
        )

    out["weight"] = w.astype(float)
    return out

def run_all(*, write_oof: bool = True, verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """Run outer CV for the baseline ML models."""
    data_dir = str(CONFIG["DATA_DIR"])
    oof_dir = str(CONFIG["OOF_DIR"])
    os.makedirs(oof_dir, exist_ok=True)

    seed = int(CONFIG["SEED"])
    val_years = tuple(int(y) for y in CONFIG["VAL_YEARS"])
    report_years = tuple(int(y) for y in CONFIG.get("REPORT_YEARS", val_years))
    report_years_set = set(report_years)

    q_event = float(CONFIG["Q_EVENT"])
    ece_bins = int(CONFIG["ECE_BINS"])
    post_start = int(CONFIG.get("POST_PANDEMIC_START", min(val_years)))

    seed_all(seed)

    weekly = du.prepare_weekly_panel(data_dir, verbose=bool(verbose))

    min_metro_weeks = int(CONFIG.get("MIN_METRO_WEEKS", CONFIG.get("MIN_OBS_WEEKS", 0)))
    if min_metro_weeks > 0:
        weekly = du.retain_metros_min_weeks(
            weekly,
            min_weeks=min_metro_weeks,
            verbose=bool(verbose),
        )

    weekly = _apply_weights(
        weekly,
        mode=str(CONFIG.get("WEIGHT_MODE", "capped")),
        cap_q=float(CONFIG.get("WEIGHT_CAP_Q", 0.95)),
    )
    feature_cols = list(FEATURE_COLS)

    require_model_columns(weekly, require_label=False)
    feature_columns_present(weekly, feature_cols)
    weight_rule(weekly)

    if verbose:
        print(
            f"[ml_models] EVENT baselines | VAL_YEARS={val_years} | "
            f"REPORT_YEARS={report_years} | q_event={q_event} | ece_bins={ece_bins}"
        )

        if str(CONFIG.get("WEIGHT_MODE", "capped")).lower().strip() == "capped":
            print(f"[ml_models] weight_mode={str(CONFIG.get("WEIGHT_MODE", "capped")).lower().strip()} | weight_cap_q={CONFIG.get('WEIGHT_CAP_Q', 0.95)}")
        else:
            print(f"[ml_models] weight_mode={str(CONFIG.get("WEIGHT_MODE", "capped")).lower().strip()}")

        print(f"[ml_models] FEATURE_COLS ({len(feature_cols)}): {', '.join(feature_cols)}")
        print("[ml_models] Metro identity: train-only OHE")

    policy = du.ThresholdPolicy(q_event=q_event, post_start_year=post_start)

    def _make_logit_l2() -> LogisticRegression:
        return LogisticRegression(
            solver="lbfgs",
            l1_ratio=0.0,
            C=float(CONFIG.get("L2_C", 1.0)),
            max_iter=2000,
            random_state=int(CONFIG["SEED"]),
        )

    def _make_logit_en() -> LogisticRegression:
        return LogisticRegression(
            solver="saga",
            l1_ratio=float(CONFIG.get("EN_L1_RATIO", 0.20)),
            C=float(CONFIG.get("EN_C", 0.20)),
            max_iter=5000,
            random_state=int(CONFIG["SEED"]),
        )

    def _make_xgb() -> XGBClassifier:
        return XGBClassifier(
            n_estimators=int(CONFIG["XGB_N_ESTIMATORS"]),
            max_depth=int(CONFIG["XGB_MAX_DEPTH"]),
            learning_rate=float(CONFIG["XGB_LEARNING_RATE"]),
            subsample=float(CONFIG["XGB_SUBSAMPLE"]),
            colsample_bytree=float(CONFIG["XGB_COLSAMPLE_BYTREE"]),
            reg_lambda=float(CONFIG["XGB_REG_LAMBDA"]),
            reg_alpha=float(CONFIG["XGB_REG_ALPHA"]),
            min_child_weight=float(CONFIG["XGB_MIN_CHILD_WEIGHT"]),
            gamma=float(CONFIG["XGB_GAMMA"]),
            random_state=seed,
            tree_method=str(CONFIG.get("XGB_TREE_METHOD", "hist")),
            objective="binary:logistic",
            eval_metric="logloss",
        )

    model_builders = {
        "ml_logit_l2_metro": _make_logit_l2,
        "panel_logit_en_metro": _make_logit_en,
        "xgb_logit_metro": _make_xgb,
    }
    model_names = list(model_builders.keys())

    val_oof_parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in model_names}
    train_oof_parts: Dict[str, List[pd.DataFrame]] = {k: [] for k in model_names}
    fold_rows_template: List[Dict[str, Any]] = []

    for fold_id, val_year in enumerate(val_years):
        fold_id = int(fold_id)
        val_year = int(val_year)

        train_df, val_df, val_ctx_df, T0p9, _ = du.make_fold_frames(
            weekly,
            val_year=val_year,
            policy=policy,
        )

        require_model_columns(train_df)
        require_model_columns(val_df)

        med, mu, sd = fit_impute_standardize(train_df, feature_cols, weight_col="weight")
        train_df_p = apply_impute_standardize(train_df, feature_cols, med, mu, sd)
        val_df_p = apply_impute_standardize(val_df, feature_cols, med, mu, sd)

        train_origin = du.filter_scored_origins(train_df_p)
        val_origin = du.filter_scored_origins(val_df_p)

        min_obs = int(CONFIG.get("MIN_OBS_WEEKS", 0))
        if min_obs > 0:
            seq_len = int(du.SEQ_LEN_WEEKS)

            train_origin = _filter_min_obs(
                train_origin,
                ctx_df=train_df_p,
                min_obs_weeks=min_obs,
                seq_len_weeks=seq_len,
            )

            val_ctx_labeled = du.inject_val_labels(val_ctx_df, val_df)
            val_origin = _filter_min_obs(
                val_origin,
                ctx_df=val_ctx_labeled,
                min_obs_weeks=min_obs,
                seq_len_weeks=seq_len,
            )

        Xtr_num = train_origin.loc[:, feature_cols].to_numpy(dtype=float)
        Xva_num = val_origin.loc[:, feature_cols].to_numpy(dtype=float)

        ytr = pd.to_numeric(train_origin["flu_label"], errors="raise").to_numpy(dtype=int)
        wtr = pd.to_numeric(train_origin["weight"], errors="raise").to_numpy(dtype=float)

        Xtr_m, Xva_m = _fit_metro_ohe(
            train_origin["metro"],
            val_origin["metro"],
            min_freq=int(CONFIG.get("METRO_OHE_MIN_FREQ", 1)),
        )

        Xtr = sparse.hstack([sparse.csr_matrix(Xtr_num), Xtr_m], format="csr")
        Xva = sparse.hstack([sparse.csr_matrix(Xva_num), Xva_m], format="csr")

        tau_eff = float(np.average(ytr, weights=wtr))

        fold_rows_template.append(
            fold_diagnostics_row(
                fold=fold_id,
                val_year=val_year,
                train_df=train_origin,
                val_df=val_origin,
                T0p9=float(T0p9),
                tau=float(tau_eff),
                thr_B=None,
            )
        )

        for name, builder in model_builders.items():
            est = builder()
            est.fit(Xtr, ytr, sample_weight=wtr)

            p_tr = est.predict_proba(Xtr)[:, 1]
            p_va = est.predict_proba(Xva)[:, 1]

            train_oof_parts[name].append(
                pack_oof_from_weekly(
                    train_origin,
                    p_tr,
                    fold=fold_id,
                    T0p9=float(T0p9),
                    tau=float(tau_eff),
                )
            )
            val_oof_parts[name].append(
                pack_oof_from_weekly(
                    val_origin,
                    p_va,
                    fold=fold_id,
                    T0p9=float(T0p9),
                    tau=float(tau_eff),
                )
            )

        del train_df, val_df, val_ctx_df
        del train_df_p, val_df_p
        del train_origin, val_origin
        del Xtr_num, Xva_num, Xtr_m, Xva_m, Xtr, Xva
        gc.collect()

    metrics_by_model: Dict[str, Dict[str, float]] = {}
    rows_for_table: List[Dict[str, float]] = []

    for name in model_names:
        if not val_oof_parts[name]:
            continue

        val_oof_all = concat_and_validate_oof(
            val_oof_parts[name],
            name=f"{name}_val_oof_all",
        )

        train_oof_all = pd.concat(train_oof_parts[name], ignore_index=True)
        train_oof_all = validate_oof_table(
            train_oof_all,
            name=f"{name}_train_oof_all",
        )

        threshold_by_fold = fold_prevalence_thresholds(train_oof_all)

        rows = [dict(r) for r in fold_rows_template]
        for r in rows:
            r["thr_B"] = float(threshold_by_fold[int(r["fold"])])

        rows_report = [r for r in rows if int(r["val_year"]) in report_years_set]
        fold_diag_report = print_fold_diagnostics(rows_report)

        val_oof_report = val_oof_all.loc[val_oof_all["year_snap"].isin(report_years_set)].copy()

        report_folds = set(
            pd.to_numeric(fold_diag_report["fold"], errors="raise").astype(int).tolist()
        )
        threshold_by_fold_report = {
            int(f): float(v)
            for f, v in threshold_by_fold.items()
            if int(f) in report_folds
        }

        metrics = evaluate_from_oof(
            val_oof_report,
            threshold_by_fold=threshold_by_fold,
            ece_bins=int(ece_bins),
        )
        metrics_by_model[name] = dict(metrics)
        rows_for_table.append({"model": name, **metrics})

        if write_oof and bool(CONFIG.get("WRITE_OOF_PARQUET", True)):
            val_path = os.path.join(oof_dir, f"{name}_oof.parquet")
            validate_oof_table(val_oof_all, name=f"{name}_oof").to_parquet(val_path, index=False)

            val_report_path = os.path.join(oof_dir, f"{name}_oof_report.parquet")
            validate_oof_table(
                val_oof_report,
                name=f"{name}_oof_report",
            ).to_parquet(val_report_path, index=False)

            if bool(CONFIG.get("WRITE_TRAIN_PARQUET", False)):
                tr_path = os.path.join(oof_dir, f"{name}_train_insample.parquet")
                validate_oof_table(
                    train_oof_all,
                    name=f"{name}_train_insample",
                ).to_parquet(tr_path, index=False)

            if bool(CONFIG.get("WRITE_TABLES", True)):
                fold_diag_report.to_csv(
                    os.path.join(oof_dir, f"{name}_fold_diag_report.csv"),
                    index=False,
                )
                pd.DataFrame(
                    [{"fold": int(k), "thr_B": float(v)} for k, v in sorted(threshold_by_fold_report.items())]
                ).to_csv(
                    os.path.join(oof_dir, f"{name}_threshold_by_fold_report.csv"),
                    index=False,
                )

    if verbose and rows_for_table:
        print("\n[ ml_models ] OOF metrics (REPORT_YEARS only)")
        if hasattr(ev, "format_metrics_table"):
            print(ev.format_metrics_table(rows_for_table, index_key="model"))
        else:
            df = pd.DataFrame(rows_for_table).sort_values("model").reset_index(drop=True)
            print(df.to_string(index=False))

    return metrics_by_model


if __name__ == "__main__":
    run_all(
        write_oof=True,
        verbose=bool(CONFIG.get("VERBOSE", True)),
    )
