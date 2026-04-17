from __future__ import annotations

"""
Build the weekly metro panel used by the model scripts.

Starting from the Step 6 report-level export, this module aggregates to a
weekly metro panel, adds padded history, and prepares fold-ready metadata.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


# ==================== STUDY CONFIG ====================
STUDY_START = pd.Timestamp("2020-04-01")
STUDY_END = pd.Timestamp("2024-12-31")
HORIZON_WEEKS = 1
SEQ_LEN_WEEKS = 52
PAD_WEEKS = SEQ_LEN_WEEKS
# ======================================================


# Features
METRO_RAW_COL = "Metro (CBSA)"
REPORT_DATE_COL = "createday"
SYMPTOM_COL = "is_symptom"
VAX_COL = "received_flu_vaccine_fully"
AGE_COL = "age_cat"
GENDER_COL = "gender"
PPT_COL = "ppt"
TMAX_COL = "tmax"
TMIN_COL = "tmin"
RAW_INPUT_COLS = [
    REPORT_DATE_COL,
    GENDER_COL,
    AGE_COL,
    SYMPTOM_COL,
    VAX_COL,
    PPT_COL,
    TMAX_COL,
    TMIN_COL,
    METRO_RAW_COL,
]


# Canonical weekly feature
METRO_COL = "metro"
WEEK_START_COL = "week_start"
TARGET_WEEK_START_COL = "target_week_start"
N_REPORTS_COL = "n_reports"
WEIGHT_COL = "weight"
RATE_COL = "p_tilde"
RATE_LAG1_COL = "p_tilde_lag_1"
RATE_LEAD1_COL = "p_tilde_lead1"
LABEL_COL = "flu_label"


def _list_data_files(data_dir: str) -> List[Path]:
    """List yearly CSV files from a flat input path."""
    p = Path(data_dir)

    if p.is_dir():
        return sorted(p.glob("*.csv"))

    if p.is_file():
        return [p]

    return []


def _read_one(path: Path) -> pd.DataFrame:
    """Read one CSV."""
    df = pd.read_csv(
        path,
        parse_dates=[REPORT_DATE_COL],
        dtype={
            METRO_RAW_COL: "string",
            GENDER_COL: "category",
            AGE_COL: "category",
            SYMPTOM_COL: "Int8",
            VAX_COL: "Int8",
            PPT_COL: "float32",
            TMAX_COL: "float32",
            TMIN_COL: "float32",
        },
        low_memory=False,
    )

    missing = [c for c in RAW_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing input columns: {missing}")

    return df


def _week_start_monday(d: pd.Series) -> pd.Series:
    """Map dates to Monday-anchored week starts."""
    dd = pd.to_datetime(d, errors="raise")
    if getattr(dd.dt, "tz", None) is not None:
        dd = dd.dt.tz_convert("UTC").dt.tz_localize(None)
    dd = dd.dt.normalize()
    return dd - pd.to_timedelta(dd.dt.weekday, unit="D")


def prepare_weekly_panel(data_dir: str, *, verbose: bool = True) -> pd.DataFrame:
    """Read preprocessed files and build the weekly metro panel."""
    files = _list_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    raw = pd.concat([_read_one(f) for f in files], ignore_index=True)

    if verbose:
        print(f"[dataset_builder] loaded {len(files)} file(s) -> rows={len(raw):,}")

    raw = raw.rename(columns={METRO_RAW_COL: METRO_COL}).copy()
    raw[REPORT_DATE_COL] = pd.to_datetime(raw[REPORT_DATE_COL], errors="raise").dt.normalize()
    raw = raw.loc[
        (raw[REPORT_DATE_COL] >= STUDY_START) & (raw[REPORT_DATE_COL] <= STUDY_END)
    ].reset_index(drop=True)

    weekly = _aggregate_reports_to_weekly(raw, verbose=verbose)
    weekly = add_target_metadata(weekly)

    study_start_week = STUDY_START - pd.Timedelta(days=int(STUDY_START.weekday()))
    study_end_week = STUDY_END - pd.Timedelta(days=int(STUDY_END.weekday()))
    pad_start_week = study_start_week - pd.Timedelta(weeks=PAD_WEEKS)

    weekly = weekly.loc[
        (weekly[WEEK_START_COL] >= pad_start_week)
        & (weekly[WEEK_START_COL] <= study_end_week)
        & (weekly[TARGET_WEEK_START_COL] <= study_end_week)
    ].reset_index(drop=True)

    return weekly


def _aggregate_reports_to_weekly(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Aggregate daily reports to a complete metro-week panel."""
    d = df.copy()
    d["date"] = pd.to_datetime(d[REPORT_DATE_COL], errors="raise").dt.normalize()
    d[WEEK_START_COL] = _week_start_monday(d["date"])

    sym = pd.to_numeric(d[SYMPTOM_COL], errors="coerce").fillna(0).astype(int).clip(0, 1)
    vax = pd.to_numeric(d[VAX_COL], errors="coerce").fillna(0).astype(int).clip(0, 1)

    g = d[GENDER_COL].astype(str).str.strip()
    a = d[AGE_COL].astype(str).str.strip()

    d["_sym"] = sym
    d["_vax"] = vax
    d["_female"] = g.eq("FEMALE").astype(int)
    d["_au19"] = a.eq("<19").astype(int)
    d["_a19_64"] = a.eq("19-64").astype(int)
    d["_a64p"] = a.eq(">64").astype(int)

    wk = (
        d.groupby([METRO_COL, WEEK_START_COL], sort=False)
        .agg(
            n_reports=("_sym", "size"),
            sym_count=("_sym", "sum"),
            vax_sum=("_vax", "sum"),
            female_sum=("_female", "sum"),
            u19_sum=("_au19", "sum"),
            a19_64_sum=("_a19_64", "sum"),
            a64p_sum=("_a64p", "sum"),
        )
        .reset_index()
    )

    wk[RATE_COL] = (wk["sym_count"].astype(float) + 0.5) / (wk[N_REPORTS_COL].astype(float) + 1.0)

    n = wk[N_REPORTS_COL].astype(float)
    wk["vacc_prop"] = (wk["vax_sum"].astype(float) + 0.5) / (n + 1.0)
    wk["gender_female_share"] = (wk["female_sum"].astype(float) + 0.5) / (n + 1.0)

    denom_age = n + 1.5
    wk["age_u19_share"] = (wk["u19_sum"].astype(float) + 0.5) / denom_age
    wk["age_19_64_share"] = (wk["a19_64_sum"].astype(float) + 0.5) / denom_age
    wk["age_64_share"] = (wk["a64p_sum"].astype(float) + 0.5) / denom_age
    wk = wk.drop(columns=["vax_sum", "female_sum", "u19_sum", "a19_64_sum", "a64p_sum"])

    d2 = d[[METRO_COL, "date", PPT_COL, TMAX_COL, TMIN_COL]].copy()
    d2[PPT_COL] = pd.to_numeric(d2[PPT_COL], errors="coerce")
    d2[TMAX_COL] = pd.to_numeric(d2[TMAX_COL], errors="coerce")
    d2[TMIN_COL] = pd.to_numeric(d2[TMIN_COL], errors="coerce")

    daily = (
        d2.groupby([METRO_COL, "date"], sort=False)
        .agg(
            ppt_day=(PPT_COL, "mean"),
            tmax_day=(TMAX_COL, "mean"),
            tmin_day=(TMIN_COL, "mean"),
        )
        .reset_index()
    )

    daily[WEEK_START_COL] = _week_start_monday(daily["date"])
    daily["tavg_day"] = (daily["tmax_day"] + daily["tmin_day"]) / 2.0
    daily["diurnal_day"] = daily["tmax_day"] - daily["tmin_day"]

    wk_w = (
        daily.groupby([METRO_COL, WEEK_START_COL], sort=False)
        .agg(
            ppt_sum=("ppt_day", "sum"),
            wet_days=("ppt_day", lambda s: float(np.sum(np.asarray(s) > 0))),
            tmin_mean=("tmin_day", "mean"),
            tmax_mean=("tmax_day", "mean"),
            tavg_mean=("tavg_day", "mean"),
            diurnal_range_mean=("diurnal_day", "mean"),
        )
        .reset_index()
    )

    wk_w["ppt_log_sum"] = np.log1p(pd.to_numeric(wk_w["ppt_sum"], errors="coerce").fillna(0.0))
    wk_w["ppt_any"] = (pd.to_numeric(wk_w["ppt_sum"], errors="coerce").fillna(0.0) > 0).astype(int)
    wk_w = wk_w.drop(columns=["ppt_sum"])

    weekly = wk.merge(wk_w, on=[METRO_COL, WEEK_START_COL], how="left")

    pad_start = STUDY_START - pd.Timedelta(weeks=PAD_WEEKS)
    pad_start = pad_start - pd.Timedelta(days=int(pad_start.weekday()))
    end_monday = STUDY_END - pd.Timedelta(days=int(STUDY_END.weekday()))
    all_weeks = pd.date_range(pad_start, end_monday, freq="W-MON")

    metros = weekly[METRO_COL].astype(str).unique()
    grid = pd.MultiIndex.from_product(
        [metros, all_weeks],
        names=[METRO_COL, WEEK_START_COL],
    ).to_frame(index=False)
    weekly = grid.merge(weekly, on=[METRO_COL, WEEK_START_COL], how="left")

    weekly[N_REPORTS_COL] = pd.to_numeric(weekly[N_REPORTS_COL], errors="coerce").fillna(0).astype(int)
    weekly["sym_count"] = pd.to_numeric(weekly["sym_count"], errors="coerce").fillna(0).astype(int)

    obs_n = weekly.loc[weekly[N_REPORTS_COL] > 0, N_REPORTS_COL].astype(float)
    cap95 = float(obs_n.quantile(0.95)) if len(obs_n) else 0.0
    mean_n = float(obs_n.mean()) if len(obs_n) else 1.0

    weekly[WEIGHT_COL] = np.minimum(weekly[N_REPORTS_COL].astype(float), cap95) / mean_n
    weekly.loc[weekly[N_REPORTS_COL] <= 0, WEIGHT_COL] = 0.0
    weekly.loc[weekly[N_REPORTS_COL] <= 0, RATE_COL] = np.nan

    weekly = weekly.sort_values([METRO_COL, WEEK_START_COL], kind="mergesort").reset_index(drop=True)
    weekly[RATE_LAG1_COL] = weekly.groupby(METRO_COL, sort=False)[RATE_COL].shift(1)

    if verbose:
        print(
            f"[dataset_builder] aggregated reports->weekly: rows={len(weekly):,} "
            f"metros={len(metros):,} weeks={len(all_weeks):,}"
        )

    return weekly


def retain_metros(df: pd.DataFrame, metros_keep: Sequence[str], *, verbose: bool = True) -> pd.DataFrame:
    """Keep only requested metros."""
    metros_keep = [str(x) for x in metros_keep]

    before_rows = len(df)
    before_metros = df[METRO_COL].nunique(dropna=True)
    out = df.loc[df[METRO_COL].astype(str).isin(metros_keep)].reset_index(drop=True)

    if verbose:
        after_rows = len(out)
        after_metros = out[METRO_COL].nunique(dropna=True)
        print(
            f"[dataset_builder] retain_metros: requested={len(metros_keep)} | "
            f"metros {after_metros}/{before_metros} | rows {after_rows:,}/{before_rows:,}"
        )

    return out


def retain_metros_min_weeks(
    weekly: pd.DataFrame,
    *,
    min_weeks: int,
    weight_col: str = WEIGHT_COL,
    verbose: bool = True,
) -> pd.DataFrame:
    """Keep metros with at least min_weeks observed weeks."""
    w = pd.to_numeric(weekly[weight_col], errors="coerce").fillna(0.0)
    obs = w > 0

    all_metros = weekly[METRO_COL].astype(str).unique()
    counts = (
        weekly.loc[obs]
        .groupby(METRO_COL, sort=False)[WEEK_START_COL]
        .nunique()
        .reindex(all_metros, fill_value=0)
    )
    metros_keep = counts.index[counts >= int(min_weeks)].astype(str).tolist()

    before_rows = len(weekly)
    before_metros = weekly[METRO_COL].nunique(dropna=True)
    out = retain_metros(weekly, metros_keep, verbose=False)

    if verbose:
        after_rows = len(out)
        after_metros = out[METRO_COL].nunique(dropna=True)
        print(
            f"[dataset_builder] retain_metros_min_weeks: min_weeks={int(min_weeks)} | "
            f"metros {after_metros}/{before_metros} | rows {after_rows:,}/{before_rows:,}"
        )

    return out


def add_target_metadata(weekly: pd.DataFrame) -> pd.DataFrame:
    """Add one-week-ahead targets and target-week metadata."""
    df = weekly.sort_values([METRO_COL, WEEK_START_COL], kind="mergesort").copy()
    df[WEEK_START_COL] = pd.to_datetime(df[WEEK_START_COL], errors="raise").dt.normalize()
    df[RATE_LEAD1_COL] = df.groupby(METRO_COL, sort=False)[RATE_COL].shift(-HORIZON_WEEKS)
    df[TARGET_WEEK_START_COL] = df[WEEK_START_COL] + pd.Timedelta(weeks=HORIZON_WEEKS)

    tiso = df[TARGET_WEEK_START_COL].dt.isocalendar()
    df["year_snap"] = tiso["year"].astype(int)
    df["iso_week"] = tiso["week"].astype(int)
    return df


@dataclass(frozen=True)
class ThresholdPolicy:
    q_event: float
    post_start_year: int


def _eligible_mask_for_threshold(df: pd.DataFrame) -> np.ndarray:
    """Rows eligible for fold threshold construction."""
    rate = pd.to_numeric(df[RATE_LEAD1_COL], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    nrep = pd.to_numeric(df[N_REPORTS_COL], errors="coerce").fillna(0).to_numpy(dtype=float)
    return (w > 0) & (nrep > 0) & np.isfinite(rate)


def compute_fold_T0p9(train_df: pd.DataFrame, policy: ThresholdPolicy) -> float:
    """Compute the fold-specific training quantile threshold."""
    base_ok = _eligible_mask_for_threshold(train_df)
    if not np.any(base_ok):
        raise ValueError("No eligible training rows for threshold computation.")

    ys = pd.to_numeric(train_df["year_snap"], errors="coerce").to_numpy(dtype=int)
    post_ok = base_ok & (ys >= int(policy.post_start_year))
    use_ok = post_ok if np.any(post_ok) else base_ok

    rate = pd.to_numeric(train_df.loc[use_ok, RATE_LEAD1_COL], errors="coerce").to_numpy(dtype=float)
    return float(np.quantile(rate, float(policy.q_event)))


def add_fold_labels(df: pd.DataFrame, T: float) -> pd.DataFrame:
    """Add binary event labels from the fold threshold."""
    out = df.copy()
    r = pd.to_numeric(out[RATE_LEAD1_COL], errors="coerce").to_numpy(dtype=float)
    y = (r >= float(T)).astype(float)
    y[~np.isfinite(r)] = np.nan
    out[LABEL_COL] = y
    return out


def weighted_prevalence(df: pd.DataFrame) -> float:
    """Exposure-weighted prevalence on eligible rows."""
    y = pd.to_numeric(df[LABEL_COL], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    keep = np.isfinite(w) & (w > 0) & np.isfinite(y)
    if not np.any(keep):
        raise ValueError("No eligible rows for prevalence computation.")

    return float(np.average(y[keep], weights=w[keep]))


def make_fold_frames(
    weekly: pd.DataFrame,
    *,
    val_year: int,
    policy: ThresholdPolicy,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    """Build train, validation, and validation-context frames for one fold."""
    w = weekly.copy()
    ysnap = pd.to_numeric(w["year_snap"], errors="coerce").to_numpy(dtype=int)

    train_df = w.loc[ysnap <= int(val_year) - 1].copy()
    val_df = w.loc[ysnap == int(val_year)].copy()
    if val_df.empty:
        raise ValueError(f"val_year={val_year} produced an empty validation frame.")

    T0p9 = compute_fold_T0p9(train_df, policy)
    train_df = add_fold_labels(train_df, T0p9)
    val_df = add_fold_labels(val_df, T0p9)
    tau = weighted_prevalence(train_df)

    val_origin_min = pd.to_datetime(val_df[WEEK_START_COL], errors="raise").min()
    val_origin_max = pd.to_datetime(val_df[WEEK_START_COL], errors="raise").max()

    ctx_start = val_origin_min - pd.Timedelta(weeks=SEQ_LEN_WEEKS - 1)
    ctx_end = val_origin_max

    val_ctx_df = w.loc[
        (pd.to_datetime(w[WEEK_START_COL], errors="raise") >= ctx_start)
        & (pd.to_datetime(w[WEEK_START_COL], errors="raise") <= ctx_end)
    ].copy()

    train_df = train_df.sort_values([METRO_COL, WEEK_START_COL], kind="mergesort").reset_index(drop=True)
    val_df = val_df.sort_values([METRO_COL, WEEK_START_COL], kind="mergesort").reset_index(drop=True)
    val_ctx_df = val_ctx_df.sort_values([METRO_COL, WEEK_START_COL], kind="mergesort").reset_index(drop=True)

    return train_df, val_df, val_ctx_df, float(T0p9), float(tau)


def inject_val_labels(val_ctx_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    """Attach validation labels to the context frame."""
    ctx = val_ctx_df.copy()
    lab = val_df.loc[:, [METRO_COL, WEEK_START_COL, LABEL_COL]].copy()

    if LABEL_COL in ctx.columns:
        ctx = ctx.drop(columns=[LABEL_COL])

    return ctx.merge(lab, on=[METRO_COL, WEEK_START_COL], how="left")


def filter_scored_origins(
    df: pd.DataFrame,
    *,
    weight_col: str = WEIGHT_COL,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Keep observed rows with binary labels."""
    w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)
    y = pd.to_numeric(df[label_col], errors="coerce")
    return df.loc[(w > 0) & y.isin([0.0, 1.0])].copy()


def _eligible_mask_min_obs_sorted(
    work: pd.DataFrame,
    *,
    min_observed_weeks: int,
    seq_len_weeks: int = SEQ_LEN_WEEKS,
    metro_col: str = METRO_COL,
    weight_col: str = WEIGHT_COL,
    label_col: str = LABEL_COL,
) -> np.ndarray:
    """Assume work is already sorted by metro and week."""
    obs = (pd.to_numeric(work[weight_col], errors="coerce").fillna(0.0) > 0).astype(np.int32)
    y = pd.to_numeric(work[label_col], errors="coerce")

    obs_in_win = (
        obs.groupby(work[metro_col].astype(str), sort=False)
        .rolling(int(seq_len_weeks), min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    label_ok = y.isin([0.0, 1.0])
    elig = (obs > 0) & label_ok & (obs_in_win >= int(min_observed_weeks))
    return elig.to_numpy(dtype=bool)


def eligible_origins_min_obs(
    ctx_df: pd.DataFrame,
    *,
    min_observed_weeks: int,
    seq_len_weeks: int = SEQ_LEN_WEEKS,
    metro_col: str = METRO_COL,
    week_col: str = WEEK_START_COL,
    weight_col: str = WEIGHT_COL,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Return origin keys with enough observed history."""
    work = ctx_df.loc[:, [metro_col, week_col, weight_col, label_col]].copy()
    work[metro_col] = work[metro_col].astype(str)
    work[week_col] = pd.to_datetime(work[week_col], errors="raise").dt.normalize()
    work = work.sort_values([metro_col, week_col], kind="mergesort").reset_index(drop=True)

    mask = _eligible_mask_min_obs_sorted(
        work,
        min_observed_weeks=int(min_observed_weeks),
        seq_len_weeks=int(seq_len_weeks),
        metro_col=metro_col,
        weight_col=weight_col,
        label_col=label_col,
    )
    return work.loc[mask, [metro_col, week_col]].drop_duplicates().reset_index(drop=True)


def filter_origins_min_obs(
    origin_df: pd.DataFrame,
    *,
    ctx_df: pd.DataFrame,
    min_observed_weeks: int,
    seq_len_weeks: int = SEQ_LEN_WEEKS,
    metro_col: str = METRO_COL,
    week_col: str = WEEK_START_COL,
) -> pd.DataFrame:
    """Filter origins to those with enough observed history."""
    keys = eligible_origins_min_obs(
        ctx_df,
        min_observed_weeks=int(min_observed_weeks),
        seq_len_weeks=int(seq_len_weeks),
        metro_col=metro_col,
        week_col=week_col,
    )

    out = origin_df.copy()
    out[metro_col] = out[metro_col].astype(str)
    out[week_col] = pd.to_datetime(out[week_col], errors="raise").dt.normalize()
    return out.merge(keys, on=[metro_col, week_col], how="inner")


def build_sequences(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    min_observed_weeks: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-length sequences for event prediction."""
    cols_needed = [
        METRO_COL,
        WEEK_START_COL,
        "year_snap",
        "iso_week",
        LABEL_COL,
        WEIGHT_COL,
        *feature_cols,
    ]

    work = df.loc[:, cols_needed].copy()
    work[METRO_COL] = work[METRO_COL].astype(str)
    work[WEEK_START_COL] = pd.to_datetime(work[WEEK_START_COL], errors="raise").dt.normalize()
    work = work.sort_values([METRO_COL, WEEK_START_COL], kind="mergesort").reset_index(drop=True)

    seq_len = int(SEQ_LEN_WEEKS)
    n_feat = int(len(feature_cols))

    elig_mask_all = _eligible_mask_min_obs_sorted(
        work.loc[:, [METRO_COL, WEEK_START_COL, WEIGHT_COL, LABEL_COL]],
        min_observed_weeks=int(min_observed_weeks),
        seq_len_weeks=seq_len,
        metro_col=METRO_COL,
        weight_col=WEIGHT_COL,
        label_col=LABEL_COL,
    )

    X_all = work.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y_all = pd.to_numeric(work[LABEL_COL], errors="coerce").to_numpy(dtype=float)
    w_all = pd.to_numeric(work[WEIGHT_COL], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    metro_all = work[METRO_COL].astype(str).to_numpy()
    year_all = pd.to_numeric(work["year_snap"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)
    week_all = pd.to_numeric(work["iso_week"], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)
    ws_all = pd.to_datetime(work[WEEK_START_COL], errors="raise").to_numpy(dtype="datetime64[ns]")

    group_indices = work.groupby(METRO_COL, sort=False).indices

    out_X: list[np.ndarray] = []
    out_y: list[int] = []
    out_w: list[float] = []
    out_m: list[object] = []
    out_year: list[int] = []
    out_week: list[int] = []
    out_ws: list[np.datetime64] = []

    for _, idxs in group_indices.items():
        idx = np.asarray(idxs, dtype=np.int64)
        if idx.size == 0:
            continue

        for j in range(idx.size):
            global_row = int(idx[j])
            if not bool(elig_mask_all[global_row]):
                continue

            start_j = max(0, j - seq_len + 1)
            win_idx = idx[start_j : j + 1]

            X_win = X_all[win_idx, :]
            if X_win.shape[0] < seq_len:
                pad = np.zeros((seq_len - X_win.shape[0], n_feat), dtype=float)
                X_win = np.vstack([pad, X_win])

            out_X.append(X_win.astype(np.float32, copy=False))
            out_y.append(int(y_all[global_row]))
            out_w.append(float(w_all[global_row]))
            out_m.append(metro_all[global_row])
            out_year.append(int(year_all[global_row]))
            out_week.append(int(week_all[global_row]))
            out_ws.append(ws_all[global_row])

    if not out_X:
        return (
            np.zeros((0, seq_len, n_feat), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=object),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype="datetime64[ns]"),
        )

    return (
        np.stack(out_X, axis=0),
        np.asarray(out_y, dtype=np.int32),
        np.asarray(out_w, dtype=np.float32),
        np.asarray(out_m, dtype=object),
        np.asarray(out_year, dtype=np.int32),
        np.asarray(out_week, dtype=np.int32),
        np.asarray(out_ws, dtype="datetime64[ns]"),
    )


__all__ = [
    "STUDY_START",
    "STUDY_END",
    "HORIZON_WEEKS",
    "SEQ_LEN_WEEKS",
    "METRO_RAW_COL",
    "REPORT_DATE_COL",
    "RAW_INPUT_COLS",
    "prepare_weekly_panel",
    "retain_metros",
    "retain_metros_min_weeks",
    "add_target_metadata",
    "ThresholdPolicy",
    "compute_fold_T0p9",
    "add_fold_labels",
    "weighted_prevalence",
    "make_fold_frames",
    "inject_val_labels",
    "filter_scored_origins",
    "eligible_origins_min_obs",
    "filter_origins_min_obs",
    "build_sequences",
]
