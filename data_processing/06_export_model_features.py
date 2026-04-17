"""
Step 6: Export the final report-level dataset for model ingestion.

Purpose
-------
Starting from the yearly file, apply the final weather policy,
fill CBSA where needed, assign Metro (CBSA), and export the exact raw
report-level schema expected by dataset_builder.py.

Input
-----
step_5/step_5_<year>/step_5_<year>.csv
ZIP_code_1.xlsx
ZIP_code_2.xlsx

Output
------
step_6/step_6_<year>/step_6_<year>.csv
summary/step_6_yearly_counts.csv
"""

from pathlib import Path
import os

import pandas as pd


# ==================== STUDY CONFIG ====================
YEARS = [2020, 2021, 2022, 2023, 2024]

BASE = Path(os.getenv("GFV_SPARK_BASE_PATH", "path"))
STEP5_BASE = BASE / "step_5"
STEP6_BASE = BASE / "step_6"
SUMMARY_DIR = BASE / "summary"

ZIP1_XLSX = BASE / "ZIP_code_1.xlsx"
ZIP2_XLSX = BASE / "ZIP_code_2.xlsx"

# Final weather handling policy:
#   "keep"   -> keep rows even if ppt/tmax/tmin remain missing (default)
#   "drop"   -> drop rows missing any of ppt/tmax/tmin
#   "impute" -> fill missing weather using CBSA-date, then CBSA-month,
#               then CBSA-overall means
WEATHER_POLICY = "drop"

WEATHER_COLS = ["ppt", "tmax", "tmin"]

# Exact raw report-level schema expected by dataset_builder.py
KEEP = [
    "createday",
    "gender",
    "age_cat",
    "is_symptom",
    "received_flu_vaccine_fully",
    "ppt",
    "tmax",
    "tmin",
    "Metro (CBSA)",
]
# ======================================================


def normalize_zip(series: pd.Series) -> pd.Series:
    """
    Standardize ZIP codes to a 5-digit U.S. ZIP representation.
    """
    s = series.astype("string").str.strip().str.replace(r"\D", "", regex=True)
    s = s.where(s.str.len() >= 1, pd.NA).str.zfill(5).str[:5]
    s = s.mask(s == "00000")
    return s


def normalize_cbsa(series: pd.Series) -> pd.Series:
    """
    Standardize CBSA codes to a 5-digit character representation.
    """
    s = series.astype("string").str.strip().str.replace(r"\D", "", regex=True)
    s = s.where(s.str.len() >= 1, pd.NA).str.zfill(5).str[:5]
    s = s.mask(s == "00000")
    return s


def to_yyyymm(series: pd.Series) -> pd.Series:
    """
    Convert a date-like series to YYYYMM string.
    """
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime("%Y%m")


def coerce_binary_01(s: pd.Series) -> pd.Series:
    """
    Convert Boolean-like inputs to Int64-compatible 0/1.
    """
    if pd.api.types.is_bool_dtype(s):
        return s.astype("Int64")

    st = s.astype("string").str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    out.loc[st.isin(["true", "t", "yes", "y", "1"])] = 1
    out.loc[st.isin(["false", "f", "no", "n", "0"])] = 0
    return out


def load_zip2_zip_to_cbsa() -> pd.DataFrame:
    """
    Load ZIP_code_2.xlsx as a ZIP -> CBSA reference.
    """
    if not ZIP2_XLSX.exists():
        return pd.DataFrame(columns=["zipcode", "CBSA_code"])

    z2 = pd.read_excel(ZIP2_XLSX, dtype="string")
    z2.columns = [c.strip() for c in z2.columns]

    required = {"ZIP", "CBSA Code"}
    if not required.issubset(z2.columns):
        raise KeyError(
            f"{ZIP2_XLSX} must contain columns {sorted(required)}. "
            f"Found: {list(z2.columns)}"
        )

    out = pd.DataFrame({
        "zipcode": normalize_zip(z2["ZIP"]),
        "CBSA_code": normalize_cbsa(z2["CBSA Code"]),
    })

    out = out.dropna(subset=["zipcode", "CBSA_code"]).drop_duplicates(
        subset=["zipcode"], keep="first"
    )
    return out


def load_cbsa_to_metro_ref2() -> pd.DataFrame:
    """
    Load ZIP_code_2.xlsx as a CBSA -> Metro (CBSA) reference.
    """
    if not ZIP2_XLSX.exists():
        return pd.DataFrame(columns=["CBSA_code", "Metro (CBSA)"])

    z2 = pd.read_excel(ZIP2_XLSX, dtype="string")
    z2.columns = [c.strip() for c in z2.columns]

    required = {"CBSA Code", "Metro (CBSA)"}
    if not required.issubset(z2.columns):
        raise KeyError(
            f"{ZIP2_XLSX} must contain columns {sorted(required)}. "
            f"Found: {list(z2.columns)}"
        )

    out = pd.DataFrame({
        "CBSA_code": normalize_cbsa(z2["CBSA Code"]),
        "Metro (CBSA)": z2["Metro (CBSA)"].astype("string").str.strip(),
    })

    out = out.dropna(subset=["CBSA_code", "Metro (CBSA)"]).drop_duplicates(
        subset=["CBSA_code"], keep="first"
    )
    return out


def load_zip_to_metro_ref1() -> pd.DataFrame:
    """
    Load ZIP_code_1.xlsx as a ZIP -> Metro (CBSA) fallback reference.
    """
    if not ZIP1_XLSX.exists():
        return pd.DataFrame(columns=["zipcode", "Metro (CBSA)"])

    z1 = pd.read_excel(ZIP1_XLSX, dtype="string")
    z1.columns = [c.strip() for c in z1.columns]

    required = {"ZIP", "Metro (CBSA)"}
    if not required.issubset(z1.columns):
        raise KeyError(
            f"{ZIP1_XLSX} must contain columns {sorted(required)}. "
            f"Found: {list(z1.columns)}"
        )

    out = pd.DataFrame({
        "zipcode": normalize_zip(z1["ZIP"]),
        "Metro (CBSA)": z1["Metro (CBSA)"].astype("string").str.strip(),
    })

    out = out.dropna(subset=["zipcode", "Metro (CBSA)"]).drop_duplicates(
        subset=["zipcode"], keep="first"
    )
    return out


def attach_cbsa_from_zip(df: pd.DataFrame, zip2_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing CBSA_code values from ZIP_code_2.xlsx using zipcode.
    """
    out = df.copy()

    if "zipcode" not in out.columns:
        if "CBSA_code" not in out.columns:
            out["CBSA_code"] = pd.NA
        return out

    out["zipcode"] = normalize_zip(out["zipcode"])

    if "CBSA_code" in out.columns:
        out["CBSA_code"] = normalize_cbsa(out["CBSA_code"])
    else:
        out["CBSA_code"] = pd.NA

    if zip2_ref.empty:
        return out

    out = out.merge(zip2_ref, on="zipcode", how="left", suffixes=("", "_ref"))
    if "CBSA_code_ref" in out.columns:
        out["CBSA_code"] = out["CBSA_code"].fillna(out["CBSA_code_ref"])
        out = out.drop(columns=["CBSA_code_ref"])

    return out


def assign_metro(
    df: pd.DataFrame,
    cbsa_map: pd.DataFrame,
    zip_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign Metro (CBSA) using CBSA first and ZIP fallback.
    """
    out = df.copy()

    if "zipcode" in out.columns:
        out["zipcode"] = normalize_zip(out["zipcode"])
    else:
        out["zipcode"] = pd.NA

    if "CBSA_code" in out.columns:
        out["CBSA_code"] = normalize_cbsa(out["CBSA_code"])
    else:
        out["CBSA_code"] = pd.NA

    if "Metro (CBSA)" in out.columns:
        out = out.drop(columns=["Metro (CBSA)"])

    if not cbsa_map.empty:
        out = out.merge(cbsa_map, on="CBSA_code", how="left")

    if not zip_map.empty:
        out = out.merge(
            zip_map.rename(columns={"Metro (CBSA)": "_Metro_zip"}),
            on="zipcode",
            how="left",
        )
        if "Metro (CBSA)" in out.columns:
            out["Metro (CBSA)"] = out["Metro (CBSA)"].where(
                out["Metro (CBSA)"].notna(), out["_Metro_zip"]
            )
        else:
            out["Metro (CBSA)"] = out["_Metro_zip"]
        out = out.drop(columns=["_Metro_zip"])

    return out


def impute_weather(df: pd.DataFrame, weather_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Impute missing weather using CBSA-date, then CBSA-month, then CBSA overall.
    """
    out = df.copy()
    qa = {}

    if not weather_cols:
        return out, qa

    for c in weather_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        qa[f"{c}_na_before"] = float(out[c].isna().mean())

    if "date" in out.columns:
        out["_month"] = to_yyyymm(out["date"])
    elif "createday" in out.columns:
        out["_month"] = to_yyyymm(out["createday"])
    else:
        out["_month"] = pd.NA

    if {"CBSA_code", "date"}.issubset(out.columns):
        grp = (
            out.groupby(["CBSA_code", "date"], as_index=False)[weather_cols]
            .mean(min_count=1)
            .add_suffix("_g")
            .rename(columns={"CBSA_code_g": "CBSA_code", "date_g": "date"})
        )
        out = out.merge(grp, on=["CBSA_code", "date"], how="left")
        for c in weather_cols:
            g = f"{c}_g"
            if g in out.columns:
                out[c] = out[c].astype("Float64").fillna(out[g])
                out = out.drop(columns=[g])

    if {"CBSA_code", "_month"}.issubset(out.columns):
        grp = (
            out.groupby(["CBSA_code", "_month"], as_index=False)[weather_cols]
            .mean(min_count=1)
            .add_suffix("_m")
            .rename(columns={"CBSA_code_m": "CBSA_code", "_month_m": "_month"})
        )
        out = out.merge(grp, on=["CBSA_code", "_month"], how="left")
        for c in weather_cols:
            m = f"{c}_m"
            if m in out.columns:
                out[c] = out[c].astype("Float64").fillna(out[m])
                out = out.drop(columns=[m])

    if "CBSA_code" in out.columns:
        grp = (
            out.groupby(["CBSA_code"], as_index=False)[weather_cols]
            .mean(min_count=1)
            .add_suffix("_c")
            .rename(columns={"CBSA_code_c": "CBSA_code"})
        )
        out = out.merge(grp, on=["CBSA_code"], how="left")
        for c in weather_cols:
            cc = f"{c}_c"
            if cc in out.columns:
                out[c] = out[c].astype("Float64").fillna(out[cc])
                out = out.drop(columns=[cc])

    out = out.drop(columns=["_month"], errors="ignore")

    for c in weather_cols:
        qa[f"{c}_na_after"] = float(out[c].isna().mean())

    return out, qa


def apply_weather_policy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Apply the final weather policy.
    """
    out = df.copy()
    present = [c for c in WEATHER_COLS if c in out.columns]

    for c in present:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    qa = {f"{c}_na_before_policy": float(out[c].isna().mean()) for c in present}

    if WEATHER_POLICY == "keep":
        qa.update({f"{c}_na_after_policy": float(out[c].isna().mean()) for c in present})
        return out, qa

    if WEATHER_POLICY == "drop":
        if present:
            out = out.dropna(subset=present).copy()
        qa.update({
            f"{c}_na_after_policy": float(out[c].isna().mean())
            for c in present if c in out.columns
        })
        return out, qa

    if WEATHER_POLICY == "impute":
        out, imp_qa = impute_weather(out, present)
        qa.update(imp_qa)
        qa.update({f"{c}_na_after_policy": float(out[c].isna().mean()) for c in present})
        return out, qa

    raise ValueError(f"Unknown WEATHER_POLICY: {WEATHER_POLICY}")


def finalize_model_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce the final model-input columns and retain exactly those columns.
    """
    out = df.copy()

    if "date" in out.columns and "createday" not in out.columns:
        out["createday"] = out["date"]

    if "metro" in out.columns and "Metro (CBSA)" not in out.columns:
        out["Metro (CBSA)"] = out["metro"]

    for c in ["ppt", "tmax", "tmin"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float64")

    for c in ["is_symptom", "received_flu_vaccine_fully"]:
        if c in out.columns:
            out[c] = coerce_binary_01(out[c])

    keep = [c for c in KEEP if c in out.columns]
    out = out[keep].drop_duplicates().copy()
    return out


def process_one_year(
    year: int,
    zip2_zip_ref: pd.DataFrame,
    cbsa_map: pd.DataFrame,
    zip_map: pd.DataFrame,
) -> dict:
    """
    Process one yearly Step 5 file into the final model-input report file.
    """
    in_fp = STEP5_BASE / f"step_5_{year}" / f"step_5_{year}.csv"
    out_dir = STEP6_BASE / f"step_6_{year}"
    out_fp = out_dir / f"step_6_{year}.csv"

    if not in_fp.exists():
        print(f"[{year}] input file missing: {in_fp}")
        return {
            "year": year,
            "rows_initial": 0,
            "rows_after_weather_policy": 0,
            "rows_with_metro": 0,
            "rows_step6_final": 0,
        }

    df = pd.read_csv(
        in_fp,
        dtype={"zipcode": "string", "zipcode_raw": "string", "CBSA_code": "string"},
        low_memory=False,
    )
    rows_initial = len(df)

    df = attach_cbsa_from_zip(df, zip2_zip_ref)
    df, qa = apply_weather_policy(df)
    rows_after_weather_policy = len(df)

    df = assign_metro(df, cbsa_map, zip_map)
    rows_with_metro = (
        int(df["Metro (CBSA)"].notna().sum())
        if "Metro (CBSA)" in df.columns else 0
    )

    df = finalize_model_input_schema(df)

    if "Metro (CBSA)" in df.columns:
        df = df.dropna(subset=["Metro (CBSA)"]).copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)

    print(f"[{year}] saved -> {out_fp}")
    print(f"[{year}] rows_initial              = {rows_initial:,}")
    print(f"[{year}] rows_after_weather_policy = {rows_after_weather_policy:,}")
    print(f"[{year}] rows_with_metro           = {rows_with_metro:,}")
    print(f"[{year}] rows_step6_final          = {len(df):,}")
    if qa:
        print(f"[{year}] weather QA = {qa}")

    return {
        "year": year,
        "rows_initial": rows_initial,
        "rows_after_weather_policy": rows_after_weather_policy,
        "rows_with_metro": rows_with_metro,
        "rows_step6_final": len(df),
    }


def main() -> None:
    """
    Run Step 6 for all configured study years and write the yearly summary.
    """
    STEP6_BASE.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    zip2_zip_ref = load_zip2_zip_to_cbsa()
    cbsa_map = load_cbsa_to_metro_ref2()
    zip_map = load_zip_to_metro_ref1()

    summaries = [
        process_one_year(year, zip2_zip_ref, cbsa_map, zip_map)
        for year in YEARS
    ]
    summary_df = pd.DataFrame(summaries)

    total_row = pd.DataFrame([{
        "year": "TOTAL",
        "rows_initial": summary_df["rows_initial"].sum(),
        "rows_after_weather_policy": summary_df["rows_after_weather_policy"].sum(),
        "rows_with_metro": summary_df["rows_with_metro"].sum(),
        "rows_step6_final": summary_df["rows_step6_final"].sum(),
    }])

    out_summary = pd.concat([summary_df, total_row], ignore_index=True)
    out_fp = SUMMARY_DIR / "step_6_yearly_counts.csv"
    out_summary.to_csv(out_fp, index=False)

    print(f"\nSaved Step 6 summary -> {out_fp}")
    print(out_summary.to_string(index=False))


if __name__ == "__main__":
    main()
