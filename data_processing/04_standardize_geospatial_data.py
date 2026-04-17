"""
Step 4: Standardize ZIP codes and attach geographic reference labels.

Purpose
-------
Normalize ZIP codes to the 5-digit U.S. ZIP format, remove records with
invalid ZIP after normalization, and attach city, state, metro, and
CBSA labels from external ZIP reference workbooks.

Input
-----
step_3/step_3_<year>/step_3_<year>.csv

# ZIP reference data sources:
# The ZIP-code lookup workbooks used in Step 4 are external U.S. ZIP
# reference files. Data sources for these United States Postal Service
# ZIP-code lookup resources include the U.S. Census Bureau and the U.S.
# Office of Housing and Urban Development (HUD). ZIP-code demographic data
# come from the American Community Survey (ACS) 5-year estimates, which are
# released annually by the U.S. Census Bureau. These reference data were
# updated as of February 2025 using the most recent data available.

ZIP_code_1.xlsx
ZIP_code_2.xlsx

Output
------
step_4/step_4_<year>/step_4_<year>.csv
summary/step_4_yearly_counts.csv
"""

from pathlib import Path
import os

import pandas as pd


# ==================== STUDY CONFIG ====================
YEARS = [2020, 2021, 2022, 2023, 2024]

BASE = Path(os.getenv("GFV_SPARK_BASE_PATH", "path"))
STEP3_BASE = BASE / "step_3"
STEP4_BASE = BASE / "step_4"
SUMMARY_DIR = BASE / "summary"

ZIP1_XLSX = BASE / "ZIP_code_1.xlsx"
ZIP2_XLSX = BASE / "ZIP_code_2.xlsx"
# ======================================================


def normalize_zip(series: pd.Series) -> pd.Series:
    """
    Standardize ZIP codes to a 5-digit U.S. ZIP representation.

    Rule:
        digits-only -> zero-pad -> truncate to 5 -> '00000' becomes missing
    """
    s = series.astype("string").str.strip().str.replace(r"\D", "", regex=True)
    s = s.str.zfill(5).str[:5]
    s = s.mask(s == "00000")
    s = s.mask(s.str.fullmatch(r"0*"), pd.NA)
    s = s.mask(s.str.len() == 0, pd.NA)
    return s


def load_zip1_reference() -> pd.DataFrame:
    """
    Load ZIP reference 1 and return:
        zipcode, city, state, metro

    ZIP_code_1.xlsx columns used:
        ZIP
        State
        USPS Default City for ZIP
        Metro (CBSA)
    """
    if not ZIP1_XLSX.exists():
        return pd.DataFrame(columns=["zipcode", "city", "state", "metro"])

    z1 = pd.read_excel(ZIP1_XLSX, dtype="string")
    z1.columns = [c.strip() for c in z1.columns]

    required = {"ZIP", "State", "USPS Default City for ZIP", "Metro (CBSA)"}
    if not required.issubset(z1.columns):
        raise KeyError(
            f"{ZIP1_XLSX} must contain columns {sorted(required)}. "
            f"Found: {list(z1.columns)}"
        )

    out = pd.DataFrame({
        "zipcode": normalize_zip(z1["ZIP"]),
        "state": z1["State"].astype("string").str.strip(),
        "city": z1["USPS Default City for ZIP"].astype("string").str.strip(),
        "metro": z1["Metro (CBSA)"].astype("string").str.strip(),
    })

    out = out.dropna(subset=["zipcode"]).drop_duplicates(subset=["zipcode"], keep="first")
    return out


def load_zip2_reference() -> pd.DataFrame:
    """
    Load ZIP reference 2 and return:
        zipcode, CBSA_code, metro_from_zip2

    ZIP_code_2.xlsx columns used:
        Metro (CBSA)
        CBSA Code
        ZIP
        USPS Default City for ZIP
        % of Metro Residents in ZIP
    """
    if not ZIP2_XLSX.exists():
        return pd.DataFrame(columns=["zipcode", "CBSA_code", "metro_from_zip2"])

    z2 = pd.read_excel(ZIP2_XLSX, dtype="string")
    z2.columns = [c.strip() for c in z2.columns]

    required = {"Metro (CBSA)", "CBSA Code", "ZIP"}
    if not required.issubset(z2.columns):
        raise KeyError(
            f"{ZIP2_XLSX} must contain columns {sorted(required)}. "
            f"Found: {list(z2.columns)}"
        )

    out = pd.DataFrame({
        "zipcode": normalize_zip(z2["ZIP"]),
        "CBSA_code": (
            z2["CBSA Code"]
            .astype("string")
            .str.replace(r"\D", "", regex=True)
            .str.zfill(5)
            .str[:5]
        ),
        "metro_from_zip2": z2["Metro (CBSA)"].astype("string").str.strip(),
    })

    out["CBSA_code"] = out["CBSA_code"].mask(out["CBSA_code"] == "00000")
    out = out.dropna(subset=["zipcode", "CBSA_code"]).drop_duplicates(subset=["zipcode"], keep="first")
    return out


def process_one_year(year: int, zip1_ref: pd.DataFrame, zip2_ref: pd.DataFrame) -> dict:
    """
    Process one yearly Step 3 file into the Step 4 geography-labeled file.
    """
    in_fp = STEP3_BASE / f"step_3_{year}" / f"step_3_{year}.csv"
    out_dir = STEP4_BASE / f"step_4_{year}"
    out_fp = out_dir / f"step_4_{year}.csv"

    if not in_fp.exists():
        print(f"[{year}] input file missing: {in_fp}")
        return {
            "year": year,
            "rows_initial": 0,
            "rows_after_zip_valid": 0,
            "rows_with_city_state": 0,
            "rows_with_cbsa": 0,
            "rows_step4_final": 0,
        }

    df = pd.read_csv(in_fp, dtype={"zipcode": "string"}, low_memory=False)
    rows_initial = len(df)

    if "zipcode" not in df.columns:
        raise KeyError(f"[{year}] Step 3 file does not contain 'zipcode'.")

    # Preserve the original ZIP before normalization
    df["zipcode_raw"] = df["zipcode"]

    # Normalize ZIP and drop records still invalid afterward
    df["zipcode"] = normalize_zip(df["zipcode"])
    df = df.dropna(subset=["zipcode"]).copy()
    rows_after_zip_valid = len(df)

    # Attach ZIP-based geographic labels
    df = df.merge(zip1_ref, on="zipcode", how="left")
    df = df.merge(zip2_ref, on="zipcode", how="left")

    # Keep one metro field with ZIP1 primary and ZIP2 fallback
    if "metro" in df.columns and "metro_from_zip2" in df.columns:
        df["metro"] = df["metro"].fillna(df["metro_from_zip2"])
        df = df.drop(columns=["metro_from_zip2"])
    elif "metro_from_zip2" in df.columns:
        df["metro"] = df["metro_from_zip2"]
        df = df.drop(columns=["metro_from_zip2"])

    # Final guard against accidental post-merge ZIP drift
    df["zipcode"] = normalize_zip(df["zipcode"])

    rows_with_city_state = (
        int((df["city"].notna() & df["state"].notna()).sum())
        if {"city", "state"}.issubset(df.columns) else 0
    )
    rows_with_cbsa = int(df["CBSA_code"].notna().sum()) if "CBSA_code" in df.columns else 0

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)

    print(f"[{year}] saved -> {out_fp}")
    print(f"[{year}] rows_initial        = {rows_initial:,}")
    print(f"[{year}] rows_after_zip_valid= {rows_after_zip_valid:,}")
    print(f"[{year}] rows_with_city_state= {rows_with_city_state:,}")
    print(f"[{year}] rows_with_cbsa      = {rows_with_cbsa:,}")
    print(f"[{year}] rows_step4_final    = {len(df):,}")

    return {
        "year": year,
        "rows_initial": rows_initial,
        "rows_after_zip_valid": rows_after_zip_valid,
        "rows_with_city_state": rows_with_city_state,
        "rows_with_cbsa": rows_with_cbsa,
        "rows_step4_final": len(df),
    }


def main() -> None:
    """
    Run Step 4 for all configured study years and write the yearly summary.
    """
    STEP4_BASE.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    zip1_ref = load_zip1_reference()
    zip2_ref = load_zip2_reference()

    summaries = [process_one_year(year, zip1_ref, zip2_ref) for year in YEARS]
    summary_df = pd.DataFrame(summaries)

    total_row = pd.DataFrame([{
        "year": "TOTAL",
        "rows_initial": summary_df["rows_initial"].sum(),
        "rows_after_zip_valid": summary_df["rows_after_zip_valid"].sum(),
        "rows_with_city_state": summary_df["rows_with_city_state"].sum(),
        "rows_with_cbsa": summary_df["rows_with_cbsa"].sum(),
        "rows_step4_final": summary_df["rows_step4_final"].sum(),
    }])

    out_summary = pd.concat([summary_df, total_row], ignore_index=True)
    out_fp = SUMMARY_DIR / "step_4_yearly_counts.csv"
    out_summary.to_csv(out_fp, index=False)

    print(f"\nSaved Step 4 summary -> {out_fp}")
    print(out_summary.to_string(index=False))


if __name__ == "__main__":
    main()
