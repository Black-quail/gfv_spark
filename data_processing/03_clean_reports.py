"""
Step 3: Apply report-level analytic preprocessing.

Purpose
-------
Starting from the yearly merged dataset, restrict to U.S. records,
retain MALE/FEMALE gender categories, apply the valid-age restriction,
and construct cleaned symptom and influenza-vaccination indicators.

Input
-----
step_2/step_2_<year>/step_2_<year>.csv

Output
------
step_3/step_3_<year>/step_3_<year>.csv
summary/step_3_yearly_counts.csv
"""

from pathlib import Path
import os

import pandas as pd


# ==================== STUDY CONFIG ====================
YEARS = [2020, 2021, 2022, 2023, 2024]

BASE = Path(os.getenv("GFV_SPARK_BASE_PATH", "path"))
STEP2_BASE = BASE / "step_2"
STEP3_BASE = BASE / "step_3"
SUMMARY_DIR = BASE / "summary"

SYMPTOM_COLS = [
    "fever",
    "fatigue",
    "cough",
    "sorethroat",
    "headaches",
    "shortnessofbreath",
    "achesandpains",
]
# ======================================================

def parse_symptom_flag(s: pd.Series) -> pd.Series:
    """
    Parse raw symptom responses into Boolean indicators.

    Survey definition:
    Participants report symptoms they are currently experiencing.
    Therefore, a blank symptom field does not mean unknown status.
    It means the participant did not report that symptom.

    Coding used in Step 3:
        TRUE  -> True
        FALSE -> False
        blank -> False
    """
    s = s.astype("string").str.strip().str.upper()
    return s.eq("TRUE").fillna(False)



def parse_flu_vaccine_flag(s: pd.Series) -> pd.Series:
    """
    Parse a cleaned Boolean influenza-vaccination indicator.

    Observed raw values include:
        yes
        no
        not going to
        blank

    Defined as:
        received_flu_vaccine_fully = True  only if raw response is 'yes'
        received_flu_vaccine_fully = False otherwise
    """
    s = s.astype("string").str.strip().str.lower()
    return s.eq("yes").fillna(False)


def process_one_year(year: int) -> dict:
    """
    Process one yearly Step 2 file into the Step 3 cleaned yearly file.
    """
    in_fp = STEP2_BASE / f"step_2_{year}" / f"step_2_{year}.csv"
    out_dir = STEP3_BASE / f"step_3_{year}"
    out_fp = out_dir / f"step_3_{year}.csv"

    if not in_fp.exists():
        print(f"[{year}] input file missing: {in_fp}")
        return {
            "year": year,
            "rows_initial": 0,
            "rows_after_country": 0,
            "rows_after_gender": 0,
            "rows_after_age": 0,
            "rows_step3_final": 0,
        }

    df = pd.read_csv(in_fp, low_memory=False)
    rows_initial = len(df)

    # Restrict to U.S. records
    df["country"] = df["country"].astype("string").str.strip().str.upper()
    df = df.loc[df["country"].eq("US")].copy()
    rows_after_country = len(df)

    # Standardize gender and retain MALE / FEMALE
    df["gender"] = df["gender"].astype("string").str.strip().str.upper()
    df = df.loc[df["gender"].isin(["MALE", "FEMALE"])].copy()
    rows_after_gender = len(df)

    # Standardize age, apply valid-age restriction, and create age groups
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df.loc[(df["age"] >= 0) & (df["age"] < 99)].copy()
    rows_after_age = len(df)

    df["age_cat"] = pd.cut(
        df["age"],
        bins=[0, 19, 65, 99],
        labels=["<19", "19-64", ">64"],
        include_lowest=True,
        right=False,
    )

    # Create cleaned symptom indicators
    for col in SYMPTOM_COLS:
        if col in df.columns:
            df[f"{col}_cl"] = parse_symptom_flag(df[col])
        else:
            df[f"{col}_cl"] = False

    cl_cols = [f"{c}_cl" for c in SYMPTOM_COLS]

    # Create symptom summary fields
    df["symptom_count"] = df[cl_cols].sum(axis=1)
    other_cols = [c for c in cl_cols if c != "fever_cl"]
    df["is_symptom"] = df["fever_cl"] & df[other_cols].any(axis=1)

    # Create cleaned influenza-vaccine field
    if "received_flu_vaccine" in df.columns:
        df["received_flu_vaccine_fully"] = parse_flu_vaccine_flag(df["received_flu_vaccine"])
    else:
        df["received_flu_vaccine_fully"] = False

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_fp, index=False)

    print(f"[{year}] saved -> {out_fp}")
    print(f"[{year}] rows_initial       = {rows_initial:,}")
    print(f"[{year}] rows_after_country = {rows_after_country:,}")
    print(f"[{year}] rows_after_gender  = {rows_after_gender:,}")
    print(f"[{year}] rows_after_age     = {rows_after_age:,}")
    print(f"[{year}] rows_step3_final   = {len(df):,}")

    return {
        "year": year,
        "rows_initial": rows_initial,
        "rows_after_country": rows_after_country,
        "rows_after_gender": rows_after_gender,
        "rows_after_age": rows_after_age,
        "rows_step3_final": len(df),
    }


def main() -> None:
    """
    Run Step 3 for all configured study years and write the yearly summary.
    """
    STEP3_BASE.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    summaries = [process_one_year(year) for year in YEARS]
    summary_df = pd.DataFrame(summaries)

    total_row = pd.DataFrame([{
        "year": "TOTAL",
        "rows_initial": summary_df["rows_initial"].sum(),
        "rows_after_country": summary_df["rows_after_country"].sum(),
        "rows_after_gender": summary_df["rows_after_gender"].sum(),
        "rows_after_age": summary_df["rows_after_age"].sum(),
        "rows_step3_final": summary_df["rows_step3_final"].sum(),
    }])

    out_summary = pd.concat([summary_df, total_row], ignore_index=True)
    out_fp = SUMMARY_DIR / "step_3_yearly_counts.csv"
    out_summary.to_csv(out_fp, index=False)

    print(f"\nSaved Step 3 summary -> {out_fp}")
    print(out_summary.to_string(index=False))


if __name__ == "__main__":
    main()
