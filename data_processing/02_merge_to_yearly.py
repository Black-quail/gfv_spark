"""
Step 2: Merge monthly parsed files into yearly report-level files.

Purpose
-------
Concatenate all parsed monthly files for each year into a single yearly
report-level dataset without applying preprocessing exclusions.

Input
-----
step_1/step_1_<year>/step_1_<year>_<month>.csv

Output
------
step_2/step_2_<year>/step_2_<year>.csv
summary/step_2_monthly_counts_<year>.csv
summary/step_2_yearly_counts.csv - Audit 
"""

from pathlib import Path
import os
import re

import pandas as pd


# ==================== STUDY CONFIG ====================
YEARS = [2020, 2021, 2022, 2023, 2024]

BASE = Path(os.getenv("GFV_SPARK_BASE_PATH", "path"))
STEP1_BASE = BASE / "step_1"
STEP2_BASE = BASE / "step_2"
SUMMARY_DIR = BASE / "summary"

DATE_COL = "createday"
# ======================================================


def month_from_name(fp: Path) -> str | None:
    """
    Extract MM from a Step 1 monthly filename.
    """
    m = re.search(r"step_1_(\d{4})_(\d{2})\.csv$", fp.name)
    return m.group(2) if m else None


def merge_one_year(year: int) -> dict:
    """
    Merge all monthly Step 1 files for one year into a single yearly file.
    """
    in_dir = STEP1_BASE / f"step_1_{year}"
    out_dir = STEP2_BASE / f"step_2_{year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"[{year}] input directory missing: {in_dir}")
        return {
            "year": year,
            "n_months": 0,
            "rows_initial": 0,
            "date_min": pd.NA,
            "date_max": pd.NA,
        }

    files = sorted(in_dir.glob(f"step_1_{year}_*.csv"))
    if not files:
        print(f"[{year}] no monthly files found in {in_dir}")
        return {
            "year": year,
            "n_months": 0,
            "rows_initial": 0,
            "date_min": pd.NA,
            "date_max": pd.NA,
        }

    dfs = []
    monthly_rows = []

    for fp in files:
        df = pd.read_csv(fp, low_memory=False)
        month = month_from_name(fp)

        monthly_rows.append({
            "year": year,
            "month": month,
            "file": fp.name,
            "rows": len(df),
        })

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True, sort=False)

    date_min = pd.NaT
    date_max = pd.NaT
    if DATE_COL in merged.columns:
        merged[DATE_COL] = pd.to_datetime(merged[DATE_COL], errors="coerce")
        merged = merged.sort_values(DATE_COL, ascending=True)
        date_min = merged[DATE_COL].min()
        date_max = merged[DATE_COL].max()

    out_fp = out_dir / f"step_2_{year}.csv"
    merged.to_csv(out_fp, index=False)

    monthly_df = pd.DataFrame(monthly_rows)
    monthly_fp = SUMMARY_DIR / f"step_2_monthly_counts_{year}.csv"
    monthly_df.to_csv(monthly_fp, index=False)

    print(f"[{year}] merged file -> {out_fp}")
    print(f"[{year}] monthly count summary -> {monthly_fp}")
    print(f"[{year}] rows_initial = {len(merged):,}")

    return {
        "year": year,
        "n_months": len(files),
        "rows_initial": int(len(merged)),
        "date_min": date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else pd.NA,
        "date_max": date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else pd.NA,
    }


def main() -> None:
    """
    Merge all configured years and write yearly summary outputs.
    """
    summaries = [merge_one_year(year) for year in YEARS]
    summary_df = pd.DataFrame(summaries)

    total_row = pd.DataFrame([{
        "year": "TOTAL",
        "n_months": summary_df["n_months"].sum(),
        "rows_initial": summary_df["rows_initial"].sum(),
        "date_min": pd.NA,
        "date_max": pd.NA,
    }])

    out_summary = pd.concat([summary_df, total_row], ignore_index=True)
    out_fp = SUMMARY_DIR / "step_2_yearly_counts.csv"
    out_summary.to_csv(out_fp, index=False)

    print(f"\nSaved yearly summary -> {out_fp}")
    print(out_summary.to_string(index=False))


if __name__ == "__main__":
    main()
