"""
Step 5: Attach daily weather and validate weather coverage.

Purpose
-------
Join PRISM daily weather data to each survey record using ZIP-by-date as
the primary key and CBSA-by-date as a fallback for unmatched rows.

Input
-----
step_4/step_4_<year>/step_4_<year>.csv
weather/Merged weathers/<year>_merged_weather.csv

Output
------
step_5/step_5_<year>/step_5_<year>.csv
summary/step_5_yearly_counts.csv
"""

from pathlib import Path
import os

import pandas as pd


# ==================== STUDY CONFIG ====================
YEARS = [2020, 2021, 2022, 2023, 2024]

BASE = Path(os.getenv("GFV_SPARK_BASE_PATH", "path"))
STEP4_BASE = BASE / "step_4"
STEP5_BASE = BASE / "step_5"
SUMMARY_DIR = BASE / "summary"

# Weather reference (public auxiliary dataset):
# PRISM Climate Group, Oregon State University
# https://prism.oregonstate.edu/
# Used here for daily gridded precipitation and temperature inputs.
WEATHER_DIR = BASE / "weather" / "Merged weathers"

# ZIP/geography reference (public auxiliary dataset):
# HUD USPS ZIP Code Crosswalk Files (U.S. Department of Housing and Urban Development).
# Additional ZIP metadata may be joined from U.S. Census/ACS reference tables.
ZIP1_XLSX = BASE / "ZIP_code_1.xlsx"
ZIP2_XLSX = BASE / "ZIP_code_2.xlsx"

WEATHER_COLS = ["ppt", "tmax", "tmin"]
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


def to_yyyymmdd(series: pd.Series) -> pd.Series:
    """
    Standardize date-like fields to YYYYMMDD string format.
    """
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.strftime("%Y%m%d")


def load_valid_zip_set() -> set[str]:
    """
    Read valid ZIPs from ZIP_code_1.xlsx.
    """
    if not ZIP1_XLSX.exists():
        return set()

    z1 = pd.read_excel(ZIP1_XLSX, dtype="string")
    z1.columns = [c.strip() for c in z1.columns]

    if "ZIP" not in z1.columns:
        raise KeyError(f"{ZIP1_XLSX} must contain column 'ZIP'.")

    return set(normalize_zip(z1["ZIP"]).dropna().astype("string").unique())


def load_zip_to_cbsa_ref() -> pd.DataFrame:
    """
    Read ZIP -> CBSA mapping from ZIP_code_2.xlsx.

    ZIP_code_2.xlsx columns used:
        Metro (CBSA)
        CBSA Code
        ZIP
        USPS Default City for ZIP
        % of Metro Residents in ZIP
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
        "CBSA_code": (
            z2["CBSA Code"]
            .astype("string")
            .str.replace(r"\D", "", regex=True)
            .str.zfill(5)
            .str[:5]
        ),
    })

    out["CBSA_code"] = out["CBSA_code"].mask(out["CBSA_code"] == "00000")
    out = out.dropna(subset=["zipcode", "CBSA_code"]).drop_duplicates(subset=["zipcode"], keep="first")
    return out


def fill_missing_cbsa_from_zip(df: pd.DataFrame, zip2_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing CBSA_code values from ZIP_code_2.xlsx using zipcode.
    """
    out = df.copy()

    if "zipcode" not in out.columns:
        if "CBSA_code" not in out.columns:
            out["CBSA_code"] = pd.NA
        return out

    out["zipcode"] = normalize_zip(out["zipcode"])

    if zip2_ref.empty:
        if "CBSA_code" not in out.columns:
            out["CBSA_code"] = pd.NA
        return out

    out = out.merge(zip2_ref, on="zipcode", how="left", suffixes=("", "_ref"))

    if "CBSA_code_ref" in out.columns:
        if "CBSA_code" in out.columns:
            out["CBSA_code"] = out["CBSA_code"].fillna(out["CBSA_code_ref"])
        else:
            out = out.rename(columns={"CBSA_code_ref": "CBSA_code"})
        out = out.drop(columns=["CBSA_code_ref"])

    return out


def load_yearly_weather(year: int) -> pd.DataFrame:
    """
    Read one yearly merged weather CSV.

    Expected columns:
        date
        zipcode
        ppt
        tmax
        tmin
    """
    fp = WEATHER_DIR / f"{year}_merged_weather.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Weather file not found: {fp}")

    wx = pd.read_csv(fp, dtype="string", low_memory=False)
    wx.columns = [c.strip() for c in wx.columns]

    required = {"date", "zipcode", "ppt", "tmax", "tmin"}
    if not required.issubset(wx.columns):
        raise KeyError(
            f"{fp} must contain columns {sorted(required)}. "
            f"Found: {list(wx.columns)}"
        )

    wx["date"] = to_yyyymmdd(wx["date"])
    wx["zipcode"] = normalize_zip(wx["zipcode"])
    wx["ppt"] = pd.to_numeric(wx["ppt"], errors="coerce")
    wx["tmax"] = pd.to_numeric(wx["tmax"], errors="coerce")
    wx["tmin"] = pd.to_numeric(wx["tmin"], errors="coerce")

    return wx[["date", "zipcode", "ppt", "tmax", "tmin"]]


def process_one_year(year: int, zip2_ref: pd.DataFrame, union_set: set[str]) -> dict:
    """
    Process one yearly Step 4 file into the Step 5 weather-attached file.
    """
    in_fp = STEP4_BASE / f"step_4_{year}" / f"step_4_{year}.csv"
    out_dir = STEP5_BASE / f"step_5_{year}"
    out_fp = out_dir / f"step_5_{year}.csv"

    if not in_fp.exists():
        print(f"[{year}] input file missing: {in_fp}")
        return {
            "year": year,
            "rows_initial": 0,
            "rows_after_zip_union": 0,
            "rows_with_weather_zip_match": 0,
            "rows_with_weather_after_cbsa_fallback": 0,
            "rows_step5_final": 0,
        }

    sx = pd.read_csv(in_fp, dtype={"zipcode": "string", "CBSA_code": "string"}, low_memory=False)
    wx = load_yearly_weather(year)

    rows_initial = len(sx)

    if "createday" not in sx.columns:
        raise KeyError(f"[{year}] Step 4 file must contain 'createday'.")

    # Create a shared date key
    sx["date"] = to_yyyymmdd(sx["createday"])
    sx["zipcode"] = normalize_zip(sx["zipcode"])

    # Restrict both survey and weather ZIPs to the accepted union
    if union_set:
        sx = sx.loc[sx["zipcode"].isin(union_set)].copy()
        wx = wx.loc[wx["zipcode"].isin(union_set)].copy()

    rows_after_zip_union = len(sx)

    # Ensure CBSA_code is available for CBSA fallback
    sx = fill_missing_cbsa_from_zip(sx, zip2_ref)
    wx_cbsa_ref = fill_missing_cbsa_from_zip(wx, zip2_ref)

    # ZIP-by-date weather join
    wx_no_cbsa = wx.drop(columns=["CBSA_code"], errors="ignore")
    merged = sx.merge(wx_no_cbsa, on=["date", "zipcode"], how="left")

    weather_cols_present = [c for c in WEATHER_COLS if c in merged.columns]
    for c in weather_cols_present:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    if weather_cols_present:
        zip_match_mask = merged[weather_cols_present].notna().any(axis=1)
        rows_with_weather_zip_match = int(zip_match_mask.sum())
    else:
        rows_with_weather_zip_match = 0

    # CBSA-by-date fallback for unmatched rows
    if weather_cols_present and "CBSA_code" in merged.columns:
        for c in weather_cols_present:
            if c in wx_cbsa_ref.columns:
                wx_cbsa_ref[c] = pd.to_numeric(wx_cbsa_ref[c], errors="coerce")

        wx_cbsa = (
            wx_cbsa_ref
            .dropna(subset=["CBSA_code"])
            .groupby(["date", "CBSA_code"], as_index=False)[weather_cols_present]
            .mean()
        )

        missing_any = ~merged[weather_cols_present].notna().any(axis=1)
        if missing_any.any():
            na_idx = merged.index[missing_any]
            fb = (
                merged.loc[na_idx, ["date", "CBSA_code"]]
                .merge(wx_cbsa, on=["date", "CBSA_code"], how="left")
            )
            fb.index = na_idx

            for c in weather_cols_present:
                merged.loc[na_idx, c] = merged.loc[na_idx, c].fillna(fb[c])

    if weather_cols_present:
        final_match_mask = merged[weather_cols_present].notna().any(axis=1)
        rows_with_weather_after_cbsa_fallback = int(final_match_mask.sum())
    else:
        rows_with_weather_after_cbsa_fallback = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_fp, index=False)

    print(f"[{year}] saved -> {out_fp}")
    print(f"[{year}] rows_initial                     = {rows_initial:,}")
    print(f"[{year}] rows_after_zip_union             = {rows_after_zip_union:,}")
    print(f"[{year}] rows_with_weather_zip_match      = {rows_with_weather_zip_match:,}")
    print(f"[{year}] rows_with_weather_after_fallback = {rows_with_weather_after_cbsa_fallback:,}")
    print(f"[{year}] rows_step5_final                 = {len(merged):,}")

    return {
        "year": year,
        "rows_initial": rows_initial,
        "rows_after_zip_union": rows_after_zip_union,
        "rows_with_weather_zip_match": rows_with_weather_zip_match,
        "rows_with_weather_after_cbsa_fallback": rows_with_weather_after_cbsa_fallback,
        "rows_step5_final": len(merged),
    }


def main() -> None:
    """
    Run Step 5 for all configured study years and write the yearly summary.
    """
    STEP5_BASE.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    zip1_set = load_valid_zip_set()
    zip2_ref = load_zip_to_cbsa_ref()
    zip2_set = set(zip2_ref["zipcode"].dropna().astype("string").unique()) if not zip2_ref.empty else set()
    union_set = zip1_set | zip2_set

    summaries = [process_one_year(year, zip2_ref, union_set) for year in YEARS]
    summary_df = pd.DataFrame(summaries)

    total_row = pd.DataFrame([{
        "year": "TOTAL",
        "rows_initial": summary_df["rows_initial"].sum(),
        "rows_after_zip_union": summary_df["rows_after_zip_union"].sum(),
        "rows_with_weather_zip_match": summary_df["rows_with_weather_zip_match"].sum(),
        "rows_with_weather_after_cbsa_fallback": summary_df["rows_with_weather_after_cbsa_fallback"].sum(),
        "rows_step5_final": summary_df["rows_step5_final"].sum(),
    }])

    out_summary = pd.concat([summary_df, total_row], ignore_index=True)
    out_fp = SUMMARY_DIR / "step_5_yearly_counts.csv"
    out_summary.to_csv(out_fp, index=False)

    print(f"\nSaved Step 5 summary -> {out_fp}")
    print(out_summary.to_string(index=False))


if __name__ == "__main__":
    main()
