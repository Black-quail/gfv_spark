"""
Step 1: Retrieve and parse monthly Outbreaks Near Me API data.

Purpose
-------
For a given study year, retrieve one month of API data at a time using
monthly intervals [month_start, next_month_start), parse the
raw payload into a dataframe, and save one parsed CSV per month.

Input
-----
Remote Outbreaks Near Me API endpoint.

Output
------
step_1/step_1_<year>/step_1_<year>_<month>.csv

Remark
-----
For the study period used here, year 2020 begins at April 1, while
years 2021 onward begin at January 1.
"""

from pathlib import Path
from datetime import date
import ast
import json
import os
import re

import pandas as pd
import requests


# ==================== STUDY CONFIG ====================
YEAR = 2024
OVERWRITE = False

BASE = Path(os.getenv("GFV_SPARK_BASE_PATH", "path"))
OUT_DIR = BASE / "step_1" / f"step_1_{YEAR}"

TARGET_URL = "https://eapi-prod.outbreaksnearme.org/research-data?date-range={date_range}"
TIMEOUT = 120
# ======================================================


def monthly_date_ranges(year: int) -> list[str]:
    """
    Generate monthly half-open date ranges for one study year.
    """
    start_month = 4 if year == 2020 else 1
    ranges = []

    for month in range(start_month, 13):
        start = date(year, month, 1)
        end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        ranges.append(start.strftime("%Y%m%d") + "-" + end.strftime("%Y%m%d"))

    return ranges


def month_tag(date_range: str) -> str:
    """
    Convert YYYYMMDD-YYYYMMDD to YYYY_MM using the interval start date.
    """
    start = date_range.split("-")[0]
    return f"{start[:4]}_{start[4:6]}"


def parse_one_object(s: str) -> dict:
    """
    Parse one record object from the raw API payload.

    The payload has shown both JSON-style records and Python-literal-style
    records, so parsing proceeds with a small fallback chain.
    """
    s = s.strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    s_py = re.sub(r"\bnull\b", "None", s)
    s_py = re.sub(r"\btrue\b", "True", s_py)
    s_py = re.sub(r"\bfalse\b", "False", s_py)

    return ast.literal_eval(s_py)


def parse_raw_api_text(raw_text: str) -> pd.DataFrame:
    """
    Extract flat record objects from the raw API payload and return a table.
    """
    text = raw_text.strip()

    if text.startswith("0\n"):
        text = text[2:].lstrip()

    obj_strings = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if not obj_strings:
        raise ValueError("No record objects found in API payload.")

    records = [parse_one_object(s) for s in obj_strings]
    return pd.DataFrame(records)


def main() -> None:
    """
    Retrieve, parse, and save all monthly files for the configured year.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    date_ranges = monthly_date_ranges(YEAR)

    session = requests.Session()

    for date_range in date_ranges:
        tag = month_tag(date_range)
        out_path = OUT_DIR / f"step_1_{tag}.csv"

        if out_path.exists() and not OVERWRITE:
            print(f"[{date_range}] skipped -> {out_path} already exists")
            continue

        try:
            pointer_resp = session.get(
                TARGET_URL.format(date_range=date_range),
                timeout=TIMEOUT,
            )
            pointer_resp.raise_for_status()
            download_url = pointer_resp.json()["download_url"]

            raw_resp = session.get(download_url, timeout=TIMEOUT)
            raw_resp.raise_for_status()

            parsed_df = parse_raw_api_text(raw_resp.text)
            parsed_df.to_csv(out_path, index=False)

            print(f"[{date_range}] saved -> {out_path}")
            print(f"[{date_range}] rows  = {len(parsed_df):,}")

        except Exception as e:
            print(f"[{date_range}] FAILED: {e}")

    session.close()
    print("Step 1 completed.")


if __name__ == "__main__":
    main()
