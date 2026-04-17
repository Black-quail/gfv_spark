# Data Preprocessing

This folder contains the report-level preprocessing pipeline used to prepare the Outbreaks Near Me survey data for downstream model ingestion.

The pipeline is organized as a sequence of six scripts. It starts from monthly API retrieval and ends with a final report-level dataset that matches the raw input contract expected by the modeling code.

## Pipeline overview

### Step 1 — `01_ingest_api_monthly.py`
Retrieve and parse monthly Outbreaks Near Me API data.

This step:
- queries the API using monthly half-open intervals
- parses the raw payload into a rectangular table
- writes one parsed CSV per month

Output:
- `step_1/step_1_<year>/step_1_<year>_<month>.csv`

Special handling:
- year 2020 begins at April 1
- years 2021 onward begin at January 1

### Step 2 — `02_merge_monthly_to_yearly.py`
Merge parsed monthly files into one yearly report-level file.

This step:
- concatenates all monthly Step 1 files for a year
- sorts by `createday` when available
- does not apply exclusions

Output:
- `step_2/step_2_<year>/step_2_<year>.csv`
- yearly and monthly row-count summaries

### Step 3 — `03_clean_yearly_reports.py`
Apply report-level analytic preprocessing.

This step:
- restricts to `country == "US"`
- retains `gender in {"MALE", "FEMALE"}`
- keeps valid ages `0 <= age < 99`
- creates `age_cat`
- creates cleaned symptom indicators
- creates `symptom_count`
- creates `is_symptom`
- creates `received_flu_vaccine_fully`

Symptom rule:
- `TRUE` means symptom present
- `FALSE` means symptom absent
- blank means symptom absent

Flu vaccine rule:
- `yes` → `True`
- `no`, `not going to`, blank → `False`

Output:
- `step_3/step_3_<year>/step_3_<year>.csv`

### Step 4 — `04_standardize_zip_and_attach_geography.py`
Standardize ZIP codes and attach geographic reference labels.

This step:
- preserves the original ZIP as `zipcode_raw`
- normalizes working ZIP codes to 5-digit U.S. ZIP format
- drops rows with invalid ZIP after normalization
- attaches `city`, `state`, and `metro` from `ZIP_code_1.xlsx`
- attaches `CBSA_code` from `ZIP_code_2.xlsx`

ZIP normalization rule:
- digits-only
- zero-pad to length 5
- truncate to first 5 digits
- treat `00000` as invalid

Output:
- `step_4/step_4_<year>/step_4_<year>.csv`

### Step 5 — `05_attach_weather_and_validate_join.py`
Attach daily weather data.

This step:
- reads the yearly Step 4 file
- reads the yearly merged weather file
- standardizes `date` and `zipcode`
- restricts survey and weather ZIPs to the accepted ZIP universe
- joins weather at `zipcode + date`
- fills remaining weather gaps using `CBSA_code + date` fallback

Weather variables:
- `ppt`
- `tmax`
- `tmin`

Output:
- `step_5/step_5_<year>/step_5_<year>.csv`

### Step 6 — `06_export_model_input.py`
Export the final report-level dataset for model ingestion.

This step:
- fills missing `CBSA_code` from `ZIP_code_2.xlsx`
- applies the final weather policy
- assigns `Metro (CBSA)` using CBSA first and ZIP fallback
- exports the exact raw report-level schema expected by the model builder

Final exported columns:
- `createday`
- `gender`
- `age_cat`
- `is_symptom`
- `received_flu_vaccine_fully`
- `ppt`
- `tmax`
- `tmin`
- `Metro (CBSA)`

Output:
- `step_6/step_6_<year>/step_6_<year>.csv`

## Required external files

The preprocessing scripts expect the following external reference files to exist under the configured data directory.

### ZIP reference files
- `ZIP_code_1.xlsx (USPS)`
- `ZIP_code_2.xlsx, (HUD)`
ZIP/geography lookup inputs are based on the HUD USPS ZIP Code Crosswalk
Files (U.S. Department of Housing and Urban Development), with optional
supplemental ZIP metadata from U.S. Census/ACS tables when needed.

### Weather files
Directory:
- `weather/Merged weathers/`

Expected yearly weather files:
- `2020_merged_weather.csv`
- `2021_merged_weather.csv`
- `2022_merged_weather.csv`
- `2023_merged_weather.csv`
- `2024_merged_weather.csv`

Expected weather columns:
- `date`
- `zipcode`
- `ppt`
- `tmax`
- `tmin`

Weather source:
- PRISM Climate Group, Oregon State University (daily gridded precipitation and temperature)
- https://prism.oregonstate.edu/

## Folder structure

A typical folder layout is:

```text
preprocessing/
  01_ingest_api_monthly.py
  02_merge_monthly_to_yearly.py
  03_clean_yearly_reports.py
  04_standardize_zip_and_attach_geography.py
  05_attach_weather_and_validate_join.py
  06_export_model_input.py
