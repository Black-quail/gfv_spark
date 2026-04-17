# SPARK / ILI Forecasting Pipeline

A repository for end-to-end preprocessing, model development, evaluation, and reporting for **one-week-ahead forecasting of unusually high local influenza-like illness (ILI) symptom activity** using **Outbreaks Near Me (ONM)** participatory surveillance data in the United States.

This repository contains the full study workflow used to:

- ingest and clean ONM reports
- attach geography and daily weather
- export the final report-level model-ingestion files
- build a weekly CBSA panel
- fit baseline, deep, and ensemble models under a strictly prospective design
- write canonical out-of-fold (OOF) prediction artifacts
- generate pooled manuscript-ready performance plots

The scientific task is addressing a **rare-event, short-horizon early-warning problem**. Each week, metropolitan areas are ranked according to their risk of experiencing unusually high local participatory symptom activity in the subsequent week.

---

## 1) Study overview

This project evaluates whether participatory surveillance can support **local early warning** of elevated ILI-like symptom activity across U.S. metropolitan areas. Using ONM reports collected from **April 1, 2020 through December 31, 2024**, the pipeline builds a weekly **Core Based Statistical Area (CBSA)** panel and predicts whether a metropolitan area will cross a fold-specific high-activity threshold **one week ahead**.

### Core study characteristics

- **Data source:** Outbreaks Near Me (ONM), accessed through Global Flu View
- **Geographic unit:** U.S. **CBSA**
- **Temporal unit:** Monday-anchored weekly panel using ISO week conventions
- **Forecast horizon:** 1 week ahead
- **Reporting lag:** 0 (no lag)
- **Primary evaluation period:** **2022--2024**
- **Historical support period:** **2020--2024** for lagged features and sequence construction
- **Evaluation design:** strictly prospective, year-ahead, fold-safe preprocessing and model fitting

### Why this repository exists

The main scientific purpose is not only to fit models, but to preserve a **leakage-aware, reproducible forecasting workflow** from raw survey reports to final out-of-fold evaluation artifacts. The repository therefore combines:

1. **report-level preprocessing**
2. **shared model-side infrastructure**
3. **model entry-point scripts**
4. **final evaluation and plotting utilities**

---

## 2) Repository structure

The repository is organized into three layers: preprocessing, shared model-side helpers, and runnable model scripts.

```text
gfv_spark_ILI/
├── data_processing/
│   ├── 01_ingest_api_monthly.py
│   ├── 02_merge_monthly_to_yearly.py
│   ├── 03_clean_yearly_reports.py
│   ├── 04_standardize_zip_and_attach_geography.py
│   ├── 05_attach_weather_and_validate_join.py
│   └── 06_export_model_input.py
├── helper_classes/
│   ├── __init__.py
│   ├── dataset_builder.py
│   ├── evaluation.py
│   ├── model_requirement.py
│   └── model_engine.py
├── model_scripts/
│   ├── __init__.py
│   ├── ml_models.py
│   ├── nbeat.py
│   ├── tcn.py
│   ├── tft.py
│   ├── ensemble_model.py
│   └── plot_models.py
├── README.md
└── LICENSE
```

### What each top-level folder contains

#### `data_processing/`
The end-to-end report-level preprocessing pipeline. These scripts convert raw ONM survey exports into the final **Step 6** CSV schema expected by the model-side code.

#### `helper_classes/`
Shared modeling infrastructure. These files define the model-side data requirement, build the weekly CBSA panel, handle fold-safe preprocessing, support deep-model training, and standardize evaluation and OOF outputs.

#### `model_scripts/`
Runnable modeling entry points. These scripts fit the baseline ML models, deep sequence models, time-safe ensembles, and final pooled performance plots.

---

## 3) End-to-end data flow

The repository implements the following workflow:

```text
Monthly ONM API pulls
    -> yearly cleaned report files
    -> ZIP / CBSA / weather enrichment
    -> final Step 6 report-level model input
    -> weekly CBSA panel construction
    -> fold-safe modeling and OOF prediction
    -> ensemble fitting from shared OOF tables
    -> pooled plots and manuscript-ready summaries
```

At a high level:

1. **Preprocessing** creates one clean yearly CSV per study year
2. **Dataset building** aggregates daily reports into a weekly CBSA panel
3. **Model runners** fit base models under rolling-origin validation
4. **Ensembling** consumes base-model OOF predictions only
5. **Plotting** reads final OOF artifacts and writes pooled PR and calibration figures

---

## 4) Scientific task and outcome definition

The forecasting target is not laboratory-confirmed influenza incidence. Instead, the repository is built around a **fold-defined surge-risk classification task** using participatory symptom activity.

### Daily symptom-positive report definition

A daily report is treated as symptom-positive when it includes:

- **fever**, and
- **at least one additional symptom** from the ILI symptom set:
  - cough
  - sore throat
  - fatigue
  - headache
  - shortness of breath
  - aches or pains

Blank symptom fields are treated as not reported. Influenza vaccination is harmonized to a binary indicator.

### Weekly outcome construction

For CBSA `m` and week `t`, the pipeline computes the weekly symptom-positive share using a **Jeffreys-smoothed** rate:

```text
p_tilde(m, t) = (symptom_positive_count + 0.5) / (n_reports + 1)
```

The binary forecasting target for week `t + 1` is then defined by whether next week’s smoothed rate exceeds a **fold-specific training-only percentile threshold**. The primary analysis uses the **90th percentile**.

This makes the study a **rare-event prediction problem** and keeps the target definition prospective.

---

## 5) Study window, cohort, and evaluation years

The full historical window spans **2020-04-01 through 2024-12-31**. The early 2020 period is retained for lagged and sequence histories, but the primary reported evaluation focuses on **2022--2024**, when symptom-defined surge events became more interpretable under post-pandemic reporting conditions.

A supplementary 2021 validation fold remains useful operationally because the ensemble stage requires strictly earlier out-of-fold data to train the first reported ensemble year.

### Analytic cohort summary

After preprocessing and aggregation:

- final cleaned daily ONM reports: **3,135,554**
- weekly CBSA observations before metro filtering: **265,783**
- CBSAs before metro filtering: **883**
- weeks in the weekly panel: **301**
- CBSAs retained for modeling after the minimum-history rule: **730**

### Minimum-history rules

Two related support rules are used in the model-side pipeline:

- each CBSA must have at least **26 observed weeks** overall to remain in the study cohort
- for sequence models, a forecast origin is eligible only if at least **26 of the previous 52 weeks** are observed

These rules reduce instability caused by extremely sparse local histories while preserving broad metropolitan coverage.

---

## 6) Preprocessing pipeline

The preprocessing layer transforms raw ONM reports into the exact report-level schema consumed by `dataset_builder.py`.

### Step 1 — `01_ingest_api_monthly.py`
Retrieves and parses monthly ONM API data.

### Step 2 — `02_merge_monthly_to_yearly.py`
Concatenates monthly files into one yearly report-level table.

### Step 3 — `03_clean_yearly_reports.py`
Applies report-level analytic cleaning, including:

- U.S.-only restriction
- age filtering
- gender harmonization
- symptom cleaning
- vaccination cleaning
- creation of `age_cat`, `symptom_count`, `is_symptom`, and cleaned vaccination indicators

### Step 4 — `04_standardize_zip_and_attach_geography.py`
Normalizes ZIP codes, preserves raw ZIP, and attaches reference geography including CBSA fields.

### Step 5 — `05_attach_weather_and_validate_join.py`
Joins daily weather by ZIP and date, with CBSA-date fallback when necessary.

### Step 6 — `06_export_model_input.py`
Exports the final report-level modeling input and enforces the exact schema expected by the model-side builder.

### Final Step 6 model-ingestion schema

```text
createday
gender
age_cat
is_symptom
received_flu_vaccine_fully
ppt
tmax
tmin
Metro (CBSA)
```

### External reference files used by preprocessing

#### ZIP / CBSA reference workbooks
- `ZIP_code_1.xlsx`
- `ZIP_code_2.xlsx`

These files support ZIP normalization, geography attachment, and CBSA / metro assignment.

#### Weather inputs
Expected yearly weather files:

- `2020_merged_weather.csv`
- `2021_merged_weather.csv`
- `2022_merged_weather.csv`
- `2023_merged_weather.csv`
- `2024_merged_weather.csv`

Expected weather columns:

```text
date
zipcode
ppt
tmax
tmin
```

The workflow uses PRISM-derived daily precipitation and temperature variables.

---

## 7) Model-side weekly panel

The report-level Step 6 files are not used directly by the models. They are first transformed into a weekly CBSA panel by `dataset_builder.py`.

### What the weekly builder does

`dataset_builder.py` is the bridge between preprocessing and downstream modeling. It:

- reads the final Step 6 yearly CSV files from one flat input directory
- aggregates daily reports to metro-week rows
- computes the smoothed weekly symptom rate
- constructs demographic and vaccination composition features
- aggregates weather to the weekly level
- builds a complete CBSA-week grid
- adds one-week lag and lead metadata
- retains only metros with adequate observed history
- constructs fixed-length sequences for deep models

### Weekly panel conventions

- weeks are **Monday-anchored**
- week labels follow **ISO week-date conventions**
- padded weeks are allowed to support fixed 52-week sequence histories
- padded rows receive **zero exposure weight** and do not contribute to training or evaluation

---

## 8) Canonical feature set

All model families use the same shared weekly predictor set so that performance comparisons remain interpretable.

### Canonical 11 predictors

```text
p_tilde
p_tilde_lag_1
ppt_log_sum
ppt_any
wet_days
vacc_prop
tavg_mean
diurnal_range_mean
gender_female_share
age_19_64_share
age_64_share
```

These features summarize:

- recent local symptom activity
- precipitation burden and occurrence
- weekly temperature patterns
- influenza vaccination share
- broad demographic composition of the reporting cohort

---

## 9) Prospective validation design

All modeling code is organized around a **strictly prospective outer rolling-origin design**.

### Fold logic

For validation year `Y`:

- training uses rows with `year_snap <= Y - 1`
- validation uses rows with `year_snap == Y`

All feature completion, standardization, threshold selection, model fitting, and calibration are performed using training data only within each fold.

### Exposure weighting

Because report counts vary substantially across CBSA-weeks, the primary analysis uses report-count weighting so that sparse and unstable weekly signals contribute less than better-supported weeks. To prevent a small number of very high-volume metro-weeks from dominating fitting or evaluation, report-count weights are capped in the primary analysis at the **95th percentile** of observed weekly counts and then normalized.

The model scripts also support alternative weighting rules for sensitivity analyses, including:

- a more restrictive **90th-percentile cap**
- an **unweighted** specification in which each scored CBSA-week contributes equal nominal importance

---

## 10) Model families in this repository

The repository evaluates three main model groups.

### A. Baseline machine learning (`ml_models.py`)
Runs the classical tabular baselines:

- L2 logistic regression
- elastic-net logistic regression
- XGBoost

For these models, metro identity is included through training-fold-only one-hot encoding.

### B. Deep sequence models

All deep models consume 52-week histories and share the same model-side data requirement.

- `tcn.py` — temporal convolutional baseline
- `tft.py` — compact LSTM + transformer-style temporal model
- `nbeat.py` — N-BEATS-style residual feedforward sequence model

All deep models use:

- learned metro embeddings
- metro-specific affine logit adjustment
- fixed-epoch training
- a shared engine in `model_engine.py`

### C. Ensembles (`ensemble_model.py`)

Four time-safe ensembles are constructed from previously written base-model OOF predictions:

- **E0** — mean subset + logistic recalibration
- **E1** — logistic stacker on logit-transformed base probabilities
- **E2** — beta-calibrated stacked model
- **E3** — family-balanced mean + recalibration

The ensemble stage does **not** consume raw Step 6 preprocessing output. It consumes validated OOF artifacts from the base and deep model scripts.

---

## 11) Shared helper modules

The helper layer centralizes repository-wide modeling conventions.

### `dataset_builder.py`
Weekly panel construction, fold frames, eligibility filtering, and sequence building.

### `model_contract.py`
Canonical feature set, preprocessing requirement, OOF packing and writing, threshold-by-fold logic, and fold diagnostics.

### `model_engine.py`
Shared deep-model infrastructure including loaders, optimizer and scheduler setup, fixed-epoch training, prediction, and outer-CV orchestration.

### `evaluation.py`
Metric computation, expected calibration error, bootstrap confidence intervals, metrics formatting, and run banners.

This split is intentional: model scripts should stay thin and delegate reusable logic here.

---

## 12) Canonical OOF artifacts

A central repository design choice is the use of strict **out-of-fold prediction tables**. These artifacts make model comparison, calibration assessment, thresholded reporting, and ensembling consistent across all model families.

### Typical OOF contents

A canonical OOF table includes:

- `metro`
- `week_start`
- `target_week_start`
- `fold`
- `year_snap`
- `iso_week`
- `y_true`
- `weight`
- `prob`
- fold-level threshold metadata such as `tau` and `T0p9`

### Why OOF standardization matters

This requirement enables:

- fair model comparison on the same scored cohort
- unified metric computation
- pooled reliability and PR plotting
- strictly time-ordered ensemble fitting

Artifact names and schemas should remain stable unless the change is repository-wide and intentional.

---

## 13) Performance assessment

The evaluation layer is designed for both **rare-event discrimination** and **probability reliability**.

### Main reported metrics

The primary validation metrics include:

- AUPRC
- AUROC
- Brier score
- expected calibration error (ECE)
- accuracy
- F1 score
- Matthews correlation coefficient (MCC)

### Confidence intervals

For AUPRC and ECE, uncertainty is quantified with **95% metro-cluster bootstrap confidence intervals** using 2,000 replicates.

### Plotting

`plot_models.py` writes pooled:

- precision–recall curves
- reliability diagrams

The plotting script is configured for the manuscript lineup of:

- best ensemble
- best single model
- best deep model
- strongest simple baseline

---

## 14) Main empirical takeaway

Under the strict prospective design used in this repository, participatory symptom reports retain meaningful one-week-ahead signal for local surge-risk ranking.

At a high level, the main study findings are:

- **ensemble methods performed best overall**
- **XGBoost was the strongest individual base learner**
- **TCN was the strongest deep sequence model**
- calibration remained an important practical criterion alongside discrimination

The repository is therefore structured not only to maximize predictive performance, but also to preserve a **calibration-aware, leakage-aware forecasting workflow** that can support short-horizon operational alerting.

---

## 15) Path configuration 

Configure the local input and output paths directly in the script `CONFIG` dictionaries before running the repository workflows.

At minimum, the preprocessing and modeling scripts require:
- a base data directory for preprocessing inputs and intermediate outputs
- a model-input directory containing the final Step 6 yearly CSV files
- an OOF / ensemble directory for model outputs

If you prefer environment-variable-based path control, you should implement that consistently across the repository before documenting it here.

---

## 16) Expected execution order

A standard run order is:

1. run preprocessing through Step 6
2. stage the Step 6 yearly CSV files into one flat model-input directory
3. run base models:
   - `ml_models.py`
   - `tcn.py`
   - `tft.py`
   - `nbeat.py`
4. verify that canonical OOF artifacts were written
5. run `ensemble_model.py`
6. run `plot_models.py` for pooled reporting figures

---

## 17) Dependencies

The full repository depends primarily on:

- `pandas`
- `numpy`
- `scikit-learn`
- `torch`
- `pyarrow`
- `xgboost`
- `matplotlib`
- `openpyxl`

Typical version requirements are:

- `pandas>=2.0`
- `numpy>=1.24`
- `scikit-learn>=1.3`
- `pyarrow>=15.0`

Version constraints for `torch`, `xgboost`, `matplotlib`, and `openpyxl` may be set with respect to the local environment and platform.

---

## 18) Development and maintenance notes

### General guidance

- keep shared logic in `helper_classes/`, not duplicated across runners
- preserve the Step 6 report-level schema unless the entire model-side requirement changes
- preserve canonical OOF column names and file names unless the migration is coordinated across the repository
- keep fold-safe preprocessing behavior intact
- do not introduce `leakage` through global preprocessing, thresholding, or calibration
---

## 19) Data availability

ONM participatory surveillance data used in this study were accessed through Global Flu View under a data sharing agreement. The underlying ONM records are not publicly available for open download. Qualified researchers may request access subject to review and execution of an appropriate agreement with the data providers.

Public auxiliary datasets used in this study include:

- **PRISM Climate Data** (PRISM Climate Group, Oregon State University): daily gridded precipitation and temperature variables used for weather feature construction.
- **HUD USPS ZIP Code Crosswalk Files** (U.S. Department of Housing and Urban Development): ZIP-to-geography crosswalk inputs used during ZIP and metro-area harmonization.

These two auxiliary resources are cited in the study references and should be listed once in downstream documentation to avoid duplicate citation notes.

---

## 20) Acknowledgments

We gratefully acknowledge the Outbreaks Near Me participatory surveillance program and the volunteers who submitted symptom reports. We also acknowledge the Boston Children’s Hospital Computational Epidemiology Group and HealthMap for maintaining Outbreaks Near Me, and the Global Flu View initiative for enabling research access to participatory surveillance streams.

The interpretations and conclusions expressed here are those of the authors and do not necessarily reflect the views of Outbreaks Near Me, Boston Children’s Hospital, HealthMap, Global Flu View, or affiliated partners.
