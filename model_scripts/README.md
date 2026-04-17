# Model Scripts

This folder contains the executable model runners for the SPARK / ILI forecasting pipeline. These scripts consume the final report-level preprocessing export, construct the weekly CBSA panel using the shared dataset builder, run outer-rolling origin validation, and optionally write out-of-fold (OOF) prediction artifacts for downstream comparison and ensembling.

The scripts in this folder are the repository entry points for model fitting. Shared preprocessing, validation, sequence construction, evaluation, and OOF formatting logic are intentionally delegated to the helper modules outside this folder.

## Folder contents

### Baseline machine learning
- `ml_models.py`  
  Runs the baseline non-sequence models on the canonical weekly panel:
  - logistic regression with L2 regularization
  - elastic-net logistic regression
  - XGBoost

### Deep sequence models
- `tcn.py`  
  Temporal convolutional network baseline with metro embedding and metro-specific affine logit adjustment.

- `tft.py`  
  Compact TFT-style sequence baseline using recurrent and transformer-style temporal modeling, plus metro embedding and metro-specific affine logit adjustment.

- `nbeat.py`  
  N-BEATS-style feedforward sequence baseline over fixed-length weekly histories, plus metro embedding and metro-specific affine logit adjustment.

### Ensemble models
- `ensemble_model.py`  
  Constructs ensemble models using previously generated base-model out-of-fold (OOF) predictions. The ensemble stage does not use raw preprocessing outputs directly; instead, it relies on validated OOF artifacts produced by the base and deep model runners.

---

## Inputs

These scripts assume that preprocessing has already been completed.

The model-side input is the final report-level dataset with:

- `createday`
- `gender`
- `age_cat`
- `is_symptom`
- `received_flu_vaccine_fully`
- `ppt`
- `tmax`
- `tmin`
- `Metro (CBSA)`

For model ingestion, yearly CSV files must be staged in a single flat input directory. Model scripts do not search through preprocessing subfolders.

Accepted model input forms are:
1. a flat directory containing preprocessed CSV files
2. a single CSV file, although the flat-directory layout is preferred

---

## Design principles

These scripts follow a compact operational design:

- model scripts remain thin entry points
- weekly panel construction is centralized in the dataset builder
- feature selection, preprocessing, OOF packing, and evaluation are centralized in shared helpers
- outer validation is year-ahead and leakage-aware
- OOF artifacts are written in a canonical schema so that downstream comparison and ensembling remain consistent across model families

---

## Import and dependency note

Model runner scripts import shared helpers through repository package paths:

- `from helper_classes import dataset_builder as du`
- `from helper_classes import model_contract as mc`
- `from helper_classes import model_engine as me`


---

## Shared modeling behavior

All model runners follow the same high-level workflow:

1. read the preprocessed report-level input
2. build the weekly CBSA panel through `dataset_builder.py`
3. retain metros according to the configured minimum observed-week rule
4. define fold-specific training and validation frames
5. fit preprocessing on training data only
6. train the model for each outer validation year
7. produce canonical OOF predictions
8. evaluate on the configured report years
9. optionally write OOF artifacts and fold diagnostics

---

## Validation structure

The default design is outer rolling-origin validation by target-year fold.

Typical usage:
- training for fold year `Y` uses rows with `year_snap <= Y - 1`
- validation uses rows with `year_snap == Y`

For deep models, a context frame is also used so that validation sequences have adequate historical support without leaking future labels.

---

## Weighting and event-definition options

The model scripts support configurable event-definition and weighting rules through the local `CONFIG` dictionaries.

### Event definition
The one-week-ahead event label is defined from a fold-specific training-only threshold on the Jeffreys-smoothed weekly symptom rate. The key configuration field is:

- `Q_EVENT`  
  Primary analysis uses `0.90`; sensitivity analyses may use alternatives such as `0.85` or `0.95`.

### Weighting
The primary analysis uses capped report-count weighting, but alternative weighting schemes are also supported. The key configuration fields are:

- `WEIGHT_MODE`  
  - `"capped"` for report-count weighting
  - `"unweighted"` for equal nominal weight across scored CBSA-weeks

- `WEIGHT_CAP_Q`  
  The quantile used to cap report-count weights when `WEIGHT_MODE="capped"`.  
  Primary analysis uses `0.95`; sensitivity analyses may use alternatives such as `0.90`.

Under the unweighted specification, `WEIGHT_CAP_Q` is ignored.

---

## OOF artifacts

Each model script may write:

- full OOF predictions across all configured validation years
- optional train-side OOF or in-sample artifacts
- fold diagnostics
- threshold-by-fold tables

These files are intended for:

- standardized performance comparison
- calibration analysis
- thresholded metric reporting
- downstream ensemble fitting

The ensemble script depends on stable OOF filenames from the base models. Artifact names should therefore not be changed casually.

---

## Model-specific notes

### `ml_models.py`
Runs non-sequence baselines on the weekly panel after fold-safe imputation and standardization. Metro identity is incorporated through train-only one-hot encoding.
The emitted model names are kept stable for ensemble compatibility.

### `tcn.py`
Implements a temporal convolutional baseline over fixed-length weekly sequences. The model includes:
- metro embedding
- metro-specific affine adjustment of the output logit

### `tft.py`
Implements a compact TFT-style baseline. This is not intended as a full production TFT implementation; it is a streamlined sequence benchmark within the common deep-model framework.

### `nbeat.py`
Implements a residual feedforward sequence baseline inspired by N-BEATS-style block structure. It uses fixed-length flattened weekly histories plus metro identity modeling.

### `ensemble_model.py`
Builds ensemble predictions from the canonical OOF outputs of the base models. Ensemble fitting is time-ordered: for target year `Y`, only prior years are used to fit the combiner. This preserves the prospective evaluation logic.

---

## Configuration (Important!!)

Each script contains a local `CONFIG` dictionary that controls:

- input and output paths
- validation years
- report years
- event-threshold policy
- weighting mode and cap
- minimum observed-week filtering
- model hyperparameters
- OOF writing options

These script-local configs are intended to remain readable and explicit.

---

## Expected execution order

A typical run order is:

1. run preprocessing to completion
2. stage the preprocessed CSVs into one flat model-input directory
3. run base models:
   - `ml_models.py`
   - `tcn.py`
   - `tft.py`
   - `nbeat.py`
4. confirm that canonical OOF artifacts were written
5. run `ensemble_model.py`

---

## Dependencies

The model scripts rely on

- `pandas, (pyarrow)`
- `numpy`
- `scikit-learn`
- `torch`
- `xgboost`
- local helper modules in `helper_classes/`

---
