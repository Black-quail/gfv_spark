# Helper Classes and Shared Utilities

This folder contains the shared infrastructure used by the model runners. These modules define the model-side data contract, weekly panel construction, fold-safe preprocessing, sequence construction, evaluation, and common training utilities.


## Folder contents

- `dataset_builder.py`
- `model_contract.py`
- `model_engine.py`
- `evaluation.py`

## Purpose of each module

### `dataset_builder.py`
Builds the canonical weekly metro panel from the final report-level preprocessing export.

Responsibilities include:
- reading the preprocessed CSV input,
- validating the raw report-level input schema,
- mapping report dates to Monday-anchored weeks,
- aggregating reports to metro-week rows,
- computing symptom-rate and covariate summaries,
- building a complete weekly metro grid with padding support,
- adding lag/lead target structure,
- defining fold-specific frames and labels,
- constructing fixed-length sequences for deep models.

This module is the model-side bridge between preprocessing output and all downstream modeling code.

### `model_contract.py`
Defines the shared modeling contract for the weekly panel and OOF outputs.

Responsibilities include:
- canonical feature selection,
- column presence checks,
- padding-row logic,
- weight-contract checks,
- fold-safe imputation and standardization,
- canonical OOF packing,
- OOF validation/writing,
- prevalence-matching threshold computation,
- evaluation dispatch,
- fold-diagnostic construction and printing.

This module is the central place for repository-wide model-side conventions.

### `model_engine.py`
Provides shared deep-model training utilities.

Responsibilities include:
- sequence loader construction,
- sequence metadata tracking,
- optimizer and scheduler construction,
- sample-weight reduction,
- fixed-epoch training loops,
- probability generation,
- thin outer-CV orchestration for deep models.

Deep model scripts should define only the architecture and local configuration, while using this engine for the common training workflow.

### `evaluation.py`
Provides metric computation and reporting utilities.

Responsibilities include:
- calibration error computation,
- fold-aware metric evaluation,
- bootstrap confidence intervals,
- metrics table formatting,
- run configuration banners.

This file operates on scored predictions and canonical OOF outputs rather than on raw preprocessing input.

## Input and output

### Raw model-side input
The helper stack assumes that preprocessing has already produced the final report-level data.

Expected raw schema:
- `createday`
- `gender`
- `age_cat`
- `is_symptom`
- `received_flu_vaccine_fully`
- `ppt`
- `tmax`
- `tmin`
- `Metro (CBSA)`

For model ingestion, yearly CSVs are assumed to be staged in one flat directory or passed as a single CSV file. The builder does not recurse through preprocessing step directories.

### Canonical weekly panel
Once built, the weekly panel provides the common structure used by all model families. Important fields include:
- `metro`
- `week_start`
- `target_week_start`
- `year_snap`
- `iso_week`
- `weight`
- `flu_label` when applicable
- canonical feature columns selected through `model_contract.py`

### Canonical OOF output
Model outputs are expected to conform to the shared OOF schema, including:
- metro and time identifiers,
- fold identifier,
- binary truth label,
- sample weight,
- predicted probability,
- fold-level threshold metadata.

This standardization is what allows unified evaluation and ensemble construction.

## Module boundaries

To keep the repository maintainable, responsibilities are separated as follows:

- `dataset_builder.py` handles report-to-panel conversion and fold construction.
- `model_contract.py` defines shared modeling conventions.
- `model_engine.py` handles reusable deep-training mechanics.
- `evaluation.py` handles scored-prediction evaluation and reporting.

Model runner scripts should not duplicate these responsibilities.

## Import/dependency note

The model stack follows the repository folder structure for imports:
- shared modules are imported from `helper_classes.*`
- runners are imported from `model_scripts.*` when needed (for example, OOF
  table validation in the contract layer).

Keep these package-style imports aligned with folder layout when adding new
modules.

## Public helper API

The intended public interface from `model_contract.py` is:

- `FEATURE_COLS`
- `pick_feature_columns`
- `feature_columns_present`
- `require_model_columns`
- `is_padding_row`
- `weight_contract`
- `fit_impute_standardize`
- `apply_impute_standardize`
- `pack_oof_from_weekly`
- `write_validated_oof`
- `fold_prevalence_thresholds`
- `evaluate_from_oof`
- `fold_diagnostics_row`
- `print_fold_diagnostics`
- `concat_and_validate_oof`

These names should be treated as the stable shared helper surface for the model scripts.

## Maintenance guidance

### When to edit `dataset_builder.py`
Edit this file only when the preprocessing output or the weekly-panel construction logic changes. Do not move model-specific logic into it.

### When to edit `model_contract.py`
Edit this file when a repository-wide model-side design changes, such as:
- feature set,
- required weekly-panel columns,
- OOF schema,
- preprocessing conventions,
- thresholding/evaluation interfaces.

### When to edit `model_engine.py`
Edit this file when deep-model training behavior needs to change across all sequence models, for example:
- optimizer setup,
- scheduler behavior,
- sequence loader construction,
- common training loop rules.

### When to edit `evaluation.py`
Edit this file when evaluation methodology or metric reporting changes. Avoid embedding model-specific assumptions here.

## Practical guidance

- Put reusable logic here, not in the runnable model scripts.
- Keep the model scripts readable by delegating common operations to these helpers.
- Avoid silent drift; if the data interface changes, update the helper layer first.
- Preserve backward compatibility of canonical OOF outputs unless a repository-wide migration is intentional.

## Dependency notes

These helpers rely on:
- `pandas`
- `numpy`
- `scikit-learn`
- `torch` for deep-model utilities

They are designed to be imported by the model scripts folder and should not depend on the preprocessing scripts directly beyond the documented Step 6 data contract.
