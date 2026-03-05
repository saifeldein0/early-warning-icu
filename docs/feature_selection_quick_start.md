# Quick Start: Feature Selection

This guide explains how to run the public feature-selection workflow included in this repository.

## Before You Start

- Install the dependencies listed in `Website/requirements.txt` plus the ML libraries used by the training scripts.
- Place your private training datasets outside this repository.
- Update any dataset paths inside the scripts to point to your local private data copies.

## Recommended Run Order

1. Run `feature_selection_icu.py` to produce ranked feature lists.
2. Review the generated CSV summaries and plots in the output folders.
3. Run `train_with_selected_features.py` to compare reduced-feature models against the full pipeline.

## Expected Outputs

The feature selection workflow writes artifacts such as:

- ranked feature tables
- feature-importance plots
- reduced-feature configuration files
- model comparison CSV summaries

These outputs are regenerated locally and are not required for browsing the public repository.

## Project Layout

```text
early-warning-icu/
|-- feature_selection_icu.py
|-- train_with_selected_features.py
|-- results/
|   |-- hematology_feature_selection_results/
|   |-- solid_feature_selection_results/
|   `-- hematology_reduced_models_100feat/
|-- Website/
`-- docs/
```

## Notes for Public Release

- The public repository excludes private datasets, deployment integrations, and production model binaries.
- Some training and validation scripts are preserved for transparency, but they require private clinical data to run end-to-end.
- The website under `Website/` is configured for demo mode and uses synthetic patient data.
- Curated public output artifacts are stored under `results/` for easier navigation.
