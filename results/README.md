# Results Guide

This folder collects the public-safe artifacts retained from the original project.

## What is here

- `comprehensive_*performance*.csv`
  Aggregate model-comparison tables for the main cohort-specific experiments.
- `external_validation_all_models_metrics.csv`
  Summary metrics from the external validation pass.
- `final_metrics_summary.csv`
  Consolidated evaluation metrics used for the final public packaging.
- `optimal_thresholds.csv`
  Threshold summary for the risk-banding workflow.
- `validation_*`
  Final confusion-matrix and probability-distribution figures kept for quick reviewer inspection.
- `hematology_feature_selection_results/` and `solid_feature_selection_results/`
  Feature-ranking tables and representative feature-selection plots for each cohort.
- `hematology_reduced_models_*` and `solid_reduced_models_*`
  Reduced-feature experiment summaries and comparison plots.
- `extended_feature_selection_results/`
  Additional aggregate summaries from the expanded feature-selection experiments.

## Why this folder exists

The original working project produced many output directories and repeated plot variants.
For the public release, the retained artifacts were gathered here so reviewers can inspect the key outputs without navigating a research scratch space.

## Important note

Some scripts in the repository still write to their original local output paths when run in a full private environment.
That is expected. The `results/` folder is the curated public snapshot, not a strict mirror of every private runtime path.
