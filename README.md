# Early Warning ICU

Public release of an ICU early-warning project developed for pediatric oncology and hematology workflows at Children's Cancer Hospital Egypt 57357.

This repository is intentionally curated for public review. It keeps the technical scope of the project across training, feature selection, validation, and dashboard delivery, while excluding private data, trained model binaries, internal deployment and integration details, and low-value duplicate artifacts.

## Best entry points

- Start with [`docs/project_summary.md`](docs/project_summary.md) for the high-level technical narrative.
- Open [`Website/`](Website/) to review the runnable dashboard demo.
- Read the core modeling scripts in the repository root for the training, validation, and thresholding workflow.
- Browse [`results/`](results/) for the curated public-safe artifacts that summarize experiments and validation outcomes.

## Repository layout

- `Website/`
  Public demo dashboard built on synthetic data.
- Repository root `*.py`
  Core modeling, feature-selection, validation, and threshold-calibration scripts.
- `results/`
  Curated aggregate outputs, comparison tables, and reviewer-facing figures.
- `docs/`
  Public project summary, release notes, and workflow guidance.

## Core scripts

- `comprehensive_icu_models_final.py`
- `comprehensive_icu_models_solid.py`
- `feature_selection_icu.py`
- `train_with_selected_features.py`
- `extended_feature_selection_training.py`
- `extract_additional_features.py`
- `complete_validation_new_dataset.py`
- `validate_overlap_patients_no_leakage.py`
- `final_threshold_calculation.py`

## Privacy and release policy

This public repo excludes:

- raw or row-level patient datasets
- model `.pkl` and `.pth` files
- internal database, HL7, scheduling, network, and credential material
- manuscript build artifacts and internal operational notes
- exploratory cleanup scripts, presentation-only assets, and repetitive figure dumps
- duplicate exports, uploads, caches, and virtual environments

Included result files are aggregate summaries or figures considered safe for public sharing.

## Running the dashboard demo

```bash
cd Website
pip install -r requirements.txt
python create_sample_data.py
python app.py
```

The dashboard will read the latest synthetic export from `Website/demo_data/` and present a public-safe demo of the ICU risk workflow.

## Working with the research code

The repository keeps the scripts that best represent the technical arc of the project rather than every intermediate experiment file.
Some training and validation scripts will not run end-to-end in this public repo because the private datasets and trained artifacts have been removed.
Committed outputs are collected under `results/` for readability, even though some scripts still generate their artifacts into legacy local folders when run inside the original private environment.

Use this repository as:

- a technical portfolio artifact
- a codebase walkthrough for reviewers
- a public companion to the fellowship application narrative

## Selected project outcomes

- Cohort-specific ICU risk modeling for hematology and non-hematology populations
- Feature-selection and reduced-model experiments across multiple model families
- Leakage-focused external validation and threshold calibration
- A clinician-facing dashboard for periodic refresh and triage review

See [`docs/project_summary.md`](docs/project_summary.md), [`docs/feature_selection_quick_start.md`](docs/feature_selection_quick_start.md), and [`results/README.md`](results/README.md) for the curated public-facing guide to the repository.
