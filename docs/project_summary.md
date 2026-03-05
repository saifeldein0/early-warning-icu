# Project Summary

## Overview

Early Warning ICU is a cohort-aware ICU deterioration prediction project built around pediatric oncology and hematology inpatient workflows. The system combines longitudinal vitals, laboratory histories, medication signals, and admission context to estimate short-term ICU escalation risk and surface the results in a clinician-facing dashboard.

## Technical scope

- Separate modeling logic for hematology and non-hematology cohorts
- Multi-model experimentation including XGBoost, random forest, logistic regression, SVM, decision tree, and BiLSTM variants
- Feature-selection pipelines and reduced-feature experiments
- External validation, threshold analysis, and leakage-focused evaluation
- A web dashboard for sorting, filtering, and periodically refreshing patient-level predictions

## What made the project difficult

- The data was longitudinal rather than tabular in a simple sense, so feature engineering had to preserve time-history signals from vitals, labs, and medications without collapsing clinically useful structure.
- The project had two materially different cohorts, which meant a single thresholding strategy was not sufficient and validation needed to respect cohort-specific behavior.
- Leakage risk was non-trivial. Some validation episodes could belong to patients seen in other datasets at different times, so evaluation had to distinguish true information leakage from legitimate repeated episodes.
- The project had to move beyond notebooks and offline experiments into an operational dashboard workflow that clinicians could actually review.

## Public release note

This repository removes patient-level source data, trained model binaries, and internal deployment and integration material. It also trims presentation assets, duplicate plot collections, and intermediate cleanup files so the public version stays focused on the core technical work. The website included here runs in a synthetic-data demo mode so reviewers can inspect the dashboard workflow without access to private clinical assets.
