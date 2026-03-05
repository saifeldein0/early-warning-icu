"""
Validate ICU vs Discharged episodes while KEEPING overlapping patients between datasets.

Goal
----
Some patients appear in BOTH ICU-admitted and discharged validation datasets (same MRN/name)
but represent different time episodes. This script keeps those overlaps to test whether the
model differentiates ICU vs discharged episodes based on clinical signals (vitals/labs/meds),
not identifiers.

Leakage control
---------------
We still REMOVE any validation rows whose MRN appears in training data, because that is
true train→validation leakage.

Outputs (written to overlap_allowed_validation/)
----------------------------------------------
1) overall_metrics.csv: metrics on the full validation set (ICU + discharged episodes)
2) overlap_subset_metrics.csv: metrics on overlap-patient subset only
3) overlap_patient_episode_report.csv: per-row predictions for overlap patients
4) overlap_patient_summary.csv: per-patient comparison (ICU vs discharged risk)

Public release note
-------------------
The public repository does not include the private datasets, binary models, or regenerated
output folders required for a full execution of this script.
"""

from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

ICU_VALIDATION_CSV = "EWS 2025 Validation.csv"
DISCHARGE_VALIDATION_CSV = "EWS_2025_Discharge_Validation.csv"
HEM_TRAIN_CSV = "FINAL_HEMATOLOGYy.csv"
SOLID_TRAIN_CSV = "FINAL_SOLID.csv"

OUT_DIR = "overlap_allowed_validation"
ANONYMIZE_OUTPUTS = True
ANON_SALT = "EWS_ICU_2025_ANON_V1"  # stable salt for consistent IDs within this project

# Models: use cohort-specific preprocessing objects (same as final pipeline)
MODEL_SPECS = {
    "Hematology": {
        "Original_XGBoost": ("Website/models/xgboost_hematology.pkl", "comprehensive_hematology_models/preprocessing_objects.pkl", 2527),
        "BIC_LogisticRegression": ("logistic_regression_BIC/bic_lr_model_hematology.pkl", "comprehensive_hematology_models/preprocessing_objects.pkl", 2527),
    },
    "Non-Hematology": {
        "Original_XGBoost": ("Website/models/xgboost.pkl", "comprehensive_solid_models/preprocessing_objects.pkl", 2310),
        "BIC_LogisticRegression": ("logistic_regression_BIC/bic_lr_model_solid.pkl", "comprehensive_solid_models/preprocessing_objects.pkl", 2310),
    },
}


# --------------------------------------------------------------------------------------
# Preprocessing (mirrors final_threshold_calculation.py / complete_clean_validation.py)
# --------------------------------------------------------------------------------------

def convert_string_to_list_of_floats(s):
    if pd.isna(s):
        return []
    if isinstance(s, (list, np.ndarray)):
        return [float(x) if not pd.isna(x) else 0.0 for x in s]
    try:
        s = str(s).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return []
        return [float(x.strip()) if x.strip() else 0.0 for x in s.split(",")]
    except Exception:
        return []


def pad_list_to_length(lst, target_length, pad_value=0.0):
    if len(lst) >= target_length:
        return lst[:target_length]
    return lst + [pad_value] * (target_length - len(lst))


def clean_med_name(med_name):
    if pd.isna(med_name) or not med_name:
        return ""
    return str(med_name).strip().lower().replace(" ", "_")


def convert_string_to_list_of_meds(s):
    if pd.isna(s) or not s:
        return []
    if isinstance(s, list):
        return [clean_med_name(m) for m in s if m]
    try:
        s = str(s).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return []
        meds = [clean_med_name(m.strip().strip("'\"")) for m in s.split(",")]
        return [m for m in meds if m]
    except Exception:
        return []


def transform_meds_to_padded_ints(med_list, vocab, max_len):
    int_list = [vocab.get(m, 0) for m in med_list[:max_len]]
    return pad_list_to_length(int_list, max_len, pad_value=0)


def preprocess_data(df_raw: pd.DataFrame, preprocessing_params: dict, expected_features: int) -> np.ndarray:
    df = df_raw.copy()

    med_vocab = preprocessing_params.get("med_vocab", {})
    max_med_lengths = preprocessing_params.get("max_med_lengths", {})
    max_length_vitals = preprocessing_params.get("max_length_vitals", 308)
    lab_history_length = preprocessing_params.get("LAB_HISTORY_LENGTH", 4)

    vital_cols = preprocessing_params.get("train_vital_sign_cols", [])
    lab_cols = preprocessing_params.get("train_lab_cols", [])
    med_cols = preprocessing_params.get("train_med_cols", [])

    for col in vital_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_string_to_list_of_floats)
    for col in lab_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_string_to_list_of_floats)
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_string_to_list_of_meds)

    features_list: List[List[float]] = []
    for _, row in df.iterrows():
        patient_features: List[float] = []

        for col in vital_cols:
            if col in df.columns:
                vals = row[col] if isinstance(row[col], list) else []
                patient_features.extend(pad_list_to_length(vals, max_length_vitals, 0.0))

        for col in lab_cols:
            if col in df.columns:
                vals = row[col] if isinstance(row[col], list) else []
                last_n = vals[-lab_history_length:] if len(vals) >= lab_history_length else vals
                patient_features.extend(pad_list_to_length(last_n, lab_history_length, 0.0))

        for col in med_cols:
            if col in df.columns:
                meds = row[col] if isinstance(row[col], list) else []
                max_len = max_med_lengths.get(col, 100)
                vocab = med_vocab.get(col, {})
                padded_ints = transform_meds_to_padded_ints(meds, vocab, max_len)
                patient_features.extend(padded_ints)
                patient_features.extend(
                    [len(meds), padded_ints[0] if padded_ints else 0, padded_ints[-1] if padded_ints else 0]
                )

        features_list.append(patient_features)

    X = np.array(features_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    imputer = preprocessing_params.get("imputer_tabular") or preprocessing_params.get("imputer")
    scaler = preprocessing_params.get("scaler_tabular") or preprocessing_params.get("scaler")

    if imputer is not None:
        imp_feat = imputer.n_features_in_
        if X.shape[1] < imp_feat:
            X = np.hstack([X, np.zeros((X.shape[0], imp_feat - X.shape[1]), dtype=X.dtype)])
        elif X.shape[1] > imp_feat:
            X = X[:, :imp_feat]
        X = imputer.transform(X)

    if scaler is not None:
        scl_feat = scaler.n_features_in_
        if X.shape[1] < scl_feat:
            X = np.hstack([X, np.zeros((X.shape[0], scl_feat - X.shape[1]), dtype=X.dtype)])
        elif X.shape[1] > scl_feat:
            X = X[:, :scl_feat]
        X = scaler.transform(X)

    if X.shape[1] < expected_features:
        X = np.hstack([X, np.zeros((X.shape[0], expected_features - X.shape[1]), dtype=X.dtype)])
    elif X.shape[1] > expected_features:
        X = X[:, :expected_features]

    return X


def is_hematology_diagnosis(diagnosis_str) -> bool:
    if pd.isna(diagnosis_str):
        return False
    diagnosis_lower = str(diagnosis_str).lower()
    hematology_keywords = ["leukemia", "lymphoma", "myeloma", "hodgkin", "aml", "all"]
    return any(keyword in diagnosis_lower for keyword in hematology_keywords)


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------

@dataclass
class Metrics:
    n: int
    prevalence: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    recall: float
    f1: float
    mcc: float
    auc_roc: float | None
    auprc: float | None
    tp: int
    fp: int
    tn: int
    fn: int


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Metrics:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0

    auc_roc = None
    auprc = None
    # AUC metrics undefined if only one class present
    if len(np.unique(y_true)) == 2:
        auc_roc = float(roc_auc_score(y_true, y_proba))
        auprc = float(average_precision_score(y_true, y_proba))

    return Metrics(
        n=int(len(y_true)),
        prevalence=float(np.mean(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        sensitivity=float(sens),
        specificity=float(spec),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        auc_roc=auc_roc,
        auprc=auprc,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def metrics_to_row(model_name: str, split: str, cohort: str, m: Metrics) -> dict:
    return {
        "Model": model_name,
        "Split": split,
        "Cohort": cohort,
        "N": m.n,
        "Prevalence": m.prevalence,
        "Accuracy": m.accuracy,
        "Balanced_Accuracy": m.balanced_accuracy,
        "Sensitivity": m.sensitivity,
        "Specificity": m.specificity,
        "Precision": m.precision,
        "Recall": m.recall,
        "F1": m.f1,
        "MCC": m.mcc,
        "AUC_ROC": m.auc_roc,
        "AUPRC": m.auprc,
        "TP": m.tp,
        "FP": m.fp,
        "TN": m.tn,
        "FN": m.fn,
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def load_model(model_path: str):
    model_data = joblib.load(model_path)
    return model_data["model"] if isinstance(model_data, dict) and "model" in model_data else model_data


def fix_diagnosis_column(df: pd.DataFrame) -> pd.DataFrame:
    # Some discharged exports have DIAGNOSIS empty and the content in Unnamed: 32 (or 31)
    if "DIAGNOSIS" not in df.columns:
        return df
    diag = df["DIAGNOSIS"]
    if diag.isna().all() or (diag.astype(str).str.strip() == "").all():
        for alt in ["Unnamed: 32", "Unnamed: 31"]:
            if alt in df.columns:
                df = df.copy()
                df["DIAGNOSIS"] = df[alt]
                break
    return df


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load validation datasets (keep overlaps between ICU and discharged)
    icu_df = pd.read_csv(ICU_VALIDATION_CSV)
    discharge_df = pd.read_csv(DISCHARGE_VALIDATION_CSV)
    discharge_df = fix_diagnosis_column(discharge_df)

    # Assign row ids for reporting only (not used in features)
    icu_df = icu_df.copy()
    discharge_df = discharge_df.copy()
    icu_df["ROW_ID"] = np.arange(1, len(icu_df) + 1)
    discharge_df["ROW_ID"] = np.arange(1, len(discharge_df) + 1)
    icu_df["SOURCE_DATASET"] = "ICU_Admitted"
    discharge_df["SOURCE_DATASET"] = "Discharged"

    # Labels: ICU=1, Discharged=0
    icu_df["Y_TRUE"] = 1
    discharge_df["Y_TRUE"] = 0

    # Leakage control: remove any MRNs present in training data
    hem_train = pd.read_csv(HEM_TRAIN_CSV)
    solid_train = pd.read_csv(SOLID_TRAIN_CSV)
    train_mrns = set(hem_train["MRN"].dropna().astype(str)).union(set(solid_train["MRN"].dropna().astype(str)))

    icu_df["MRN_str"] = icu_df["MRN"].astype(str)
    discharge_df["MRN_str"] = discharge_df["MRN"].astype(str)
    icu_df_noleak = icu_df[~icu_df["MRN_str"].isin(train_mrns)].copy()
    discharge_df_noleak = discharge_df[~discharge_df["MRN_str"].isin(train_mrns)].copy()

    print("=" * 90)
    print("OVERLAP-ALLOWED VALIDATION (NO TRAINING LEAKAGE)")
    print("=" * 90)
    print(f"ICU rows: {len(icu_df):,} -> {len(icu_df_noleak):,} after removing training overlap")
    print(f"Discharged rows: {len(discharge_df):,} -> {len(discharge_df_noleak):,} after removing training overlap")

    # Identify overlap patients between ICU and discharged (after leakage removal)
    overlap_mrns = set(icu_df_noleak["MRN_str"]).intersection(set(discharge_df_noleak["MRN_str"]))
    print(f"Overlap MRNs kept (ICU INTERSECT Discharged): {len(overlap_mrns):,}")

    def patient_key(mrn_str: str) -> str:
        # Deterministic anonymized identifier (no reversible MRN/name in outputs)
        h = hashlib.sha256((ANON_SALT + "::" + str(mrn_str)).encode("utf-8")).hexdigest()
        return f"P{h[:12]}"

    # Add anonymized patient key
    icu_df_noleak["PATIENT_KEY"] = icu_df_noleak["MRN_str"].apply(patient_key)
    discharge_df_noleak["PATIENT_KEY"] = discharge_df_noleak["MRN_str"].apply(patient_key)

    # Cohort flags
    for df in (icu_df_noleak, discharge_df_noleak):
        if "DIAGNOSIS" not in df.columns:
            df["DIAGNOSIS"] = np.nan
        df["IS_HEMATOLOGY"] = df["DIAGNOSIS"].apply(is_hematology_diagnosis)
        df["COHORT"] = np.where(df["IS_HEMATOLOGY"], "Hematology", "Non-Hematology")

    # Combine for "whole dataset" evaluation
    combined = pd.concat([icu_df_noleak, discharge_df_noleak], ignore_index=True)

    # Evaluate each model type; for each row pick the cohort-specific model
    metrics_rows: List[dict] = []
    overlap_metrics_rows: List[dict] = []
    overlap_episode_rows: List[dict] = []

    for model_name in ["Original_XGBoost", "BIC_LogisticRegression"]:
        print(f"\n--- Evaluating: {model_name} ---")

        # Predict per cohort and stitch back together
        preds_all = []
        for cohort in ["Hematology", "Non-Hematology"]:
            model_path, preproc_path, expected_features = MODEL_SPECS[cohort][model_name]
            model = load_model(model_path)
            preproc = joblib.load(preproc_path)

            subset = combined[combined["COHORT"] == cohort].copy()
            if subset.empty:
                continue

            X = preprocess_data(subset, preproc, expected_features)
            y_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)

            base_cols = ["ROW_ID", "SOURCE_DATASET", "PATIENT_KEY", "COHORT", "Y_TRUE"]
            # Keep human identifiers only if explicitly requested
            if not ANONYMIZE_OUTPUTS:
                base_cols.extend(["MRN_str"])
                if "PATIENT_NAME" in subset.columns:
                    base_cols.extend(["PATIENT_NAME"])

            subset_out = subset[base_cols].copy()
            subset_out["Y_PROBA"] = y_proba
            subset_out["Y_PRED"] = y_pred
            subset_out["MODEL"] = model_name
            preds_all.append(subset_out)

        pred_df = pd.concat(preds_all, ignore_index=True).sort_values(["SOURCE_DATASET", "ROW_ID"])

        # Whole dataset metrics (all cohorts together)
        y_true = pred_df["Y_TRUE"].to_numpy()
        y_pred = pred_df["Y_PRED"].to_numpy()
        y_proba = pred_df["Y_PROBA"].to_numpy()
        m_all = compute_metrics(y_true, y_pred, y_proba)
        metrics_rows.append(metrics_to_row(model_name, "All_Validation_Episodes", "All", m_all))

        # Per cohort metrics too (high-signal to interpret behavior)
        for cohort in ["Hematology", "Non-Hematology"]:
            cdf = pred_df[pred_df["COHORT"] == cohort]
            if len(cdf) == 0:
                continue
            cm = compute_metrics(cdf["Y_TRUE"].to_numpy(), cdf["Y_PRED"].to_numpy(), cdf["Y_PROBA"].to_numpy())
            metrics_rows.append(metrics_to_row(model_name, "All_Validation_Episodes", cohort, cm))

        # Overlap subset metrics + per-episode report
        overlap_keys = set(pd.Series(list(overlap_mrns)).apply(patient_key))
        overlap_df = pred_df[pred_df["PATIENT_KEY"].isin(overlap_keys)].copy()
        if not overlap_df.empty:
            om = compute_metrics(overlap_df["Y_TRUE"].to_numpy(), overlap_df["Y_PRED"].to_numpy(), overlap_df["Y_PROBA"].to_numpy())
            overlap_metrics_rows.append(metrics_to_row(model_name, "Overlap_Patients_Only", "All", om))
            for cohort in ["Hematology", "Non-Hematology"]:
                cdf = overlap_df[overlap_df["COHORT"] == cohort]
                if len(cdf) == 0:
                    continue
                cm = compute_metrics(cdf["Y_TRUE"].to_numpy(), cdf["Y_PRED"].to_numpy(), cdf["Y_PROBA"].to_numpy())
                overlap_metrics_rows.append(metrics_to_row(model_name, "Overlap_Patients_Only", cohort, cm))

            overlap_episode_rows.append(overlap_df)

        # Save the full prediction table for this model (useful for spot checks)
        pred_df.to_csv(os.path.join(OUT_DIR, f"predictions_{model_name}.csv"), index=False)

    # Save metrics
    overall_metrics = pd.DataFrame(metrics_rows)
    overall_metrics.to_csv(os.path.join(OUT_DIR, "overall_metrics.csv"), index=False)

    overlap_metrics = pd.DataFrame(overlap_metrics_rows)
    overlap_metrics.to_csv(os.path.join(OUT_DIR, "overlap_subset_metrics.csv"), index=False)

    # Save overlap episode report and per-patient summary
    if overlap_episode_rows:
        overlap_all = pd.concat(overlap_episode_rows, ignore_index=True)
        overlap_all.to_csv(os.path.join(OUT_DIR, "overlap_patient_episode_report.csv"), index=False)

        # Patient-level summary: compare mean ICU proba vs mean Discharged proba per MRN & model & cohort
        summary_rows = []
        grouped = overlap_all.groupby(["MODEL", "COHORT", "PATIENT_KEY"], dropna=False)
        for (model, cohort, pkey), g in grouped:
            icu_g = g[g["SOURCE_DATASET"] == "ICU_Admitted"]
            dis_g = g[g["SOURCE_DATASET"] == "Discharged"]
            if icu_g.empty or dis_g.empty:
                continue
            icu_mean = float(icu_g["Y_PROBA"].mean())
            dis_mean = float(dis_g["Y_PROBA"].mean())
            # "episode separation" heuristic: ICU risk > Discharged risk
            separated = icu_mean > dis_mean
            # classification correctness rates
            icu_acc = float(np.mean(icu_g["Y_PRED"].to_numpy() == 1))
            dis_acc = float(np.mean(dis_g["Y_PRED"].to_numpy() == 0))
            summary_rows.append(
                {
                    "MODEL": model,
                    "COHORT": cohort,
                    "PATIENT_KEY": pkey,
                    "ICU_EPISODES": int(len(icu_g)),
                    "DISCH_EPISODES": int(len(dis_g)),
                    "ICU_MEAN_PROBA": icu_mean,
                    "DISCH_MEAN_PROBA": dis_mean,
                    "ICU_GT_DISCH": separated,
                    "ICU_PRED_POS_RATE": icu_acc,
                    "DISCH_PRED_NEG_RATE": dis_acc,
                }
            )
        patient_summary = pd.DataFrame(summary_rows).sort_values(["MODEL", "COHORT", "ICU_GT_DISCH"], ascending=[True, True, False])
        patient_summary.to_csv(os.path.join(OUT_DIR, "overlap_patient_summary.csv"), index=False)

    print("\n[OK] Done.")
    print(f"Outputs written to: {OUT_DIR}/")
    print(f"- overall_metrics.csv")
    print(f"- overlap_subset_metrics.csv")
    print(f"- predictions_*.csv (per model)")
    if overlap_episode_rows:
        print(f"- overlap_patient_episode_report.csv")
        print(f"- overlap_patient_summary.csv")


if __name__ == "__main__":
    main()

