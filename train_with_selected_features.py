"""
Train Models Using Selected Features
====================================

This script trains models using features selected by the feature_selection_icu.py script.
It allows comparison between full-feature models and reduced-feature models.

Usage:
    1. Run feature_selection_icu.py first to generate feature sets
    2. Configure this script (dataset type, feature count)
    3. Run this script to train and evaluate models

Module: Reduced-Feature Training Workflow
Date: 2025
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
import re
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, accuracy_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

print("=" * 80)
print("Training Models with Selected Features")
print("=" * 80)

# --- Configuration ---
DATASET_TYPE = "solid"  # Change to "solid" for solid tumor dataset
N_FEATURES = 500  # Options: 50, 100, 200, 500, or "all" for full feature set

print(f"\nConfiguration:")
print(f"  Dataset Type: {DATASET_TYPE.upper()}")
print(f"  Number of Features: {N_FEATURES if N_FEATURES != 'all' else 'ALL (full feature set)'}")

# Determine input/output paths
if DATASET_TYPE == "hematology":
    INPUT_FILE = 'FINAL_HEMATOLOGYy.csv'
    OUTPUT_PREFIX = 'hematology'
elif DATASET_TYPE == "solid":
    INPUT_FILE = 'FINAL_SOLID.csv'
    OUTPUT_PREFIX = 'solid'
else:
    raise ValueError("DATASET_TYPE must be 'hematology' or 'solid'")

FEATURE_SELECTION_DIR = f"{OUTPUT_PREFIX}_feature_selection_results"
OUTPUT_DIR = f"{OUTPUT_PREFIX}_reduced_models_{N_FEATURES}feat"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# --- Feature Configuration ---
train_vital_sign_cols = [
    'HEART_RATE', 'PULSE_OXIMETRY', 'TEMPERATURE',
    'SYSTOLIC_BLOOD_PRESSURE', 'MEAN_ARTERIAL_PRESSURE',
    'DIASTOLIC_BLOOD_PRESSURE', 'RESPIRATION_RATE'
]
train_lab_cols = [
    'AST_RESULT', 'CREATININE_RESULT', 'TOTAL_BILIRUBIN_RESULT',
    'DIRECT_BILIRUBIN_RESULT', 'POTASSIUM_RESULT', 'HEMOGLOBIN_RESULT',
    'LEUKOCYTE_COUNT_RESULT', 'ABSOLUTE_NEUTROPHILS', 'PLATELET_COUNT_RESULT',
    'PROTHROMBIN_CONCENTRATION'
]
train_med_cols = [
    'ANTIBIOTICS', 'NEUROLOGY_DRUGS', 'CARDIOLOGY_DRUGS', 'FUNGAL_DRUGS'
]
train_target_col = 'ICU_RISK'

PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
EMPTY_SEQ_ID = 0
LAB_HISTORY_LENGTH = 4

# --- Helper Functions ---
def convert_string_to_list_of_floats(x):
    if isinstance(x, str):
        cleaned_x = ''.join(c if c.isdigit() or c == '.' or c == ',' or c == '-' or c.isspace() else ' ' for c in x)
        values = [val.strip() for val in cleaned_x.split(',') if val.strip()]
        float_values = []
        for val in values:
            if val and val not in ['.', '-']:
                try:
                    float_values.append(float(val))
                except ValueError:
                    continue
        return float_values
    elif isinstance(x, (int, float)):
        return [float(x)]
    return []

def pad_list_to_length(lst, length, pad_value=PAD_TOKEN_ID):
    lst = lst[:length]
    return lst + [pad_value] * (length - len(lst))

def clean_med_name(name):
    return re.sub(r'\s+', ' ', name).strip().lower()

def convert_string_to_list_of_meds(x):
    if pd.isna(x) or not isinstance(x, str) or x.strip() == '':
        return []
    meds = [clean_med_name(med) for med in x.split(',') if clean_med_name(med)]
    return meds

def transform_meds_to_padded_ints(med_list, vocab, max_len):
    if not isinstance(med_list, list):
        med_list = []
    unk_token_int = vocab.get("UNK", UNK_TOKEN_ID)
    int_sequence = [vocab.get(med, unk_token_int) for med in med_list]
    int_sequence = int_sequence[:max_len]
    padded_sequence = int_sequence + [PAD_TOKEN_ID] * (max_len - len(int_sequence))
    return padded_sequence

# --- Load Data ---
print("\n" + "=" * 80)
print("STEP 1: Loading Data")
print("=" * 80)

print(f"\nLoading {INPUT_FILE}...")
data = pd.read_csv(INPUT_FILE)
print(f"[OK] Loaded {len(data)} patient records")

# Convert columns
print("\nConverting data columns...")
for col in train_vital_sign_cols + train_lab_cols:
    if col in data.columns:
        data[col] = data[col].apply(convert_string_to_list_of_floats)
    else:
        data[col] = pd.Series([[] for _ in range(len(data))], index=data.index)

for col in train_med_cols:
    if col in data.columns:
        data[col] = data[col].apply(convert_string_to_list_of_meds)
    else:
        data[col] = pd.Series([[] for _ in range(len(data))], index=data.index)

# Prepare target
data['EVENT'] = data[train_target_col].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
y = data['EVENT']

print(f"[OK] Target distribution: ICU Risk={sum(y==1)}, No Risk={sum(y==0)}")

# Train/Test Split
data_train, data_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[OK] Train: {len(data_train)}, Test: {len(data_test)}")

# --- Load Preprocessing Objects ---
print("\n" + "=" * 80)
print("STEP 2: Loading Preprocessing Objects")
print("=" * 80)

preprocessing_path = os.path.join(FEATURE_SELECTION_DIR, f'{OUTPUT_PREFIX}_preprocessing_objects.pkl')
print(f"\nLoading preprocessing objects from: {preprocessing_path}")

if not os.path.exists(preprocessing_path):
    print(f"[ERROR] Error: Preprocessing objects not found!")
    print(f"   Please run feature_selection_icu.py first for {DATASET_TYPE} dataset.")
    exit()

preprocessing_objects = joblib.load(preprocessing_path)
med_vocab = preprocessing_objects['med_vocab']
max_med_lengths = preprocessing_objects['max_med_lengths']
max_length_vitals = preprocessing_objects['max_length_vitals']
imputer = preprocessing_objects['imputer']
scaler = preprocessing_objects['scaler']
all_feature_names = preprocessing_objects['feature_names']

print(f"[OK] Loaded preprocessing objects")
print(f"  - Total features: {len(all_feature_names)}")
print(f"  - Max vital length: {max_length_vitals}")
print(f"  - Medication vocab size: {len(med_vocab)}")

# --- Feature Processing ---
print("\n" + "=" * 80)
print("STEP 3: Feature Engineering")
print("=" * 80)

def process_features_tabular(df):
    processed_features_list = []

    # Vital Signs
    for col in train_vital_sign_cols:
        if col in df.columns:
            padded_col = df[col].apply(lambda x: pad_list_to_length(
                x if isinstance(x, list) else [], max_length_vitals, pad_value=0.0
            ))
            col_array = np.array(padded_col.tolist(), dtype=float)
            if col_array.shape != (len(df), max_length_vitals):
                col_array = np.zeros((len(df), max_length_vitals))
            processed_features_list.append(col_array)
        else:
            processed_features_list.append(np.zeros((len(df), max_length_vitals)))

    # Lab Results
    for col in train_lab_cols:
        if col in df.columns:
            proc_col = df[col].apply(lambda x: [0.0]*(LAB_HISTORY_LENGTH - len(x)) + x[-LAB_HISTORY_LENGTH:]
                                     if isinstance(x, list) else [0.0]*LAB_HISTORY_LENGTH)
            col_array = np.array(proc_col.tolist(), dtype=float)
            if col_array.shape != (len(df), LAB_HISTORY_LENGTH):
                col_array = np.zeros((len(df), LAB_HISTORY_LENGTH))
            processed_features_list.append(col_array)
        else:
            processed_features_list.append(np.zeros((len(df), LAB_HISTORY_LENGTH)))

    # Medications
    for col in train_med_cols:
        max_len = max_med_lengths.get(col, 0)
        med_lists = df[col].apply(lambda x: x if isinstance(x, list) else []) if col in df else pd.Series([[] for _ in range(len(df))], index=df.index)

        # Sequences
        if col in df.columns:
            transformed_col = med_lists.apply(lambda x: transform_meds_to_padded_ints(x, med_vocab, max_len))
            col_array_seq = np.array(transformed_col.tolist(), dtype=int)
            if col_array_seq.shape != (len(df), max_len):
                col_array_seq = np.zeros((len(df), max_len), dtype=int)
            processed_features_list.append(col_array_seq)
        else:
            processed_features_list.append(np.zeros((len(df), max_len), dtype=int))

        # Aggregates
        lengths = med_lists.apply(len).values.reshape(-1, 1)
        processed_features_list.append(lengths.astype(float))
        processed_features_list.append((lengths > 0).astype(float))
        processed_features_list.append(med_lists.apply(
            lambda x: med_vocab.get(x[0], UNK_TOKEN_ID) if len(x) > 0 else EMPTY_SEQ_ID
        ).values.reshape(-1, 1).astype(float))
        processed_features_list.append(med_lists.apply(
            lambda x: med_vocab.get(x[-1], UNK_TOKEN_ID) if len(x) > 0 else EMPTY_SEQ_ID
        ).values.reshape(-1, 1).astype(float))
        processed_features_list.append(med_lists.apply(lambda x: len(set(x))).values.reshape(-1, 1).astype(float))

    valid_features = [arr for arr in processed_features_list if arr.ndim == 2 and arr.shape[0] == len(df) and arr.shape[1] > 0]
    if not valid_features:
        return np.empty((len(df), 0))

    X_processed = np.concatenate(valid_features, axis=1)
    return X_processed

print("\nProcessing features...")
X_train_processed = process_features_tabular(data_train)
X_test_processed = process_features_tabular(data_test)

print(f"[OK] Processed features")
print(f"  - Train shape: {X_train_processed.shape}")
print(f"  - Test shape: {X_test_processed.shape}")

# Apply imputation and scaling
print("\nApplying imputation and scaling...")
X_train_imputed = imputer.transform(X_train_processed)
X_test_imputed = imputer.transform(X_test_processed)

X_train_full = scaler.transform(X_train_imputed)
X_test_full = scaler.transform(X_test_imputed)
print("[OK] Complete")

# --- Load Selected Features ---
print("\n" + "=" * 80)
print("STEP 4: Loading Selected Features")
print("=" * 80)

if N_FEATURES != "all":
    feature_set_path = os.path.join(FEATURE_SELECTION_DIR, f'{OUTPUT_PREFIX}_top{N_FEATURES}_features.pkl')

    if not os.path.exists(feature_set_path):
        print(f"[ERROR] Error: Feature set not found at {feature_set_path}")
        print(f"   Please run feature_selection_icu.py first.")
        exit()

    print(f"\nLoading selected features from: {feature_set_path}")
    feature_set = joblib.load(feature_set_path)
    selected_indices = feature_set['indices']
    selected_names = feature_set['names']

    print(f"[OK] Loaded {len(selected_indices)} selected features")
    print(f"\nTop 10 Selected Features:")
    for i, name in enumerate(selected_names[:10]):
        print(f"  {i+1:2d}. {name}")

    # Select features
    X_train_final = X_train_full[:, selected_indices]
    X_test_final = X_test_full[:, selected_indices]

else:
    print("\nUsing ALL features (full feature set)")
    selected_indices = list(range(X_train_full.shape[1]))
    selected_names = all_feature_names
    X_train_final = X_train_full
    X_test_final = X_test_full

print(f"\nFinal feature matrix shapes:")
print(f"  - Train: {X_train_final.shape}")
print(f"  - Test: {X_test_final.shape}")

# --- Train Models ---
print("\n" + "=" * 80)
print("STEP 5: Training Models")
print("=" * 80)

results = []

# Model 1: XGBoost
print("\n1. Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=13,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train_final, y_train)

y_pred_xgb = xgb_model.predict(X_test_final)
y_pred_proba_xgb = xgb_model.predict_proba(X_test_final)[:, 1]

xgb_metrics = {
    'Model': 'XGBoost',
    'Accuracy': accuracy_score(y_test, y_pred_xgb),
    'Precision': precision_score(y_test, y_pred_xgb),
    'Recall': recall_score(y_test, y_pred_xgb),
    'F1-Score': f1_score(y_test, y_pred_xgb),
    'AUC-ROC': roc_auc_score(y_test, y_pred_proba_xgb),
    'MCC': matthews_corrcoef(y_test, y_pred_xgb)
}
results.append(xgb_metrics)

print(f"  [OK] Accuracy: {xgb_metrics['Accuracy']:.4f}, AUC-ROC: {xgb_metrics['AUC-ROC']:.4f}")

# Save model
joblib.dump({'model': xgb_model, 'selected_indices': selected_indices, 'selected_names': selected_names},
            os.path.join(OUTPUT_DIR, 'xgboost_model.pkl'))

# Model 2: Logistic Regression
print("\n2. Training Logistic Regression...")
lr_model = LogisticRegression(
    C=1.0,
    penalty='l1',
    solver='saga',
    max_iter=1000,
    random_state=42,
    n_jobs=-1
)
lr_model.fit(X_train_final, y_train)

y_pred_lr = lr_model.predict(X_test_final)
y_pred_proba_lr = lr_model.predict_proba(X_test_final)[:, 1]

lr_metrics = {
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-Score': f1_score(y_test, y_pred_lr),
    'AUC-ROC': roc_auc_score(y_test, y_pred_proba_lr),
    'MCC': matthews_corrcoef(y_test, y_pred_lr)
}
results.append(lr_metrics)

print(f"  [OK] Accuracy: {lr_metrics['Accuracy']:.4f}, AUC-ROC: {lr_metrics['AUC-ROC']:.4f}")

joblib.dump({'model': lr_model, 'selected_indices': selected_indices, 'selected_names': selected_names},
            os.path.join(OUTPUT_DIR, 'logistic_regression_model.pkl'))

# Model 3: Random Forest
print("\n3. Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_final, y_train)

y_pred_rf = rf_model.predict(X_test_final)
y_pred_proba_rf = rf_model.predict_proba(X_test_final)[:, 1]

rf_metrics = {
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1-Score': f1_score(y_test, y_pred_rf),
    'AUC-ROC': roc_auc_score(y_test, y_pred_proba_rf),
    'MCC': matthews_corrcoef(y_test, y_pred_rf)
}
results.append(rf_metrics)

print(f"  [OK] Accuracy: {rf_metrics['Accuracy']:.4f}, AUC-ROC: {rf_metrics['AUC-ROC']:.4f}")

joblib.dump({'model': rf_model, 'selected_indices': selected_indices, 'selected_names': selected_names},
            os.path.join(OUTPUT_DIR, 'random_forest_model.pkl'))

# --- Results Summary ---
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

# Save results
results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_performance.csv'), index=False)
print(f"\n[OK] Results saved to {OUTPUT_DIR}/model_performance.csv")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'AUC-ROC', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    ax.bar(results_df['Model'], results_df[metric], color=['steelblue', 'coral', 'mediumseagreen'])
    ax.set_ylabel(metric, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(results_df[metric]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

fig.suptitle(f'Model Performance - {DATASET_TYPE.upper()} ({N_FEATURES} Features)',
             fontsize=16, fontweight='bold')
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, 'performance_comparison.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Performance plot saved to {plot_path}")

print("\n" + "=" * 80)
print("[OK] TRAINING COMPLETE!")
print("=" * 80)

print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
print("\nFiles created:")
print("  - xgboost_model.pkl")
print("  - logistic_regression_model.pkl")
print("  - random_forest_model.pkl")
print("  - model_performance.csv")
print("  - performance_comparison.png")

print("\n" + "=" * 80)
