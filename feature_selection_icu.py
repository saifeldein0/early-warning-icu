"""
Feature Selection for ICU Risk Prediction System
================================================

This script implements two complementary feature selection algorithms:
1. ANOVA F-test (Univariate): Evaluates each feature independently
2. Random Forest Importance (Multivariate): Captures feature interactions and temporal patterns

Purpose:
- Identify most predictive features from ~2,500 dimensional feature space
- Create multiple feature sets (top 50, 100, 200, 500) for experimentation
- Analyze separately for hematology and solid tumor datasets

Module: Feature Selection Workflow
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
from collections import Counter
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("ICU Risk Prediction - Feature Selection")
print("=" * 80)

# --- Configuration ---
DATASET_TYPE = "solid"  # Change to "solid" for solid tumor dataset

if DATASET_TYPE == "hematology":
    INPUT_FILE = 'FINAL_HEMATOLOGYy.csv'
    OUTPUT_PREFIX = 'hematology'
elif DATASET_TYPE == "solid":
    INPUT_FILE = 'FINAL_SOLID.csv'
    OUTPUT_PREFIX = 'solid'
else:
    raise ValueError("DATASET_TYPE must be 'hematology' or 'solid'")

print(f"\nDataset Type: {DATASET_TYPE.upper()}")
print(f"Input File: {INPUT_FILE}")
print(f"Output Prefix: {OUTPUT_PREFIX}")

# Create output directory for results
OUTPUT_DIR = f"{OUTPUT_PREFIX}_feature_selection_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# --- Feature Configuration (Same as training scripts) ---
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

# --- Constants for Special Tokens ---
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
EMPTY_SEQ_ID = 0
LAB_HISTORY_LENGTH = 4

# --- Helper Functions (From existing training scripts) ---

def convert_string_to_list_of_floats(x):
    """Convert string representation of numbers to list of floats."""
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
    """Pad or truncate list to specified length."""
    lst = lst[:length]
    return lst + [pad_value] * (length - len(lst))

def clean_med_name(name):
    """Clean and normalize medication names."""
    return re.sub(r'\s+', ' ', name).strip().lower()

def convert_string_to_list_of_meds(x):
    """Convert comma-separated medication string to list."""
    if pd.isna(x) or not isinstance(x, str) or x.strip() == '':
        return []
    meds = [clean_med_name(med) for med in x.split(',') if clean_med_name(med)]
    return meds

def transform_meds_to_padded_ints(med_list, vocab, max_len):
    """Transform medication list to padded integer sequence."""
    if not isinstance(med_list, list):
        med_list = []
    unk_token_int = vocab.get("UNK", UNK_TOKEN_ID)
    int_sequence = [vocab.get(med, unk_token_int) for med in med_list]
    int_sequence = int_sequence[:max_len]
    padded_sequence = int_sequence + [PAD_TOKEN_ID] * (max_len - len(int_sequence))
    return padded_sequence

# --- Data Loading ---
print("\n" + "=" * 80)
print("STEP 1: Data Loading")
print("=" * 80)

print(f"\nLoading training data: {INPUT_FILE}...")
try:
    data = pd.read_csv(INPUT_FILE)
    print(f"[OK] Successfully loaded {len(data)} patient records")
except FileNotFoundError:
    print(f"[ERROR] {INPUT_FILE} not found. Please ensure the file exists.")
    exit()

# --- Data Conversion ---
print("\nConverting data columns to appropriate formats...")
print("  - Converting vital signs and lab results to float lists")
for col in train_vital_sign_cols + train_lab_cols:
    if col in data.columns:
        data[col] = data[col].apply(convert_string_to_list_of_floats)
    else:
        print(f"  Warning: Column '{col}' not found. Adding empty lists.")
        data[col] = pd.Series([[] for _ in range(len(data))], index=data.index)

print("  - Converting medications to string lists")
for col in train_med_cols:
    if col in data.columns:
        data[col] = data[col].apply(convert_string_to_list_of_meds)
    else:
        print(f"  Warning: Column '{col}' not found. Adding empty lists.")
        data[col] = pd.Series([[] for _ in range(len(data))], index=data.index)

# --- Prepare Target Variable ---
print("\nPreparing target variable (ICU Risk)...")
if train_target_col not in data.columns:
    print(f"[ERROR] Error: Target column '{train_target_col}' not found.")
    exit()

data['EVENT'] = data[train_target_col].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
y = data['EVENT']

print(f"[OK] Target distribution:")
print(f"  - ICU Risk (1): {(y == 1).sum()} patients ({(y == 1).sum()/len(y)*100:.1f}%)")
print(f"  - No ICU Risk (0): {(y == 0).sum()} patients ({(y == 0).sum()/len(y)*100:.1f}%)")

# --- Train/Test Split ---
print("\n" + "=" * 80)
print("STEP 2: Train/Test Split")
print("=" * 80)

print("\nSplitting data (80% train / 20% test, stratified)...")
data_train, data_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[OK] Training set: {len(data_train)} patients")
print(f"[OK] Testing set: {len(data_test)} patients")

# --- Preprocessing Fitting (Training Data Only) ---
print("\n" + "=" * 80)
print("STEP 3: Preprocessing Configuration (Training Data)")
print("=" * 80)

print("\nBuilding medication vocabulary from training data...")
med_vocab = {"PAD": PAD_TOKEN_ID, "UNK": UNK_TOKEN_ID}
current_index = 2
max_med_lengths = {}

for col in train_med_cols:
    if col in data_train.columns:
        med_lists = data_train[col].apply(lambda x: x if isinstance(x, list) else [])
        all_meds_train = list(itertools.chain.from_iterable(med_lists.tolist()))
        for med in all_meds_train:
            if med not in med_vocab:
                med_vocab[med] = current_index
                current_index += 1
        max_len_col = med_lists.apply(len).max() if not med_lists.empty else 0
        max_med_lengths[col] = int(max_len_col)
    else:
        max_med_lengths[col] = 0

print(f"[OK] Total unique medications: {len(med_vocab) - 2}")
print(f"[OK] Vocabulary size (with special tokens): {len(med_vocab)}")

print("\nDetermining max sequence lengths from training data...")
max_length_vitals = 0
for col in train_vital_sign_cols:
    if col in data_train.columns:
        lengths = data_train[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
        max_col_length = lengths.max() if not lengths.empty else 0
        if max_col_length > max_length_vitals:
            max_length_vitals = int(max_col_length)

print(f"[OK] Max vital sign sequence length: {max_length_vitals}")
print(f"[OK] Lab history length: {LAB_HISTORY_LENGTH}")

# --- Feature Transformation ---
print("\n" + "=" * 80)
print("STEP 4: Feature Engineering")
print("=" * 80)

def process_features_tabular(df, is_train_phase=True):
    """Process all features into tabular format for ML models."""
    print(f"\nProcessing {'training' if is_train_phase else 'testing'} features...")
    processed_features_list = []
    feature_names = []

    # 1. Vital Signs (Time Series)
    print("  - Processing vital signs (time series)...")
    for col in train_vital_sign_cols:
        if col in df.columns:
            padded_col = df[col].apply(lambda x: pad_list_to_length(
                x if isinstance(x, list) else [], max_length_vitals, pad_value=0.0
            ))
            col_array = np.array(padded_col.tolist(), dtype=float)
            expected_shape = (len(df), max_length_vitals if max_length_vitals > 0 else 0)
            if col_array.shape != expected_shape:
                col_array = np.zeros(expected_shape)
            processed_features_list.append(col_array)
            # Generate feature names for each time point
            for t in range(max_length_vitals):
                feature_names.append(f"{col}_t{t}")
        else:
            processed_features_list.append(np.zeros((len(df), max_length_vitals if max_length_vitals > 0 else 0)))
            for t in range(max_length_vitals):
                feature_names.append(f"{col}_t{t}")

    # 2. Lab Results (Historical Values)
    print("  - Processing lab results (last 4 measurements)...")
    for col in train_lab_cols:
        if col in df.columns:
            proc_col = df[col].apply(lambda x: [0.0]*(LAB_HISTORY_LENGTH - len(x)) + x[-LAB_HISTORY_LENGTH:]
                                     if isinstance(x, list) else [0.0]*LAB_HISTORY_LENGTH)
            col_array = np.array(proc_col.tolist(), dtype=float)
            expected_shape = (len(df), LAB_HISTORY_LENGTH)
            if col_array.shape != expected_shape:
                col_array = np.zeros(expected_shape)
            processed_features_list.append(col_array)
            # Generate feature names for each historical value
            for h in range(LAB_HISTORY_LENGTH):
                feature_names.append(f"{col}_h{h}")
        else:
            processed_features_list.append(np.zeros((len(df), LAB_HISTORY_LENGTH)))
            for h in range(LAB_HISTORY_LENGTH):
                feature_names.append(f"{col}_h{h}")

    # 3. Medications (Sequences + Aggregates)
    print("  - Processing medications (sequences + aggregates)...")
    for col in train_med_cols:
        max_len = max_med_lengths.get(col, 0)
        med_lists = df[col].apply(lambda x: x if isinstance(x, list) else []) if col in df else pd.Series([[] for _ in range(len(df))], index=df.index)

        # 3a. Padded Integer Sequence
        if col in df.columns:
            transformed_col = med_lists.apply(lambda x: transform_meds_to_padded_ints(x, med_vocab, max_len))
            col_array_seq = np.array(transformed_col.tolist(), dtype=int)
            expected_shape = (len(df), max_len if max_len > 0 else 0)
            if col_array_seq.shape != expected_shape:
                col_array_seq = np.zeros(expected_shape, dtype=int)
            processed_features_list.append(col_array_seq)
            # Feature names for medication sequences
            for s in range(max_len):
                feature_names.append(f"{col}_seq_{s}")
        else:
            processed_features_list.append(np.zeros((len(df), max_len if max_len > 0 else 0), dtype=int))
            for s in range(max_len):
                feature_names.append(f"{col}_seq_{s}")

        # 3b. Aggregate Features
        lengths = med_lists.apply(len).values.reshape(-1, 1)
        processed_features_list.append(lengths.astype(float))
        feature_names.append(f"{col}_length")

        processed_features_list.append((lengths > 0).astype(float))
        feature_names.append(f"{col}_has_med")

        processed_features_list.append(med_lists.apply(
            lambda x: med_vocab.get(x[0], UNK_TOKEN_ID) if len(x) > 0 else EMPTY_SEQ_ID
        ).values.reshape(-1, 1).astype(float))
        feature_names.append(f"{col}_first_med_id")

        processed_features_list.append(med_lists.apply(
            lambda x: med_vocab.get(x[-1], UNK_TOKEN_ID) if len(x) > 0 else EMPTY_SEQ_ID
        ).values.reshape(-1, 1).astype(float))
        feature_names.append(f"{col}_last_med_id")

        processed_features_list.append(med_lists.apply(lambda x: len(set(x))).values.reshape(-1, 1).astype(float))
        feature_names.append(f"{col}_unique_count")

    # Concatenate all features
    valid_features = [arr for arr in processed_features_list if arr.ndim == 2 and arr.shape[0] == len(df) and arr.shape[1] > 0]
    if not valid_features:
        return np.empty((len(df), 0)), []

    X_processed = np.concatenate(valid_features, axis=1)
    print(f"  [OK] Final feature matrix shape: {X_processed.shape}")
    print(f"  [OK] Total features: {X_processed.shape[1]}")

    return X_processed, feature_names

# Process training and testing data
X_train_processed, feature_names = process_features_tabular(data_train, is_train_phase=True)
X_test_processed, _ = process_features_tabular(data_test, is_train_phase=False)

# Verify feature names count matches feature count
print(f"\n[OK] Generated {len(feature_names)} feature names")
print(f"[OK] Feature matrix has {X_train_processed.shape[1]} features")

# Ensure feature names match actual features
if len(feature_names) != X_train_processed.shape[1]:
    print(f"[WARNING] Feature name mismatch: {len(feature_names)} names vs {X_train_processed.shape[1]} features")
    print(f"[WARNING] Truncating feature names to match feature count")
    feature_names = feature_names[:X_train_processed.shape[1]]

# --- Imputation and Scaling ---
print("\n" + "=" * 80)
print("STEP 5: Imputation and Scaling")
print("=" * 80)

print("\nApplying median imputation for missing values...")
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_processed)
X_test_imputed = imputer.transform(X_test_processed)
print("[OK] Imputation complete")

print("\nApplying Min-Max scaling to [0, 1] range...")
scaler = MinMaxScaler()
X_train_final = scaler.fit_transform(X_train_imputed)
X_test_final = scaler.transform(X_test_imputed)
print("[OK] Scaling complete")

print(f"\nFinal feature matrix shapes:")
print(f"  - Training: {X_train_final.shape}")
print(f"  - Testing: {X_test_final.shape}")

# Adjust feature_names if scaler dropped constant features
if X_train_final.shape[1] != len(feature_names):
    print(f"[WARNING] Scaler reduced features from {len(feature_names)} to {X_train_final.shape[1]}")
    print(f"[WARNING] Adjusting feature names to match (likely constant features removed)")
    feature_names = feature_names[:X_train_final.shape[1]]

# Save preprocessing objects for future use
preprocessing_objects = {
    'med_vocab': med_vocab,
    'max_med_lengths': max_med_lengths,
    'max_length_vitals': max_length_vitals,
    'LAB_HISTORY_LENGTH': LAB_HISTORY_LENGTH,
    'imputer': imputer,
    'scaler': scaler,
    'feature_names': feature_names
}
joblib.dump(preprocessing_objects, os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_preprocessing_objects.pkl'))
print(f"\n[OK] Saved preprocessing objects to {OUTPUT_DIR}/{OUTPUT_PREFIX}_preprocessing_objects.pkl")

# --- Algorithm 1: ANOVA F-test (Univariate Feature Selection) ---
print("\n" + "=" * 80)
print("STEP 6: ANOVA F-test Feature Selection (Univariate)")
print("=" * 80)

print("\nComputing F-statistics and p-values for all features...")
print("This evaluates each feature independently against ICU risk outcome...")

F_scores, p_values = f_classif(X_train_final, y_train)

print(f"[OK] Computed F-statistics for {len(F_scores)} features")
print(f"  - Features with p < 0.05: {(p_values < 0.05).sum()}")
print(f"  - Features with p < 0.01: {(p_values < 0.01).sum()}")
print(f"  - Features with p < 0.001: {(p_values < 0.001).sum()}")

# Handle any NaN or infinite values
F_scores = np.nan_to_num(F_scores, nan=0.0, posinf=0.0, neginf=0.0)
p_values = np.nan_to_num(p_values, nan=1.0, posinf=1.0, neginf=1.0)

# Normalize F-scores to [0, 1] for ranking
F_scores_normalized = (F_scores - F_scores.min()) / (F_scores.max() - F_scores.min() + 1e-10)

print(f"\n[OK] F-test feature selection complete")
print(f"  - Top feature: {feature_names[F_scores.argmax()]} (F-score: {F_scores.max():.2f})")
print(f"  - Mean F-score: {F_scores.mean():.2f}")
print(f"  - Median F-score: {np.median(F_scores):.2f}")

# --- Algorithm 2: Random Forest Feature Importance (Multivariate) ---
print("\n" + "=" * 80)
print("STEP 7: Random Forest Feature Importance (Multivariate)")
print("=" * 80)

print("\nTraining Random Forest classifier for feature importance...")
print("This captures feature interactions and temporal patterns...")

# Use a validation split for permutation importance
X_train_rf, X_val_rf, y_train_rf, y_val_rf = train_test_split(
    X_train_final, y_train, test_size=0.25, random_state=42, stratify=y_train
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print(f"  - Training on {len(X_train_rf)} samples, validating on {len(X_val_rf)} samples")
rf_model.fit(X_train_rf, y_train_rf)
print("[OK] Random Forest training complete")

# Evaluate on validation set
y_val_pred_proba = rf_model.predict_proba(X_val_rf)[:, 1]
val_auc = roc_auc_score(y_val_rf, y_val_pred_proba)
val_acc = accuracy_score(y_val_rf, rf_model.predict(X_val_rf))
print(f"  - Validation AUC-ROC: {val_auc:.4f}")
print(f"  - Validation Accuracy: {val_acc:.4f}")

# Get Gini-based feature importance
print("\nComputing Gini-based feature importance...")
gini_importance = rf_model.feature_importances_
print("[OK] Gini importance computed")

# Get permutation-based feature importance
print("\nComputing permutation-based feature importance (10 repeats)...")
print("This may take a few minutes...")
perm_importance_result = permutation_importance(
    rf_model, X_val_rf, y_val_rf,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)
perm_importance = perm_importance_result.importances_mean
print("[OK] Permutation importance computed")

# Handle any NaN or infinite values
gini_importance = np.nan_to_num(gini_importance, nan=0.0, posinf=0.0, neginf=0.0)
perm_importance = np.nan_to_num(perm_importance, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize to [0, 1]
gini_importance_normalized = (gini_importance - gini_importance.min()) / (gini_importance.max() - gini_importance.min() + 1e-10)
perm_importance_normalized = (perm_importance - perm_importance.min()) / (perm_importance.max() - perm_importance.min() + 1e-10)

# Combined RF importance (70% Gini, 30% Permutation)
rf_importance_combined = 0.7 * gini_importance_normalized + 0.3 * perm_importance_normalized

print(f"\n[OK] Random Forest feature importance complete")
print(f"  - Top feature (Gini): {feature_names[gini_importance.argmax()]} (importance: {gini_importance.max():.4f})")
print(f"  - Top feature (Permutation): {feature_names[perm_importance.argmax()]} (importance: {perm_importance.max():.4f})")
print(f"  - Mean importance: {rf_importance_combined.mean():.4f}")

# --- Combined Feature Ranking ---
print("\n" + "=" * 80)
print("STEP 8: Combined Feature Ranking")
print("=" * 80)

print("\nCombining F-test and Random Forest importance scores...")
print("  - Weighting: 50% F-test + 50% Random Forest")

# Combined score (equal weighting)
combined_score = 0.5 * F_scores_normalized + 0.5 * rf_importance_combined

# Create comprehensive results dataframe
results_df = pd.DataFrame({
    'feature_index': range(len(feature_names)),
    'feature_name': feature_names,
    'F_score': F_scores,
    'p_value': p_values,
    'F_score_normalized': F_scores_normalized,
    'RF_gini_importance': gini_importance,
    'RF_perm_importance': perm_importance,
    'RF_combined_importance': rf_importance_combined,
    'combined_score': combined_score
})

# Sort by combined score (descending)
results_df = results_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
results_df['rank'] = range(1, len(results_df) + 1)

print("[OK] Combined ranking complete")
print(f"\nTop 10 Features by Combined Score:")
print("-" * 80)
for idx, row in results_df.head(10).iterrows():
    print(f"  {row['rank']:3d}. {row['feature_name']:40s} | Score: {row['combined_score']:.4f}")

# Save complete results
results_csv_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_feature_selection_results.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"\n[OK] Saved complete results to {results_csv_path}")

# --- Extract Multiple Feature Sets ---
print("\n" + "=" * 80)
print("STEP 9: Extracting Multiple Feature Sets")
print("=" * 80)

feature_tiers = [50, 100, 200, 500]
print(f"\nCreating feature sets for: {feature_tiers}")

for n_features in feature_tiers:
    if n_features > len(results_df):
        print(f"  Warning: Requested {n_features} features but only {len(results_df)} available. Using all features.")
        n_features = len(results_df)

    top_features = results_df.head(n_features)

    feature_set = {
        'n_features': n_features,
        'indices': top_features['feature_index'].tolist(),
        'names': top_features['feature_name'].tolist(),
        'scores': top_features['combined_score'].tolist(),
        'F_scores': top_features['F_score'].tolist(),
        'p_values': top_features['p_value'].tolist(),
        'RF_importance': top_features['RF_combined_importance'].tolist()
    }

    output_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_top{n_features}_features.pkl')
    joblib.dump(feature_set, output_path)
    print(f"  [OK] Saved top {n_features} features to {output_path}")

# --- Visualization ---
print("\n" + "=" * 80)
print("STEP 10: Creating Visualizations")
print("=" * 80)

# Plot 1: Top 30 Features by Combined Score
print("\nGenerating feature importance bar chart...")
fig, ax = plt.subplots(figsize=(12, 10))
top_30 = results_df.head(30)
y_pos = np.arange(len(top_30))

ax.barh(y_pos, top_30['combined_score'], align='center', color='steelblue', alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(top_30['feature_name'], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Combined Score', fontsize=12, fontweight='bold')
ax.set_title(f'Top 30 Features - {DATASET_TYPE.upper()} Dataset', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()

plot1_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_feature_importance_plot.png')
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved feature importance plot to {plot1_path}")

# Plot 2: Comparison of F-test vs RF Importance
print("\nGenerating F-test vs Random Forest comparison plot...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

top_20 = results_df.head(20)

# Subplot 1: F-score
axes[0, 0].barh(range(len(top_20)), top_20['F_score_normalized'], color='coral', alpha=0.7)
axes[0, 0].set_yticks(range(len(top_20)))
axes[0, 0].set_yticklabels(top_20['feature_name'], fontsize=8)
axes[0, 0].invert_yaxis()
axes[0, 0].set_xlabel('Normalized F-score', fontweight='bold')
axes[0, 0].set_title('ANOVA F-test Scores', fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# Subplot 2: RF Importance
axes[0, 1].barh(range(len(top_20)), top_20['RF_combined_importance'], color='mediumseagreen', alpha=0.7)
axes[0, 1].set_yticks(range(len(top_20)))
axes[0, 1].set_yticklabels(top_20['feature_name'], fontsize=8)
axes[0, 1].invert_yaxis()
axes[0, 1].set_xlabel('RF Importance', fontweight='bold')
axes[0, 1].set_title('Random Forest Importance', fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Subplot 3: Combined Score
axes[1, 0].barh(range(len(top_20)), top_20['combined_score'], color='steelblue', alpha=0.7)
axes[1, 0].set_yticks(range(len(top_20)))
axes[1, 0].set_yticklabels(top_20['feature_name'], fontsize=8)
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Combined Score', fontweight='bold')
axes[1, 0].set_title('Combined Ranking', fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Subplot 4: Scatter plot F-test vs RF
axes[1, 1].scatter(results_df['F_score_normalized'], results_df['RF_combined_importance'],
                   alpha=0.5, c=results_df['combined_score'], cmap='viridis', s=20)
axes[1, 1].set_xlabel('Normalized F-score', fontweight='bold')
axes[1, 1].set_ylabel('RF Importance', fontweight='bold')
axes[1, 1].set_title('F-test vs Random Forest Correlation', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# Add colorbar for combined score
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Combined Score', fontweight='bold')

fig.suptitle(f'Feature Selection Comparison - {DATASET_TYPE.upper()} Dataset',
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()

plot2_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_selection_comparison.png')
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved comparison plot to {plot2_path}")

# Plot 3: Feature Category Distribution in Top Features
print("\nGenerating feature category distribution plot...")

def categorize_feature(feature_name):
    """Categorize feature by type."""
    if any(vs in feature_name for vs in train_vital_sign_cols):
        return 'Vital Signs'
    elif any(lab in feature_name for lab in train_lab_cols):
        return 'Lab Results'
    elif any(med in feature_name for med in train_med_cols):
        if '_seq_' in feature_name:
            return 'Medication Sequences'
        else:
            return 'Medication Aggregates'
    return 'Other'

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for idx, n_features in enumerate([50, 100, 200, 500]):
    if n_features <= len(results_df):
        top_n = results_df.head(n_features)
        categories = top_n['feature_name'].apply(categorize_feature)
        category_counts = categories.value_counts()

        colors = {'Vital Signs': 'lightcoral', 'Lab Results': 'lightblue',
                  'Medication Sequences': 'lightgreen', 'Medication Aggregates': 'lightyellow'}

        axes[idx].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                      colors=[colors.get(cat, 'gray') for cat in category_counts.index],
                      startangle=90)
        axes[idx].set_title(f'Top {n_features} Features', fontweight='bold')

fig.suptitle(f'Feature Category Distribution - {DATASET_TYPE.upper()} Dataset',
             fontsize=16, fontweight='bold')
plt.tight_layout()

plot3_path = os.path.join(OUTPUT_DIR, f'{OUTPUT_PREFIX}_category_distribution.png')
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Saved category distribution plot to {plot3_path}")

# --- Final Summary ---
print("\n" + "=" * 80)
print("FEATURE SELECTION SUMMARY")
print("=" * 80)

print(f"\nDataset: {DATASET_TYPE.upper()}")
print(f"Total Features: {len(feature_names)}")
print(f"Total Patients: {len(data)} ({len(data_train)} train, {len(data_test)} test)")

print(f"\nAlgorithms Used:")
print(f"  1. ANOVA F-test (Univariate)")
print(f"  2. Random Forest Importance (Multivariate)")

print(f"\nFeature Sets Created:")
for n in feature_tiers:
    if n <= len(results_df):
        print(f"  - Top {n} features")

print(f"\nOutput Files:")
print(f"  - {OUTPUT_PREFIX}_feature_selection_results.csv (complete rankings)")
print(f"  - {OUTPUT_PREFIX}_top{50/100/200/500}_features.pkl (feature sets)")
print(f"  - {OUTPUT_PREFIX}_preprocessing_objects.pkl (preprocessing pipeline)")
print(f"  - {OUTPUT_PREFIX}_feature_importance_plot.png (visualization)")
print(f"  - {OUTPUT_PREFIX}_selection_comparison.png (algorithm comparison)")
print(f"  - {OUTPUT_PREFIX}_category_distribution.png (feature categories)")

print(f"\nAll results saved to: {OUTPUT_DIR}/")

print("\n" + "=" * 80)
print("[OK] FEATURE SELECTION COMPLETE!")
print("=" * 80)

print("\nNext Steps:")
print("  1. Review the feature rankings in the CSV file")
print("  2. Examine the visualization plots")
print("  3. Use the .pkl feature sets to train models with reduced features")
print("  4. Compare performance: full features vs. selected features")
print("  5. Interpret clinically relevant features for domain experts")

print("\n" + "=" * 80)
