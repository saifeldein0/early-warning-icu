"""
Complete Validation on New ICU Dataset
======================================
Validates all models (Original XGBoost, BIC LR, Feature Selection) on the new
"icu (1).xlsx" dataset using actual Length of Stay (LOS) calculations.

Public release note:
- This script regenerates validation output folders that are not committed in the
  trimmed public repository.
- Private datasets and model binaries required for a full run are intentionally omitted.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from datetime import datetime
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, 
                             precision_recall_curve, average_precision_score)

warnings.filterwarnings('ignore')

print("=" * 80)
print("COMPLETE VALIDATION ON NEW ICU DATASET")
print("=" * 80)

# ============================================================================
# STEP 0: DELETE OLD PLOTS (FRESH START)
# ============================================================================
print("\n[0/6] Deleting old plots for fresh start...")

folders_to_clean = [
    'validation_graphs',
    'logistic_regression_BIC',
    'feature_selection_validation_graphs'
]

for folder in folders_to_clean:
    if os.path.exists(folder):
        for root, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith('.png'):
                    try:
                        os.remove(os.path.join(root, f))
                    except:
                        pass
                    
print("   [OK] Old plots deleted")

# ============================================================================
# STEP 1: LOAD NEW DATASET
# ============================================================================
print("\n[1/6] Loading new dataset 'icu (1).xlsx'...")

df = pd.read_excel('icu (1).xlsx')
print(f"   Total patients: {df.shape[0]}")

# Calculate Length of Stay (LOS) in hours
df['LOS_hours'] = (df['ICU_ADMISSION_DATE'] - df['ADMISSION_DATE']).dt.total_seconds() / 3600
df['LOS_days'] = df['LOS_hours'] / 24

print(f"   LOS range: {df['LOS_hours'].min():.1f} to {df['LOS_hours'].max():.1f} hours")
print(f"   LOS mean: {df['LOS_hours'].mean():.1f} hours ({df['LOS_days'].mean():.1f} days)")

# Split by diagnosis (hematology vs solid/non-hematology)
def is_hematology_diagnosis(diagnosis_str):
    if pd.isna(diagnosis_str):
        return False
    diagnosis_lower = str(diagnosis_str).lower()
    hematology_keywords = ['leukemia', 'lymphoma', 'myeloma', 'hodgkin', 'aml', 'all']
    return any(keyword in diagnosis_lower for keyword in hematology_keywords)

df['IS_HEMATOLOGY'] = df['DIAGNOSIS'].apply(is_hematology_diagnosis)
df_hematology = df[df['IS_HEMATOLOGY']].copy()
df_solid = df[~df['IS_HEMATOLOGY']].copy()

print(f"   Hematology patients: {len(df_hematology)}")
print(f"   Non-Hematology patients: {len(df_solid)}")

# ============================================================================
# STEP 2: PREPROCESSING FUNCTIONS
# ============================================================================
print("\n[2/6] Setting up preprocessing...")

def convert_string_to_list_of_floats(s):
    """Convert string representation of list to actual list of floats."""
    if pd.isna(s):
        return []
    if isinstance(s, (list, np.ndarray)):
        return [float(x) if not pd.isna(x) else 0.0 for x in s]
    try:
        s = str(s).strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        if not s:
            return []
        values = [float(x.strip()) if x.strip() else 0.0 for x in s.split(',')]
        return values
    except:
        return []

def pad_list_to_length(lst, target_length, pad_value=0.0):
    """Pad list to target length."""
    if len(lst) >= target_length:
        return lst[:target_length]
    return lst + [pad_value] * (target_length - len(lst))

def clean_med_name(med_name):
    """Clean medication name."""
    if pd.isna(med_name) or not med_name:
        return ''
    return str(med_name).strip().lower().replace(' ', '_')

def convert_string_to_list_of_meds(s):
    """Convert medication string to list."""
    if pd.isna(s) or not s:
        return []
    if isinstance(s, list):
        return [clean_med_name(m) for m in s if m]
    try:
        s = str(s).strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        if not s:
            return []
        meds = [clean_med_name(m.strip().strip("'\"")) for m in s.split(',')]
        return [m for m in meds if m]
    except:
        return []

def transform_meds_to_padded_ints(med_list, vocab, max_len):
    """Transform medication list to padded integer list."""
    int_list = [vocab.get(m, 0) for m in med_list[:max_len]]
    return pad_list_to_length(int_list, max_len, pad_value=0)

def preprocess_validation_data(df_raw, preprocessing_params):
    """
    Preprocess validation data to match training format.
    Uses parameters from the saved preprocessing objects.
    """
    df = df_raw.copy()
    
    # Get preprocessing parameters
    imputer = preprocessing_params.get('imputer_tabular', preprocessing_params.get('imputer'))
    scaler = preprocessing_params.get('scaler_tabular', preprocessing_params.get('scaler'))
    med_vocab = preprocessing_params.get('med_vocab', {})
    max_med_lengths = preprocessing_params.get('max_med_lengths', {})
    max_length_vitals = preprocessing_params.get('max_length_vitals', 308)
    LAB_HISTORY_LENGTH = preprocessing_params.get('LAB_HISTORY_LENGTH', 4)
    
    # Get column lists from preprocessing params
    train_vital_sign_cols = preprocessing_params.get('train_vital_sign_cols', 
        ['HEART_RATE', 'PULSE_OXIMETRY', 'TEMPERATURE', 'SYSTOLIC_BLOOD_PRESSURE', 
         'MEAN_ARTERIAL_PRESSURE', 'DIASTOLIC_BLOOD_PRESSURE', 'RESPIRATION_RATE'])
    train_lab_cols = preprocessing_params.get('train_lab_cols',
        ['AST_RESULT', 'CREATININE_RESULT', 'TOTAL_BILIRUBIN_RESULT', 'DIRECT_BILIRUBIN_RESULT',
         'POTASSIUM_RESULT', 'HEMOGLOBIN_RESULT', 'LEUKOCYTE_COUNT_RESULT', 'ABSOLUTE_NEUTROPHILS',
         'PLATELET_COUNT_RESULT', 'PROTHROMBIN_CONCENTRATION'])
    train_med_cols = preprocessing_params.get('train_med_cols',
        ['ANTIBIOTICS', 'NEUROLOGY_DRUGS', 'CARDIOLOGY_DRUGS', 'FUNGAL_DRUGS'])
    
    # Process vital signs
    for col in train_vital_sign_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_string_to_list_of_floats)
    
    # Process labs
    for col in train_lab_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_string_to_list_of_floats)
    
    # Process medications
    for col in train_med_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_string_to_list_of_meds)
    
    # Build feature matrix
    features_list = []
    
    for idx, row in df.iterrows():
        patient_features = []
        
        # Vital signs (padded to max_length_vitals)
        for col in train_vital_sign_cols:
            if col in df.columns:
                vals = row[col] if isinstance(row[col], list) else []
                padded = pad_list_to_length(vals, max_length_vitals, 0.0)
                patient_features.extend(padded)
        
        # Labs (last LAB_HISTORY_LENGTH values)
        for col in train_lab_cols:
            if col in df.columns:
                vals = row[col] if isinstance(row[col], list) else []
                last_n = vals[-LAB_HISTORY_LENGTH:] if len(vals) >= LAB_HISTORY_LENGTH else vals
                padded = pad_list_to_length(last_n, LAB_HISTORY_LENGTH, 0.0)
                patient_features.extend(padded)
        
        # Medications
        for col in train_med_cols:
            if col in df.columns:
                meds = row[col] if isinstance(row[col], list) else []
                max_len = max_med_lengths.get(col, 100)
                vocab = med_vocab.get(col, {})
                padded_ints = transform_meds_to_padded_ints(meds, vocab, max_len)
                patient_features.extend(padded_ints)
                
                # Aggregate features (3 per medication column)
                patient_features.append(len(meds))  # sequence length
                patient_features.append(padded_ints[0] if padded_ints else 0)  # first med
                patient_features.append(padded_ints[-1] if padded_ints else 0)  # last med
        
        features_list.append(patient_features)
    
    X = np.array(features_list, dtype=np.float32)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Impute and scale
    if imputer is not None:
        try:
            X = imputer.transform(X)
        except Exception as e:
            print(f"      Imputer warning: {e}")
    
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print(f"      Scaler warning: {e}")
    
    return X

print("   [OK] Preprocessing functions ready")

# ============================================================================
# STEP 3: VALIDATE ALL MODELS
# ============================================================================
print("\n[3/6] Validating all models...")

# Store all results
all_results = []

# ---------- 3A: Original XGBoost Models ----------
print("\n   --- Original XGBoost Models ---")

for dataset_type, df_subset in [('Hematology', df_hematology), ('Non-Hematology', df_solid)]:
    if dataset_type == 'Hematology':
        model_path = 'Website/models/xgboost_hematology.pkl'
        preproc_path = 'comprehensive_hematology_models/preprocessing_objects.pkl'
    else:
        model_path = 'Website/models/xgboost.pkl'
        preproc_path = 'comprehensive_solid_models/preprocessing_objects.pkl'
    
    try:
        model_data = joblib.load(model_path)
        model = model_data['model'] if isinstance(model_data, dict) else model_data
        preprocessing_params = joblib.load(preproc_path)
        
        X = preprocess_validation_data(df_subset.copy(), preprocessing_params)
        
        expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else X.shape[1]
        
        # Adjust features if needed
        if X.shape[1] != expected_features:
            print(f"      {dataset_type}: Feature adjustment {X.shape[1]} -> {expected_features}")
            if X.shape[1] < expected_features:
                X = np.hstack([X, np.zeros((X.shape[0], expected_features - X.shape[1]))])
            else:
                X = X[:, :expected_features]
        
        # Predict (all should be ICU=1)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        y_true = np.ones(len(df_subset))  # All are ICU patients
        
        acc = accuracy_score(y_true, y_pred)
        
        # Store results with LOS
        for i, (idx, row) in enumerate(df_subset.iterrows()):
            all_results.append({
                'Model_Type': 'Original_XGBoost',
                'Dataset': dataset_type,
                'Patient_Idx': idx,
                'LOS_hours': row['LOS_hours'],
                'LOS_days': row['LOS_days'],
                'True_Label': 1,
                'Predicted': y_pred[i],
                'Probability': y_proba[i],
                'Correct': y_pred[i] == 1
            })
        
        print(f"      {dataset_type}: {acc*100:.1f}% accuracy ({int(sum(y_pred))}/{len(y_pred)} correct)")
        
    except Exception as e:
        print(f"      {dataset_type}: ERROR - {e}")

# ---------- 3B: BIC Logistic Regression Models ----------
print("\n   --- BIC Logistic Regression Models ---")

for dataset_type, df_subset in [('Hematology', df_hematology), ('Non-Hematology', df_solid)]:
    dataset_key = 'hematology' if dataset_type == 'Hematology' else 'solid'
    model_path = f'logistic_regression_BIC/bic_lr_model_{dataset_key}.pkl'
    
    if dataset_type == 'Hematology':
        preproc_path = 'comprehensive_hematology_models/preprocessing_objects.pkl'
    else:
        preproc_path = 'comprehensive_solid_models/preprocessing_objects.pkl'
    
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_mask = model_data['feature_mask'].flatten()
        preprocessing_params = joblib.load(preproc_path)
        
        X = preprocess_validation_data(df_subset.copy(), preprocessing_params)
        
        # Adjust features to match mask size
        if X.shape[1] < len(feature_mask):
            X = np.hstack([X, np.zeros((X.shape[0], len(feature_mask) - X.shape[1]))])
        elif X.shape[1] > len(feature_mask):
            X = X[:, :len(feature_mask)]
        
        # Apply feature mask
        X_selected = X[:, feature_mask]
        
        y_pred = model.predict(X_selected)
        y_proba = model.predict_proba(X_selected)[:, 1]
        y_true = np.ones(len(df_subset))
        
        acc = accuracy_score(y_true, y_pred)
        
        for i, (idx, row) in enumerate(df_subset.iterrows()):
            all_results.append({
                'Model_Type': 'BIC_LogisticRegression',
                'Dataset': dataset_type,
                'Patient_Idx': idx,
                'LOS_hours': row['LOS_hours'],
                'LOS_days': row['LOS_days'],
                'True_Label': 1,
                'Predicted': y_pred[i],
                'Probability': y_proba[i],
                'Correct': y_pred[i] == 1
            })
        
        print(f"      {dataset_type}: {acc*100:.1f}% accuracy ({int(sum(y_pred))}/{len(y_pred)} correct)")
        
    except Exception as e:
        print(f"      {dataset_type}: ERROR - {e}")

# ---------- 3C: Feature Selection Models ----------
print("\n   --- Feature Selection Models ---")

BEST_FS_MODELS = {
    'Hematology': {
        'XGBoost': 2000, 'RandomForest': 2250, 'LogisticRegression': 1500,
        'SVM': 2000, 'DecisionTree': 750
    },
    'Non-Hematology': {
        'XGBoost': 1750, 'RandomForest': 2250, 'LogisticRegression': 750,
        'SVM': 2250, 'DecisionTree': 1500
    }
}

for dataset_type, df_subset in [('Hematology', df_hematology), ('Non-Hematology', df_solid)]:
    dataset_key = 'hematology' if dataset_type == 'Hematology' else 'solid'
    preproc_path = f'{dataset_key}_feature_selection_results/{dataset_key}_preprocessing_objects.pkl'
    
    print(f"\n      {dataset_type}:")
    
    for model_name, n_features in BEST_FS_MODELS[dataset_type].items():
        model_dir = f'extended_feature_selection_results/{dataset_key}/models/{model_name.lower()}_{n_features}feat'
        model_file = f'{model_dir}/model.pkl'
        indices_file = f'{dataset_key}_feature_selection_results/{dataset_key}_top{n_features}_features.pkl'
        
        try:
            model = joblib.load(model_file)
            feature_data = joblib.load(indices_file)
            feature_indices = feature_data['indices']
            preprocessing_params = joblib.load(preproc_path)
            
            X = preprocess_validation_data(df_subset.copy(), preprocessing_params)
            
            # Ensure we have enough features
            max_idx = max(feature_indices) if len(feature_indices) > 0 else 0
            if X.shape[1] <= max_idx:
                X = np.hstack([X, np.zeros((X.shape[0], max_idx - X.shape[1] + 1))])
            
            # Select features
            X_selected = X[:, feature_indices]
            
            y_pred = model.predict(X_selected)
            y_proba = model.predict_proba(X_selected)[:, 1]
            y_true = np.ones(len(df_subset))
            
            acc = accuracy_score(y_true, y_pred)
            
            for i, (idx, row) in enumerate(df_subset.iterrows()):
                all_results.append({
                    'Model_Type': f'FS_{model_name}',
                    'Dataset': dataset_type,
                    'Patient_Idx': idx,
                    'LOS_hours': row['LOS_hours'],
                    'LOS_days': row['LOS_days'],
                    'True_Label': 1,
                    'Predicted': y_pred[i],
                    'Probability': y_proba[i],
                    'Correct': y_pred[i] == 1
                })
            
            print(f"         {model_name}: {acc*100:.1f}%")
            
        except Exception as e:
            print(f"         {model_name}: ERROR - {str(e)[:60]}")

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

if len(results_df) == 0:
    print("\n   ERROR: No results collected! Check model paths and preprocessing.")
    exit(1)

results_df.to_csv('complete_validation_results.csv', index=False)
print(f"\n   [OK] Saved {len(results_df)} validation results")

# ============================================================================
# STEP 4: GENERATE ALL PLOTS
# ============================================================================
print("\n[4/6] Generating all plots...")

# Ensure output folders exist
os.makedirs('validation_graphs', exist_ok=True)
os.makedirs('logistic_regression_BIC', exist_ok=True)
os.makedirs('feature_selection_validation_graphs', exist_ok=True)

# Get unique model types
model_types = results_df['Model_Type'].unique()

# Calculate summary metrics per model
summary_metrics = []
for model_type in model_types:
    for dataset in ['Hematology', 'Non-Hematology']:
        subset = results_df[(results_df['Model_Type'] == model_type) & (results_df['Dataset'] == dataset)]
        if len(subset) == 0:
            continue
        
        acc = subset['Correct'].mean()
        avg_prob = subset['Probability'].mean()
        avg_prob_correct = subset[subset['Correct']]['Probability'].mean() if subset['Correct'].any() else 0
        avg_los_correct = subset[subset['Correct']]['LOS_hours'].mean() if subset['Correct'].any() else 0
        avg_los_incorrect = subset[~subset['Correct']]['LOS_hours'].mean() if (~subset['Correct']).any() else 0
        
        summary_metrics.append({
            'Model': model_type,
            'Dataset': dataset,
            'Accuracy': acc,
            'N_Patients': len(subset),
            'N_Correct': int(subset['Correct'].sum()),
            'Avg_Probability': avg_prob,
            'Avg_Prob_Correct': avg_prob_correct,
            'Avg_LOS_Correct': avg_los_correct,
            'Avg_LOS_Incorrect': avg_los_incorrect
        })

summary_df = pd.DataFrame(summary_metrics)
summary_df.to_csv('complete_validation_summary.csv', index=False)

# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def plot_accuracy_comparison(summary_df, output_path):
    """Plot accuracy comparison for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        ax = axes[i]
        data = summary_df[summary_df['Dataset'] == dataset].sort_values('Accuracy', ascending=True)
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(dataset)
            continue
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))
        bars = ax.barh(range(len(data)), data['Accuracy'] * 100, color=colors)
        
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels(data['Model'].values)
        ax.set_xlabel('Accuracy (%)')
        ax.set_title(f'{dataset}\n(n={data["N_Patients"].iloc[0]})', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 100)
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add percentage labels
        for bar, acc in zip(bars, data['Accuracy']):
            ax.text(acc * 100 + 1, bar.get_y() + bar.get_height()/2, 
                   f'{acc*100:.1f}%', va='center', fontsize=9)
    
    plt.suptitle('Model Accuracy Comparison - ICU (1) Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_probability_distribution(results_df, output_path):
    """Plot probability distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        ax = axes[i]
        data = results_df[results_df['Dataset'] == dataset]
        
        # Get unique models
        models = data['Model_Type'].unique()
        
        for model in models:
            model_data = data[data['Model_Type'] == model]
            ax.hist(model_data['Probability'], bins=20, alpha=0.5, label=model)
        
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Number of Patients')
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Probability Distribution - All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_los_analysis(results_df, summary_df, output_path):
    """Plot Length of Stay analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: LOS vs Prediction Accuracy (Bar) - Aggregated across all models
    ax = axes[0, 0]
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        data = results_df[results_df['Dataset'] == dataset]
        correct = data[data['Correct']]['LOS_hours'].mean()
        incorrect = data[~data['Correct']]['LOS_hours'].mean() if (~data['Correct']).any() else 0
        
        x = np.array([0, 1]) + i * 2.5
        colors = ['#4CAF50', '#F44336'] if i == 0 else ['#81C784', '#E57373']
        ax.bar(x, [correct, incorrect], width=0.8, color=colors)
        
        ax.text(x[0], correct + 10, f'{correct:.1f}h', ha='center', fontsize=9)
        if incorrect > 0:
            ax.text(x[1], incorrect + 10, f'{incorrect:.1f}h', ha='center', fontsize=9)
    
    ax.set_xticks([0.5, 3])
    ax.set_xticklabels(['Hematology', 'Non-Hematology'])
    ax.set_ylabel('Average LOS (hours)')
    ax.set_title('Average Length of Stay: Correct vs Incorrect Predictions\n(All Models Combined)', fontweight='bold')
    ax.legend(['Correctly Predicted', 'Incorrectly Predicted'], loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: LOS Distribution by Prediction (Box plots)
    ax = axes[0, 1]
    plot_data = []
    labels = []
    positions = []
    colors_bp = []
    
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        data = results_df[results_df['Dataset'] == dataset]
        correct_los = data[data['Correct']]['LOS_hours'].values
        incorrect_los = data[~data['Correct']]['LOS_hours'].values
        
        if len(correct_los) > 0:
            plot_data.append(correct_los)
            positions.append(i*3)
            colors_bp.append('#4CAF50')
        if len(incorrect_los) > 0:
            plot_data.append(incorrect_los)
            positions.append(i*3 + 1)
            colors_bp.append('#F44336')
    
    if len(plot_data) > 0:
        bp = ax.boxplot(plot_data, positions=positions, widths=0.6, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color)
    
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(['Hematology', 'Non-Hematology'])
    ax.set_ylabel('LOS (hours)')
    ax.set_title('LOS Distribution by Prediction Outcome', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: LOS vs Probability scatter (Original XGBoost only)
    ax = axes[1, 0]
    for dataset in ['Hematology', 'Non-Hematology']:
        data = results_df[(results_df['Dataset'] == dataset) & 
                          (results_df['Model_Type'] == 'Original_XGBoost')]
        if len(data) > 0:
            ax.scatter(data['LOS_hours'], data['Probability'], 
                      alpha=0.5, label=dataset, s=50)
    
    ax.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
    ax.set_xlabel('Length of Stay (hours)')
    ax.set_ylabel('Predicted Probability')
    ax.set_title('LOS vs Predicted Probability (Original XGBoost)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary Table
    ax = axes[1, 1]
    ax.axis('off')
    
    # Get LOS stats
    hem_correct_los = results_df[(results_df['Dataset'] == 'Hematology') & results_df['Correct']]['LOS_hours']
    hem_incorrect_los = results_df[(results_df['Dataset'] == 'Hematology') & ~results_df['Correct']]['LOS_hours']
    solid_correct_los = results_df[(results_df['Dataset'] == 'Non-Hematology') & results_df['Correct']]['LOS_hours']
    solid_incorrect_los = results_df[(results_df['Dataset'] == 'Non-Hematology') & ~results_df['Correct']]['LOS_hours']
    
    summary_text = f"""
LENGTH OF STAY (LOS) ANALYSIS SUMMARY
=====================================

Dataset: icu (1).xlsx
Total Patients: {len(df)}
LOS Calculation: ICU_ADMISSION_DATE - ADMISSION_DATE

HEMATOLOGY ({len(df_hematology)} patients):
- Avg LOS (Correct): {hem_correct_los.mean():.1f} hours ({hem_correct_los.mean()/24:.1f} days)
- Avg LOS (Incorrect): {hem_incorrect_los.mean() if len(hem_incorrect_los) > 0 else 0:.1f} hours

NON-HEMATOLOGY ({len(df_solid)} patients):
- Avg LOS (Correct): {solid_correct_los.mean():.1f} hours ({solid_correct_los.mean()/24:.1f} days)
- Avg LOS (Incorrect): {solid_incorrect_los.mean() if len(solid_incorrect_los) > 0 else 0:.1f} hours

KEY FINDING:
Patients correctly predicted as ICU tend to have 
{('LONGER' if hem_correct_los.mean() > hem_incorrect_los.mean() else 'SHORTER')} hospital stays 
before ICU admission, suggesting the model benefits 
from more observation data.
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Length of Stay Analysis - ICU (1) Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(results_df, output_path):
    """Plot confusion matrices for all models."""
    models = results_df['Model_Type'].unique()
    n_models = len(models)
    
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models))
    
    for i, model in enumerate(models):
        for j, dataset in enumerate(['Hematology', 'Non-Hematology']):
            ax = axes[i, j] if n_models > 1 else axes[j]
            
            data = results_df[(results_df['Model_Type'] == model) & (results_df['Dataset'] == dataset)]
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(f'{model} - {dataset}')
                continue
            
            y_true = data['True_Label'].values
            y_pred = data['Predicted'].values
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Non-ICU', 'ICU'], yticklabels=['Non-ICU', 'ICU'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            acc = (y_pred == y_true).mean() * 100
            ax.set_title(f'{model}\n{dataset} ({acc:.1f}%)', fontsize=10)
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_comparison_heatmap(summary_df, output_path):
    """Create heatmap comparing all models across metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        ax = axes[i]
        data = summary_df[summary_df['Dataset'] == dataset].copy()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax.set_title(dataset)
            continue
        
        data = data.set_index('Model')
        
        # Select numeric columns for heatmap
        heatmap_data = data[['Accuracy', 'Avg_Probability', 'Avg_Prob_Correct']].copy()
        heatmap_data.columns = ['Accuracy', 'Avg Prob', 'Avg Prob (Correct)']
        
        # Sort by accuracy
        heatmap_data = heatmap_data.sort_values('Accuracy', ascending=False)
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax,
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'})
        ax.set_title(f'{dataset}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
    
    plt.suptitle('Model Performance Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Generate all plots
print("\n   Generating accuracy comparison...")
plot_accuracy_comparison(summary_df, 'validation_graphs/accuracy_comparison_all_models.png')

print("   Generating probability distribution...")
plot_probability_distribution(results_df, 'validation_graphs/probability_distribution_all_models.png')

print("   Generating LOS analysis...")
plot_los_analysis(results_df, summary_df, 'validation_graphs/length_of_stay_analysis.png')

print("   Generating confusion matrices...")
plot_confusion_matrices(results_df, 'validation_graphs/confusion_matrices_all_models.png')

print("   Generating model comparison heatmap...")
plot_model_comparison_heatmap(summary_df, 'validation_graphs/model_comparison_heatmap.png')

# Copy relevant plots to other folders
print("\n   Copying plots to BIC and FS folders...")

# BIC folder plots
shutil.copy('validation_graphs/accuracy_comparison_all_models.png', 
            'logistic_regression_BIC/accuracy_comparison_all_models.png')
shutil.copy('validation_graphs/length_of_stay_analysis.png',
            'logistic_regression_BIC/length_of_stay_analysis.png')

# Feature selection folder plots
shutil.copy('validation_graphs/accuracy_comparison_all_models.png',
            'feature_selection_validation_graphs/accuracy_comparison_all_models.png')
shutil.copy('validation_graphs/length_of_stay_analysis.png',
            'feature_selection_validation_graphs/length_of_stay_analysis.png')

# ============================================================================
# STEP 5: INDIVIDUAL FOLDER PLOTS
# ============================================================================
print("\n[5/6] Generating folder-specific plots...")

# --- Validation Graphs Folder (Original Models) ---
print("\n   --- validation_graphs/ ---")

orig_results = results_df[results_df['Model_Type'] == 'Original_XGBoost']

if len(orig_results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        ax = axes[i]
        data = orig_results[orig_results['Dataset'] == dataset]
        
        if len(data) > 0:
            correct_data = data[data['Correct']]['Probability']
            incorrect_data = data[~data['Correct']]['Probability']
            
            if len(correct_data) > 0:
                ax.hist(correct_data, bins=15, alpha=0.7, color='#4CAF50', label='Correct')
            if len(incorrect_data) > 0:
                ax.hist(incorrect_data, bins=15, alpha=0.7, color='#F44336', label='Incorrect')
            
            ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'{dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Original XGBoost - Probability Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('validation_graphs/original_xgboost_probability.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- BIC Folder ---
print("   --- logistic_regression_BIC/ ---")

bic_results = results_df[results_df['Model_Type'] == 'BIC_LogisticRegression']

if len(bic_results) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        ax = axes[i]
        data = bic_results[bic_results['Dataset'] == dataset]
        
        if len(data) > 0:
            correct_data = data[data['Correct']]['Probability']
            incorrect_data = data[~data['Correct']]['Probability']
            
            if len(correct_data) > 0:
                ax.hist(correct_data, bins=15, alpha=0.7, color='#4CAF50', label='Correct')
            if len(incorrect_data) > 0:
                ax.hist(incorrect_data, bins=15, alpha=0.7, color='#F44336', label='Incorrect')
            
            ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'{dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('BIC Logistic Regression - Probability Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('logistic_regression_BIC/bic_lr_probability.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # LOS by BIC model
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        ax = axes[i]
        data = bic_results[bic_results['Dataset'] == dataset]
        
        if len(data) > 0:
            correct_los = data[data['Correct']]['LOS_hours']
            incorrect_los = data[~data['Correct']]['LOS_hours']
            
            vals = [correct_los.mean() if len(correct_los) > 0 else 0,
                   incorrect_los.mean() if len(incorrect_los) > 0 else 0]
            ax.bar([0, 1], vals, color=['#4CAF50', '#F44336'], width=0.6)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Correct', 'Incorrect'])
            ax.set_ylabel('Average LOS (hours)')
            ax.set_title(f'{dataset}')
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('BIC LR - Length of Stay Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('logistic_regression_BIC/bic_lr_los_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- Feature Selection Folder ---
print("   --- feature_selection_validation_graphs/ ---")

fs_results = results_df[results_df['Model_Type'].str.startswith('FS_')]
fs_models = fs_results['Model_Type'].unique()

# Create subfolders and plots for each model
for model_type in fs_models:
    model_name = model_type.replace('FS_', '')
    model_folder = f'feature_selection_validation_graphs/{model_name}'
    os.makedirs(model_folder, exist_ok=True)
    
    model_data = fs_results[fs_results['Model_Type'] == model_type]
    
    # Probability plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        ax = axes[i]
        data = model_data[model_data['Dataset'] == dataset]
        
        if len(data) > 0:
            correct_data = data[data['Correct']]['Probability']
            incorrect_data = data[~data['Correct']]['Probability']
            
            if len(correct_data) > 0:
                ax.hist(correct_data, bins=15, alpha=0.7, color='#4CAF50', label='Correct')
            if len(incorrect_data) > 0:
                ax.hist(incorrect_data, bins=15, alpha=0.7, color='#F44336', label='Incorrect')
            
            ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
        
        ax.set_xlabel('Probability')
        ax.set_ylabel('Count')
        ax.set_title(f'{dataset}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Probability Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_folder}/probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # LOS plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, dataset in enumerate(['Hematology', 'Non-Hematology']):
        ax = axes[i]
        data = model_data[model_data['Dataset'] == dataset]
        
        if len(data) > 0:
            correct_los = data[data['Correct']]['LOS_hours']
            incorrect_los = data[~data['Correct']]['LOS_hours']
            
            vals = [correct_los.mean() if len(correct_los) > 0 else 0,
                   incorrect_los.mean() if len(incorrect_los) > 0 else 0]
            ax.bar([0, 1], vals, color=['#4CAF50', '#F44336'], width=0.6)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Correct', 'Incorrect'])
        ax.set_ylabel('Average LOS (hours)')
        ax.set_title(f'{dataset}')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle(f'{model_name} - Length of Stay Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{model_folder}/los_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# STEP 6: FINAL COMPARISON
# ============================================================================
print("\n[6/6] Creating final comparison...")

# Create comprehensive comparison plot
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Plot 1: All models accuracy comparison (Hematology)
ax = axes[0, 0]
hem_data = summary_df[summary_df['Dataset'] == 'Hematology'].sort_values('Accuracy', ascending=True)
if len(hem_data) > 0:
    colors = plt.cm.tab20(np.linspace(0, 1, len(hem_data)))
    bars = ax.barh(range(len(hem_data)), hem_data['Accuracy'] * 100, color=colors)
    ax.set_yticks(range(len(hem_data)))
    ax.set_yticklabels(hem_data['Model'].values, fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Hematology - Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    for bar, acc in zip(bars, hem_data['Accuracy']):
        ax.text(acc * 100 + 1, bar.get_y() + bar.get_height()/2, f'{acc*100:.1f}%', va='center', fontsize=8)
ax.grid(True, axis='x', alpha=0.3)

# Plot 2: All models accuracy comparison (Non-Hematology)
ax = axes[0, 1]
solid_data = summary_df[summary_df['Dataset'] == 'Non-Hematology'].sort_values('Accuracy', ascending=True)
if len(solid_data) > 0:
    colors = plt.cm.tab20(np.linspace(0, 1, len(solid_data)))
    bars = ax.barh(range(len(solid_data)), solid_data['Accuracy'] * 100, color=colors)
    ax.set_yticks(range(len(solid_data)))
    ax.set_yticklabels(solid_data['Model'].values, fontsize=9)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Non-Hematology - Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    for bar, acc in zip(bars, solid_data['Accuracy']):
        ax.text(acc * 100 + 1, bar.get_y() + bar.get_height()/2, f'{acc*100:.1f}%', va='center', fontsize=8)
ax.grid(True, axis='x', alpha=0.3)

# Plot 3: Best models comparison
ax = axes[1, 0]
if len(hem_data) >= 3 and len(solid_data) >= 3:
    best_hem = hem_data.nlargest(3, 'Accuracy')
    best_solid = solid_data.nlargest(3, 'Accuracy')

    x = np.arange(3)
    width = 0.35
    bars1 = ax.bar(x - width/2, best_hem['Accuracy'].values * 100, width, label='Hematology', color='#3F51B5')
    bars2 = ax.bar(x + width/2, best_solid['Accuracy'].values * 100, width, label='Non-Hematology', color='#FF9800')

    ax.set_xticks(x)
    ax.set_xticklabels([f'#{i+1}' for i in range(3)])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Top 3 Models per Dataset', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 100)

    # Add model names as annotations
    for i, (h, s) in enumerate(zip(best_hem['Model'].values, best_solid['Model'].values)):
        ax.text(i - width/2, 5, h[:15], ha='center', va='bottom', rotation=90, fontsize=6, color='white')
        ax.text(i + width/2, 5, s[:15], ha='center', va='bottom', rotation=90, fontsize=6, color='white')

ax.grid(True, axis='y', alpha=0.3)

# Plot 4: Summary table
ax = axes[1, 1]
ax.axis('off')

if len(hem_data) >= 3 and len(solid_data) >= 3:
    summary_text = f"""
FINAL VALIDATION SUMMARY - ICU (1) Dataset
==========================================

Dataset: icu (1).xlsx
Total Patients: {len(df)}
- Hematology: {len(df_hematology)}
- Non-Hematology: {len(df_solid)}

LOS Calculation: ICU_ADMISSION_DATE - ADMISSION_DATE
Average LOS: {df['LOS_hours'].mean():.1f} hours ({df['LOS_days'].mean():.1f} days)

BEST PERFORMING MODELS:

HEMATOLOGY:
  1. {hem_data.nlargest(1, 'Accuracy')['Model'].values[0]}: {hem_data.nlargest(1, 'Accuracy')['Accuracy'].values[0]*100:.1f}%
  2. {hem_data.nlargest(2, 'Accuracy').iloc[1]['Model']}: {hem_data.nlargest(2, 'Accuracy').iloc[1]['Accuracy']*100:.1f}%
  3. {hem_data.nlargest(3, 'Accuracy').iloc[2]['Model']}: {hem_data.nlargest(3, 'Accuracy').iloc[2]['Accuracy']*100:.1f}%

NON-HEMATOLOGY:
  1. {solid_data.nlargest(1, 'Accuracy')['Model'].values[0]}: {solid_data.nlargest(1, 'Accuracy')['Accuracy'].values[0]*100:.1f}%
  2. {solid_data.nlargest(2, 'Accuracy').iloc[1]['Model']}: {solid_data.nlargest(2, 'Accuracy').iloc[1]['Accuracy']*100:.1f}%
  3. {solid_data.nlargest(3, 'Accuracy').iloc[2]['Model']}: {solid_data.nlargest(3, 'Accuracy').iloc[2]['Accuracy']*100:.1f}%

MODEL TYPES COMPARED:
- Original XGBoost (full features)
- BIC Logistic Regression (~30 features)
- Feature Selection Models (750-2250 features)
"""
else:
    summary_text = "Insufficient data for summary"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.suptitle('Complete Model Comparison - ICU (1) Validation', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('validation_graphs/FINAL_COMPARISON_ALL_MODELS.png', dpi=300, bbox_inches='tight')
plt.savefig('logistic_regression_BIC/FINAL_COMPARISON_ALL_MODELS.png', dpi=300, bbox_inches='tight')
plt.savefig('feature_selection_validation_graphs/FINAL_COMPARISON_ALL_MODELS.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# FINAL OUTPUT
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION COMPLETE!")
print("=" * 80)

print("\nResults saved to:")
print("  - complete_validation_results.csv (all predictions)")
print("  - complete_validation_summary.csv (summary metrics)")

print("\nPlots generated in:")
print("  - validation_graphs/")
print("  - logistic_regression_BIC/")
print("  - feature_selection_validation_graphs/")

print("\n--- BEST MODELS ---")
for dataset in ['Hematology', 'Non-Hematology']:
    data = summary_df[summary_df['Dataset'] == dataset]
    if len(data) > 0:
        best = data.nlargest(1, 'Accuracy').iloc[0]
        print(f"\n{dataset}:")
        print(f"  Best: {best['Model']} with {best['Accuracy']*100:.1f}% accuracy")

print("\n" + "=" * 80)
