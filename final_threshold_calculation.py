"""
Final Threshold Calculation and Plot Generation
================================================
1. Validate XGBoost and BIC LR on clean ICU + Discharged data
2. Generate all plots (ICU, Discharged, Combined)
3. Calculate optimal thresholds for risk categories

Public release note:
- This script expects private validation spreadsheets and model binaries that are not
  distributed in the public repository.
- Plot output directories referenced below are regenerated locally when the full private
  environment is available.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.metrics import (confusion_matrix, accuracy_score, roc_auc_score, 
                            precision_score, recall_score, f1_score, 
                            roc_curve, auc, precision_recall_curve)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 80)
print("FINAL THRESHOLD CALCULATION AND PLOT GENERATION")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD AND CLEAN DATA
# ============================================================================
print("\n[1/6] Loading and cleaning data...")

# Load validation datasets
icu_data = pd.read_excel('icu (1).xlsx')
discharge_data = pd.read_excel('EWS 2025 Discharge Validation.xlsx')

# Fix discharge diagnosis column
if 'Unnamed: 32' in discharge_data.columns:
    discharge_data['DIAGNOSIS'] = discharge_data['Unnamed: 32']

# Load training data for overlap detection
hematology_train = pd.read_csv('FINAL_HEMATOLOGYy.csv')
solid_train = pd.read_csv('FINAL_SOLID.csv')

# Get MRN sets
icu_mrns = set(icu_data['MRN'].dropna().astype(str))
discharge_mrns = set(discharge_data['MRN'].dropna().astype(str))
train_mrns = set(hematology_train['MRN'].dropna().astype(str)).union(
              set(solid_train['MRN'].dropna().astype(str)))

# Find and remove overlaps
overlap_icu_discharge = icu_mrns.intersection(discharge_mrns)
overlap_icu_train = icu_mrns.intersection(train_mrns)
overlap_discharge_train = discharge_mrns.intersection(train_mrns)

remove_from_icu = overlap_icu_discharge.union(overlap_icu_train)
remove_from_discharge = overlap_icu_discharge.union(overlap_discharge_train)

icu_data['MRN_str'] = icu_data['MRN'].astype(str)
icu_clean = icu_data[~icu_data['MRN_str'].isin(remove_from_icu)].copy()

discharge_data['MRN_str'] = discharge_data['MRN'].astype(str)
discharge_clean = discharge_data[~discharge_data['MRN_str'].isin(remove_from_discharge)].copy()

print(f"   ICU: {len(icu_data)} -> {len(icu_clean)} patients (cleaned)")
print(f"   Discharged: {len(discharge_data)} -> {len(discharge_clean)} patients (cleaned)")

# Split by diagnosis
def is_hematology_diagnosis(diagnosis_str):
    if pd.isna(diagnosis_str):
        return False
    diagnosis_lower = str(diagnosis_str).lower()
    hematology_keywords = ['leukemia', 'lymphoma', 'myeloma', 'hodgkin', 'aml', 'all']
    return any(keyword in diagnosis_lower for keyword in hematology_keywords)

icu_clean['IS_HEMATOLOGY'] = icu_clean['DIAGNOSIS'].apply(is_hematology_diagnosis)
icu_hem = icu_clean[icu_clean['IS_HEMATOLOGY']].copy()
icu_solid = icu_clean[~icu_clean['IS_HEMATOLOGY']].copy()

discharge_clean['IS_HEMATOLOGY'] = discharge_clean['DIAGNOSIS'].apply(is_hematology_diagnosis)
discharge_hem = discharge_clean[discharge_clean['IS_HEMATOLOGY']].copy()
discharge_solid = discharge_clean[~discharge_clean['IS_HEMATOLOGY']].copy()

print(f"\n   Hematology: {len(icu_hem)} ICU + {len(discharge_hem)} Discharged = {len(icu_hem)+len(discharge_hem)} total")
print(f"   Non-Hematology: {len(icu_solid)} ICU + {len(discharge_solid)} Discharged = {len(icu_solid)+len(discharge_solid)} total")

# ============================================================================
# STEP 2: PREPROCESSING FUNCTIONS
# ============================================================================
print("\n[2/6] Setting up preprocessing...")

def convert_string_to_list_of_floats(s):
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
        return [float(x.strip()) if x.strip() else 0.0 for x in s.split(',')]
    except:
        return []

def pad_list_to_length(lst, target_length, pad_value=0.0):
    if len(lst) >= target_length:
        return lst[:target_length]
    return lst + [pad_value] * (target_length - len(lst))

def clean_med_name(med_name):
    if pd.isna(med_name) or not med_name:
        return ''
    return str(med_name).strip().lower().replace(' ', '_')

def convert_string_to_list_of_meds(s):
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
    int_list = [vocab.get(m, 0) for m in med_list[:max_len]]
    return pad_list_to_length(int_list, max_len, pad_value=0)

def preprocess_data(df_raw, preprocessing_params, expected_features):
    df = df_raw.copy()
    
    med_vocab = preprocessing_params.get('med_vocab', {})
    max_med_lengths = preprocessing_params.get('max_med_lengths', {})
    max_length_vitals = preprocessing_params.get('max_length_vitals', 308)
    LAB_HISTORY_LENGTH = preprocessing_params.get('LAB_HISTORY_LENGTH', 4)
    
    vital_cols = preprocessing_params.get('train_vital_sign_cols', [])
    lab_cols = preprocessing_params.get('train_lab_cols', [])
    med_cols = preprocessing_params.get('train_med_cols', [])
    
    for col in vital_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_string_to_list_of_floats)
    for col in lab_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_string_to_list_of_floats)
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].apply(convert_string_to_list_of_meds)
    
    features_list = []
    for idx, row in df.iterrows():
        patient_features = []
        
        for col in vital_cols:
            if col in df.columns:
                vals = row[col] if isinstance(row[col], list) else []
                patient_features.extend(pad_list_to_length(vals, max_length_vitals, 0.0))
        
        for col in lab_cols:
            if col in df.columns:
                vals = row[col] if isinstance(row[col], list) else []
                last_n = vals[-LAB_HISTORY_LENGTH:] if len(vals) >= LAB_HISTORY_LENGTH else vals
                patient_features.extend(pad_list_to_length(last_n, LAB_HISTORY_LENGTH, 0.0))
        
        for col in med_cols:
            if col in df.columns:
                meds = row[col] if isinstance(row[col], list) else []
                max_len = max_med_lengths.get(col, 100)
                vocab = med_vocab.get(col, {})
                padded_ints = transform_meds_to_padded_ints(meds, vocab, max_len)
                patient_features.extend(padded_ints)
                patient_features.extend([len(meds), padded_ints[0] if padded_ints else 0, 
                                        padded_ints[-1] if padded_ints else 0])
        
        features_list.append(patient_features)
    
    X = np.array(features_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    imputer = preprocessing_params.get('imputer_tabular') or preprocessing_params.get('imputer')
    scaler = preprocessing_params.get('scaler_tabular') or preprocessing_params.get('scaler')
    
    if imputer is not None:
        imp_feat = imputer.n_features_in_
        if X.shape[1] < imp_feat:
            X = np.hstack([X, np.zeros((X.shape[0], imp_feat - X.shape[1]))])
        elif X.shape[1] > imp_feat:
            X = X[:, :imp_feat]
        X = imputer.transform(X)
    
    if scaler is not None:
        scl_feat = scaler.n_features_in_
        if X.shape[1] < scl_feat:
            X = np.hstack([X, np.zeros((X.shape[0], scl_feat - X.shape[1]))])
        elif X.shape[1] > scl_feat:
            X = X[:, :scl_feat]
        X = scaler.transform(X)
    
    if X.shape[1] < expected_features:
        X = np.hstack([X, np.zeros((X.shape[0], expected_features - X.shape[1]))])
    elif X.shape[1] > expected_features:
        X = X[:, :expected_features]
    
    return X

# ============================================================================
# STEP 3: VALIDATE MODELS
# ============================================================================
print("\n[3/6] Validating models on clean data...")

# Models to validate
MODELS = {
    'Hematology': [
        ('Original_XGBoost', 'Website/models/xgboost_hematology.pkl', 
         'comprehensive_hematology_models/preprocessing_objects.pkl', 2527),
        ('BIC_LogisticRegression', 'logistic_regression_BIC/bic_lr_model_hematology.pkl', 
         'comprehensive_hematology_models/preprocessing_objects.pkl', 2527),
    ],
    'Non-Hematology': [
        ('Original_XGBoost', 'Website/models/xgboost.pkl', 
         'comprehensive_solid_models/preprocessing_objects.pkl', 2310),
        ('BIC_LogisticRegression', 'logistic_regression_BIC/bic_lr_model_solid.pkl', 
         'comprehensive_solid_models/preprocessing_objects.pkl', 2310),
    ]
}

all_results = []

for dataset_type, models in MODELS.items():
    if dataset_type == 'Hematology':
        icu_subset = icu_hem
        discharge_subset = discharge_hem
    else:
        icu_subset = icu_solid
        discharge_subset = discharge_solid
    
    for model_name, model_path, preproc_path, expected_features in models:
        print(f"\n   {model_name} - {dataset_type}:")
        
        try:
            # Load model and preprocessing
            model_data = joblib.load(model_path)
            model = model_data['model'] if isinstance(model_data, dict) else model_data
            preprocessing_params = joblib.load(preproc_path)
            
            # Process ICU patients
            if len(icu_subset) > 0:
                X_icu = preprocess_data(icu_subset.copy(), preprocessing_params, expected_features)
                y_pred_icu = model.predict(X_icu)
                y_proba_icu = model.predict_proba(X_icu)[:, 1]
                
                for i in range(len(icu_subset)):
                    all_results.append({
                        'Model': model_name, 'Dataset': dataset_type, 'Patient_Type': 'ICU',
                        'True_Label': 1, 'Predicted': y_pred_icu[i], 'Probability': y_proba_icu[i]
                    })
                
                sens = (y_pred_icu == 1).mean()
                print(f"      ICU ({len(icu_subset)}): Sensitivity = {sens*100:.1f}%")
            
            # Process Discharged patients
            if len(discharge_subset) > 0:
                X_dis = preprocess_data(discharge_subset.copy(), preprocessing_params, expected_features)
                y_pred_dis = model.predict(X_dis)
                y_proba_dis = model.predict_proba(X_dis)[:, 1]
                
                for i in range(len(discharge_subset)):
                    all_results.append({
                        'Model': model_name, 'Dataset': dataset_type, 'Patient_Type': 'Discharged',
                        'True_Label': 0, 'Predicted': y_pred_dis[i], 'Probability': y_proba_dis[i]
                    })
                
                spec = (y_pred_dis == 0).mean()
                print(f"      Discharged ({len(discharge_subset)}): Specificity = {spec*100:.1f}%")
                
        except Exception as e:
            print(f"      ERROR: {str(e)[:60]}")

results_df = pd.DataFrame(all_results)
results_df.to_csv('final_validation_results.csv', index=False)

# ============================================================================
# STEP 4: GENERATE ALL PLOTS
# ============================================================================
print("\n[4/6] Generating all plots...")

# Create folder structure
folders = {
    'main': 'validation_graphs',
    'bic': 'logistic_regression_BIC',
}

for folder in folders.values():
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'{folder}/icu_admitted', exist_ok=True)
    os.makedirs(f'{folder}/discharged', exist_ok=True)
    os.makedirs(f'{folder}/combined_summary', exist_ok=True)

colors = {'Original_XGBoost': '#2196F3', 'BIC_LogisticRegression': '#FF9800'}

# --- PLOT 1: Probability Distribution (ICU, Discharged, Combined) ---
for dataset_type in ['Hematology', 'Non-Hematology']:
    for model_name in ['Original_XGBoost', 'BIC_LogisticRegression']:
        data = results_df[(results_df['Model'] == model_name) & (results_df['Dataset'] == dataset_type)]
        
        if len(data) == 0:
            continue
        
        icu_probs = data[data['Patient_Type'] == 'ICU']['Probability']
        dis_probs = data[data['Patient_Type'] == 'Discharged']['Probability']
        
        # ICU only plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(icu_probs, bins=20, alpha=0.7, color='red', edgecolor='black')
        ax.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
        ax.set_xlabel('Predicted Probability (ICU Risk)', fontsize=12)
        ax.set_ylabel('Number of Patients', fontsize=12)
        ax.set_title(f'{model_name} - {dataset_type}\nICU Patients Probability Distribution (n={len(icu_probs)})', fontsize=14)
        ax.set_xlim(0, 1)
        ax.legend()
        for folder in folders.values():
            plt.savefig(f'{folder}/icu_admitted/prob_dist_{model_name}_{dataset_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Discharged only plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(dis_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(x=0.5, color='black', linestyle='--', label='Default Threshold (0.5)')
        ax.set_xlabel('Predicted Probability (ICU Risk)', fontsize=12)
        ax.set_ylabel('Number of Patients', fontsize=12)
        ax.set_title(f'{model_name} - {dataset_type}\nDischarged Patients Probability Distribution (n={len(dis_probs)})', fontsize=14)
        ax.set_xlim(0, 1)
        ax.legend()
        for folder in folders.values():
            plt.savefig(f'{folder}/discharged/prob_dist_{model_name}_{dataset_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Combined plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(icu_probs, bins=20, alpha=0.6, color='red', label=f'ICU (n={len(icu_probs)})', edgecolor='darkred')
        ax.hist(dis_probs, bins=20, alpha=0.6, color='green', label=f'Discharged (n={len(dis_probs)})', edgecolor='darkgreen')
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Default Threshold (0.5)')
        ax.set_xlabel('Predicted Probability (ICU Risk)', fontsize=12)
        ax.set_ylabel('Number of Patients', fontsize=12)
        ax.set_title(f'{model_name} - {dataset_type}\nProbability Distribution: ICU vs Discharged', fontsize=14)
        ax.set_xlim(0, 1)
        ax.legend()
        for folder in folders.values():
            plt.savefig(f'{folder}/combined_summary/prob_dist_combined_{model_name}_{dataset_type}.png', dpi=150, bbox_inches='tight')
        plt.close()

# --- PLOT 2: ROC Curves ---
for dataset_type in ['Hematology', 'Non-Hematology']:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model_name in ['Original_XGBoost', 'BIC_LogisticRegression']:
        data = results_df[(results_df['Model'] == model_name) & (results_df['Dataset'] == dataset_type)]
        if len(data) == 0:
            continue
        
        y_true = data['True_Label']
        y_proba = data['Probability']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[model_name], linewidth=2, 
               label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(f'{dataset_type} - ROC Curves (Clean Data)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    for folder in folders.values():
        plt.savefig(f'{folder}/combined_summary/roc_curves_{dataset_type}.png', dpi=150, bbox_inches='tight')
    plt.close()

# --- PLOT 3: Confusion Matrices ---
for dataset_type in ['Hematology', 'Non-Hematology']:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, model_name in enumerate(['Original_XGBoost', 'BIC_LogisticRegression']):
        data = results_df[(results_df['Model'] == model_name) & (results_df['Dataset'] == dataset_type)]
        if len(data) == 0:
            continue
        
        y_true = data['True_Label']
        y_pred = data['Predicted']
        
        cm = confusion_matrix(y_true, y_pred)
        
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Discharged', 'ICU'],
                   yticklabels=['Discharged', 'ICU'])
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=12)
    
    plt.suptitle(f'{dataset_type} - Confusion Matrices (Clean Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    for folder in folders.values():
        plt.savefig(f'{folder}/combined_summary/confusion_matrices_{dataset_type}.png', dpi=150, bbox_inches='tight')
    plt.close()

# --- PLOT 4: Metrics Comparison Bar Chart ---
metrics_summary = []
for model_name in ['Original_XGBoost', 'BIC_LogisticRegression']:
    for dataset_type in ['Hematology', 'Non-Hematology']:
        data = results_df[(results_df['Model'] == model_name) & (results_df['Dataset'] == dataset_type)]
        if len(data) == 0:
            continue
        
        y_true = data['True_Label'].values
        y_pred = data['Predicted'].values
        y_proba = data['Probability'].values
        
        TP = ((y_true == 1) & (y_pred == 1)).sum()
        TN = ((y_true == 0) & (y_pred == 0)).sum()
        FP = ((y_true == 0) & (y_pred == 1)).sum()
        FN = ((y_true == 1) & (y_pred == 0)).sum()
        
        metrics_summary.append({
            'Model': model_name,
            'Dataset': dataset_type,
            'Sensitivity': TP / (TP + FN) if (TP + FN) > 0 else 0,
            'Specificity': TN / (TN + FP) if (TN + FP) > 0 else 0,
            'Accuracy': (TP + TN) / len(y_true),
            'F1_Score': f1_score(y_true, y_pred),
            'AUC_ROC': roc_auc_score(y_true, y_proba),
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
        })

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv('final_metrics_summary.csv', index=False)

# Metrics comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
metrics_to_plot = ['Sensitivity', 'Specificity', 'Accuracy', 'F1_Score', 'AUC_ROC']

for idx, dataset_type in enumerate(['Hematology', 'Non-Hematology']):
    ax = axes[idx]
    data = metrics_df[metrics_df['Dataset'] == dataset_type]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, model_name in enumerate(['Original_XGBoost', 'BIC_LogisticRegression']):
        model_data = data[data['Model'] == model_name]
        if len(model_data) == 0:
            continue
        values = [model_data[m].values[0] for m in metrics_to_plot]
        offset = width * (i - 0.5)
        bars = ax.bar(x + offset, [v * 100 for v in values], width, 
                     label=model_name, color=colors[model_name])
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'{dataset_type}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend()

plt.suptitle('Model Performance Comparison (Clean Data - No Leakage)', fontsize=14, fontweight='bold')
plt.tight_layout()
for folder in folders.values():
    plt.savefig(f'{folder}/combined_summary/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("   Generated all comparison plots")

# ============================================================================
# STEP 5: CALCULATE OPTIMAL THRESHOLDS
# ============================================================================
print("\n[5/6] Calculating optimal thresholds...")

threshold_results = []

for dataset_type in ['Hematology', 'Non-Hematology']:
    print(f"\n   {dataset_type}:")
    
    # Use Original XGBoost (best model)
    data = results_df[(results_df['Model'] == 'Original_XGBoost') & (results_df['Dataset'] == dataset_type)]
    
    if len(data) == 0:
        continue
    
    icu_probs = data[data['Patient_Type'] == 'ICU']['Probability'].values
    dis_probs = data[data['Patient_Type'] == 'Discharged']['Probability'].values
    
    # Method 1: Youden's J statistic (maximize Sensitivity + Specificity - 1)
    best_threshold_youden = 0.5
    best_j = 0
    
    for threshold in np.arange(0.05, 0.95, 0.01):
        sens = (icu_probs >= threshold).mean()
        spec = (dis_probs < threshold).mean()
        j = sens + spec - 1
        
        if j > best_j:
            best_j = j
            best_threshold_youden = threshold
    
    # Method 2: High sensitivity threshold (>= 95% sensitivity)
    for threshold in np.arange(0.95, 0.05, -0.01):
        sens = (icu_probs >= threshold).mean()
        if sens >= 0.95:
            best_threshold_high_sens = threshold
            break
    else:
        best_threshold_high_sens = 0.3
    
    # Method 3: Risk tiers based on probability distribution
    # LOW: very unlikely to be ICU (high confidence discharge)
    # MEDIUM: uncertain, needs monitoring
    # HIGH: likely ICU, immediate attention
    
    # Find threshold where most discharged patients fall below
    low_threshold = np.percentile(dis_probs, 75)  # 75% of discharged below this
    high_threshold = np.percentile(icu_probs, 25)  # 75% of ICU above this
    
    # Ensure low < high
    if low_threshold >= high_threshold:
        low_threshold = 0.30
        high_threshold = 0.60
    
    # Calculate metrics at Youden threshold
    sens_youden = (icu_probs >= best_threshold_youden).mean()
    spec_youden = (dis_probs < best_threshold_youden).mean()
    
    print(f"      Youden's J Optimal Threshold: {best_threshold_youden:.2f}")
    print(f"         Sensitivity: {sens_youden*100:.1f}%")
    print(f"         Specificity: {spec_youden*100:.1f}%")
    print(f"      High Sensitivity Threshold (>=95%): {best_threshold_high_sens:.2f}")
    print(f"      Risk Tier Thresholds:")
    print(f"         LOW RISK: 0 - {low_threshold:.2f}")
    print(f"         MEDIUM RISK: {low_threshold:.2f} - {high_threshold:.2f}")
    print(f"         HIGH RISK: {high_threshold:.2f} - 1.0")
    
    threshold_results.append({
        'Dataset': dataset_type,
        'Youden_Threshold': best_threshold_youden,
        'Youden_Sensitivity': sens_youden,
        'Youden_Specificity': spec_youden,
        'High_Sens_Threshold': best_threshold_high_sens,
        'Low_Risk_Upper': low_threshold,
        'High_Risk_Lower': high_threshold
    })

threshold_df = pd.DataFrame(threshold_results)
threshold_df.to_csv('optimal_thresholds.csv', index=False)

# ============================================================================
# STEP 6: THRESHOLD VISUALIZATION
# ============================================================================
print("\n[6/6] Generating threshold visualization...")

for dataset_type in ['Hematology', 'Non-Hematology']:
    data = results_df[(results_df['Model'] == 'Original_XGBoost') & (results_df['Dataset'] == dataset_type)]
    
    if len(data) == 0:
        continue
    
    icu_probs = data[data['Patient_Type'] == 'ICU']['Probability'].values
    dis_probs = data[data['Patient_Type'] == 'Discharged']['Probability'].values
    
    thresholds = threshold_df[threshold_df['Dataset'] == dataset_type].iloc[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Probability distribution with thresholds
    ax = axes[0, 0]
    ax.hist(icu_probs, bins=25, alpha=0.6, color='red', label=f'ICU (n={len(icu_probs)})', density=True)
    ax.hist(dis_probs, bins=25, alpha=0.6, color='green', label=f'Discharged (n={len(dis_probs)})', density=True)
    ax.axvline(x=thresholds['Youden_Threshold'], color='blue', linestyle='--', linewidth=2, 
               label=f"Youden's Optimal ({thresholds['Youden_Threshold']:.2f})")
    ax.axvline(x=thresholds['Low_Risk_Upper'], color='orange', linestyle=':', linewidth=2,
               label=f"Low/Medium ({thresholds['Low_Risk_Upper']:.2f})")
    ax.axvline(x=thresholds['High_Risk_Lower'], color='purple', linestyle=':', linewidth=2,
               label=f"Medium/High ({thresholds['High_Risk_Lower']:.2f})")
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Probability Distribution with Thresholds', fontsize=12)
    ax.set_xlim(0, 1)
    ax.legend()
    
    # Plot 2: Sensitivity/Specificity vs Threshold
    ax = axes[0, 1]
    thresholds_range = np.arange(0.05, 0.95, 0.01)
    sensitivities = [(icu_probs >= t).mean() for t in thresholds_range]
    specificities = [(dis_probs < t).mean() for t in thresholds_range]
    
    ax.plot(thresholds_range, sensitivities, 'r-', linewidth=2, label='Sensitivity')
    ax.plot(thresholds_range, specificities, 'g-', linewidth=2, label='Specificity')
    ax.axvline(x=thresholds['Youden_Threshold'], color='blue', linestyle='--', alpha=0.7,
               label=f"Youden's Optimal ({thresholds['Youden_Threshold']:.2f})")
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Rate', fontsize=11)
    ax.set_title('Sensitivity & Specificity vs Threshold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Risk Tiers Visualization
    ax = axes[1, 0]
    tier_colors = ['#4CAF50', '#FFC107', '#F44336']  # Green, Yellow, Red
    
    # Count patients in each tier
    low_icu = (icu_probs < thresholds['Low_Risk_Upper']).sum()
    med_icu = ((icu_probs >= thresholds['Low_Risk_Upper']) & (icu_probs < thresholds['High_Risk_Lower'])).sum()
    high_icu = (icu_probs >= thresholds['High_Risk_Lower']).sum()
    
    low_dis = (dis_probs < thresholds['Low_Risk_Upper']).sum()
    med_dis = ((dis_probs >= thresholds['Low_Risk_Upper']) & (dis_probs < thresholds['High_Risk_Lower'])).sum()
    high_dis = (dis_probs >= thresholds['High_Risk_Lower']).sum()
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, [low_icu, med_icu, high_icu], width, label='ICU', color='red', alpha=0.7)
    ax.bar(x + width/2, [low_dis, med_dis, high_dis], width, label='Discharged', color='green', alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels(['LOW RISK\n(Safe)', 'MEDIUM RISK\n(Monitor)', 'HIGH RISK\n(Urgent)'])
    ax.set_ylabel('Number of Patients', fontsize=11)
    ax.set_title('Patient Distribution by Risk Tier', fontsize=12)
    ax.legend()
    
    # Add annotations
    for i, (icu_count, dis_count) in enumerate([(low_icu, low_dis), (med_icu, med_dis), (high_icu, high_dis)]):
        ax.text(i - width/2, icu_count + 1, str(icu_count), ha='center', fontsize=10)
        ax.text(i + width/2, dis_count + 1, str(dis_count), ha='center', fontsize=10)
    
    # Plot 4: Summary Table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    RECOMMENDED THRESHOLDS FOR {dataset_type.upper()}
    ================================================
    
    OPTIMAL SINGLE THRESHOLD (Youden's J):
        Threshold: {thresholds['Youden_Threshold']:.2f}
        Sensitivity: {thresholds['Youden_Sensitivity']*100:.1f}%
        Specificity: {thresholds['Youden_Specificity']*100:.1f}%
    
    RISK TIER THRESHOLDS:
        LOW RISK:    0.00 - {thresholds['Low_Risk_Upper']:.2f}
        MEDIUM RISK: {thresholds['Low_Risk_Upper']:.2f} - {thresholds['High_Risk_Lower']:.2f}
        HIGH RISK:   {thresholds['High_Risk_Lower']:.2f} - 1.00
    
    RISK TIER PATIENT COUNTS:
        LOW RISK:    {low_dis} discharged, {low_icu} ICU (FN)
        MEDIUM RISK: {med_dis} discharged, {med_icu} ICU
        HIGH RISK:   {high_dis} discharged (FP), {high_icu} ICU
    
    INTERPRETATION:
        - ICU patients in LOW RISK = FALSE NEGATIVES (missed)
        - Discharged in HIGH RISK = FALSE POSITIVES (over-alert)
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle(f'{dataset_type} - Threshold Analysis (Original XGBoost)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    for folder in folders.values():
        plt.savefig(f'{folder}/combined_summary/threshold_analysis_{dataset_type}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

print("\n" + "-" * 80)
print("MODEL PERFORMANCE (Clean Data - No Leakage)")
print("-" * 80)
print(f"{'Model':<25} {'Dataset':<15} {'Sens':<8} {'Spec':<8} {'FP':<6} {'FN':<6} {'AUC':<8}")
print("-" * 80)
for _, row in metrics_df.iterrows():
    print(f"{row['Model']:<25} {row['Dataset']:<15} {row['Sensitivity']*100:>5.1f}%  {row['Specificity']*100:>5.1f}%  "
          f"{row['FP']:>4.0f}  {row['FN']:>4.0f}  {row['AUC_ROC']:.3f}")
print("-" * 80)

print("\n" + "-" * 80)
print("RECOMMENDED THRESHOLDS")
print("-" * 80)
for _, row in threshold_df.iterrows():
    print(f"\n{row['Dataset']}:")
    print(f"   Optimal Single Threshold: {row['Youden_Threshold']:.2f}")
    print(f"   Risk Tiers:")
    print(f"      LOW RISK:    0.00 - {row['Low_Risk_Upper']:.2f}")
    print(f"      MEDIUM RISK: {row['Low_Risk_Upper']:.2f} - {row['High_Risk_Lower']:.2f}")
    print(f"      HIGH RISK:   {row['High_Risk_Lower']:.2f} - 1.00")

print("\n" + "=" * 80)
print("FILES GENERATED:")
print("   - final_validation_results.csv")
print("   - final_metrics_summary.csv")
print("   - optimal_thresholds.csv")
print("   - All plots in validation_graphs/ and logistic_regression_BIC/")
print("=" * 80)

