import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, accuracy_score, confusion_matrix,
    roc_curve, auc, precision_recall_curve, make_scorer
)
from sklearn.calibration import calibration_curve
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
# from dcurves import dca, plot_graphs # dca for calculation, plot_graphs for plotting multiple  # Commented out due to Python 3.13 compatibility issues

import warnings
import itertools
import re
from collections import Counter
from tabulate import tabulate

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Suppress specific XGBoost warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Suppress XGBoost device warnings specifically
import logging
logging.getLogger('xgboost').setLevel(logging.ERROR)

# Additional XGBoost warning suppression
import xgboost as xgb
xgb.set_config(verbosity=0)

# Suppress sklearn warnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Create comprehensive solid models plots directory
if not os.path.exists('comprehensive_solid_models_plots'):
    os.makedirs('comprehensive_solid_models_plots')

# --- GPU Configuration ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
xgb_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Configuration: Define Column Names ---
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
all_feature_cols_original = train_vital_sign_cols + train_lab_cols + train_med_cols

# --- Constants for Special Tokens ---
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
EMPTY_SEQ_ID = 0

# --- Helper Functions ---
def convert_string_to_list_of_floats(x):
    if isinstance(x, str):
        cleaned_x = ''.join(c if c.isdigit() or c == '.' or c == ',' or c == '-' or c.isspace() else ' ' for c in x)
        values = [val.strip() for val in cleaned_x.split(',') if val.strip()]
        float_values = []
        for val in values:
            if val and val not in ['.', '-']:
                try: float_values.append(float(val))
                except ValueError: continue
        return float_values
    elif isinstance(x, (int, float)): return [float(x)]
    return []

def pad_list_to_length(lst, length, pad_value=PAD_TOKEN_ID):
    lst = lst[:length]
    return lst + [pad_value] * (length - len(lst))

def clean_med_name(name):
    return re.sub(r'\s+', ' ', name).strip().lower()

def convert_string_to_list_of_meds(x):
    if pd.isna(x) or not isinstance(x, str) or x.strip() == '': return []
    meds = [clean_med_name(med) for med in x.split(',') if clean_med_name(med)]
    return meds

def transform_meds_to_padded_ints(med_list, vocab, max_len):
    if not isinstance(med_list, list): med_list = []
    unk_token_int = vocab.get("UNK", UNK_TOKEN_ID)
    int_sequence = [vocab.get(med, unk_token_int) for med in med_list]
    int_sequence = int_sequence[:max_len]
    padded_sequence = int_sequence + [PAD_TOKEN_ID] * (max_len - len(int_sequence))
    return padded_sequence

# --- Data Loading ---
INPUT_FILE = 'FINAL_SOLID.csv' # Using the solid tumor dataset file
print(f"Loading training data ({INPUT_FILE})...")
try:
    data = pd.read_csv(INPUT_FILE)
    print(f"Successfully loaded {INPUT_FILE} with {len(data)} rows.")
except FileNotFoundError:
    print(f"Error: {INPUT_FILE} not found. Please ensure the file is in the correct directory.")
    exit()

# --- Initial Data Conversion ---
print("Converting vital sign, lab, and medication columns to lists...")
for col in train_vital_sign_cols + train_lab_cols:
    if col in data.columns: data[col] = data[col].apply(convert_string_to_list_of_floats)
    else:
        print(f"Warning: Column '{col}' not found during initial conversion. Adding empty lists.")
        data[col] = pd.Series([[] for _ in range(len(data))], index=data.index)
for col in train_med_cols:
    if col in data.columns: data[col] = data[col].apply(convert_string_to_list_of_meds)
    else:
        print(f"Warning: Column '{col}' not found during initial conversion. Adding empty lists.")
        data[col] = pd.Series([[] for _ in range(len(data))], index=data.index)

# --- Prepare Target Variable (y) ---
print("Preparing target variable...")
if train_target_col not in data.columns:
     print(f"Error: Target column '{train_target_col}' not found.")
     exit()
data['EVENT'] = data[train_target_col].apply(lambda x: 1 if str(x).strip().upper() == 'YES' else 0)
y = data['EVENT']

print(f"Target variable distribution:")
print(f"ICU Risk (1): {(y == 1).sum()}")
print(f"No ICU Risk (0): {(y == 0).sum()}")

# --- Train/Test Split (Raw Data) ---
print("Splitting data into training and testing sets...")
data_train, data_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {len(data_train)}, Test set size: {len(data_test)}")
if len(data_train) == 0 or len(data_test) == 0: 
    print("Error: Training or testing set is empty."); exit()

# --- Preprocessing Fitting (on Training Data Only) ---
print("Fitting medication vocabulary and determining max lengths (on training data)...")
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
print(f"Total unique medications in training vocab: {len(med_vocab) - 2}")

max_length_vitals = 0
for col in train_vital_sign_cols:
    if col in data_train.columns:
        lengths = data_train[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
        max_col_length = lengths.max() if not lengths.empty else 0
        if max_col_length > max_length_vitals: max_length_vitals = int(max_col_length)
print(f"Determined max vital sign padding length (from train): {max_length_vitals}")

LAB_HISTORY_LENGTH = 4
print(f"Using fixed lab history length: {LAB_HISTORY_LENGTH}")

# --- Feature Transformation for Tabular Models ---
def process_features_tabular(df, is_train_phase=True):
    print(f"Processing tabular features for {'training' if is_train_phase else 'testing'} data...")
    processed_features_list = []
    # Vital Signs
    for col in train_vital_sign_cols:
        if col in df.columns:
            padded_col = df[col].apply(lambda x: pad_list_to_length(x if isinstance(x, list) else [], max_length_vitals, pad_value=0.0))
            col_array = np.array(padded_col.tolist(), dtype=float)
            expected_shape=(len(df), max_length_vitals if max_length_vitals > 0 else 0)
            if col_array.shape != expected_shape : col_array = np.zeros(expected_shape)
            processed_features_list.append(col_array)
        else: processed_features_list.append(np.zeros((len(df), max_length_vitals if max_length_vitals > 0 else 0)))
    # Lab Results
    for col in train_lab_cols:
        if col in df.columns:
            proc_col = df[col].apply(lambda x: [0.0]*(LAB_HISTORY_LENGTH - len(x)) + x[-LAB_HISTORY_LENGTH:] if isinstance(x, list) else [0.0]*LAB_HISTORY_LENGTH)
            col_array = np.array(proc_col.tolist(), dtype=float)
            expected_shape=(len(df), LAB_HISTORY_LENGTH)
            if col_array.shape != expected_shape : col_array = np.zeros(expected_shape)
            processed_features_list.append(col_array)
        else: processed_features_list.append(np.zeros((len(df), LAB_HISTORY_LENGTH)))
    # Medications
    for col in train_med_cols:
        max_len = max_med_lengths.get(col, 0)
        med_lists = df[col].apply(lambda x: x if isinstance(x, list) else []) if col in df else pd.Series([[] for _ in range(len(df))], index=df.index)
        # Padded Integer Sequence
        if col in df.columns:
            transformed_col = med_lists.apply(lambda x: transform_meds_to_padded_ints(x, med_vocab, max_len))
            col_array_seq = np.array(transformed_col.tolist(), dtype=int)
            expected_shape = (len(df), max_len if max_len > 0 else 0)
            if col_array_seq.shape != expected_shape: col_array_seq = np.zeros(expected_shape, dtype=int)
            processed_features_list.append(col_array_seq)
        else: processed_features_list.append(np.zeros((len(df), max_len if max_len > 0 else 0), dtype=int))
        # Aggregate Features
        lengths = med_lists.apply(len).values.reshape(-1, 1)
        processed_features_list.append(lengths.astype(float))
        processed_features_list.append((lengths > 0).astype(float))
        processed_features_list.append(med_lists.apply(lambda x: med_vocab.get(x[0], UNK_TOKEN_ID) if len(x)>0 else EMPTY_SEQ_ID).values.reshape(-1, 1).astype(float))
        processed_features_list.append(med_lists.apply(lambda x: med_vocab.get(x[-1], UNK_TOKEN_ID) if len(x)>0 else EMPTY_SEQ_ID).values.reshape(-1, 1).astype(float))
        processed_features_list.append(med_lists.apply(lambda x: len(set(x))).values.reshape(-1, 1).astype(float))

    valid_features = [arr for arr in processed_features_list if arr.ndim == 2 and arr.shape[0] == len(df) and arr.shape[1] > 0]
    if not valid_features: return np.empty((len(df), 0))
    X_processed = np.concatenate(valid_features, axis=1)
    print(f"  Final processed tabular feature matrix shape: {X_processed.shape}")
    return X_processed

X_train_processed_tabular = process_features_tabular(data_train, is_train_phase=True)
X_test_processed_tabular = process_features_tabular(data_test, is_train_phase=False)

# --- Imputation and Scaling for Tabular Models ---
print("Applying imputation and scaling for tabular models...")
imputer_tabular = SimpleImputer(strategy='median')
X_train_imputed_tabular = imputer_tabular.fit_transform(X_train_processed_tabular)
X_test_imputed_tabular = imputer_tabular.transform(X_test_processed_tabular)

scaler_tabular = MinMaxScaler()
X_train_final_tabular = scaler_tabular.fit_transform(X_train_imputed_tabular)
X_test_final_tabular = scaler_tabular.transform(X_test_imputed_tabular)
print(f"Final training features shape for tabular models: {X_train_final_tabular.shape}")

# --- Feature Processing for Bi-LSTM ---
def create_bilstm_input(df, vital_cols, lab_cols, med_cols,
                        max_vitals_len, lab_hist_len,
                        med_vocab_dict, max_med_len_dict, is_train_phase=True):
    print(f"Processing Bi-LSTM features for {'training' if is_train_phase else 'testing'} data...")
    N = len(df)
    # 1. Vital Signs (N, max_vitals_len, num_vital_cols)
    vital_sequences = []
    for col in vital_cols:
        padded_seqs = df[col].apply(lambda x: pad_list_to_length(x if isinstance(x,list) else [], max_vitals_len, 0.0))
        vital_sequences.append(np.array(padded_seqs.tolist()))
    stacked_vitals = np.stack(vital_sequences, axis=-1) if vital_sequences else np.zeros((N, max_vitals_len, 0))

    # 2. Lab Values (N, max_vitals_len, num_lab_cols) - align sequence length with vitals
    lab_sequences_aligned = []
    for col in lab_cols:
        processed_labs = df[col].apply(lambda x: ([0.0]*(lab_hist_len - len(x)) + x[-lab_hist_len:]) if isinstance(x,list) else [0.0]*lab_hist_len)
        aligned_labs = []
        for seq in processed_labs:
            full_seq = []
            for i in range(max_vitals_len):
                full_seq.append(seq[min(i, lab_hist_len-1)] if lab_hist_len > 0 else 0.0)
            aligned_labs.append(full_seq)
        lab_sequences_aligned.append(np.array(aligned_labs))
    stacked_labs = np.stack(lab_sequences_aligned, axis=-1) if lab_sequences_aligned else np.zeros((N,max_vitals_len,0))

    # 3. Medication Aggregate Features (N, max_vitals_len, num_med_cols * 5_aggregates) - repeat across time
    med_aggregate_features_patient = []
    for col in med_cols:
        med_lists = df[col].apply(lambda x: x if isinstance(x, list) else [])
        lengths = med_lists.apply(len).values.reshape(-1, 1)
        med_aggregate_features_patient.append(lengths)
        med_aggregate_features_patient.append((lengths > 0).astype(int))
        med_aggregate_features_patient.append(med_lists.apply(lambda x: med_vocab_dict.get(x[0], UNK_TOKEN_ID) if len(x)>0 else EMPTY_SEQ_ID).values.reshape(-1, 1))
        med_aggregate_features_patient.append(med_lists.apply(lambda x: med_vocab_dict.get(x[-1], UNK_TOKEN_ID) if len(x)>0 else EMPTY_SEQ_ID).values.reshape(-1, 1))
        med_aggregate_features_patient.append(med_lists.apply(lambda x: len(set(x))).values.reshape(-1, 1))
    
    if med_aggregate_features_patient:
        stacked_med_aggregates_static = np.concatenate(med_aggregate_features_patient, axis=1).astype(float)
        stacked_med_aggregates_temporal = np.repeat(stacked_med_aggregates_static[:, np.newaxis, :], max_vitals_len, axis=1)
    else:
        stacked_med_aggregates_temporal = np.zeros((N, max_vitals_len, 0))

    # Concatenate all feature types
    final_bilstm_input = np.concatenate([stacked_vitals, stacked_labs, stacked_med_aggregates_temporal], axis=2)
    print(f"  Final Bi-LSTM input feature matrix shape: {final_bilstm_input.shape}")
    return final_bilstm_input

if max_length_vitals == 0:
    print("Warning: max_length_vitals is 0. Bi-LSTM sequence length will be 1.")
    X_train_bilstm_3d = np.zeros((len(data_train), 1, 1))
    X_test_bilstm_3d = np.zeros((len(data_test), 1, 1))
else:
    X_train_bilstm_3d = create_bilstm_input(data_train, train_vital_sign_cols, train_lab_cols, train_med_cols,
                                        max_length_vitals, LAB_HISTORY_LENGTH, med_vocab, max_med_lengths)
    X_test_bilstm_3d = create_bilstm_input(data_test, train_vital_sign_cols, train_lab_cols, train_med_cols,
                                       max_length_vitals, LAB_HISTORY_LENGTH, med_vocab, max_med_lengths, is_train_phase=False)

# Scale Bi-LSTM 3D data
num_samples_train, seq_len_train, num_features_train = X_train_bilstm_3d.shape
print(f"X_train_bilstm_3d BEFORE scaling - NaNs: {np.isnan(X_train_bilstm_3d).sum()}, Infs: {np.isinf(X_train_bilstm_3d).sum()}")
print(f"X_test_bilstm_3d BEFORE scaling - NaNs: {np.isnan(X_test_bilstm_3d).sum()}, Infs: {np.isinf(X_test_bilstm_3d).sum()}")

if num_features_train == 0:
    print("ERROR: BiLSTM training data has 0 features after processing. Cannot scale.")
    X_train_bilstm_3d_scaled = X_train_bilstm_3d
    X_test_bilstm_3d_scaled = X_test_bilstm_3d
else:
    scaler_bilstm = MinMaxScaler()
    X_train_bilstm_3d_reshaped = X_train_bilstm_3d.reshape(-1, num_features_train)
    
    if np.isnan(X_train_bilstm_3d_reshaped).any() or np.isinf(X_train_bilstm_3d_reshaped).any():
        print("Warning: NaNs/Infs found in BiLSTM training data BEFORE scaling. Applying median imputation...")
        imputer_bilstm_temp = SimpleImputer(strategy='median')
        X_train_bilstm_3d_reshaped = imputer_bilstm_temp.fit_transform(X_train_bilstm_3d_reshaped)

    X_train_bilstm_3d_scaled = scaler_bilstm.fit_transform(X_train_bilstm_3d_reshaped).reshape(num_samples_train, seq_len_train, num_features_train)

    num_samples_test, seq_len_test, num_features_test = X_test_bilstm_3d.shape
    if num_features_test == 0:
        print("ERROR: BiLSTM test data has 0 features after processing. Cannot scale.")
        X_test_bilstm_3d_scaled = X_test_bilstm_3d
    else:
        X_test_bilstm_3d_reshaped = X_test_bilstm_3d.reshape(-1, num_features_test)
        if np.isnan(X_test_bilstm_3d_reshaped).any() or np.isinf(X_test_bilstm_3d_reshaped).any():
            print("Warning: NaNs/Infs found in BiLSTM test data BEFORE scaling. Applying median imputation (fitted on train)...")
            if 'imputer_bilstm_temp' in locals():
                X_test_bilstm_3d_reshaped = imputer_bilstm_temp.transform(X_test_bilstm_3d_reshaped)
            else:
                temp_imputer_for_test = SimpleImputer(strategy='median')
                X_test_bilstm_3d_reshaped = temp_imputer_for_test.fit_transform(X_test_bilstm_3d_reshaped)

        X_test_bilstm_3d_scaled = scaler_bilstm.transform(X_test_bilstm_3d_reshaped).reshape(num_samples_test, seq_len_test, num_features_test)

print(f"X_train_bilstm_3d AFTER scaling - NaNs: {np.isnan(X_train_bilstm_3d_scaled).sum()}, Infs: {np.isinf(X_train_bilstm_3d_scaled).sum()}")
print(f"X_test_bilstm_3d AFTER scaling - NaNs: {np.isnan(X_test_bilstm_3d_scaled).sum()}, Infs: {np.isinf(X_test_bilstm_3d_scaled).sum()}")

# --- Store results ---
all_results = []
NUM_RUNS = 50  # For sklearn models
NUM_RUNS_BILSTM = 25  # Reduced runs for BiLSTM to save time

# --- Calculate scale_pos_weight ---
neg_count = (y_train == 0).sum(); pos_count = (y_train == 1).sum()
scale_pos_weight_value = neg_count / pos_count if pos_count > 0 and neg_count > 0 else 1.0
print(f"Calculated scale_pos_weight for XGBoost: {scale_pos_weight_value:.2f}")

# --- Define Cross-Validation Strategy ---
kf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Using {kf_cv.get_n_splits()}-Fold CV for all GridSearchCV operations.")

# --- BiLSTM Model Definition ---
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout_rate=0.3):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=True, 
                           dropout=dropout_rate if num_layers > 1 else 0)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out

# --- Model Evaluation Function ---
def calculate_metrics(y_true, y_pred_binary, y_pred_proba, model_name_eval="Model", run_number=1):
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_binary)
    
    roc_auc_val = float('nan')
    if len(np.unique(y_true)) > 1:
        try: roc_auc_val = roc_auc_score(y_true, y_pred_proba)
        except ValueError: pass

    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = 0,0,0,0
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        if len(y_true[y_true==0]) > 0 and np.all(y_pred_binary == 1):
            tn = 0; fp = len(y_true[y_true==0])
        elif len(y_true[y_true==1]) > 0 and np.all(y_pred_binary == 0):
            fn = len(y_true[y_true==1]); tp = 0
        specificity = 0.0

    # Calculate AUPRC using sklearn's auc function
    precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = auc(recall_pr, precision_pr)

    return {
        'Model': model_name_eval, 'Run': run_number, 'Accuracy': accuracy, 'Precision': precision,
        'Recall': recall, 'F1-Score': f1, 'AUC-ROC': roc_auc_val,
        'MCC': mcc, 'Specificity': specificity, 'AUPRC': auprc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

print("\n" + "="*80)
print("PHASE 1: GRID SEARCH FOR OPTIMAL PARAMETERS (SOLID TUMOR DATASET)")
print("="*80)

# --- XGBoost Grid Search ---
print("\n🔍 XGBoost Grid Search...")
xgb_base = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    device=xgb_device
)

# Using GridSearchCV with a more realistic grid for a powerful machine
xgb_param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [5, 9, 13],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'min_child_weight': [1, 5],
    'gamma': [0.1, 0.3],
    'scale_pos_weight': [scale_pos_weight_value]
}

print(f"   Testing {len(list(ParameterGrid(xgb_param_grid)))} parameter combinations using GridSearchCV...")
print(f"   This may take several minutes. Please wait...")

xgb_grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=xgb_param_grid,
    cv=kf_cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

try:
    print(f"   🚀 Starting XGBoost grid search...")
    start_time = time.time()
    
    xgb_grid.fit(X_train_final_tabular, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"   ⏱️ XGBoost grid search completed in {elapsed_time/60:.1f} minutes")
    
    best_xgb_params = xgb_grid.best_params_
    # Remove scale_pos_weight from params to avoid duplicate parameter error
    if 'scale_pos_weight' in best_xgb_params:
        del best_xgb_params['scale_pos_weight']
    print(f"   ✅ Best XGBoost parameters: {best_xgb_params}")
    print(f"   ✅ Best CV Score: {xgb_grid.best_score_:.4f}")
except Exception as e:
    print(f"   ❌ Error in XGBoost grid search: {e}")
    best_xgb_params = {
        'colsample_bytree': 0.8, 
        'gamma': 0, 
        'learning_rate': 0.15, 
        'max_depth': 3, 
        'min_child_weight': 1, 
        'n_estimators': 200, 
        'subsample': 0.9
    }

# --- Logistic Regression Grid Search ---
print("\n🔍 Logistic Regression Grid Search...")

# Expanded parameter grid for Logistic Regression
lr_param_grid = [
    {
        'penalty': ['l1'],
        'solver': ['liblinear', 'saga'],
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'max_iter': [1000, 2000],
        'class_weight': [None, 'balanced'],
        'tol': [1e-4, 1e-3]
    },
    {
        'penalty': ['l2'],
        'solver': ['liblinear', 'saga'],
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'max_iter': [1000, 2000],
        'class_weight': [None, 'balanced'],
        'tol': [1e-4, 1e-3]
    },
    {
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'max_iter': [1000, 2000],
        'class_weight': [None, 'balanced'],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
        'tol': [1e-4, 1e-3]
    }
]

print(f"   Testing valid parameter combinations...")
print(f"   This may take 3-5 minutes. Please wait...")

lr_base = LogisticRegression(random_state=42)

lr_grid = GridSearchCV(
    lr_base, 
    lr_param_grid,
    cv=kf_cv, 
    scoring='roc_auc', 
    n_jobs=-1, 
    verbose=1
)

try:
    print(f"   🚀 Starting Logistic Regression grid search...")
    start_time = time.time()
    
    lr_grid.fit(X_train_final_tabular, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"   ⏱️ Logistic Regression grid search completed in {elapsed_time/60:.1f} minutes")
    
    best_lr_params = lr_grid.best_params_
    print(f"   ✅ Best Logistic Regression parameters: {best_lr_params}")
    print(f"   ✅ Best CV Score: {lr_grid.best_score_:.4f}")
except Exception as e:
    print(f"   ❌ Error in Logistic Regression grid search: {e}")
    best_lr_params = {'C': 1.0, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 2000}

# --- Random Forest Grid Search ---
print("\n🔍 Random Forest Grid Search...")
rf_base = RandomForestClassifier(random_state=42)
rf_param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy'],
}

print(f"   Testing {len(list(ParameterGrid(rf_param_grid)))} parameter combinations using GridSearchCV...")
print(f"   This may take several minutes. Please wait...")

rf_grid = GridSearchCV(
    estimator=rf_base,
    param_grid=rf_param_grid,
    cv=kf_cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

try:
    print(f"   🚀 Starting Random Forest grid search...")
    start_time = time.time()
    
    rf_grid.fit(X_train_final_tabular, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"   ⏱️ Random Forest grid search completed in {elapsed_time/60:.1f} minutes")
    
    best_rf_params = rf_grid.best_params_
    print(f"   ✅ Best Random Forest parameters: {best_rf_params}")
    print(f"   ✅ Best CV Score: {rf_grid.best_score_:.4f}")
except Exception as e:
    print(f"   ❌ Error in Random Forest grid search: {e}")
    best_rf_params = {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5, 
                      'min_samples_leaf': 2, 'max_features': 'sqrt'}

# --- SVM Grid Search ---
print("\n🔍 SVM Grid Search...")

svm_param_grid = [
    {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10, 100, 500, 1000],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 'scale'],
        'class_weight': [None, 'balanced'],
        'shrinking': [True, False],
    },
    {
        'kernel': ['linear'],
        'C': [0.1, 1, 10, 100, 500, 1000],
        'class_weight': [None, 'balanced'],
    },
    {
        'kernel': ['poly'],
        'C': [0.1, 1, 10, 100],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
        'coef0': [0.0, 0.5, 1.0],
        'class_weight': [None, 'balanced'],
    }
]

print(f"   Testing valid parameter combinations...")
print(f"   This may take 8-12 minutes. Please wait...")

svm_base = SVC(probability=True, random_state=42)

svm_grid = GridSearchCV(
    svm_base, 
    svm_param_grid,
    cv=kf_cv, 
    scoring='roc_auc', 
    n_jobs=-1, 
    verbose=1
)

try:
    print(f"   🚀 Starting SVM grid search...")
    start_time = time.time()
    
    svm_grid.fit(X_train_final_tabular, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"   ⏱️ SVM grid search completed in {elapsed_time/60:.1f} minutes")
    
    best_svm_params = svm_grid.best_params_
    print(f"   ✅ Best SVM parameters: {best_svm_params}")
    print(f"   ✅ Best CV Score: {svm_grid.best_score_:.4f}")
except Exception as e:
    print(f"   ❌ Error in SVM grid search: {e}")
    best_svm_params = {'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}

# --- Decision Tree Grid Search ---
print("\n🔍 Decision Tree Grid Search...")
dt_base = DecisionTreeClassifier(random_state=42)
dt_param_grid = {
    'max_depth': [5, 15, 25, None],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
}

print(f"   Testing {len(list(ParameterGrid(dt_param_grid)))} parameter combinations using GridSearchCV...")
print(f"   This may take a few minutes. Please wait...")

dt_grid = GridSearchCV(
    estimator=dt_base,
    param_grid=dt_param_grid,
    cv=kf_cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

try:
    print(f"   🚀 Starting Decision Tree grid search...")
    start_time = time.time()
    
    dt_grid.fit(X_train_final_tabular, y_train)
    
    elapsed_time = time.time() - start_time
    print(f"   ⏱️ Decision Tree grid search completed in {elapsed_time/60:.1f} minutes")
    
    best_dt_params = dt_grid.best_params_
    print(f"   ✅ Best Decision Tree parameters: {best_dt_params}")
    print(f"   ✅ Best CV Score: {dt_grid.best_score_:.4f}")
except Exception as e:
    print(f"   ❌ Error in Decision Tree grid search: {e}")
    best_dt_params = {'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt'}

# --- BiLSTM Grid Search ---
print("\n🔍 BiLSTM Grid Search...")
if X_train_bilstm_3d_scaled.shape[1] > 0 and X_train_bilstm_3d_scaled.shape[2] > 0:
    input_dim_bilstm = X_train_bilstm_3d_scaled.shape[2]
    
    # BiLSTM parameter combinations (reduced for faster search)
    bilstm_param_combinations = [
        # Efficient configurations for solid tumor dataset
        {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.2, 'lr': 0.001, 'batch_size': 32},
        {'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.3, 'lr': 0.001, 'batch_size': 32},
        {'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001, 'batch_size': 32},
        {'hidden_dim': 128, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.001, 'batch_size': 64},
        {'hidden_dim': 256, 'num_layers': 1, 'dropout': 0.3, 'lr': 0.0005, 'batch_size': 32},
        {'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.4, 'lr': 0.0005, 'batch_size': 32},
        {'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.3, 'lr': 0.002, 'batch_size': 32},
    ]

    best_bilstm_score = 0
    best_bilstm_params = bilstm_param_combinations[1]  # Default to a good middle option
    
    print(f"   Testing {len(bilstm_param_combinations)} parameter combinations using {kf_cv.get_n_splits()}-fold CV...")
    print(f"   This may take a significant amount of time. Please wait...")
    
    for params in tqdm(bilstm_param_combinations, desc="BiLSTM Grid Search", ncols=100):
        fold_scores = []
        try:
            for fold, (train_idx, val_idx) in enumerate(kf_cv.split(X_train_bilstm_3d_scaled, y_train)):
                
                # --- Data for this fold ---
                X_train_fold, X_val_fold = X_train_bilstm_3d_scaled[train_idx], X_train_bilstm_3d_scaled[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                X_train_torch = torch.tensor(X_train_fold, dtype=torch.float32).to(device)
                y_train_torch = torch.tensor(y_train_fold.values, dtype=torch.float32).unsqueeze(1).to(device)
                X_val_torch = torch.tensor(X_val_fold, dtype=torch.float32).to(device)

                # --- Model and training setup for this fold ---
                model = BiLSTM(input_dim_bilstm, params['hidden_dim'], 
                              params['num_layers'], params['dropout']).to(device)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(model.parameters(), lr=params['lr'])
                
                train_dataset = TensorDataset(X_train_torch, y_train_torch)
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                
                # --- Training on this fold ---
                for epoch in range(50): # 50 epochs per fold for grid search
                    model.train()
                    for X_batch, y_batch in train_loader:
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                
                # --- Evaluation on this fold ---
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_torch).cpu().numpy().flatten()
                    score = roc_auc_score(y_val_fold, val_outputs)
                    fold_scores.append(score)

            # --- Average score for this param set ---
            avg_score = np.mean(fold_scores)
            if avg_score > best_bilstm_score:
                best_bilstm_score = avg_score
                best_bilstm_params = params
                    
        except Exception as e:
            print(f"   ⚠️ Error testing BiLSTM params {params}: {e}")
    
    print(f"   ✅ Best BiLSTM parameters (from CV): {best_bilstm_params}")
    print(f"   ✅ Best CV Score: {best_bilstm_score:.4f}")
else:
    best_bilstm_params = {'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.3, 'lr': 0.001, 'batch_size': 32}
    print("   ⚠️ Skipping BiLSTM grid search due to invalid input shape. Using default parameters.")

print(f"\n✅ Grid Search Complete! Found optimal parameters for solid tumor dataset:")
print(f"   XGBoost: {best_xgb_params}")
print(f"   Logistic Regression: {best_lr_params}")
print(f"   Random Forest: {best_rf_params}")
print(f"   SVM: {best_svm_params}")
print(f"   Decision Tree: {best_dt_params}")
print(f"   BiLSTM: {best_bilstm_params}")

print("\n" + "="*80)
print("PHASE 2: RUNNING MODELS 50 TIMES WITH BEST PARAMETERS")
print("="*80)

# Create models directory for saving
if not os.path.exists('comprehensive_solid_models'):
    os.makedirs('comprehensive_solid_models')

# Get input dimension for BiLSTM if needed
if X_train_bilstm_3d_scaled.shape[2] > 0:
    input_dim_bilstm = X_train_bilstm_3d_scaled.shape[2]
else:
    input_dim_bilstm = 1  # fallback value

# Function to train and evaluate a single model
def train_and_evaluate_model(model_name, model_class, params, run_num):
    if model_name == 'BiLSTM':
        if X_train_bilstm_3d_scaled.shape[1] > 0 and X_train_bilstm_3d_scaled.shape[2] > 0:
            # Set random seed for reproducibility
            torch.manual_seed(42 + run_num)
            np.random.seed(42 + run_num)
            
            model = BiLSTM(input_dim_bilstm, params['hidden_dim'], 
                          params['num_layers'], params['dropout']).to(device)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])
            
            X_train_torch = torch.tensor(X_train_bilstm_3d_scaled, dtype=torch.float32).to(device)
            y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
            X_test_torch = torch.tensor(X_test_bilstm_3d_scaled, dtype=torch.float32).to(device)
            
            train_dataset = TensorDataset(X_train_torch, y_train_torch)
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, 
                                    num_workers=0, pin_memory=torch.cuda.is_available())
            
            # Training - Optimized for speed vs. quality trade-off
            if run_num == 1:
                # First run: Train thoroughly for best model saving
                epochs = 100  # Reduced from 150 for faster training
                patience = 15  # Early stopping patience
                print(f"   🎯 Training BiLSTM thoroughly for model saving (up to {epochs} epochs)")
            else:
                # Subsequent runs: Quick training for statistical robustness
                epochs = 50  # Much faster for repeated runs
                patience = 8   # Earlier stopping
            
            # Early stopping setup
            best_loss = float('inf')
            patience_counter = 0
            
            # Learning rate scheduler for faster convergence
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                batch_count = 0
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                
                # Calculate average loss for this epoch
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
                scheduler.step(avg_epoch_loss)
                
                # Early stopping check (only for first run to ensure quality)
                if run_num == 1:
                    if avg_epoch_loss < best_loss:
                        best_loss = avg_epoch_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        if run_num == 1:
                            print(f"   ⏰ Early stopping at epoch {epoch+1}/{epochs} (best loss: {best_loss:.4f})")
                        break
            
            # Save BiLSTM model in .pth format (PyTorch native)
            if run_num == 1:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_params': params,
                    'input_dim': input_dim_bilstm,
                    'scaler_state': scaler_bilstm
                }, f'comprehensive_solid_models/bilstm_best_model.pth')
                print(f"   💾 Saved BiLSTM model to comprehensive_solid_models/bilstm_best_model.pth (pth format)")
            
            # Prediction
            model.eval()
            with torch.no_grad():
                y_pred_prob = model(X_test_torch).cpu().numpy().flatten()
                
            return model, y_pred_prob
        else:
            return None, None
    else:
        # Set random seed for sklearn models
        np.random.seed(42 + run_num)
        
        # Create model with best parameters - FIXED: Remove duplicate scale_pos_weight
        if model_name == 'XGBoost':
            model = XGBClassifier(
                eval_metric='logloss',
                random_state=42 + run_num,
                scale_pos_weight=scale_pos_weight_value,  # Set here only once
                tree_method='hist',
                device=xgb_device,
                **params  # params no longer contains scale_pos_weight
            )
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(random_state=42 + run_num, **params)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(random_state=42 + run_num, **params)
        elif model_name == 'SVM':
            model = SVC(probability=True, random_state=42 + run_num, **params)
        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier(random_state=42 + run_num, **params)
        
        # Train and predict
        model.fit(X_train_final_tabular, y_train)
        y_pred_prob = model.predict_proba(X_test_final_tabular)[:, 1]
        
        # Save sklearn model in .pkl format (as requested)
        if run_num == 1:
            model_filename = f'comprehensive_solid_models/{model_name.lower()}_best_model.pkl'
            joblib.dump({
                'model': model,
                'model_params': params,
                'scaler': scaler_tabular,
                'imputer': imputer_tabular
            }, model_filename)
            print(f"   💾 Saved {model_name} model to {model_filename} (pkl format)")
        
        return model, y_pred_prob
    
    # This section should not be reached
    return None, None

# Run all models 50 times
models_to_run = {
    'XGBoost': best_xgb_params,
    'LogisticRegression': best_lr_params,
    'RandomForest': best_rf_params,
    'SVM': best_svm_params,
    'DecisionTree': best_dt_params,
    'BiLSTM': best_bilstm_params
}

total_runs = (len(models_to_run) - 1) * NUM_RUNS + NUM_RUNS_BILSTM  # Account for BiLSTM having fewer runs
current_run = 0

print(f"\n🚀 Running models with optimized configurations:")
print(f"   📊 Sklearn models: 5 models × {NUM_RUNS} runs = {5 * NUM_RUNS} runs")
print(f"   🧠 BiLSTM model: 1 model × {NUM_RUNS_BILSTM} runs = {NUM_RUNS_BILSTM} runs")
print(f"   🎯 Total: {total_runs} runs (time optimized for BiLSTM)")

# Store the best models for later use in plotting
best_models = {}
latest_results = {}

for model_name, best_params in models_to_run.items():
    # Use different run counts for different model types
    current_num_runs = NUM_RUNS_BILSTM if model_name == 'BiLSTM' else NUM_RUNS
    print(f"\n📊 {model_name} - Running {current_num_runs} times")
    print(f"   Using parameters: {best_params}")
    
    for run in tqdm(range(1, current_num_runs + 1), desc=f"{model_name} Runs", ncols=100):
        try:
            model, y_pred_prob = train_and_evaluate_model(model_name, None, best_params, run)
            if model is not None and y_pred_prob is not None:
                # Store the first model and predictions for plotting
                if run == 1:
                    best_models[model_name] = model
                    latest_results[model_name] = y_pred_prob
                
                # Calculate metrics
                y_pred_binary = (y_pred_prob >= 0.5).astype(int)
                metrics = calculate_metrics(y_test, y_pred_binary, y_pred_prob, model_name, run)
                all_results.append(metrics)
            current_run += 1
        except Exception as e:
            print(f"   ❌ Error in {model_name} run {run}: {e}")

print(f"\n✅ Completed {len(all_results)} successful runs out of {total_runs} total runs")

# Save all results to CSV
results_df = pd.DataFrame(all_results)
results_df.to_csv("comprehensive_solid_models_performance.csv", index=False)
print(f"💾 Saved results to comprehensive_solid_models_performance.csv")

print("\n" + "="*80)
print("PHASE 3: CALCULATING AVERAGE METRICS AND GENERATING PLOTS")
print("="*80)

# Calculate average metrics for plotting
avg_metrics = results_df.groupby('Model').agg({
    'Accuracy': 'mean',
    'Precision': 'mean', 
    'Recall': 'mean',
    'F1-Score': 'mean',
    'AUC-ROC': 'mean',
    'MCC': 'mean',
    'Specificity': 'mean',
    'AUPRC': 'mean'
}).reset_index()

# Sort models by AUC-ROC for consistent ordering
avg_metrics = avg_metrics.sort_values('AUC-ROC', ascending=False)
model_order = avg_metrics['Model'].tolist()

print("\n📈 Average Performance Metrics (ordered by AUC-ROC):")
print(tabulate(avg_metrics, headers='keys', tablefmt='psql', floatfmt=".4f"))

# Use predictions already generated during model runs
print("\n🎨 Using predictions from the first run of each model for plotting...")
print(f"Available models for plotting: {list(latest_results.keys())}")

print(f"\n🎨 Generating comprehensive plots...")

# Create color palette for consistent plotting
colors = plt.cm.Set3(np.linspace(0, 1, len(model_order)))

# 1. Combined ROC Curve (ordered by performance)
plt.figure(figsize=(12, 8))
for i, model_name in enumerate(model_order):
    if model_name in latest_results:
        y_prob = latest_results[model_name]
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc_val = auc(fpr, tpr)
            avg_auc = avg_metrics[avg_metrics['Model'] == model_name]['AUC-ROC'].iloc[0]
            plt.plot(fpr, tpr, lw=3, color=colors[i], 
                    label=f'{model_name} (Avg AUC = {avg_auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Comprehensive Model Comparison\n(Ordered by Average AUC-ROC)', fontsize=14, pad=20)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comprehensive_solid_models_plots/roc_curves_ordered.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Combined Precision-Recall Curve (ordered by performance)
plt.figure(figsize=(12, 8))
no_skill_level = len(y_test[y_test==1]) / len(y_test) if len(y_test) > 0 else 0

for i, model_name in enumerate(model_order):
    if model_name in latest_results:
        y_prob = latest_results[model_name]
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
        auprc_val = auc(recall_vals, precision_vals)
        avg_auprc = avg_metrics[avg_metrics['Model'] == model_name]['AUPRC'].iloc[0]
        plt.plot(recall_vals, precision_vals, lw=3, color=colors[i],
                label=f'{model_name} (Avg AUPRC = {avg_auprc:.3f})')

plt.plot([0, 1], [no_skill_level, no_skill_level], linestyle='--', color='grey', 
         alpha=0.8, label=f'No Skill ({no_skill_level:.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Comprehensive Model Comparison\n(Ordered by Average AUC-ROC)', fontsize=14, pad=20)
plt.legend(loc="lower left", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comprehensive_solid_models_plots/precision_recall_curves_ordered.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Combined Calibration Plot (ordered by performance)
plt.figure(figsize=(12, 8))
plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)

for i, model_name in enumerate(model_order):
    if model_name in latest_results:
        y_prob = latest_results[model_name]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", color=colors[i],
                linewidth=2, markersize=8, label=f'{model_name}')

plt.ylabel("Fraction of positives (Actual)", fontsize=12)
plt.xlabel("Mean predicted probability (Predicted)", fontsize=12)
plt.ylim([-0.05, 1.05])
plt.title('Calibration Plots - Comprehensive Model Comparison\n(Ordered by Average AUC-ROC)', fontsize=14, pad=20)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comprehensive_solid_models_plots/calibration_curves_ordered.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Decision Curve Analysis (ordered by performance)
print("   ⚠️ Skipping Decision Curve Analysis (DCA) plot due to Python 3.13 compatibility issues with dcurves package.")

# 5. Performance Comparison Bar Charts (ordered by performance)
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'MCC', 'Specificity', 'AUPRC']

for metric in tqdm(metrics_to_plot, desc="Generating Metric Plots", ncols=100):
    plt.figure(figsize=(14, 8))
    
    # Get ordered data for this metric
    metric_data = []
    model_names_ordered = []
    for model_name in model_order:
        if model_name in avg_metrics['Model'].values:
            value = avg_metrics[avg_metrics['Model'] == model_name][metric].iloc[0]
            metric_data.append(value)
            model_names_ordered.append(model_name)
    
    # Create bar plot with colors
    bars = plt.bar(range(len(model_names_ordered)), metric_data, 
                   color=colors[:len(model_names_ordered)], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on top of bars
    for i, (bar, value) in enumerate(zip(bars, metric_data)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Customize plot
    plt.title(f'{metric} Comparison - Comprehensive Model Analysis\n(Ordered by Average AUC-ROC)', 
              fontsize=14, pad=20, fontweight='bold')
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel(f'Average {metric}', fontsize=12, fontweight='bold')
    plt.xticks(range(len(model_names_ordered)), model_names_ordered, rotation=45, ha='right')
    
    # Set y-axis limits
    min_val = min(metric_data)
    max_val = max(metric_data)
    range_val = max_val - min_val
    plt.ylim(max(0, min_val - range_val * 0.1), min(1, max_val + range_val * 0.15))
    
    # Add grid and mean line
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    mean_val = np.mean(metric_data)
    plt.axhline(y=mean_val, color='red', linestyle='--', alpha=0.8, linewidth=2,
                label=f'Mean: {mean_val:.3f}')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'comprehensive_solid_models_plots/{metric.lower().replace("-", "_")}_comparison_ordered.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

# 6. Comprehensive Performance Summary Heatmap
plt.figure(figsize=(12, 8))
heatmap_data = avg_metrics.set_index('Model')[metrics_to_plot]
heatmap_data = heatmap_data.reindex(model_order)  # Ensure correct order

sns.heatmap(heatmap_data.T, annot=True, cmap='RdYlBu_r', center=0.5, 
            fmt='.3f', cbar_kws={'label': 'Performance Score'}, 
            linewidths=0.5, square=False)
plt.title('Comprehensive Performance Heatmap\n(Models Ordered by Average AUC-ROC)', 
          fontsize=14, pad=20, fontweight='bold')
plt.xlabel('Models', fontsize=12, fontweight='bold')
plt.ylabel('Metrics', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('comprehensive_solid_models_plots/performance_heatmap_ordered.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ All comprehensive plots saved to 'comprehensive_solid_models_plots' directory")
print(f"📊 Total results: {len(all_results)} (optimized: {NUM_RUNS} sklearn + {NUM_RUNS_BILSTM} BiLSTM runs)")
print(f"💾 Results saved to: comprehensive_solid_models_performance.csv")

# Save final model comparison summary
print("\n📋 Saving final model comparison summary...")
final_summary = avg_metrics.copy()
final_summary['Best_Params'] = final_summary['Model'].map(models_to_run)
final_summary.to_csv('comprehensive_solid_model_comparison_summary.csv', index=False)
print("💾 Model comparison summary saved to: comprehensive_solid_model_comparison_summary.csv")

# Save preprocessing objects for future use
print("\n💾 Saving preprocessing objects...")
preprocessing_objects = {
    'scaler_tabular': scaler_tabular,
    'imputer_tabular': imputer_tabular,
    'scaler_bilstm': scaler_bilstm,
    'med_vocab': med_vocab,
    'max_med_lengths': max_med_lengths,
    'max_length_vitals': max_length_vitals,
    'LAB_HISTORY_LENGTH': LAB_HISTORY_LENGTH,
    'train_vital_sign_cols': train_vital_sign_cols,
    'train_lab_cols': train_lab_cols,
    'train_med_cols': train_med_cols,
    'scale_pos_weight_value': scale_pos_weight_value
}

joblib.dump(preprocessing_objects, 'comprehensive_solid_models/preprocessing_objects.pkl')
print("💾 Preprocessing objects saved to: comprehensive_solid_models/preprocessing_objects.pkl")

print("\n" + "="*80)
print("🎉 COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
print("="*80)
print("\n📁 Generated Files and Directories:")
print("  📊 Metrics: comprehensive_solid_models_performance.csv")
print("  📈 Summary: comprehensive_solid_model_comparison_summary.csv")
print("  🤖 Models: comprehensive_solid_models/ directory")
print("     ├── xgboost_best_model.pkl")
print("     ├── logisticregression_best_model.pkl")
print("     ├── randomforest_best_model.pkl")
print("     ├── svm_best_model.pkl")
print("     ├── decisiontree_best_model.pkl")
print("     ├── bilstm_best_model.pth")
print("     └── preprocessing_objects.pkl")
print("  📈 Plots: comprehensive_solid_models_plots/ directory")
print("     ├── roc_curves_ordered.png")
print("     ├── precision_recall_curves_ordered.png")
print("     ├── calibration_curves_ordered.png")
print("     ├── performance_heatmap_ordered.png")
print("     └── [metric]_comparison_ordered.png (8 plots)")

print(f"\n✨ Best performing model: {avg_metrics.iloc[0]['Model']} (AUC-ROC: {avg_metrics.iloc[0]['AUC-ROC']:.4f})")
print("\n🔄 To reload models for inference, use:")
print("   📝 Sklearn models: joblib.load('comprehensive_solid_models/[model_name]_best_model.pkl')")
print("   🧠 BiLSTM model: torch.load('comprehensive_solid_models/bilstm_best_model.pth')")
print("   ⚙️  Preprocessing: joblib.load('comprehensive_solid_models/preprocessing_objects.pkl')")

print(f"\n📊 Performance Summary:")
print(tabulate(avg_metrics[['Model', 'AUC-ROC', 'F1-Score', 'Precision', 'Recall']], 
               headers=['Model', 'AUC-ROC', 'F1-Score', 'Precision', 'Recall'], 
               tablefmt='psql', floatfmt=".4f")) 