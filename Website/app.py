from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import glob
import time
import warnings
from datetime import datetime, timedelta
import logging
from werkzeug.utils import secure_filename
import json
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'public-demo-secret-key')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, 'models')
DEMO_DATA_FOLDER = os.path.join(BASE_DIR, 'demo_data')
HEMATOLOGY_DATA_FILE = 'hematology_patients.xlsx'
SOLID_DATA_FILE = 'solid_patients.xlsx'
PREDICTION_INTERVAL_HOURS = 6  # Can be changed to 12
PUBLIC_DEMO_MODE = os.getenv('EWS_PUBLIC_DEMO', '1') == '1'
AUTO_SCAN_PATH = DEMO_DATA_FOLDER if PUBLIC_DEMO_MODE else os.getenv('EWS_PRIVATE_EXPORT_DIR', DEMO_DATA_FOLDER)
AUTO_SCAN_PATTERN = 'export_all_columns_'  # File name pattern to match

# Model validation configuration
VALIDATION_CONFIG = {
    'min_probability': 0.0,
    'max_probability': 1.0,
    'expected_risk_levels': [
        'Low Risk (Safe)',
        'Medium Risk (Monitoring)',
        'High Risk (Urgent)'
    ],
    'required_features_count': {
        'hematology': 2500,  # Updated based on actual model output
        'solid': 2300
    },
    'max_prediction_time_seconds': 5.0  # Maximum time for a single prediction
}

# Ensure public-demo data exists without recreating private-model folders.
if not PUBLIC_DEMO_MODE:
    os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(DEMO_DATA_FOLDER, exist_ok=True)

# Global variables to store prediction data
hematology_predictions = []
solid_predictions = []
last_prediction_time = {"hematology": None, "solid": None}

# Model information (simplified - no metrics)
MODEL_INFO = {
    "name": "XGBoost",
    "description": "ICU Risk Prediction System",
    "mode": "Public Demo" if PUBLIC_DEMO_MODE else "Private Deployment"
}

# Constants from training files
PAD_TOKEN_ID = 0
UNK_TOKEN_ID = 1
EMPTY_SEQ_ID = 0

# Compatibility patch helpers
def ensure_fill_dtype(imputer):
    """Backfill SimpleImputer private attrs for older pickles on newer sklearn."""
    try:
        if imputer is None:
            return
        if not hasattr(imputer, "_fill_dtype"):
            # Fall back to the dtype used at fit time or statistics dtype
            if hasattr(imputer, "_fit_dtype"):
                imputer._fill_dtype = imputer._fit_dtype
            elif hasattr(imputer, "statistics_"):
                imputer._fill_dtype = imputer.statistics_.dtype
            else:
                imputer._fill_dtype = np.dtype("float64")
            logger.info("Patched imputer _fill_dtype for compatibility")
    except Exception as e:
        logger.warning(f"Could not patch imputer _fill_dtype: {e}")

# Column definitions from training files
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

# Preprocessing functions from training files
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


def latest_numeric_value(value, default=0.0):
    """Return the most recent numeric value from a parsed history list."""
    if isinstance(value, list) and value:
        try:
            return float(value[-1])
        except (TypeError, ValueError):
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_probability(value):
    return max(0.01, min(0.98, float(value)))


def demo_probability_from_row(row, model_type):
    """Heuristic ICU risk score for the public demo when private models are unavailable."""
    heart_rate = latest_numeric_value(row.get('HEART_RATE'), 85.0)
    pulse_ox = latest_numeric_value(row.get('PULSE_OXIMETRY'), 98.0)
    temperature = latest_numeric_value(row.get('TEMPERATURE'), 37.0)
    map_value = latest_numeric_value(row.get('MEAN_ARTERIAL_PRESSURE'), 80.0)
    respiration = latest_numeric_value(row.get('RESPIRATION_RATE'), 18.0)
    creatinine = latest_numeric_value(row.get('CREATININE_RESULT'), 0.9)
    bilirubin = latest_numeric_value(row.get('TOTAL_BILIRUBIN_RESULT'), 0.8)
    hemoglobin = latest_numeric_value(row.get('HEMOGLOBIN_RESULT'), 12.0)
    leukocytes = latest_numeric_value(row.get('LEUKOCYTE_COUNT_RESULT'), 8000.0)
    neutrophils = latest_numeric_value(row.get('ABSOLUTE_NEUTROPHILS'), 4500.0)
    platelets = latest_numeric_value(row.get('PLATELET_COUNT_RESULT'), 220000.0)

    antibiotics = row.get('ANTIBIOTICS') if isinstance(row.get('ANTIBIOTICS'), list) else []
    fungal_drugs = row.get('FUNGAL_DRUGS') if isinstance(row.get('FUNGAL_DRUGS'), list) else []

    risk = 0.05
    risk += max(0.0, min(0.18, (heart_rate - 95.0) / 220.0))
    risk += max(0.0, min(0.20, (95.0 - pulse_ox) / 18.0))
    risk += max(0.0, min(0.12, (respiration - 20.0) / 80.0))
    risk += max(0.0, min(0.10, abs(temperature - 37.0) / 20.0))
    risk += max(0.0, min(0.15, (75.0 - map_value) / 90.0))
    risk += max(0.0, min(0.08, (creatinine - 1.2) / 15.0))
    risk += max(0.0, min(0.06, (bilirubin - 1.0) / 12.0))
    risk += max(0.0, min(0.07, (10.5 - hemoglobin) / 25.0))
    risk += max(0.0, min(0.06, (platelets < 150000) * ((150000 - platelets) / 150000.0)))
    risk += max(0.0, min(0.04, (leukocytes - 11000.0) / 50000.0))
    risk += max(0.0, min(0.04, (neutrophils - 7000.0) / 30000.0))
    risk += 0.03 if antibiotics else 0.0
    risk += 0.02 if fungal_drugs else 0.0

    if model_type == 'hematology':
        risk += 0.05
        risk += max(0.0, min(0.05, (120000.0 - platelets) / 120000.0))
    else:
        risk += max(0.0, min(0.04, (70.0 - map_value) / 80.0))

    token = sum(ord(ch) for ch in str(row.get('MRN', 'demo')))
    jitter = ((token % 11) - 5) / 250.0
    return clamp_probability(risk + jitter)

# Load models
def load_models():
    """Load the trained XGBoost models and preprocessing objects"""
    try:
        hematology_model_path = os.path.join(MODELS_FOLDER, 'xgboost_hematology.pkl')
        solid_model_path = os.path.join(MODELS_FOLDER, 'xgboost.pkl')
        
        # Load preprocessing objects
        hematology_preprocessing_path = os.path.join('..', 'comprehensive_hematology_models', 'preprocessing_objects.pkl')
        solid_preprocessing_path = os.path.join('..', 'comprehensive_solid_models', 'preprocessing_objects.pkl')
        
        if os.path.exists(hematology_model_path) and os.path.exists(solid_model_path):
            hematology_model_data = joblib.load(hematology_model_path)
            solid_model_data = joblib.load(solid_model_path)
            
            # Extract model and preprocessing objects
            hematology_model = hematology_model_data['model']
            solid_model = solid_model_data['model']
            
            # Store preprocessing objects globally
            global hematology_scaler, hematology_imputer, solid_scaler, solid_imputer
            global hematology_preprocessing_params, solid_preprocessing_params
            hematology_scaler = hematology_model_data['scaler']
            hematology_imputer = hematology_model_data['imputer']
            solid_scaler = solid_model_data['scaler']
            solid_imputer = solid_model_data['imputer']

            # Patch older imputers to satisfy newer sklearn internals
            ensure_fill_dtype(hematology_imputer)
            ensure_fill_dtype(solid_imputer)
            
            # Load preprocessing parameters
            if os.path.exists(hematology_preprocessing_path):
                hematology_preprocessing_params = joblib.load(hematology_preprocessing_path)
                logger.info("Loaded hematology preprocessing parameters")
            else:
                logger.warning("Hematology preprocessing parameters not found, using defaults")
                hematology_preprocessing_params = {
                    'max_length_vitals': 10,
                    'LAB_HISTORY_LENGTH': 4,
                    'max_med_lengths': {'ANTIBIOTICS': 5, 'NEUROLOGY_DRUGS': 5, 'CARDIOLOGY_DRUGS': 5, 'FUNGAL_DRUGS': 5},
                    'med_vocab': {"PAD": 0, "UNK": 1}
                }
            
            if os.path.exists(solid_preprocessing_path):
                solid_preprocessing_params = joblib.load(solid_preprocessing_path)
                logger.info("Loaded solid preprocessing parameters")
            else:
                logger.warning("Solid preprocessing parameters not found, using defaults")
                solid_preprocessing_params = {
                    'max_length_vitals': 10,
                    'LAB_HISTORY_LENGTH': 4,
                    'max_med_lengths': {'ANTIBIOTICS': 5, 'NEUROLOGY_DRUGS': 5, 'CARDIOLOGY_DRUGS': 5, 'FUNGAL_DRUGS': 5},
                    'med_vocab': {"PAD": 0, "UNK": 1}
                }
            
            logger.info("Models and preprocessing objects loaded successfully")
            return hematology_model, solid_model
        else:
            if PUBLIC_DEMO_MODE:
                logger.info("Model files not found. Continuing in public demo mode.")
            else:
                logger.error("Model files not found")
            return None, None
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return None, None

# Load models at startup
hematology_model, solid_model = load_models()

@app.route('/')
def index():
    """Redirect root to predictions dashboard"""
    return redirect(url_for('predictions_dashboard'))

@app.route('/predictions')
def predictions_dashboard():
    """Unified predictions dashboard"""
    return render_template('predictions.html', 
                         model_info=MODEL_INFO,
                         predictions=hematology_predictions + solid_predictions,
                         last_update=last_prediction_time.get("predictions"),
                         prediction_interval=PREDICTION_INTERVAL_HOURS)

@app.route('/api/predictions')
def get_predictions_api():
    """API endpoint to get current predictions"""
    return jsonify({
        'predictions': hematology_predictions + solid_predictions,
        'last_update': last_prediction_time.get("predictions"),
        'model_info': MODEL_INFO
    })

@app.route('/api/trigger_prediction', methods=['POST'])
def trigger_prediction():
    """Manually trigger prediction update"""
    try:
        success = auto_scan_and_process()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Predictions refreshed from the latest public demo export' if PUBLIC_DEMO_MODE else 'Predictions updated from latest file on shared drive',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No suitable demo file found' if PUBLIC_DEMO_MODE else 'No suitable file found on shared drive',
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        logger.error(f"Error triggering prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/validation_status')
def get_validation_status():
    """Get validation status for all predictions"""
    try:
        all_predictions = hematology_predictions + solid_predictions
        
        # Calculate validation statistics
        total_predictions = len(all_predictions)
        passed_validations = sum(1 for p in all_predictions if p.get('validation_passed', True))
        failed_validations = total_predictions - passed_validations
        
        # Collect all validation errors and warnings
        all_errors = []
        all_warnings = []
        for pred in all_predictions:
            all_errors.extend(pred.get('validation_errors', []))
            all_warnings.extend(pred.get('validation_warnings', []))
        
        # Group errors by type
        error_types = {}
        for error in all_errors:
            error_type = error.split(':')[0] if ':' in error else 'Unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        validation_summary = {
            'total_predictions': total_predictions,
            'passed_validations': passed_validations,
            'failed_validations': failed_validations,
            'validation_success_rate': (passed_validations / total_predictions * 100) if total_predictions > 0 else 0,
            'total_errors': len(all_errors),
            'total_warnings': len(all_warnings),
            'error_types': error_types,
            'recent_errors': all_errors[-10:] if all_errors else [],  # Last 10 errors
            'recent_warnings': all_warnings[-10:] if all_warnings else [],  # Last 10 warnings
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(validation_summary)
        
    except Exception as e:
        logger.error(f"Error getting validation status: {str(e)}")
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls', 'csv']

def find_latest_excel_file():
    """Find the latest Excel/CSV file in the configured scan directory."""
    try:
        if not os.path.exists(AUTO_SCAN_PATH):
            logger.warning(f"Auto-scan path {AUTO_SCAN_PATH} does not exist")
            return None
        
        # Search for files matching the pattern (both .xlsx and .csv)
        matching_files = []
        for ext in ['*.xlsx', '*.csv']:
            pattern = os.path.join(AUTO_SCAN_PATH, f"{AUTO_SCAN_PATTERN}{ext}")
            matching_files.extend(glob.glob(pattern))
        
        if not matching_files:
            logger.info(f"No files found matching pattern: {AUTO_SCAN_PATTERN}*.xlsx or {AUTO_SCAN_PATTERN}*.csv")
            return None
        
        # Sort files by modification time (newest first)
        latest_file = max(matching_files, key=os.path.getmtime)
        logger.info(f"Found latest file: {latest_file}")
        return latest_file
        
    except Exception as e:
        logger.error(f"Error finding latest Excel/CSV file: {str(e)}")
        return None

def auto_scan_and_process():
    """Load the latest available export file and refresh predictions."""
    try:
        latest_file = find_latest_excel_file()
        if latest_file:
            logger.info(f"Auto-processing file: {latest_file}")
            process_uploaded_file('predictions', specific_file=latest_file)
            return True
        else:
            logger.info("No suitable file found for auto-processing")
            return False
    except Exception as e:
        logger.error(f"Error in auto-scan and process: {str(e)}")
        return False

def process_features_tabular(df, preprocessing_params, is_train_phase=True):
    """Process features using the same logic as training files"""
    # Silent processing - no logging
    processed_features_list = []
    
    # Extract preprocessing parameters
    max_length_vitals = preprocessing_params.get('max_length_vitals', 10)
    LAB_HISTORY_LENGTH = preprocessing_params.get('LAB_HISTORY_LENGTH', 4)
    max_med_lengths = preprocessing_params.get('max_med_lengths', {})
    med_vocab = preprocessing_params.get('med_vocab', {"PAD": 0, "UNK": 1})
    
    # Vital Signs
    for col in train_vital_sign_cols:
        if col in df.columns:
            padded_col = df[col].apply(lambda x: pad_list_to_length(x if isinstance(x, list) else [], max_length_vitals, pad_value=0.0))
            col_array = np.array(padded_col.tolist(), dtype=float)
            expected_shape = (len(df), max_length_vitals)
            if col_array.shape != expected_shape: 
                col_array = np.zeros(expected_shape)
            processed_features_list.append(col_array)
        else: 
            processed_features_list.append(np.zeros((len(df), max_length_vitals)))
    
    # Lab Results
    for col in train_lab_cols:
        if col in df.columns:
            proc_col = df[col].apply(lambda x: [0.0]*(LAB_HISTORY_LENGTH - len(x)) + x[-LAB_HISTORY_LENGTH:] if isinstance(x, list) else [0.0]*LAB_HISTORY_LENGTH)
            col_array = np.array(proc_col.tolist(), dtype=float)
            expected_shape = (len(df), LAB_HISTORY_LENGTH)
            if col_array.shape != expected_shape: 
                col_array = np.zeros(expected_shape)
            processed_features_list.append(col_array)
        else: 
            processed_features_list.append(np.zeros((len(df), LAB_HISTORY_LENGTH)))
    
    # Medications
    for col in train_med_cols:
        max_len = max_med_lengths.get(col, 0)
        med_lists = df[col].apply(lambda x: x if isinstance(x, list) else []) if col in df else pd.Series([[] for _ in range(len(df))], index=df.index)
        
        # Padded Integer Sequence
        if col in df.columns:
            transformed_col = med_lists.apply(lambda x: transform_meds_to_padded_ints(x, med_vocab, max_len))
            col_array_seq = np.array(transformed_col.tolist(), dtype=int)
            expected_shape = (len(df), max_len)
            if col_array_seq.shape != expected_shape: 
                col_array_seq = np.zeros(expected_shape, dtype=int)
            processed_features_list.append(col_array_seq)
        else: 
            processed_features_list.append(np.zeros((len(df), max_len), dtype=int))
        
        # Aggregate Features
        lengths = med_lists.apply(len).values.reshape(-1, 1)
        processed_features_list.append(lengths.astype(float))
        processed_features_list.append((lengths > 0).astype(float))
        processed_features_list.append(med_lists.apply(lambda x: med_vocab.get(x[0], UNK_TOKEN_ID) if len(x)>0 else EMPTY_SEQ_ID).values.reshape(-1, 1).astype(float))
        processed_features_list.append(med_lists.apply(lambda x: med_vocab.get(x[-1], UNK_TOKEN_ID) if len(x)>0 else EMPTY_SEQ_ID).values.reshape(-1, 1).astype(float))
        processed_features_list.append(med_lists.apply(lambda x: len(set(x))).values.reshape(-1, 1).astype(float))

    valid_features = [arr for arr in processed_features_list if arr.ndim == 2 and arr.shape[0] == len(df) and arr.shape[1] > 0]
    if not valid_features: 
        return np.empty((len(df), 0))
    X_processed = np.concatenate(valid_features, axis=1)
    # Silent processing - no shape logging
    return X_processed

def process_uploaded_file(patient_type, specific_file=None):
    """Process uploaded file(s) and generate predictions. If specific_file is provided, only that file is processed."""
    global hematology_predictions, solid_predictions, last_prediction_time
    
    if not PUBLIC_DEMO_MODE and (hematology_model is None or solid_model is None):
        logger.error("Models not loaded")
        return []
    
    try:
        # Determine which files to process
        all_predictions = []

        files_to_process = []
        if specific_file:
            files_to_process = [specific_file]
        else:
            # If no specific file, scan Z: drive for latest file
            latest_file = find_latest_excel_file()
            if latest_file:
                files_to_process = [latest_file]

        for filepath in files_to_process:
            filename = os.path.basename(filepath)
            # Read file based on extension
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(filepath)
            elif ext == '.csv':
                df = pd.read_csv(filepath)
            else:
                continue
            logger.info(f"Processing {len(df)} patients from {os.path.basename(filepath)}")

            # Check required columns
            required_columns = ['PATIENT_NAME', 'MRN', 'LOCATION', 'ROOM', 'ADMISSION_ORDER']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.warning(f"Missing required columns in {filename}: {missing_columns}")
                continue

            # Convert columns to lists (same as training preprocessing)
            # Silent processing
            for col in train_vital_sign_cols + train_lab_cols:
                if col in df.columns:
                    df[col] = df[col].apply(convert_string_to_list_of_floats)
                else:
                    logger.warning(f"Column '{col}' not found. Adding empty lists.")
                    df[col] = pd.Series([[] for _ in range(len(df))], index=df.index)

            for col in train_med_cols:
                if col in df.columns:
                    df[col] = df[col].apply(convert_string_to_list_of_meds)
                else:
                    logger.warning(f"Column '{col}' not found. Adding empty lists.")
                    df[col] = pd.Series([[] for _ in range(len(df))], index=df.index)

            file_predictions = []

            for index, row in df.iterrows():
                try:
                    # Start timing for validation
                    prediction_start_time = time.time()
                    
                    # Determine which model to use based on ADMISSION ORDER
                    admission_order = str(row['ADMISSION_ORDER']).strip()

                    if admission_order == "Admission Hematology":
                        model_type = "hematology"
                        preprocessing_params = globals().get('hematology_preprocessing_params', {
                            'max_length_vitals': 10,
                            'LAB_HISTORY_LENGTH': 4,
                            'max_med_lengths': {'ANTIBIOTICS': 5, 'NEUROLOGY_DRUGS': 5, 'CARDIOLOGY_DRUGS': 5, 'FUNGAL_DRUGS': 5},
                            'med_vocab': {"PAD": 0, "UNK": 1}
                        })
                    else:
                        model_type = "solid"
                        preprocessing_params = globals().get('solid_preprocessing_params', {
                            'max_length_vitals': 10,
                            'LAB_HISTORY_LENGTH': 4,
                            'max_med_lengths': {'ANTIBIOTICS': 5, 'NEUROLOGY_DRUGS': 5, 'CARDIOLOGY_DRUGS': 5, 'FUNGAL_DRUGS': 5},
                            'med_vocab': {"PAD": 0, "UNK": 1}
                        })

                    # Create a single-row DataFrame for processing
                    row_df = pd.DataFrame([row])

                    if PUBLIC_DEMO_MODE or hematology_model is None or solid_model is None:
                        prediction_prob = demo_probability_from_row(row, model_type)
                        risk_level = classify_risk_level(prediction_prob, model_type)
                        validation_results = {
                            'passed': True,
                            'errors': [],
                            'warnings': ['Public demo mode: scores are generated from a deterministic heuristic over synthetic data.']
                        }
                    else:
                        # Process features using the same logic as training
                        X_processed = process_features_tabular(row_df, preprocessing_params, is_train_phase=False)

                        # Apply imputation and scaling
                        if model_type == "hematology":
                            model = hematology_model
                            scaler = hematology_scaler
                            imputer = hematology_imputer
                        else:
                            model = solid_model
                            scaler = solid_scaler
                            imputer = solid_imputer

                        X_imputed = imputer.transform(X_processed)
                        X_scaled = scaler.transform(X_imputed)

                        # Make prediction
                        prediction_prob = model.predict_proba(X_scaled)[0][1]  # Probability of positive class
                        risk_level = classify_risk_level(prediction_prob, model_type)

                        # Comprehensive validation (silent - no logging)
                        validation_results = comprehensive_prediction_validation(
                            row, X_processed, prediction_prob, risk_level, model_type, prediction_start_time
                        )

                    # Create prediction result
                    diagnosis_value = ''
                    for diag_col in ['DIAGNOSIS', 'Diagnosis', 'PRIMARY_DIAGNOSIS', 'PRIMARY DIAGNOSIS', 'DIAGNOSIS_NAME']:
                        if diag_col in row and str(row[diag_col]).strip():
                            diagnosis_value = str(row[diag_col]).strip()
                            break
                    prediction_result = {
                        'patient_name': str(row['PATIENT_NAME']),
                        'mrn': str(row['MRN']),
                        'location': str(row['LOCATION']),
                        'room': str(row['ROOM']),
                        'admission_date': str(row['ADMISSION_DATE']) if 'ADMISSION_DATE' in row else '',
                        'admission_order': admission_order,
                        'diagnosis': diagnosis_value,
                        'model_used': model_type,
                        'icu_risk_probability': float(prediction_prob),
                        'risk_level': risk_level,
                        'prediction_timestamp': datetime.now().isoformat(),
                        'demo_mode': PUBLIC_DEMO_MODE,
                        'validation_passed': validation_results['passed'],
                        'validation_errors': validation_results['errors'],
                        'validation_warnings': validation_results['warnings']
                    }

                    file_predictions.append(prediction_result)

                except Exception as e:
                    logger.error(f"Error processing row {index}: {str(e)}")
                    continue

            all_predictions.extend(file_predictions)
            # Silent processing
        
        # Update global predictions
        hematology_predictions = [p for p in all_predictions if p['model_used'] == 'hematology']
        solid_predictions = [p for p in all_predictions if p['model_used'] == 'solid']
        last_prediction_time["predictions"] = datetime.now().isoformat()
        
        logger.info(f"✅ Successfully processed {len(all_predictions)} patients ({len(hematology_predictions)} hematology, {len(solid_predictions)} solid)")
        
        return all_predictions

    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise

def classify_risk_level(probability, model_type):
    """Classify risk level based on probability and cohort-specific thresholds."""
    prob_pct = float(probability) * 100
    if model_type == "hematology":
        # Hematology: 0-8 low, 9-70 medium, 71-100 high
        if prob_pct <= 8:
            return 'Low Risk (Safe)'
        elif prob_pct <= 70:
            return 'Medium Risk (Monitoring)'
        else:
            return 'High Risk (Urgent)'
    else:
        # Non-hematology: 0-32 low, 33-88 medium, 89-100 high
        if prob_pct <= 32:
            return 'Low Risk (Safe)'
        elif prob_pct <= 88:
            return 'Medium Risk (Monitoring)'
        else:
            return 'High Risk (Urgent)'

# Model Validation Functions
def validate_prediction_probability(probability, model_type):
    """Validate that prediction probability is within expected range"""
    try:
        prob = float(probability)
        if not (VALIDATION_CONFIG['min_probability'] <= prob <= VALIDATION_CONFIG['max_probability']):
            logger.error(f"Invalid probability {prob} for {model_type} model")
            return False, f"Probability {prob} outside valid range [{VALIDATION_CONFIG['min_probability']}, {VALIDATION_CONFIG['max_probability']}]"
        return True, "Probability validation passed"
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid probability type for {model_type} model: {e}")
        return False, f"Invalid probability type: {type(probability)}"

def validate_risk_level_classification(risk_level, model_type):
    """Validate that risk level classification is correct"""
    if risk_level not in VALIDATION_CONFIG['expected_risk_levels']:
        logger.error(f"Invalid risk level '{risk_level}' for {model_type} model")
        return False, f"Risk level '{risk_level}' not in expected values: {VALIDATION_CONFIG['expected_risk_levels']}"
    return True, "Risk level validation passed"

def validate_feature_matrix(X_processed, model_type):
    """Validate feature matrix shape and content"""
    try:
        if not isinstance(X_processed, np.ndarray):
            return False, f"Feature matrix is not numpy array, got {type(X_processed)}"
        
        if X_processed.ndim != 2:
            return False, f"Feature matrix should be 2D, got {X_processed.ndim}D"
        
        if X_processed.shape[0] == 0:
            return False, "Feature matrix has 0 rows"
        
        expected_features = VALIDATION_CONFIG['required_features_count'].get(model_type, 50)
        # Silent validation - no logging of feature count mismatches
        
        # Check for NaN or infinite values - but be more lenient
        nan_count = np.sum(np.isnan(X_processed))
        inf_count = np.sum(np.isinf(X_processed))
        
        # Silent validation - no logging of NaN/inf counts
        
        return True, f"Feature matrix validation passed: shape {X_processed.shape} (NaN: {nan_count}, Inf: {inf_count})"
    
    except Exception as e:
        logger.error(f"Error validating feature matrix for {model_type}: {e}")
        return False, f"Feature matrix validation error: {str(e)}"

def validate_model_consistency(prediction_prob, risk_level, model_type):
    """Validate that probability and risk level are consistent"""
    prob = float(prediction_prob)
    
    # Check if risk level matches probability
    expected_risk = classify_risk_level(prob, model_type)
    if risk_level != expected_risk:
        logger.error(f"Risk level inconsistency: prob={prob}, got='{risk_level}', expected='{expected_risk}'")
        return False, f"Risk level '{risk_level}' doesn't match probability {prob} (expected '{expected_risk}')"
    
    return True, "Model consistency validation passed"

def validate_input_data(row, required_columns):
    """Validate input data row"""
    try:
        # Check required columns exist
        missing_cols = [col for col in required_columns if col not in row.index]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for empty/null critical fields
        critical_fields = ['PATIENT_NAME', 'MRN', 'ADMISSION_ORDER']
        for field in critical_fields:
            if field in row and (pd.isna(row[field]) or str(row[field]).strip() == ''):
                return False, f"Empty/null value in critical field: {field}"
        
        # Validate admission order (any value is valid, we just need it to exist)
        admission_order = str(row['ADMISSION_ORDER']).strip()
        if not admission_order:
            return False, f"Empty admission order value"
        
        return True, "Input data validation passed"
    
    except Exception as e:
        logger.error(f"Error validating input data: {e}")
        return False, f"Input data validation error: {str(e)}"

def validate_prediction_timing(start_time, model_type):
    """Validate that prediction doesn't take too long"""
    elapsed_time = time.time() - start_time
    if elapsed_time > VALIDATION_CONFIG['max_prediction_time_seconds']:
        logger.warning(f"Prediction for {model_type} took {elapsed_time:.2f}s (max: {VALIDATION_CONFIG['max_prediction_time_seconds']}s)")
        return False, f"Prediction too slow: {elapsed_time:.2f}s"
    return True, f"Prediction timing OK: {elapsed_time:.3f}s"

def comprehensive_prediction_validation(row, X_processed, prediction_prob, risk_level, model_type, start_time):
    """Comprehensive validation of a single prediction"""
    validation_results = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'details': {}
    }
    
    # Input data validation
    required_columns = ['PATIENT_NAME', 'MRN', 'LOCATION', 'ROOM', 'ADMISSION_ORDER']
    input_valid, input_msg = validate_input_data(row, required_columns)
    if not input_valid:
        validation_results['passed'] = False
        validation_results['errors'].append(f"Input validation failed: {input_msg}")
    else:
        validation_results['details']['input_validation'] = input_msg
    
    # Feature matrix validation
    feature_valid, feature_msg = validate_feature_matrix(X_processed, model_type)
    if not feature_valid:
        validation_results['passed'] = False
        validation_results['errors'].append(f"Feature validation failed: {feature_msg}")
    else:
        validation_results['details']['feature_validation'] = feature_msg
    
    # Probability validation
    prob_valid, prob_msg = validate_prediction_probability(prediction_prob, model_type)
    if not prob_valid:
        validation_results['passed'] = False
        validation_results['errors'].append(f"Probability validation failed: {prob_msg}")
    else:
        validation_results['details']['probability_validation'] = prob_msg
    
    # Risk level validation
    risk_valid, risk_msg = validate_risk_level_classification(risk_level, model_type)
    if not risk_valid:
        validation_results['passed'] = False
        validation_results['errors'].append(f"Risk level validation failed: {risk_msg}")
    else:
        validation_results['details']['risk_validation'] = risk_msg
    
    # Model consistency validation
    consistency_valid, consistency_msg = validate_model_consistency(prediction_prob, risk_level, model_type)
    if not consistency_valid:
        validation_results['passed'] = False
        validation_results['errors'].append(f"Consistency validation failed: {consistency_msg}")
    else:
        validation_results['details']['consistency_validation'] = consistency_msg
    
    # Timing validation
    timing_valid, timing_msg = validate_prediction_timing(start_time, model_type)
    if not timing_valid:
        validation_results['warnings'].append(f"Timing warning: {timing_msg}")
    else:
        validation_results['details']['timing_validation'] = timing_msg
    
    return validation_results

# Validation logging removed - validation runs silently in background

def simulate_predictions(patient_type):
    """Generate deterministic synthetic predictions when no demo file is available."""
    global hematology_predictions, solid_predictions, last_prediction_time

    np.random.seed(42)

    if patient_type == 'hematology':
        num_patients = 18
        predictions = []

        for i in range(num_patients):
            patient = {
                'patient_name': f'Demo Hematology Patient {i + 1}',
                'mrn': f'HEM_{i + 1:03d}',
                'location': f'Ward {np.random.randint(1, 5)}',
                'room': f'H{np.random.randint(100, 500)}',
                'admission_order': 'Admission Hematology',
                'model_used': 'hematology',
                'icu_risk_probability': np.random.beta(2, 5),
                'risk_level': '',
                'prediction_timestamp': datetime.now().isoformat(),
                'demo_mode': True,
                'validation_passed': True,
                'validation_errors': [],
                'validation_warnings': ['Public demo mode fallback: synthetic rows generated in memory.']
            }
            patient['risk_level'] = classify_risk_level(patient['icu_risk_probability'], 'hematology')
            predictions.append(patient)

        hematology_predictions = predictions
        last_prediction_time["hematology"] = datetime.now().isoformat()

    elif patient_type == 'solid':
        num_patients = 22
        predictions = []

        for i in range(num_patients):
            patient = {
                'patient_name': f'Demo Solid Patient {i + 1}',
                'mrn': f'SOL_{i + 1:03d}',
                'location': f'Ward {np.random.randint(1, 5)}',
                'room': f'S{np.random.randint(100, 500)}',
                'admission_order': 'Admission Oncology',
                'model_used': 'solid',
                'icu_risk_probability': np.random.beta(2.5, 4),
                'risk_level': '',
                'prediction_timestamp': datetime.now().isoformat(),
                'demo_mode': True,
                'validation_passed': True,
                'validation_errors': [],
                'validation_warnings': ['Public demo mode fallback: synthetic rows generated in memory.']
            }
            patient['risk_level'] = classify_risk_level(patient['icu_risk_probability'], 'solid')
            predictions.append(patient)

        solid_predictions = predictions
        last_prediction_time["solid"] = datetime.now().isoformat()

    logger.info(f"Generated {len(predictions)} predictions for {patient_type} patients")

def schedule_auto_scan():
    """Schedule automatic scanning every 6 hours"""
    import threading
    import time
    
    def auto_scan_worker():
        while True:
            try:
                logger.info("Running scheduled auto-scan...")
                auto_scan_and_process()
                # Sleep for 6 hours (6 * 60 * 60 seconds)
                time.sleep(6 * 60 * 60)
            except Exception as e:
                logger.error(f"Error in scheduled auto-scan: {str(e)}")
                # Sleep for 1 hour before retrying if there's an error
                time.sleep(60 * 60)
    
    # Start the auto-scan thread
    auto_scan_thread = threading.Thread(target=auto_scan_worker, daemon=True)
    auto_scan_thread.start()
    logger.info("Auto-scan scheduler started (every 6 hours)")


if PUBLIC_DEMO_MODE:
    logger.info("Public demo mode enabled. Loading synthetic demo files from %s", DEMO_DATA_FOLDER)
    if not auto_scan_and_process():
        simulate_predictions('hematology')
        simulate_predictions('solid')
elif hematology_model is None or solid_model is None:
    logger.warning("Models not loaded, using simulated data")
    simulate_predictions('hematology')
    simulate_predictions('solid')

if __name__ == '__main__':
    # Start the auto-scan scheduler
    schedule_auto_scan()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 
