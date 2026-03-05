"""
Extract Additional Feature Sets from Existing Feature Selection Results
========================================================================

This script extracts top 750, 1000, 1250, 1500, 1750, 2000, and 2250 features
from the already-computed feature selection rankings.

NO need to re-run feature selection - just slices existing ranked results.

Usage:
    python extract_additional_features.py --dataset hematology
    python extract_additional_features.py --dataset solid
    python extract_additional_features.py --dataset all  (runs both)
"""

import pandas as pd
import joblib
import os
import argparse
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_COUNTS = [750, 1000, 1250, 1500, 1750, 2000, 2250]
DATASETS = ['hematology', 'solid']

# ============================================================================
# FUNCTIONS
# ============================================================================

def extract_features_for_dataset(dataset_type):
    """
    Extract additional feature sets from existing feature selection results.

    Parameters:
    -----------
    dataset_type : str
        Either 'hematology' or 'solid'
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTING ADDITIONAL FEATURES FOR: {dataset_type.upper()}")
    print(f"{'='*70}\n")

    # Define paths
    results_dir = f"{dataset_type}_feature_selection_results"
    results_csv = os.path.join(results_dir, f"{dataset_type}_feature_selection_results.csv")

    # Check if results file exists
    if not os.path.exists(results_csv):
        print(f"[ERROR] Feature selection results not found: {results_csv}")
        print(f"[ERROR] Please run feature_selection_icu.py first for {dataset_type} dataset")
        return False

    # Load complete feature rankings
    print(f"[1/3] Loading feature selection results from: {results_csv}")
    results_df = pd.read_csv(results_csv)
    total_features = len(results_df)
    print(f"      Total features available: {total_features}")

    # Extract each feature count
    print(f"\n[2/3] Extracting feature sets:")
    extracted_counts = []

    for n_features in FEATURE_COUNTS:
        # Check if we have enough features
        if n_features > total_features:
            print(f"      [SKIP] {n_features} features - Only {total_features} available")
            continue

        # Extract top N features
        top_features = results_df.head(n_features)

        # Create feature set dictionary
        feature_set = {
            'n_features': n_features,
            'indices': top_features['feature_index'].tolist(),
            'names': top_features['feature_name'].tolist(),
            'scores': top_features['combined_score'].tolist(),
            'F_scores': top_features['F_score'].tolist(),
            'p_values': top_features['p_value'].tolist(),
            'RF_importance': top_features['RF_combined_importance'].tolist()
        }

        # Save to pickle file
        output_file = os.path.join(results_dir, f"{dataset_type}_top{n_features}_features.pkl")
        joblib.dump(feature_set, output_file)

        print(f"      [OK] Extracted top {n_features} features -> {output_file}")
        extracted_counts.append(n_features)

    # Summary
    print(f"\n[3/3] Extraction Summary:")
    print(f"      Dataset: {dataset_type}")
    print(f"      Feature counts extracted: {extracted_counts}")
    print(f"      Total feature sets saved: {len(extracted_counts)}")
    print(f"      Output directory: {results_dir}")
    print(f"\n[SUCCESS] Feature extraction completed for {dataset_type}!\n")

    return True


def verify_extracted_features(dataset_type):
    """
    Verify that all expected feature files exist.

    Parameters:
    -----------
    dataset_type : str
        Either 'hematology' or 'solid'
    """
    print(f"\nVerifying extracted features for {dataset_type}:")
    results_dir = f"{dataset_type}_feature_selection_results"

    # Load original results to check max features
    results_csv = os.path.join(results_dir, f"{dataset_type}_feature_selection_results.csv")
    results_df = pd.read_csv(results_csv)
    total_features = len(results_df)

    all_exist = True
    for n_features in FEATURE_COUNTS:
        if n_features > total_features:
            continue

        file_path = os.path.join(results_dir, f"{dataset_type}_top{n_features}_features.pkl")
        if os.path.exists(file_path):
            # Load and verify
            feature_set = joblib.load(file_path)
            if len(feature_set['indices']) == n_features:
                print(f"  [OK] {n_features} features verified")
            else:
                print(f"  [ERROR] {n_features} features file corrupted!")
                all_exist = False
        else:
            print(f"  [MISSING] {n_features} features file not found")
            all_exist = False

    return all_exist


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract additional feature sets from existing feature selection results'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['hematology', 'solid', 'all'],
        default='all',
        help='Dataset to extract features for (default: all)'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("ADDITIONAL FEATURE EXTRACTION TOOL")
    print("="*70)
    print(f"Feature counts to extract: {FEATURE_COUNTS}")
    print(f"Datasets: {args.dataset}")
    print("="*70)

    # Determine which datasets to process
    if args.dataset == 'all':
        datasets_to_process = DATASETS
    else:
        datasets_to_process = [args.dataset]

    # Process each dataset
    success_count = 0
    for dataset in datasets_to_process:
        if extract_features_for_dataset(dataset):
            if verify_extracted_features(dataset):
                success_count += 1

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Datasets processed successfully: {success_count}/{len(datasets_to_process)}")

    if success_count == len(datasets_to_process):
        print("\n[SUCCESS] All feature extractions completed successfully!")
        print("\nNext steps:")
        print("  1. Run: python extended_feature_selection_training.py --dataset hematology")
        print("  2. Run: python extended_feature_selection_training.py --dataset solid")
    else:
        print("\n[WARNING] Some extractions failed. Please check error messages above.")

    print("="*70 + "\n")


if __name__ == "__main__":
    main()
