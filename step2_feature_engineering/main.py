import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle

from config import ZIGZAG_CONFIG, SEQUENCE_CONFIG, OUTPUT_CONFIG, TECHNICAL_INDICATORS
from zigzag import calculate_zigzag, get_zigzag_statistics, validate_zigzag_points
from feature_extractor import FeatureExtractor
from sequence_builder import prepare_training_data


def setup_output_directory():
    """
    Create output directory if it doesn't exist.
    """
    output_dir = Path(OUTPUT_CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {output_dir.absolute()}")
    return output_dir


def load_and_validate_data(csv_path: str) -> pd.DataFrame:
    """
    Load CSV file and validate required columns.
    
    Args:
        csv_path (str): Path to input CSV file
    
    Returns:
        pd.DataFrame: Loaded data
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Missing values:\n{df[required_cols].isnull().sum()}")
    
    return df


def process_single_file(csv_path: str, output_dir: Path, timeframe: str = '15m') -> dict:
    """
    Complete pipeline for single CSV file.
    
    Processing steps:
    1. Load and validate input data
    2. Calculate Zigzag turning points
    3. Extract technical features
    4. Normalize features
    5. Generate sequences and labels
    6. Save all outputs
    
    Args:
        csv_path (str): Input CSV file path
        output_dir (Path): Output directory path
        timeframe (str): Timeframe identifier for output files
    
    Returns:
        dict: Processing results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Processing {timeframe} data")
    print(f"{'='*80}")
    
    results = {
        'timeframe': timeframe,
        'input_file': csv_path,
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'error': None
    }
    
    try:
        # Step 1: Load data
        print("\n[STEP 1] Loading and validating data...")
        df = load_and_validate_data(csv_path)
        
        # Step 2: Calculate Zigzag
        print("\n[STEP 2] Calculating Zigzag turning points...")
        threshold = ZIGZAG_CONFIG['threshold_percent']
        df_zigzag, turning_points = calculate_zigzag(df, threshold=threshold)
        
        zigzag_stats = get_zigzag_statistics(turning_points)
        print(f"Zigzag calculation complete.")
        print(f"  Total turning points: {zigzag_stats['total_points']}")
        print(f"  HH (High-High): {zigzag_stats['hh_count']} ({zigzag_stats['hh_percentage']}%)")
        print(f"  LL (Low-Low): {zigzag_stats['ll_count']} ({zigzag_stats['ll_percentage']}%)")
        
        if not validate_zigzag_points(df_zigzag, turning_points):
            print("Warning: Zigzag point validation issues detected")
        
        # Step 3: Extract technical features
        print("\n[STEP 3] Extracting technical features...")
        extractor = FeatureExtractor(lookback=ZIGZAG_CONFIG['lookback'])
        df_features = extractor.create_technical_features(df_zigzag)
        
        # Step 4: Normalize features
        print("\n[STEP 4] Normalizing features...")
        df_normalized, scaler = extractor.normalize_features(df_features, fit=True)
        
        feature_stats = extractor.get_feature_statistics(df_features)
        extractor.validate_features(df_normalized)
        
        # Step 5: Generate sequences and labels
        print("\n[STEP 5] Preparing training data...")
        training_data = prepare_training_data(
            df_normalized,
            df_zigzag,
            turning_points,
            sequence_length=SEQUENCE_CONFIG['sequence_length'],
            feature_names=TECHNICAL_INDICATORS
        )
        
        X_class = training_data['X_class']
        y_class = training_data['y_class']
        X_reg = training_data['X_reg']
        y_reg = training_data['y_reg']
        
        print(f"\nTraining data summary:")
        print(f"  Classification - X shape: {X_class.shape}, y shape: {y_class.shape}")
        print(f"  Regression - X shape: {X_reg.shape}, y shape: {y_reg.shape}")
        print(f"  Classification label stats: {training_data['class_stats']}")
        print(f"  Regression label stats: {training_data['reg_stats']}")
        
        # Step 6: Save outputs
        print("\n[STEP 6] Saving outputs...")
        
        base_name = f"{timeframe}"
        
        np.save(output_dir / f"{base_name}_X_sequences.npy", X_class)
        np.save(output_dir / f"{base_name}_y_class.npy", y_class)
        np.save(output_dir / f"{base_name}_y_reg.npy", y_reg)
        
        with open(output_dir / f"{base_name}_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(output_dir / f"{base_name}_zigzag_points.json", 'w') as f:
            json.dump(turning_points, f, indent=2)
        
        # Prepare statistics
        statistics = {
            'timeframe': timeframe,
            'input_file': csv_path,
            'total_samples': len(df),
            'sequence_length': SEQUENCE_CONFIG['sequence_length'],
            'total_sequences': len(X_class),
            'n_features': training_data['n_features'],
            'feature_names': training_data['feature_names'],
            'zigzag': zigzag_stats,
            'feature_statistics': feature_stats,
            'classification_labels': training_data['class_stats'],
            'regression_labels': training_data['reg_stats'],
            'data_shapes': {
                'X_class': X_class.shape,
                'y_class': y_class.shape,
                'X_reg': X_reg.shape,
                'y_reg': y_reg.shape
            },
            'output_files': [
                f"{base_name}_X_sequences.npy",
                f"{base_name}_y_class.npy",
                f"{base_name}_y_reg.npy",
                f"{base_name}_scaler.pkl",
                f"{base_name}_zigzag_points.json"
            ]
        }
        
        with open(output_dir / f"{base_name}_statistics.json", 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"\nOutputs saved to: {output_dir}")
        print(f"  - X_sequences: {X_class.shape}")
        print(f"  - y_class labels: {y_class.shape}")
        print(f"  - y_reg labels: {y_reg.shape}")
        print(f"  - Scaler: {base_name}_scaler.pkl")
        print(f"  - Zigzag points: {len(turning_points)} points")
        print(f"  - Statistics: {base_name}_statistics.json")
        
        results['success'] = True
        results['statistics'] = statistics
        
    except Exception as e:
        results['error'] = str(e)
        print(f"\nError processing {timeframe}: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def main():
    """
    Main execution function.
    """
    print("\n" + "="*80)
    print("ZIGZAG LSTM PREDICTOR - STEP 2: FEATURE ENGINEERING")
    print("="*80)
    print(f"Start time: {datetime.now()}")
    
    # Setup
    output_dir = setup_output_directory()
    
    # Input file paths
    base_path = Path(r"C:\Users\zong\PycharmProjects\zigzag-lstm-predictor\step1_data_extraction\step1_output")
    
    input_files = {
        'BTC_15m': base_path / "BTC_15m.csv",
        'BTC_1h': base_path / "BTC_1h.csv"
    }
    
    # Process each file
    all_results = {}
    
    for timeframe, csv_path in input_files.items():
        results = process_single_file(str(csv_path), output_dir, timeframe=timeframe)
        all_results[timeframe] = results
    
    # Save execution log
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    execution_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'zigzag': ZIGZAG_CONFIG,
            'sequence': SEQUENCE_CONFIG,
            'output': OUTPUT_CONFIG
        },
        'results': all_results
    }
    
    log_path = output_dir / "EXECUTION_LOG.json"
    with open(log_path, 'w') as f:
        json.dump(execution_log, f, indent=2)
    
    success_count = sum(1 for r in all_results.values() if r['success'])
    total_count = len(all_results)
    
    print(f"\nProcessing results: {success_count}/{total_count} files successful")
    
    for timeframe, results in all_results.items():
        status = "SUCCESS" if results['success'] else "FAILED"
        print(f"  {timeframe}: {status}")
        if results['success']:
            stats = results['statistics']
            print(f"    - Sequences: {stats['total_sequences']}")
            print(f"    - Zigzag points: {stats['zigzag']['total_points']}")
        else:
            print(f"    - Error: {results['error']}")
    
    print(f"\nExecution log saved: {log_path}")
    print(f"End time: {datetime.now()}")
    print("\nFeature engineering pipeline complete.")


if __name__ == "__main__":
    main()
