import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle

from config import ZIGZAG_CONFIG, SEQUENCE_CONFIG, OUTPUT_CONFIG, TECHNICAL_INDICATORS
from zigzag import calculate_zigzag, get_zigzag_statistics, validate_zigzag_points, get_zigzag_type_mapping
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
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {csv_path}")
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"CSV missing columns: {missing}")
    
    print(f"  Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Missing values:\n{df[required_cols].isnull().sum()}")
    
    null_count = df[required_cols].isnull().sum().sum()
    if null_count > 0:
        print(f"  Filling {null_count} missing values...")
        df[required_cols] = df[required_cols].fillna(method='ffill')
    
    return df


def process_single_file(csv_path: str, output_dir: Path, timeframe: str = '15m') -> dict:
    """
    Complete pipeline for single CSV file.
    
    Processing pipeline:
    1. Load and validate input data
    2. Calculate Zigzag turning points with HH/LL/HL/LH classification
    3. Extract 13 technical features
    4. Normalize features using StandardScaler
    5. Generate classification labels (4 classes + unknown)
    6. Generate regression labels (distance to next turning point)
    7. Create sequences for LSTM (30-bar sliding windows)
    8. Save all outputs and statistics
    
    Args:
        csv_path (str): Input CSV file path
        output_dir (Path): Output directory path
        timeframe (str): Timeframe identifier for output files (e.g., '15m', '1h')
    
    Returns:
        dict: Processing results containing statistics and metadata
    """
    print(f"\n{'='*90}")
    print(f"Processing {timeframe} data")
    print(f"{'='*90}")
    
    results = {
        'timeframe': timeframe,
        'input_file': csv_path,
        'output_dir': str(output_dir),
        'timestamp': datetime.now().isoformat(),
        'success': False,
        'error': None
    }
    
    try:
        print("\n[STEP 1] Loading and validating data...")
        df = load_and_validate_data(csv_path)
        print(f"  Successfully loaded {len(df)} records")
        
        print("\n[STEP 2] Calculating Zigzag turning points with HH/LL/HL/LH classification...")
        threshold = ZIGZAG_CONFIG['threshold_percent']
        df_zigzag, turning_points = calculate_zigzag(df, threshold=threshold)
        
        zigzag_stats = get_zigzag_statistics(turning_points)
        print(f"  Zigzag calculation complete: {zigzag_stats['total_points']} turning points found")
        print(f"    - HH (Higher High): {zigzag_stats['hh_count']} ({zigzag_stats['hh_percent']}%)")
        print(f"    - LL (Lower Low): {zigzag_stats['ll_count']} ({zigzag_stats['ll_percent']}%)")
        print(f"    - HL (Higher Low): {zigzag_stats['hl_count']} ({zigzag_stats['hl_percent']}%)")
        print(f"    - LH (Lower High): {zigzag_stats['lh_count']} ({zigzag_stats['lh_percent']}%)")
        print(f"    - Trend continuation ratio: {zigzag_stats['continuation_ratio']}%")
        print(f"    - Trend reversal ratio: {zigzag_stats['reversal_ratio']}%")
        
        if not validate_zigzag_points(df_zigzag, turning_points):
            print("  Warning: Some zigzag validation issues detected")
        
        print("\n[STEP 3] Extracting 13 technical features...")
        extractor = FeatureExtractor(lookback=ZIGZAG_CONFIG['lookback'])
        df_features = extractor.create_technical_features(df_zigzag)
        print(f"  Features extracted: {df_features.shape}")
        
        print("\n[STEP 4] Normalizing features using StandardScaler...")
        df_normalized, scaler = extractor.normalize_features(df_features, fit=True)
        
        feature_stats = extractor.get_feature_statistics(df_features)
        extractor.validate_features(df_normalized)
        
        print("\n[STEP 5] Preparing training sequences and labels...")
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
        
        print(f"\n  Sequence shapes:")
        print(f"    - X_class: {X_class.shape} (sequences, timesteps, features)")
        print(f"    - y_class: {y_class.shape} (classification labels)")
        print(f"    - X_reg: {X_reg.shape}")
        print(f"    - y_reg: {y_reg.shape} (regression labels - distance to next point)")
        
        print("\n[STEP 6] Saving outputs to disk...")
        base_name = timeframe
        
        np.save(output_dir / f"{base_name}_X_sequences.npy", X_class)
        np.save(output_dir / f"{base_name}_y_class.npy", y_class)
        np.save(output_dir / f"{base_name}_y_reg.npy", y_reg)
        
        with open(output_dir / f"{base_name}_scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(output_dir / f"{base_name}_zigzag_points.json", 'w') as f:
            json.dump(turning_points, f, indent=2)
        
        statistics = {
            'metadata': {
                'timeframe': timeframe,
                'input_file': csv_path,
                'output_dir': str(output_dir),
                'timestamp': datetime.now().isoformat(),
                'total_raw_samples': len(df),
                'total_sequences': len(X_class)
            },
            'sequence_config': {
                'sequence_length': SEQUENCE_CONFIG['sequence_length'],
                'n_features': training_data['n_features'],
                'feature_names': training_data['feature_names']
            },
            'data_shapes': {
                'X_class': X_class.shape,
                'y_class': y_class.shape,
                'X_reg': X_reg.shape,
                'y_reg': y_reg.shape
            },
            'zigzag_analysis': {
                'total_turning_points': zigzag_stats['total_points'],
                'hh_count': zigzag_stats['hh_count'],
                'hh_percent': zigzag_stats['hh_percent'],
                'll_count': zigzag_stats['ll_count'],
                'll_percent': zigzag_stats['ll_percent'],
                'hl_count': zigzag_stats['hl_count'],
                'hl_percent': zigzag_stats['hl_percent'],
                'lh_count': zigzag_stats['lh_count'],
                'lh_percent': zigzag_stats['lh_percent'],
                'continuation_ratio_percent': zigzag_stats['continuation_ratio'],
                'reversal_ratio_percent': zigzag_stats['reversal_ratio'],
                'threshold_percent': threshold
            },
            'classification_labels': training_data['class_stats'],
            'regression_labels': training_data['reg_stats'],
            'feature_statistics': feature_stats,
            'output_files': [
                f"{base_name}_X_sequences.npy",
                f"{base_name}_y_class.npy",
                f"{base_name}_y_reg.npy",
                f"{base_name}_scaler.pkl",
                f"{base_name}_zigzag_points.json",
                f"{base_name}_statistics.json"
            ]
        }
        
        with open(output_dir / f"{base_name}_statistics.json", 'w') as f:
            json.dump(statistics, f, indent=2)
        
        print(f"  Outputs saved:")
        print(f"    - {base_name}_X_sequences.npy: {X_class.shape}")
        print(f"    - {base_name}_y_class.npy: {y_class.shape}")
        print(f"    - {base_name}_y_reg.npy: {y_reg.shape}")
        print(f"    - {base_name}_scaler.pkl")
        print(f"    - {base_name}_zigzag_points.json: {len(turning_points)} points")
        print(f"    - {base_name}_statistics.json")
        
        results['success'] = True
        results['statistics'] = statistics
        
    except Exception as e:
        results['error'] = str(e)
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    return results


def main():
    """
    Main execution function for complete step2 pipeline.
    """
    print("\n" + "="*90)
    print("ZIGZAG LSTM PREDICTOR - STEP 2: FEATURE ENGINEERING")
    print("Four-class Zigzag Classification: HH (0), LL (1), HL (2), LH (3)")
    print("="*90)
    print(f"Start time: {datetime.now()}")
    
    output_dir = setup_output_directory()
    
    base_path = Path(r"C:\Users\zong\PycharmProjects\zigzag-lstm-predictor\step1_data_extraction\step1_output")
    
    input_files = {
        'BTC_15m': base_path / "BTC_15m.csv",
        'BTC_1h': base_path / "BTC_1h.csv"
    }
    
    print(f"\nProcessing input files:")
    for tf, path in input_files.items():
        exists = "[EXISTS]" if path.exists() else "[NOT FOUND]"
        print(f"  {tf}: {path} {exists}")
    
    all_results = {}
    
    for timeframe, csv_path in input_files.items():
        results = process_single_file(str(csv_path), output_dir, timeframe=timeframe)
        all_results[timeframe] = results
    
    print("\n" + "="*90)
    print("EXECUTION SUMMARY")
    print("="*90)
    
    for timeframe, results in all_results.items():
        status = "SUCCESS" if results['success'] else "FAILED"
        print(f"\n{timeframe}: {status}")
        
        if results['success']:
            stats = results['statistics']
            print(f"  Metadata:")
            print(f"    Raw samples: {stats['metadata']['total_raw_samples']}")
            print(f"    Total sequences: {stats['metadata']['total_sequences']}")
            
            print(f"  Zigzag:")
            print(f"    Turning points: {stats['zigzag_analysis']['total_turning_points']}")
            print(f"    HH: {stats['zigzag_analysis']['hh_count']} ({stats['zigzag_analysis']['hh_percent']}%)")
            print(f"    LL: {stats['zigzag_analysis']['ll_count']} ({stats['zigzag_analysis']['ll_percent']}%)")
            print(f"    HL: {stats['zigzag_analysis']['hl_count']} ({stats['zigzag_analysis']['hl_percent']}%)")
            print(f"    LH: {stats['zigzag_analysis']['lh_count']} ({stats['zigzag_analysis']['lh_percent']}%)")
            
            class_stats = stats['classification_labels']
            print(f"  Classification labels:")
            print(f"    Valid sequences: {class_stats['valid']}/{stats['metadata']['total_sequences']} "
                  f"({class_stats['valid_percent']}%)")
            if 'class_details' in class_stats:
                for label, info in class_stats['class_details'].items():
                    print(f"    {label}: {info['count']} ({info['percentage']}%)")
            
            reg_stats = stats['regression_labels']
            print(f"  Regression labels:")
            print(f"    Valid sequences: {reg_stats['valid']}/{stats['metadata']['total_sequences']} "
                  f"({reg_stats['valid_percent']}%)")
            print(f"    Mean distance: {reg_stats['mean_distance']:.1f} bars")
            print(f"    Distance range: {reg_stats['min_distance']}-{reg_stats['max_distance']} bars")
        else:
            print(f"  Error: {results['error']}")
    
    execution_log = {
        'start_time': datetime.now().isoformat(),
        'config': {
            'zigzag': ZIGZAG_CONFIG,
            'sequence': SEQUENCE_CONFIG,
            'output': OUTPUT_CONFIG,
            'technical_indicators': TECHNICAL_INDICATORS
        },
        'results': all_results
    }
    
    log_path = output_dir / "execution_summary.json"
    with open(log_path, 'w') as f:
        json.dump(execution_log, f, indent=2)
    
    success_count = sum(1 for r in all_results.values() if r['success'])
    total_count = len(all_results)
    
    print(f"\n{'='*90}")
    print(f"Final Result: {success_count}/{total_count} files processed successfully")
    print(f"Execution summary saved: {log_path}")
    print(f"End time: {datetime.now()}")
    print("="*90)


if __name__ == "__main__":
    main()
