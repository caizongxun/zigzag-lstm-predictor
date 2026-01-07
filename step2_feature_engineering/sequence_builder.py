import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional


def generate_classification_labels(df: pd.DataFrame, turning_points: List[Dict]) -> np.ndarray:
    """
    Generate four-class classification labels for next turning point type.
    
    Label mapping (4 classes + unknown):
    0 = HH (Higher High) - uptrend continuation
    1 = LL (Lower Low) - downtrend continuation
    2 = HL (Higher Low) - uptrend reversal signal
    3 = LH (Lower High) - downtrend reversal signal
    -1 = Unknown (no next turning point in remaining data)
    
    For each row i, label is the type of the NEXT turning point after i.
    This allows the model to predict both trend direction and reversal signals.
    
    Args:
        df (pd.DataFrame): DataFrame with zigzag_type column
        turning_points (List[Dict]): List of turning points with indices and types
    
    Returns:
        np.ndarray: Classification labels array of shape (len(df),)
    """
    n = len(df)
    labels = np.full(n, -1, dtype=np.int32)
    
    type_to_label = {'HH': 0, 'LL': 1, 'HL': 2, 'LH': 3}
    
    if not turning_points:
        print("Warning: No turning points provided for classification labels")
        return labels
    
    tp_indices = sorted([tp['index'] for tp in turning_points])
    tp_dict = {tp['index']: tp['type'] for tp in turning_points}
    
    for i in range(n):
        future_points = [idx for idx in tp_indices if idx > i]
        
        if future_points:
            next_point_idx = future_points[0]
            next_type = tp_dict[next_point_idx]
            labels[i] = type_to_label.get(next_type, -1)
    
    return labels


def generate_regression_labels(df: pd.DataFrame, turning_points: List[Dict]) -> np.ndarray:
    """
    Generate regression labels for distance to next turning point.
    
    For each row i, calculate how many bars until the next zigzag turning point.
    If no next turning point exists, use -1 (will be filtered during training).
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        turning_points (List[Dict]): List of turning points with indices
    
    Returns:
        np.ndarray: Distance to next turning point (in bars), or -1 if none
    """
    n = len(df)
    labels = np.full(n, -1, dtype=np.int32)
    
    if not turning_points:
        print("Warning: No turning points provided for regression labels")
        return labels
    
    tp_indices = sorted([tp['index'] for tp in turning_points])
    
    for i in range(n):
        future_points = [idx for idx in tp_indices if idx > i]
        
        if future_points:
            next_point_idx = future_points[0]
            distance = next_point_idx - i
            labels[i] = distance
    
    return labels


def create_sequences(features: np.ndarray,
                    labels: np.ndarray,
                    sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series LSTM prediction.
    
    Converts (n_samples, n_features) to (n_sequences, sequence_length, n_features)
    
    Sequence construction:
    - For position i, extract features[i:i+sequence_length] as one sequence
    - Label for this sequence is labels[i+sequence_length-1] (label at end of sequence)
    - This creates overlapping sliding windows
    
    Example with sequence_length=3:
    features: [f0, f1, f2, f3, f4, f5]
    labels:   [l0, l1, l2, l3, l4, l5]
    
    Output sequences:
    X[0] = [f0, f1, f2] with label l2
    X[1] = [f1, f2, f3] with label l3
    X[2] = [f2, f3, f4] with label l4
    X[3] = [f3, f4, f5] with label l5
    
    Args:
        features (np.ndarray): Feature array of shape (n_samples, n_features)
        labels (np.ndarray): Label array of shape (n_samples,)
        sequence_length (int): Length of sequences (default: 30)
    
    Returns:
        tuple: (X sequences, y labels)
               X: shape (n_sequences, sequence_length, n_features)
               y: shape (n_sequences,)
    """
    if len(features) != len(labels):
        raise ValueError(f"Features and labels length mismatch: {len(features)} vs {len(labels)}")
    
    if sequence_length <= 0:
        raise ValueError(f"Sequence length must be positive: {sequence_length}")
    
    if len(features) < sequence_length:
        raise ValueError(f"Not enough data for sequence_length {sequence_length}. Data length: {len(features)}")
    
    n_samples = len(features)
    n_features = features.shape[1] if len(features.shape) > 1 else 1
    n_sequences = n_samples - sequence_length + 1
    
    X = np.zeros((n_sequences, sequence_length, n_features), dtype=np.float32)
    y = np.zeros(n_sequences, dtype=labels.dtype)
    
    for i in range(n_sequences):
        X[i] = features[i:i + sequence_length]
        y[i] = labels[i + sequence_length - 1]
    
    return X, y


def validate_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> bool:
    """
    Validate sequence integrity and consistency.
    
    Checks:
    1. X and y have matching first dimension (number of sequences)
    2. X has correct 3D shape (sequences, timesteps, features)
    3. X.shape[1] matches sequence_length
    4. No NaN values in X (after normalization)
    
    Args:
        X (np.ndarray): Feature sequences of shape (n_seq, sequence_length, n_features)
        y (np.ndarray): Labels of shape (n_seq,)
        sequence_length (int): Expected sequence length
    
    Returns:
        bool: True if validation passes
    """
    if len(X) != len(y):
        print(f"Error: X and y length mismatch - {len(X)} vs {len(y)}")
        return False
    
    if len(X.shape) != 3:
        print(f"Error: X should be 3D (samples, timesteps, features), got shape {X.shape}")
        return False
    
    if X.shape[1] != sequence_length:
        print(f"Error: Sequence length mismatch - {X.shape[1]} vs {sequence_length}")
        return False
    
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        print(f"Warning: {nan_count} NaN values found in X. This may cause training issues.")
    
    return True


def analyze_label_distribution(labels: np.ndarray, label_type: str = 'classification') -> Dict:
    """
    Analyze distribution of labels for class balance assessment.
    
    For classification labels, provides:
    - Total sequences
    - Valid sequences (not -1)
    - Distribution of each class (0, 1, 2, 3)
    - Percentage for each class
    - Class balance ratio (highest / lowest)
    
    For regression labels, provides:
    - Statistics about distances
    - Min, max, mean distance to next turning point
    
    Args:
        labels (np.ndarray): Labels array
        label_type (str): Type of labels ('classification' or 'regression')
    
    Returns:
        dict: Label distribution statistics
    """
    total = len(labels)
    valid_labels = labels[labels != -1]
    n_valid = len(valid_labels)
    n_invalid = total - n_valid
    
    if n_valid == 0:
        return {
            'total': total,
            'valid': 0,
            'invalid': total,
            'distribution': {},
            'class_balance': 0.0
        }
    
    if label_type == 'classification':
        unique, counts = np.unique(valid_labels, return_counts=True)
        distribution = {int(label): int(count) for label, count in zip(unique, counts)}
        
        class_percentages = {}
        for label in [0, 1, 2, 3]:
            count = distribution.get(label, 0)
            percentage = (count / n_valid) * 100
            class_percentages[f'class_{label}'] = {
                'count': count,
                'percentage': round(percentage, 2)
            }
        
        if len(distribution) > 0:
            max_count = max(distribution.values())
            min_count = min(distribution.values())
            balance_ratio = round(max_count / min_count, 2) if min_count > 0 else 0.0
        else:
            balance_ratio = 0.0
        
        return {
            'total': total,
            'valid': n_valid,
            'invalid': n_invalid,
            'valid_percent': round((n_valid / total) * 100, 2),
            'distribution': distribution,
            'class_details': class_percentages,
            'balance_ratio': balance_ratio
        }
    
    else:
        return {
            'total': total,
            'valid': n_valid,
            'invalid': n_invalid,
            'valid_percent': round((n_valid / total) * 100, 2),
            'mean_distance': float(np.mean(valid_labels)),
            'std_distance': float(np.std(valid_labels)),
            'min_distance': int(np.min(valid_labels)),
            'max_distance': int(np.max(valid_labels)),
            'median_distance': int(np.median(valid_labels))
        }


def prepare_training_data(features_df: pd.DataFrame,
                        zigzag_df: pd.DataFrame,
                        turning_points: List[Dict],
                        sequence_length: int = 30,
                        feature_names: Optional[List[str]] = None) -> Dict:
    """
    Complete workflow to prepare training data for LSTM model.
    
    Pipeline:
    1. Extract feature array from features_df
    2. Handle NaN values in features
    3. Generate classification labels (4-class: HH/LL/HL/LH + unknown)
    4. Generate regression labels (distance to next turning point)
    5. Create sequences using sliding window approach
    6. Validate sequences
    7. Analyze label distributions
    
    Args:
        features_df (pd.DataFrame): DataFrame with calculated and normalized features
        zigzag_df (pd.DataFrame): DataFrame with zigzag information
        turning_points (List[Dict]): List of turning points from zigzag calculation
        sequence_length (int): Sequence length for LSTM (default: 30)
        feature_names (Optional[List[str]]): Names of feature columns to use
    
    Returns:
        dict: Dictionary containing:
            - X_class: Classification sequences (n_seq, 30, n_features)
            - y_class: Classification labels (n_seq,) with values in [0, 1, 2, 3, -1]
            - X_reg: Regression sequences (n_seq, 30, n_features)
            - y_reg: Regression labels (n_seq,) with bar distances or -1
            - n_features: Number of features used
            - feature_names: List of feature column names
            - class_stats: Classification label statistics
            - reg_stats: Regression label statistics
            - valid_class_count: Number of valid (not -1) classification labels
            - valid_reg_count: Number of valid regression labels
    """
    if feature_names is None:
        feature_names = [
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'atr_14',
            'high_low_ratio', 'close_range', 'highest_20', 'lowest_20',
            'volume_sma_20', 'volume_ratio', 'returns', 'returns_volatility'
        ]
    
    print("\nPreparing training data...")
    print(f"  Features DataFrame shape: {features_df.shape}")
    print(f"  Zigzag DataFrame shape: {zigzag_df.shape}")
    print(f"  Turning points count: {len(turning_points)}")
    
    if len(features_df) != len(zigzag_df):
        raise ValueError(f"Features and zigzag DataFrames length mismatch: "
                        f"{len(features_df)} vs {len(zigzag_df)}")
    
    feature_columns = [col for col in feature_names if col in features_df.columns]
    if len(feature_columns) == 0:
        raise ValueError(f"No feature columns found in DataFrame. Available columns: {features_df.columns.tolist()}")
    
    print(f"  Using {len(feature_columns)} features")
    
    features_array = features_df[feature_columns].values.astype(np.float32)
    
    nan_count = np.isnan(features_array).sum()
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaN values. Replacing with 0...")
        features_array = np.nan_to_num(features_array, nan=0.0)
    
    print("  Generating classification labels (HH=0, LL=1, HL=2, LH=3, unknown=-1)...")
    y_class = generate_classification_labels(zigzag_df, turning_points)
    class_distribution_raw = analyze_label_distribution(y_class, 'classification')
    print(f"    Raw labels - Valid: {class_distribution_raw['valid']}, "
          f"Invalid: {class_distribution_raw['invalid']}")
    
    print("  Generating regression labels (bars to next turning point)...")
    y_reg = generate_regression_labels(zigzag_df, turning_points)
    reg_distribution_raw = analyze_label_distribution(y_reg, 'regression')
    print(f"    Raw labels - Valid: {reg_distribution_raw['valid']}, "
          f"Invalid: {reg_distribution_raw['invalid']}")
    
    print(f"  Creating sequences with length {sequence_length}...")
    X_class, y_class_seq = create_sequences(features_array, y_class, sequence_length)
    X_reg, y_reg_seq = create_sequences(features_array, y_reg, sequence_length)
    
    print(f"    Classification: X shape {X_class.shape}, y shape {y_class_seq.shape}")
    print(f"    Regression: X shape {X_reg.shape}, y shape {y_reg_seq.shape}")
    
    validate_sequences(X_class, y_class_seq, sequence_length)
    validate_sequences(X_reg, y_reg_seq, sequence_length)
    
    print("  Analyzing label distributions...")
    class_stats = analyze_label_distribution(y_class_seq, 'classification')
    reg_stats = analyze_label_distribution(y_reg_seq, 'regression')
    
    valid_class = class_stats['valid']
    valid_reg = reg_stats['valid']
    total_seq = len(y_class_seq)
    
    print(f"    Classification - Valid: {valid_class}/{total_seq} "
          f"({class_stats['valid_percent']}%)")
    if 'class_details' in class_stats:
        for class_label, class_info in class_stats['class_details'].items():
            print(f"      {class_label}: {class_info['count']} ({class_info['percentage']}%)")
    
    print(f"    Regression - Valid: {valid_reg}/{total_seq} "
          f"({reg_stats['valid_percent']}%)")
    print(f"      Mean distance: {reg_stats['mean_distance']:.1f} bars")
    print(f"      Distance range: {reg_stats['min_distance']}-{reg_stats['max_distance']} bars")
    
    return {
        'X_class': X_class,
        'y_class': y_class_seq,
        'X_reg': X_reg,
        'y_reg': y_reg_seq,
        'n_features': len(feature_columns),
        'feature_names': feature_columns,
        'class_stats': class_stats,
        'reg_stats': reg_stats,
        'valid_class_count': valid_class,
        'valid_reg_count': valid_reg,
        'sequence_length': sequence_length
    }
