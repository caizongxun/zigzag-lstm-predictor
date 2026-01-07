import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional


def generate_classification_labels(df: pd.DataFrame, turning_points: List[Dict]) -> np.ndarray:
    """
    Generate classification labels for next turning point type.
    
    For each row, determine what the next zigzag turning point type will be:
    0 = High-High (HH) - uptrend turning point
    1 = Low-Low (LL) - downtrend turning point
    -1 = Unknown (no next turning point in data)
    
    Args:
        df (pd.DataFrame): DataFrame with zigzag_type column
        turning_points (List[Dict]): List of turning points with indices and types
    
    Returns:
        np.ndarray: Classification labels array
    """
    n = len(df)
    labels = np.full(n, -1, dtype=np.int32)
    
    if not turning_points:
        print("Warning: No turning points provided for classification labels")
        return labels
    
    tp_indices = [tp['index'] for tp in turning_points]
    tp_types = [tp['type'] for tp in turning_points]
    
    for i in range(n):
        future_points = [idx for idx in tp_indices if idx > i]
        
        if future_points:
            next_point_idx = min(future_points)
            next_point_pos = tp_indices.index(next_point_idx)
            next_type = tp_types[next_point_pos]
            
            labels[i] = 0 if next_type == 'HH' else 1
    
    return labels


def generate_regression_labels(df: pd.DataFrame, turning_points: List[Dict]) -> np.ndarray:
    """
    Generate regression labels for distance to next turning point.
    
    For each row, calculate how many bars until the next zigzag turning point.
    If no next turning point, use -1.
    
    Args:
        df (pd.DataFrame): DataFrame with price data
        turning_points (List[Dict]): List of turning points with indices
    
    Returns:
        np.ndarray: Distance to next turning point (in bars)
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
            next_point_idx = min(future_points)
            distance = next_point_idx - i
            labels[i] = distance
    
    return labels


def create_sequences(features: np.ndarray, 
                    labels: np.ndarray, 
                    sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction.
    
    Converts (n_samples, n_features) to (n_samples-sequence_length, sequence_length, n_features)
    Each sequence uses a label from the END of that sequence (lookahead = 0)
    
    Args:
        features (np.ndarray): Feature array of shape (n_samples, n_features)
        labels (np.ndarray): Label array of shape (n_samples,)
        sequence_length (int): Length of sequences (default: 30)
    
    Returns:
        tuple: (X sequences, y labels)
               X: shape (n_samples - sequence_length + 1, sequence_length, n_features)
               y: shape (n_samples - sequence_length + 1,)
    """
    if len(features) != len(labels):
        raise ValueError(f"Features and labels length mismatch: {len(features)} vs {len(labels)}")
    
    if sequence_length <= 0:
        raise ValueError(f"Sequence length must be positive: {sequence_length}")
    
    n_samples = len(features)
    n_features = features.shape[1] if len(features.shape) > 1 else 1
    
    X = []
    y = []
    
    for i in range(n_samples - sequence_length + 1):
        X.append(features[i:i + sequence_length])
        y.append(labels[i + sequence_length - 1])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
    
    return X, y


def validate_sequences(X: np.ndarray, y: np.ndarray, sequence_length: int) -> bool:
    """
    Validate sequence integrity.
    
    Args:
        X (np.ndarray): Feature sequences
        y (np.ndarray): Labels
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
    
    print(f"Sequence validation passed. Total sequences: {len(X)}")
    return True


def analyze_label_distribution(labels: np.ndarray, label_type: str = 'classification') -> Dict:
    """
    Analyze distribution of labels for class balance assessment.
    
    Args:
        labels (np.ndarray): Labels array
        label_type (str): Type of labels ('classification' or 'regression')
    
    Returns:
        dict: Label distribution statistics
    """
    valid_labels = labels[labels != -1]
    
    if len(valid_labels) == 0:
        return {'total': 0, 'valid': 0, 'distribution': {}}
    
    if label_type == 'classification':
        unique, counts = np.unique(valid_labels, return_counts=True)
        distribution = {
            f'class_{int(label)}': int(count) for label, count in zip(unique, counts)
        }
        
        total = len(labels)
        valid = len(valid_labels)
        invalid = total - valid
        
        class_balance = {}
        for label, count in distribution.items():
            class_balance[label] = round((count / valid) * 100, 2)
        
        return {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'distribution': distribution,
            'class_balance_percent': class_balance
        }
    
    else:  # regression
        return {
            'total': len(labels),
            'valid': len(valid_labels),
            'invalid': len(labels) - len(valid_labels),
            'mean_distance': float(np.mean(valid_labels)),
            'std_distance': float(np.std(valid_labels)),
            'min_distance': int(np.min(valid_labels)),
            'max_distance': int(np.max(valid_labels))
        }


def prepare_training_data(features_df: pd.DataFrame, 
                        zigzag_df: pd.DataFrame,
                        turning_points: List[Dict],
                        sequence_length: int = 30,
                        feature_names: Optional[List[str]] = None) -> Dict:
    """
    Complete workflow to prepare training data for LSTM.
    
    Integrates feature creation, sequencing, and label generation.
    
    Args:
        features_df (pd.DataFrame): DataFrame with calculated features
        zigzag_df (pd.DataFrame): DataFrame with zigzag information
        turning_points (List[Dict]): List of turning points
        sequence_length (int): Sequence length for LSTM
        feature_names (Optional[List[str]]): Names of feature columns to use
    
    Returns:
        dict: Dictionary containing:
            - X_class: Classification sequences
            - y_class: Classification labels
            - X_reg: Regression sequences
            - y_reg: Regression labels
            - class_stats: Classification label statistics
            - reg_stats: Regression label statistics
    """
    if feature_names is None:
        feature_names = [
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'atr_14',
            'high_low_ratio', 'close_range', 'highest_20', 'lowest_20',
            'volume_sma_20', 'volume_ratio', 'returns', 'returns_volatility'
        ]
    
    print("\nPreparing training data...")
    
    if len(features_df) != len(zigzag_df):
        raise ValueError("Features and zigzag DataFrames length mismatch")
    
    feature_columns = [col for col in feature_names if col in features_df.columns]
    if len(feature_columns) == 0:
        raise ValueError(f"No feature columns found in DataFrame")
    
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    features_array = features_df[feature_columns].values
    
    # Handle NaN values
    nan_count = np.isnan(features_array).sum()
    if nan_count > 0:
        print(f"Warning: Found {nan_count} NaN values. Forward filling...")
        features_array = np.nan_to_num(features_array, nan=0.0)
    
    # Generate labels
    print("Generating classification labels...")
    y_class = generate_classification_labels(zigzag_df, turning_points)
    
    print("Generating regression labels...")
    y_reg = generate_regression_labels(zigzag_df, turning_points)
    
    # Create sequences
    print(f"Creating sequences with length {sequence_length}...")
    X_class, y_class_seq = create_sequences(features_array, y_class, sequence_length)
    X_reg, y_reg_seq = create_sequences(features_array, y_reg, sequence_length)
    
    # Validate sequences
    validate_sequences(X_class, y_class_seq, sequence_length)
    validate_sequences(X_reg, y_reg_seq, sequence_length)
    
    # Analyze label distribution
    print("\nAnalyzing label distribution...")
    class_stats = analyze_label_distribution(y_class_seq, 'classification')
    reg_stats = analyze_label_distribution(y_reg_seq, 'regression')
    
    return {
        'X_class': X_class,
        'y_class': y_class_seq,
        'X_reg': X_reg,
        'y_reg': y_reg_seq,
        'class_stats': class_stats,
        'reg_stats': reg_stats,
        'n_features': len(feature_columns),
        'feature_names': feature_columns
    }
