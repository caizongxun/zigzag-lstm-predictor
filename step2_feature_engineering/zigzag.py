import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


def calculate_zigzag(df: pd.DataFrame, threshold: float = 5.0) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Calculate Zigzag turning points with complete HH/LL/HL/LH classification.
    
    Zigzag Pattern Types:
    - HH (Higher High): Current swing high is higher than previous swing high (uptrend continuation)
    - LL (Lower Low): Current swing low is lower than previous swing low (downtrend continuation)
    - HL (Higher Low): Current swing low is higher than previous swing low (uptrend reversal signal)
    - LH (Lower High): Current swing high is lower than previous swing high (downtrend reversal signal)
    
    Implementation logic:
    1. Track price movement direction (up/down trend)
    2. Identify swing highs and swing lows
    3. Record turning points when reverse movement exceeds threshold%
    4. Compare current extreme with previous extreme of same type to classify HH/LL/HL/LH
    
    Args:
        df (pd.DataFrame): Input data with 'high', 'low', 'close' columns
        threshold (float): Percentage threshold for identifying turns (default: 5.0)
    
    Returns:
        tuple: (
            DataFrame with new columns:
                - zigzag_type: 'HH', 'LL', 'HL', 'LH', or None
                - zigzag_point: True/False indicator
            List of turning points: [
                {
                    'index': bar_index,
                    'type': 'HH'|'LL'|'HL'|'LH',
                    'price': float,
                    'high': float (for context),
                    'low': float (for context)
                },
                ...
            ]
        )
    """
    df = df.copy()
    
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    if df[required_cols].isnull().any().any():
        print("Warning: Missing values found. Forward filling...")
        df[required_cols] = df[required_cols].fillna(method='ffill')
    
    n = len(df)
    zigzag_type = [None] * n
    zigzag_point = [False] * n
    turning_points = []
    
    if n < 2:
        df['zigzag_type'] = zigzag_type
        df['zigzag_point'] = zigzag_point
        return df, turning_points
    
    high = df['high'].values
    low = df['low'].values
    
    last_extreme_idx = 0
    last_extreme_price = high[0]
    last_extreme_type = 'high'
    previous_high = None
    previous_low = None
    
    trend = None
    
    for i in range(1, n):
        current_high = high[i]
        current_low = low[i]
        threshold_amount = (last_extreme_price * threshold) / 100.0
        
        if trend is None:
            if current_high > last_extreme_price:
                last_extreme_price = current_high
                last_extreme_idx = i
                last_extreme_type = 'high'
                trend = 'up'
            elif current_low < last_extreme_price:
                last_extreme_price = current_low
                last_extreme_idx = i
                last_extreme_type = 'low'
                trend = 'down'
        
        elif trend == 'up':
            if current_high >= last_extreme_price:
                last_extreme_price = current_high
                last_extreme_idx = i
            
            elif current_low < last_extreme_price - threshold_amount:
                zz_type = classify_turning_point(
                    'high', last_extreme_price, previous_high, previous_low
                )
                zigzag_type[last_extreme_idx] = zz_type
                zigzag_point[last_extreme_idx] = True
                
                turning_points.append({
                    'index': last_extreme_idx,
                    'type': zz_type,
                    'price': last_extreme_price,
                    'high': last_extreme_price,
                    'low': current_low
                })
                
                previous_high = last_extreme_price
                last_extreme_price = current_low
                last_extreme_idx = i
                last_extreme_type = 'low'
                trend = 'down'
        
        elif trend == 'down':
            if current_low <= last_extreme_price:
                last_extreme_price = current_low
                last_extreme_idx = i
            
            elif current_high > last_extreme_price + threshold_amount:
                zz_type = classify_turning_point(
                    'low', last_extreme_price, previous_high, previous_low
                )
                zigzag_type[last_extreme_idx] = zz_type
                zigzag_point[last_extreme_idx] = True
                
                turning_points.append({
                    'index': last_extreme_idx,
                    'type': zz_type,
                    'price': last_extreme_price,
                    'high': current_high,
                    'low': last_extreme_price
                })
                
                previous_low = last_extreme_price
                last_extreme_price = current_high
                last_extreme_idx = i
                last_extreme_type = 'high'
                trend = 'up'
    
    df['zigzag_type'] = zigzag_type
    df['zigzag_point'] = zigzag_point
    
    return df, turning_points


def classify_turning_point(extreme_type: str, current_price: float, 
                          previous_high: Optional[float], 
                          previous_low: Optional[float]) -> str:
    """
    Classify turning point based on extreme type and comparison with previous extremes.
    
    Classification rules:
    - High turning point:
      * If previous_high is None or current > previous_high: HH (Higher High)
      * If previous_high is not None and current < previous_high: LH (Lower High)
    - Low turning point:
      * If previous_low is None or current < previous_low: LL (Lower Low)
      * If previous_low is not None and current > previous_low: HL (Higher Low)
    
    Args:
        extreme_type (str): 'high' or 'low'
        current_price (float): Current extreme price
        previous_high (Optional[float]): Previous swing high price
        previous_low (Optional[float]): Previous swing low price
    
    Returns:
        str: Classification type 'HH', 'LL', 'HL', or 'LH'
    """
    if extreme_type == 'high':
        if previous_high is None:
            return 'HH'
        return 'HH' if current_price > previous_high else 'LH'
    
    else:
        if previous_low is None:
            return 'LL'
        return 'LL' if current_price < previous_low else 'HL'


def get_zigzag_statistics(turning_points: List[Dict]) -> Dict:
    """
    Calculate comprehensive statistics about identified turning points.
    
    Args:
        turning_points (List[Dict]): List of turning point dictionaries
    
    Returns:
        dict: Statistics including counts and distributions for all four types
    """
    if not turning_points:
        return {
            'total_points': 0,
            'hh_count': 0,
            'll_count': 0,
            'hl_count': 0,
            'lh_count': 0,
            'hh_percent': 0.0,
            'll_percent': 0.0,
            'hl_percent': 0.0,
            'lh_percent': 0.0,
            'continuation_ratio': 0.0,
            'reversal_ratio': 0.0
        }
    
    hh_count = sum(1 for tp in turning_points if tp['type'] == 'HH')
    ll_count = sum(1 for tp in turning_points if tp['type'] == 'LL')
    hl_count = sum(1 for tp in turning_points if tp['type'] == 'HL')
    lh_count = sum(1 for tp in turning_points if tp['type'] == 'LH')
    total = len(turning_points)
    
    continuation_count = hh_count + ll_count
    reversal_count = hl_count + lh_count
    
    return {
        'total_points': total,
        'hh_count': hh_count,
        'll_count': ll_count,
        'hl_count': hl_count,
        'lh_count': lh_count,
        'hh_percent': round((hh_count / total) * 100, 2) if total > 0 else 0.0,
        'll_percent': round((ll_count / total) * 100, 2) if total > 0 else 0.0,
        'hl_percent': round((hl_count / total) * 100, 2) if total > 0 else 0.0,
        'lh_percent': round((lh_count / total) * 100, 2) if total > 0 else 0.0,
        'continuation_ratio': round((continuation_count / total) * 100, 2) if total > 0 else 0.0,
        'reversal_ratio': round((reversal_count / total) * 100, 2) if total > 0 else 0.0
    }


def validate_zigzag_points(df: pd.DataFrame, turning_points: List[Dict]) -> bool:
    """
    Validate integrity of calculated zigzag points.
    
    Checks:
    1. Count consistency between DataFrame and turning_points list
    2. Index consistency
    3. Type completeness (all four types present if expected)
    4. No duplicate indices
    
    Args:
        df (pd.DataFrame): DataFrame with zigzag columns
        turning_points (List[Dict]): List of turning points
    
    Returns:
        bool: True if validation passes
    """
    if len(turning_points) == 0:
        print("Warning: No turning points found")
        return True
    
    zigzag_count = df['zigzag_point'].sum()
    
    if zigzag_count != len(turning_points):
        print(f"Validation warning: Point count mismatch - "
              f"zigzag_point: {zigzag_count}, turning_points: {len(turning_points)}")
        return False
    
    indices_from_df = set(df[df['zigzag_point']].index.tolist())
    indices_from_list = set(tp['index'] for tp in turning_points)
    
    if indices_from_df != indices_from_list:
        print("Validation warning: Index mismatch between DataFrame and turning_points")
        return False
    
    types_in_list = set(tp['type'] for tp in turning_points)
    required_types = {'HH', 'LL', 'HL', 'LH'}
    
    if len(types_in_list) < 4 and len(turning_points) > 10:
        missing_types = required_types - types_in_list
        print(f"Warning: Missing zigzag types in points (likely normal for short series): {missing_types}")
    
    duplicate_indices = len(indices_from_list) != len(turning_points)
    if duplicate_indices:
        print("Validation warning: Duplicate indices detected in turning_points")
        return False
    
    return True


def get_zigzag_type_mapping() -> Dict[str, int]:
    """
    Get mapping of zigzag types to integer labels.
    
    Returns:
        dict: Mapping {type: label}
    """
    return {
        'HH': 0,
        'LL': 1,
        'HL': 2,
        'LH': 3
    }
