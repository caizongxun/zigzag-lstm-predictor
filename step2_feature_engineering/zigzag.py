import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


def calculate_zigzag(df: pd.DataFrame, threshold: float = 5.0) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Calculate Zigzag turning points based on percentage threshold.
    
    Implementation logic:
    1. Track price movement direction (up/down)
    2. Record turning points when reverse movement exceeds threshold%
    3. Identify High-High (HH) and Low-Low (LL) patterns
    4. Return both marked DataFrame and detailed turning point list
    
    Args:
        df (pd.DataFrame): Input data with 'high', 'low', 'close' columns
        threshold (float): Percentage threshold for identifying turns (default: 5.0)
    
    Returns:
        tuple: (
            DataFrame with new columns:
                - zigzag_type: 'HH', 'LL', or None
                - zigzag_point: True/False
            List of turning points: [(index, type, price), ...]
        )
    """
    df = df.copy()
    
    # Validate input
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Check for missing values
    if df[required_cols].isnull().any().any():
        print("Warning: Missing values found in price data. Filling forward...")
        df[required_cols] = df[required_cols].fillna(method='ffill')
    
    n = len(df)
    zigzag_type = [None] * n
    zigzag_point = [False] * n
    turning_points = []
    
    # Initialize tracking variables
    last_extreme_idx = 0
    last_extreme_price = df.iloc[0]['high']
    trend = None
    
    for i in range(1, n):
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        
        threshold_amount = (last_extreme_price * threshold) / 100.0
        
        if trend is None:
            if current_high > last_extreme_price:
                last_extreme_price = current_high
                last_extreme_idx = i
                trend = 'up'
            elif current_low < last_extreme_price:
                last_extreme_price = current_low
                last_extreme_idx = i
                trend = 'down'
        
        elif trend == 'up':
            if current_high >= last_extreme_price:
                last_extreme_price = current_high
                last_extreme_idx = i
            
            elif current_low < last_extreme_price - threshold_amount:
                zigzag_type[last_extreme_idx] = 'HH'
                zigzag_point[last_extreme_idx] = True
                
                turning_points.append({
                    'index': last_extreme_idx,
                    'type': 'HH',
                    'price': last_extreme_price
                })
                
                last_extreme_price = current_low
                last_extreme_idx = i
                trend = 'down'
        
        elif trend == 'down':
            if current_low <= last_extreme_price:
                last_extreme_price = current_low
                last_extreme_idx = i
            
            elif current_high > last_extreme_price + threshold_amount:
                zigzag_type[last_extreme_idx] = 'LL'
                zigzag_point[last_extreme_idx] = True
                
                turning_points.append({
                    'index': last_extreme_idx,
                    'type': 'LL',
                    'price': last_extreme_price
                })
                
                last_extreme_price = current_high
                last_extreme_idx = i
                trend = 'up'
    
    df['zigzag_type'] = zigzag_type
    df['zigzag_point'] = zigzag_point
    
    return df, turning_points


def get_zigzag_statistics(turning_points: List[Dict]) -> Dict:
    """
    Calculate statistics about identified turning points.
    
    Args:
        turning_points (List[Dict]): List of turning point dictionaries
    
    Returns:
        dict: Statistics including counts and distributions
    """
    if not turning_points:
        return {
            'total_points': 0,
            'hh_count': 0,
            'll_count': 0,
            'hh_percentage': 0,
            'll_percentage': 0
        }
    
    hh_count = sum(1 for tp in turning_points if tp['type'] == 'HH')
    ll_count = sum(1 for tp in turning_points if tp['type'] == 'LL')
    total = len(turning_points)
    
    return {
        'total_points': total,
        'hh_count': hh_count,
        'll_count': ll_count,
        'hh_percentage': round((hh_count / total) * 100, 2) if total > 0 else 0,
        'll_percentage': round((ll_count / total) * 100, 2) if total > 0 else 0
    }


def validate_zigzag_points(df: pd.DataFrame, turning_points: List[Dict]) -> bool:
    """
    Validate integrity of calculated zigzag points.
    
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
        print(f"Validation warning: Point count mismatch - zigzag_point count: {zigzag_count}, "
              f"turning_points count: {len(turning_points)}")
        return False
    
    indices_from_df = set(df[df['zigzag_point']].index.tolist())
    indices_from_list = set(tp['index'] for tp in turning_points)
    
    if indices_from_df != indices_from_list:
        print("Validation warning: Index mismatch between DataFrame and turning_points list")
        return False
    
    return True
