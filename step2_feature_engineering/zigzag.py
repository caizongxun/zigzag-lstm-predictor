import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional


def calculate_zigzag(df: pd.DataFrame, threshold: float = 5.0) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Calculate zigzag turning points with HH/LL/HL/LH classification.

    Zigzag Pattern Types:
    - HH (Higher High): Current swing high > previous swing high (uptrend continuation)
    - LL (Lower Low): Current swing low < previous swing low (downtrend continuation)
    - HL (Higher Low): Current swing low > previous swing low (uptrend reversal signal)
    - LH (Lower High): Current swing high < previous swing high (downtrend reversal signal)

    Implementation:
    1. Track price movement direction (up/down trend)
    2. Identify swing highs and swing lows
    3. Record turning points when reverse movement exceeds threshold%
    4. Compare current extreme with previous extreme of same type to classify HH/LL/HL/LH

    Args:
        df (pd.DataFrame): Input DataFrame with 'high', 'low' columns (and optionally others)
        threshold (float): Percentage threshold for identifying reversals (default: 5.0%)

    Returns:
        tuple: (
            DataFrame with new columns:
                - zigzag_type: 'HH', 'LL', 'HL', 'LH', or None
                - zigzag_point: True/False indicator
            List of turning points: [
                {
                    'index': int,
                    'type': str ('HH'|'LL'|'HL'|'LH'),
                    'price': float
                },
                ...
            ]
        )

    Raises:
        ValueError: If required columns are missing
    """
    df = df.copy()

    required_cols = ["high", "low"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if df[required_cols].isnull().any().any():
        df[required_cols] = df[required_cols].ffill()

    n = len(df)
    zigzag_type: List[Optional[str]] = [None] * n
    zigzag_point: List[bool] = [False] * n
    turning_points: List[Dict] = []

    if n < 2:
        df["zigzag_type"] = zigzag_type
        df["zigzag_point"] = zigzag_point
        return df, turning_points

    high_vals = df["high"].values.astype(float)
    low_vals = df["low"].values.astype(float)

    last_extreme_idx = 0
    last_extreme_price = high_vals[0]
    last_extreme_is_high = True
    trend: Optional[str] = None

    prev_swing_high: Optional[float] = None
    prev_swing_low: Optional[float] = None

    for i in range(1, n):
        current_high = high_vals[i]
        current_low = low_vals[i]
        threshold_amount = last_extreme_price * (threshold / 100.0)

        if trend is None:
            if current_high > last_extreme_price:
                last_extreme_price = current_high
                last_extreme_idx = i
                last_extreme_is_high = True
                trend = "up"
            elif current_low < last_extreme_price:
                last_extreme_price = current_low
                last_extreme_idx = i
                last_extreme_is_high = False
                trend = "down"
            continue

        if trend == "up":
            if current_high >= last_extreme_price:
                last_extreme_price = current_high
                last_extreme_idx = i
                last_extreme_is_high = True
            elif current_low <= last_extreme_price - threshold_amount:
                swing_price = last_extreme_price
                zz_type = _classify_zigzag_type(
                    is_high_extreme=True,
                    price=swing_price,
                    prev_high=prev_swing_high,
                    prev_low=prev_swing_low,
                )
                zigzag_type[last_extreme_idx] = zz_type
                zigzag_point[last_extreme_idx] = True

                turning_points.append(
                    {
                        "index": int(last_extreme_idx),
                        "type": zz_type,
                        "price": float(swing_price),
                    }
                )

                prev_swing_high = swing_price

                last_extreme_price = current_low
                last_extreme_idx = i
                last_extreme_is_high = False
                trend = "down"

        elif trend == "down":
            if current_low <= last_extreme_price:
                last_extreme_price = current_low
                last_extreme_idx = i
                last_extreme_is_high = False
            elif current_high >= last_extreme_price + threshold_amount:
                swing_price = last_extreme_price
                zz_type = _classify_zigzag_type(
                    is_high_extreme=False,
                    price=swing_price,
                    prev_high=prev_swing_high,
                    prev_low=prev_swing_low,
                )
                zigzag_type[last_extreme_idx] = zz_type
                zigzag_point[last_extreme_idx] = True

                turning_points.append(
                    {
                        "index": int(last_extreme_idx),
                        "type": zz_type,
                        "price": float(swing_price),
                    }
                )

                prev_swing_low = swing_price

                last_extreme_price = current_high
                last_extreme_idx = i
                last_extreme_is_high = True
                trend = "up"

    df["zigzag_type"] = zigzag_type
    df["zigzag_point"] = zigzag_point

    return df, turning_points


def _classify_zigzag_type(
    is_high_extreme: bool,
    price: float,
    prev_high: Optional[float],
    prev_low: Optional[float],
) -> str:
    """
    Classify turning point as HH/LL/HL/LH based on extreme type and previous extremes.

    Classification Logic:
    - For high extreme (swing high):
        * If no previous high or price >= prev_high: HH (Higher High)
        * If price < prev_high: LH (Lower High)
    - For low extreme (swing low):
        * If no previous low or price <= prev_low: LL (Lower Low)
        * If price > prev_low: HL (Higher Low)

    Args:
        is_high_extreme (bool): True if current extreme is a high, False if low
        price (float): Current extreme price
        prev_high (Optional[float]): Previous swing high price
        prev_low (Optional[float]): Previous swing low price

    Returns:
        str: One of 'HH', 'LL', 'HL', 'LH'
    """
    if is_high_extreme:
        if prev_high is None or price >= prev_high:
            return "HH"
        else:
            return "LH"
    else:
        if prev_low is None or price <= prev_low:
            return "LL"
        else:
            return "HL"


def get_zigzag_statistics(turning_points: List[Dict]) -> Dict:
    """
    Calculate comprehensive statistics about identified turning points.

    Args:
        turning_points (List[Dict]): List of turning point dictionaries with 'type' key

    Returns:
        dict: Statistics including counts and percentages for all four types
    """
    if not turning_points:
        return {
            "total_points": 0,
            "hh_count": 0,
            "ll_count": 0,
            "hl_count": 0,
            "lh_count": 0,
            "hh_percent": 0.0,
            "ll_percent": 0.0,
            "hl_percent": 0.0,
            "lh_percent": 0.0,
            "continuation_ratio": 0.0,
            "reversal_ratio": 0.0,
        }

    hh_count = sum(1 for tp in turning_points if tp["type"] == "HH")
    ll_count = sum(1 for tp in turning_points if tp["type"] == "LL")
    hl_count = sum(1 for tp in turning_points if tp["type"] == "HL")
    lh_count = sum(1 for tp in turning_points if tp["type"] == "LH")
    total = len(turning_points)

    continuation_count = hh_count + ll_count
    reversal_count = hl_count + lh_count

    hh_pct = round((hh_count / total) * 100, 2) if total > 0 else 0.0
    ll_pct = round((ll_count / total) * 100, 2) if total > 0 else 0.0
    hl_pct = round((hl_count / total) * 100, 2) if total > 0 else 0.0
    lh_pct = round((lh_count / total) * 100, 2) if total > 0 else 0.0
    cont_pct = round((continuation_count / total) * 100, 2) if total > 0 else 0.0
    rev_pct = round((reversal_count / total) * 100, 2) if total > 0 else 0.0

    return {
        "total_points": total,
        "hh_count": hh_count,
        "ll_count": ll_count,
        "hl_count": hl_count,
        "lh_count": lh_count,
        "hh_percent": hh_pct,
        "ll_percent": ll_pct,
        "hl_percent": hl_pct,
        "lh_percent": lh_pct,
        "continuation_ratio": cont_pct,
        "reversal_ratio": rev_pct,
    }


def validate_zigzag_points(df: pd.DataFrame, turning_points: List[Dict]) -> bool:
    """
    Validate integrity of calculated zigzag points.

    Checks:
    1. Count consistency between DataFrame and turning_points list
    2. Index consistency
    3. Type completeness
    4. No duplicate indices

    Args:
        df (pd.DataFrame): DataFrame with zigzag columns
        turning_points (List[Dict]): List of turning points

    Returns:
        bool: True if validation passes, False otherwise
    """
    if len(turning_points) == 0:
        return True

    if "zigzag_point" not in df.columns:
        print("Error: zigzag_point column not found in DataFrame")
        return False

    zigzag_count = df["zigzag_point"].sum()

    if zigzag_count != len(turning_points):
        print(
            f"Validation error: Point count mismatch - "
            f"zigzag_point: {zigzag_count}, turning_points: {len(turning_points)}"
        )
        return False

    indices_from_df = set(df[df["zigzag_point"]].index.tolist())
    indices_from_list = set(tp["index"] for tp in turning_points)

    if indices_from_df != indices_from_list:
        print("Validation error: Index mismatch between DataFrame and turning_points")
        return False

    types_in_list = set(tp["type"] for tp in turning_points)
    required_types = {"HH", "LL", "HL", "LH"}

    if len(types_in_list) < 4 and len(turning_points) > 10:
        missing_types = required_types - types_in_list
        print(
            f"Warning: Missing zigzag types in turning_points (may be normal for short series): {missing_types}"
        )

    valid_types = {"HH", "LL", "HL", "LH"}
    invalid_types = types_in_list - valid_types
    if invalid_types:
        print(f"Validation error: Invalid zigzag types found: {invalid_types}")
        return False

    duplicate_check = len(indices_from_list) == len(turning_points)
    if not duplicate_check:
        print("Validation error: Duplicate indices detected in turning_points")
        return False

    return True


def get_zigzag_type_mapping() -> Dict[str, int]:
    """
    Get mapping of zigzag types to integer labels for classification.

    Returns:
        dict: Mapping {zigzag_type_str: label_int}
              'HH': 0, 'LL': 1, 'HL': 2, 'LH': 3
    """
    return {"HH": 0, "LL": 1, "HL": 2, "LH": 3}
