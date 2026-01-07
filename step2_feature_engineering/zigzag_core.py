import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


def calculate_zigzag(df: pd.DataFrame, threshold: float = 5.0) -> Tuple[pd.DataFrame, List[Dict]]:
    """Calculate zigzag turning points and HH/LL/HL/LH structure.

    This implementation:
    - 使用 high / low 價格
    - 透過百分比 threshold 判斷反向波動是否足以構成轉折
    - 將 swing high / swing low 依前一個同類極值分類為 HH / LL / HL / LH
    - 在 df 上新增:
        * zigzag_type: 'HH','LL','HL','LH' 或 None
        * zigzag_point: bool
    - 回傳 turning_points: [{'index', 'type', 'price'}, ...]
    """
    df = df.copy()
    required = ["high", "low"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"DataFrame must contain columns: {required}")

    if df[required].isnull().any().any():
        df[required] = df[required].ffill()

    n = len(df)
    zigzag_type: List[Optional[str]] = [None] * n
    zigzag_flag: List[bool] = [False] * n
    turning_points: List[Dict] = []

    if n < 2:
        df["zigzag_type"] = zigzag_type
        df["zigzag_point"] = zigzag_flag
        return df, turning_points

    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)

    idx_last = 0
    price_last = highs[0]
    last_is_high = True
    trend: Optional[str] = None  # 'up' or 'down'

    prev_swing_high: Optional[float] = None
    prev_swing_low: Optional[float] = None

    for i in range(1, n):
        h = highs[i]
        l = lows[i]
        thr_val = price_last * (threshold / 100.0)

        if trend is None:
            if h > price_last:
                price_last = h
                idx_last = i
                last_is_high = True
                trend = "up"
            elif l < price_last:
                price_last = l
                idx_last = i
                last_is_high = False
                trend = "down"
            continue

        if trend == "up":
            if h >= price_last:
                price_last = h
                idx_last = i
                last_is_high = True
            elif l <= price_last - thr_val:
                swing_price = price_last
                zz_type = _classify_turn(
                    is_high=True,
                    price=swing_price,
                    prev_high=prev_swing_high,
                    prev_low=prev_swing_low,
                )
                zigzag_type[idx_last] = zz_type
                zigzag_flag[idx_last] = True
                turning_points.append(
                    {
                        "index": int(idx_last),
                        "type": zz_type,
                        "price": float(swing_price),
                    }
                )
                prev_swing_high = swing_price

                price_last = l
                idx_last = i
                last_is_high = False
                trend = "down"

        else:  # trend == 'down'
            if l <= price_last:
                price_last = l
                idx_last = i
                last_is_high = False
            elif h >= price_last + thr_val:
                swing_price = price_last
                zz_type = _classify_turn(
                    is_high=False,
                    price=swing_price,
                    prev_high=prev_swing_high,
                    prev_low=prev_swing_low,
                )
                zigzag_type[idx_last] = zz_type
                zigzag_flag[idx_last] = True
                turning_points.append(
                    {
                        "index": int(idx_last),
                        "type": zz_type,
                        "price": float(swing_price),
                    }
                )
                prev_swing_low = swing_price

                price_last = h
                idx_last = i
                last_is_high = True
                trend = "up"

    df["zigzag_type"] = zigzag_type
    df["zigzag_point"] = zigzag_flag
    return df, turning_points


def _classify_turn(
    is_high: bool,
    price: float,
    prev_high: Optional[float],
    prev_low: Optional[float],
) -> str:
    """根據前一同類極值分類 HH/LL/HL/LH。"""
    if is_high:
        if prev_high is None or price >= prev_high:
            return "HH"
        return "LH"
    else:
        if prev_low is None or price <= prev_low:
            return "LL"
        return "HL"
