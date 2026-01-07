import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI) using vectorized operations.

    RSI formula:
    RS = Average Gain / Average Loss
    RSI = 100 - (100 / (1 + RS))

    Implementation:
    1. Calculate price changes (delta)
    2. Separate positive changes (gains) and negative changes (losses)
    3. Use smoothed averages (Wilder's smoothing)
    4. Calculate RSI from RS ratio

    Args:
        close (np.ndarray): Close price array
        period (int): Period for RSI calculation (default: 14)

    Returns:
        np.ndarray: RSI values (0-100), with NaN for first 'period' bars
    """
    close = np.asarray(close, dtype=float)
    n = len(close)

    if n < period + 1:
        return np.full(n, np.nan)

    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    rsi = np.full(n, np.nan, dtype=float)

    avg_gain = np.mean(gains[1 : period + 1])
    avg_loss = np.mean(losses[1 : period + 1])

    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0.0:
            rsi[i] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def calculate_macd(
    close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence) using vectorized operations.

    MACD formula:
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(signal) of MACD
    Histogram = MACD - Signal

    Implementation:
    1. Calculate fast and slow exponential moving averages
    2. Calculate MACD line as difference
    3. Calculate signal line as EMA of MACD
    4. Calculate histogram as MACD - Signal

    Args:
        close (np.ndarray): Close price array
        fast (int): Fast EMA period (default: 12)
        slow (int): Slow EMA period (default: 26)
        signal (int): Signal line EMA period (default: 9)

    Returns:
        tuple: (MACD array, Signal array, Histogram array)
    """
    close = np.asarray(close, dtype=float)

    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average."""
        n = len(data)
        result = np.zeros(n, dtype=float)
        multiplier = 2.0 / (period + 1)
        result[0] = data[0]

        for i in range(1, n):
            result[i] = data[i] * multiplier + result[i - 1] * (1 - multiplier)

        return result

    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)

    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_atr(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
) -> np.ndarray:
    """
    Calculate Average True Range (ATR) using vectorized operations.

    ATR formula:
    True Range = max(H-L, abs(H-PC), abs(L-PC))
    ATR = EMA of True Range

    where:
    H = Current High
    L = Current Low
    PC = Previous Close

    Implementation:
    1. Calculate three components of true range
    2. Take maximum of three components
    3. Apply EMA smoothing to true range

    Args:
        high (np.ndarray): High price array
        low (np.ndarray): Low price array
        close (np.ndarray): Close price array
        period (int): Period for ATR calculation (default: 14)

    Returns:
        np.ndarray: ATR values
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))

    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    true_range[0] = tr1[0]

    atr = np.zeros_like(true_range, dtype=float)
    multiplier = 2.0 / (period + 1)
    atr[0] = true_range[0]

    for i in range(1, len(true_range)):
        atr[i] = true_range[i] * multiplier + atr[i - 1] * (1 - multiplier)

    return atr


def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using vectorized operations.

    SMA = Sum of prices over period / period

    Args:
        data (np.ndarray): Input data array
        period (int): Period for SMA

    Returns:
        np.ndarray: SMA values with NaN for first 'period-1' bars
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    sma = np.full(n, np.nan, dtype=float)

    if n < period:
        return sma

    cumsum = np.cumsum(data)
    sma[period - 1 :] = (cumsum[period - 1 :] - np.concatenate(([0], cumsum[:-period]))) / period

    return sma


def calculate_high_low_ratio(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Calculate High-Low ratio as percentage.

    Formula: ((High - Low) / Low) * 100

    Args:
        high (np.ndarray): High price array
        low (np.ndarray): Low price array

    Returns:
        np.ndarray: High-Low ratio percentage
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    low_safe = np.where(low != 0, low, 1.0)
    return ((high - low) / low_safe) * 100.0


def calculate_close_range(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Calculate where close price falls within high-low range (0-1 normalized).

    Formula: (Close - Low) / (High - Low)

    Returns:
    0 = Close at Low
    1 = Close at High
    0.5 = Close at midpoint

    Args:
        close (np.ndarray): Close price array
        high (np.ndarray): High price array
        low (np.ndarray): Low price array

    Returns:
        np.ndarray: Close range ratio (0-1)
    """
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)

    range_val = high - low
    range_val = np.where(range_val != 0, range_val, 1.0)

    return (close - low) / range_val


def calculate_highest_lowest(
    data: np.ndarray, period: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate highest and lowest values over period using rolling window.

    Implementation:
    - For each bar i, find max and min of data[max(0, i-period+1):i+1]
    - Lookback period includes current bar

    Args:
        data (np.ndarray): Input data array
        period (int): Period for calculation

    Returns:
        tuple: (Highest array, Lowest array)
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    highest = np.zeros(n, dtype=float)
    lowest = np.zeros(n, dtype=float)

    for i in range(n):
        start = max(0, i - period + 1)
        highest[i] = np.max(data[start : i + 1])
        lowest[i] = np.min(data[start : i + 1])

    return highest, lowest


def calculate_returns(close: np.ndarray) -> np.ndarray:
    """
    Calculate log returns.

    Formula: log(Close_t / Close_t-1)

    Returns at bar 0 is 0 (no prior close)
    This handles the first bar by prepending the first close value

    Args:
        close (np.ndarray): Close price array

    Returns:
        np.ndarray: Log returns array
    """
    close = np.asarray(close, dtype=float)
    log_close = np.log(close)
    returns = np.diff(log_close, prepend=log_close[0])
    return returns


def calculate_volatility(returns: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Calculate rolling volatility (standard deviation of returns).

    Implementation:
    - For each bar i, calculate std dev of returns[max(0, i-period+1):i+1]
    - Lookback period includes current bar

    Args:
        returns (np.ndarray): Returns array
        period (int): Period for volatility calculation (default: 20)

    Returns:
        np.ndarray: Volatility (std dev) array
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    volatility = np.zeros(n, dtype=float)

    for i in range(n):
        start = max(0, i - period + 1)
        volatility[i] = np.std(returns[start : i + 1])

    return volatility
