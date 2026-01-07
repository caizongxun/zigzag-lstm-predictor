import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using vectorized operations.
    
    RSI formula:
    RS = Average Gain / Average Loss
    RSI = 100 - (100 / (1 + RS))
    
    Args:
        close (np.ndarray): Close price array
        period (int): Period for RSI calculation (default: 14)
    
    Returns:
        np.ndarray: RSI values array
    """
    # Calculate price changes
    delta = np.diff(close, prepend=close[0])
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Initialize RSI array
    rsi = np.zeros_like(close, dtype=float)
    
    # Calculate first average gain and loss
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    for i in range(period, len(close)):
        if i == period:
            avg_gain = np.mean(gains[1:period+1])
            avg_loss = np.mean(losses[1:period+1])
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    rsi[:period] = np.nan
    return rsi


def calculate_macd(close: np.ndarray, 
                   fast: int = 12, 
                   slow: int = 26, 
                   signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence) using vectorized operations.
    
    MACD formula:
    MACD = EMA12 - EMA26
    Signal = EMA9 of MACD
    Histogram = MACD - Signal
    
    Args:
        close (np.ndarray): Close price array
        fast (int): Fast EMA period (default: 12)
        slow (int): Slow EMA period (default: 26)
        signal (int): Signal line EMA period (default: 9)
    
    Returns:
        tuple: (MACD array, Signal array, Histogram array)
    """
    def ema(data: np.ndarray, period: int) -> np.ndarray:
        result = np.zeros_like(data, dtype=float)
        multiplier = 2.0 / (period + 1)
        result[0] = data[0]
        
        for i in range(1, len(data)):
            result[i] = data[i] * multiplier + result[i-1] * (1 - multiplier)
        
        return result
    
    # Calculate EMAs
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal line
    signal_line = ema(macd_line, signal)
    
    # Calculate Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_atr(high: np.ndarray, 
                  low: np.ndarray, 
                  close: np.ndarray, 
                  period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using vectorized operations.
    
    ATR formula:
    True Range = max(H-L, abs(H-PC), abs(L-PC))
    ATR = EMA of True Range
    
    where:
    H = Current High
    L = Current Low
    PC = Previous Close
    
    Args:
        high (np.ndarray): High price array
        low (np.ndarray): Low price array
        close (np.ndarray): Close price array
        period (int): Period for ATR calculation (default: 14)
    
    Returns:
        np.ndarray: ATR values array
    """
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    true_range[0] = tr1[0]
    
    # Calculate ATR using EMA
    atr = np.zeros_like(true_range, dtype=float)
    multiplier = 2.0 / (period + 1)
    atr[0] = true_range[0]
    
    for i in range(1, len(true_range)):
        atr[i] = true_range[i] * multiplier + atr[i-1] * (1 - multiplier)
    
    return atr


def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using vectorized operations.
    
    Args:
        data (np.ndarray): Input data array
        period (int): Period for SMA
    
    Returns:
        np.ndarray: SMA values array
    """
    sma = np.zeros_like(data, dtype=float)
    cumsum = np.cumsum(data)
    
    sma[period-1:] = (cumsum[period-1:] - np.concatenate(([0], cumsum[:-period]))) / period
    sma[:period-1] = np.nan
    
    return sma


def calculate_high_low_ratio(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Calculate High-Low ratio as percentage of low.
    
    Args:
        high (np.ndarray): High price array
        low (np.ndarray): Low price array
    
    Returns:
        np.ndarray: High-Low ratio array
    """
    return ((high - low) / low) * 100


def calculate_close_range(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Calculate where close price falls within high-low range (0-1).
    
    Args:
        close (np.ndarray): Close price array
        high (np.ndarray): High price array
        low (np.ndarray): Low price array
    
    Returns:
        np.ndarray: Close range ratio (0-1)
    """
    range_val = high - low
    range_val = np.where(range_val == 0, 1, range_val)
    
    return (close - low) / range_val


def calculate_highest_lowest(data: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate highest and lowest values over period using vectorized rolling window.
    
    Args:
        data (np.ndarray): Input data array
        period (int): Period for calculation
    
    Returns:
        tuple: (Highest array, Lowest array)
    """
    highest = np.zeros_like(data, dtype=float)
    lowest = np.zeros_like(data, dtype=float)
    
    for i in range(len(data)):
        start = max(0, i - period + 1)
        highest[i] = np.max(data[start:i+1])
        lowest[i] = np.min(data[start:i+1])
    
    return highest, lowest


def calculate_returns(close: np.ndarray) -> np.ndarray:
    """
    Calculate log returns.
    
    Returns = log(close_t / close_t-1)
    
    Args:
        close (np.ndarray): Close price array
    
    Returns:
        np.ndarray: Log returns array
    """
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    return returns


def calculate_volatility(returns: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Calculate rolling volatility (standard deviation of returns).
    
    Args:
        returns (np.ndarray): Returns array
        period (int): Period for volatility calculation
    
    Returns:
        np.ndarray: Volatility array
    """
    volatility = np.zeros_like(returns, dtype=float)
    
    for i in range(len(returns)):
        start = max(0, i - period + 1)
        volatility[i] = np.std(returns[start:i+1])
    
    return volatility
