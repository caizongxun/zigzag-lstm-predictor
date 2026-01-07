import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

from indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    calculate_sma,
    calculate_high_low_ratio,
    calculate_close_range,
    calculate_highest_lowest,
    calculate_returns,
    calculate_volatility
)


class FeatureExtractor:
    """
    Extract and normalize technical features for LSTM model.
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.scaler = StandardScaler()
        self.feature_names = [
            'rsi_14',
            'macd',
            'macd_signal',
            'macd_histogram',
            'atr_14',
            'high_low_ratio',
            'close_range',
            'highest_20',
            'lowest_20',
            'volume_sma_20',
            'volume_ratio',
            'returns',
            'returns_volatility'
        ]
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all technical indicator features from OHLCV data.
        
        Output features (13 total):
        1. rsi_14: Relative Strength Index (14-period)
        2. macd: MACD line
        3. macd_signal: MACD Signal line
        4. macd_histogram: MACD Histogram
        5. atr_14: Average True Range (14-period)
        6. high_low_ratio: (High-Low)/Low ratio
        7. close_range: Close position in High-Low range
        8. highest_20: Highest price in 20 periods
        9. lowest_20: Lowest price in 20 periods
        10. volume_sma_20: Volume SMA (20-period)
        11. volume_ratio: Current volume / Volume SMA
        12. returns: Log returns
        13. returns_volatility: Volatility of returns
        
        Args:
            df (pd.DataFrame): Input data with OHLCV columns
        
        Returns:
            pd.DataFrame: DataFrame with calculated features
        """
        features_df = df.copy()
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Convert to numpy arrays for calculations
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        
        print("Calculating technical indicators...")
        
        # RSI calculation
        features_df['rsi_14'] = calculate_rsi(close, period=14)
        
        # MACD calculation
        macd_line, signal_line, histogram = calculate_macd(close, fast=12, slow=26, signal=9)
        features_df['macd'] = macd_line
        features_df['macd_signal'] = signal_line
        features_df['macd_histogram'] = histogram
        
        # ATR calculation
        features_df['atr_14'] = calculate_atr(high, low, close, period=14)
        
        # High-Low ratio
        features_df['high_low_ratio'] = calculate_high_low_ratio(high, low)
        
        # Close range
        features_df['close_range'] = calculate_close_range(close, high, low)
        
        # Highest and Lowest over 20 periods
        highest_20, lowest_20 = calculate_highest_lowest(high, self.lookback)
        features_df['highest_20'] = highest_20
        features_df['lowest_20'] = lowest_20
        
        # Volume indicators
        volume_sma = calculate_sma(volume, self.lookback)
        features_df['volume_sma_20'] = volume_sma
        
        volume_sma_safe = np.where(volume_sma > 0, volume_sma, 1)
        features_df['volume_ratio'] = volume / volume_sma_safe
        
        # Returns calculation
        log_returns = calculate_returns(close)
        features_df['returns'] = log_returns
        
        # Volatility of returns
        features_df['returns_volatility'] = calculate_volatility(log_returns, period=self.lookback)
        
        print(f"Technical indicators calculated. Shape: {features_df.shape}")
        
        return features_df
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Normalize technical features using StandardScaler (Z-score normalization).
        
        StandardScaler is preferred over MinMax scaling for:
        - Data with bell curve distribution
        - Handling outliers better
        - LSTM performance improvement (typical 10% accuracy boost)
        
        Args:
            df (pd.DataFrame): DataFrame with features
            fit (bool): If True, fit scaler on this data. If False, use pre-fitted scaler.
        
        Returns:
            tuple: (Normalized DataFrame, fitted scaler object)
        """
        if fit:
            print("Fitting StandardScaler...")
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df[self.feature_names])
        else:
            print("Applying pre-fitted StandardScaler...")
            scaled_data = self.scaler.transform(df[self.feature_names])
        
        # Create normalized DataFrame
        normalized_df = df.copy()
        normalized_df[self.feature_names] = scaled_data
        
        print(f"Features normalized. Mean: {scaled_data.mean():.6f}, Std: {scaled_data.std():.6f}")
        
        return normalized_df, self.scaler
    
    def get_feature_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calculate statistics for each feature.
        
        Args:
            df (pd.DataFrame): DataFrame with features
        
        Returns:
            dict: Statistics for each feature
        """
        stats = {}
        for feature in self.feature_names:
            if feature in df.columns:
                stats[feature] = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max()),
                    'median': float(df[feature].median()),
                    'missing_count': int(df[feature].isna().sum())
                }
        
        return stats
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validate features for consistency and missing values.
        
        Args:
            df (pd.DataFrame): DataFrame with features
        
        Returns:
            bool: True if validation passes
        """
        for feature in self.feature_names:
            if feature not in df.columns:
                print(f"Error: Missing feature column: {feature}")
                return False
            
            missing = df[feature].isna().sum()
            if missing > 0:
                print(f"Warning: Feature '{feature}' has {missing} missing values")
        
        print(f"Feature validation passed. Total samples: {len(df)}")
        return True
