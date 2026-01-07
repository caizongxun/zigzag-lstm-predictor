import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict

from indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_atr,
    calculate_sma,
    calculate_high_low_ratio,
    calculate_close_range,
    calculate_highest_lowest,
    calculate_returns,
    calculate_volatility,
)


class FeatureExtractor:
    """
    Extract and normalize 13 technical indicator features for LSTM model training.

    Feature Set (13 features):
    1. rsi_14: Relative Strength Index (14-period)
    2. macd: MACD line
    3. macd_signal: MACD Signal line
    4. macd_histogram: MACD Histogram
    5. atr_14: Average True Range (14-period)
    6. high_low_ratio: (High-Low)/Low ratio
    7. close_range: Close position in High-Low range [0,1]
    8. highest_20: Highest price in last 20 periods
    9. lowest_20: Lowest price in last 20 periods
    10. volume_sma_20: Volume SMA (20-period)
    11. volume_ratio: Current volume / Volume SMA
    12. returns: Log returns
    13. returns_volatility: Rolling volatility of returns (20-period)
    """

    def __init__(self, lookback: int = 20):
        """
        Initialize feature extractor.

        Args:
            lookback (int): Lookback period for rolling calculations (default: 20)
        """
        self.lookback = lookback
        self.scaler = StandardScaler()
        self.feature_names = [
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_histogram",
            "atr_14",
            "high_low_ratio",
            "close_range",
            "highest_20",
            "lowest_20",
            "volume_sma_20",
            "volume_ratio",
            "returns",
            "returns_volatility",
        ]

    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all 13 technical indicator features from OHLCV data.

        Processing steps:
        1. Validate input data (OHLCV columns present)
        2. Calculate momentum indicators (RSI, MACD, ATR)
        3. Calculate price-based indicators (High-Low ratio, Close range)
        4. Calculate volatility indicators (highest/lowest, volume, volatility)
        5. Calculate return indicators (log returns, volatility of returns)

        Args:
            df (pd.DataFrame): Input data with OHLCV columns
                Required columns: 'open', 'high', 'low', 'close', 'volume'

        Returns:
            pd.DataFrame: Input DataFrame with 13 new feature columns appended

        Raises:
            ValueError: If required columns are missing
        """
        features_df = df.copy()

        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)
        volume = df["volume"].values.astype(float)

        print("  Calculating technical features...")

        print("    - RSI (14)")
        features_df["rsi_14"] = calculate_rsi(close, period=14)

        print("    - MACD (12, 26, 9)")
        macd_line, signal_line, histogram = calculate_macd(close, fast=12, slow=26, signal=9)
        features_df["macd"] = macd_line
        features_df["macd_signal"] = signal_line
        features_df["macd_histogram"] = histogram

        print("    - ATR (14)")
        features_df["atr_14"] = calculate_atr(high, low, close, period=14)

        print("    - High-Low ratio")
        features_df["high_low_ratio"] = calculate_high_low_ratio(high, low)

        print("    - Close range")
        features_df["close_range"] = calculate_close_range(close, high, low)

        print("    - Highest/Lowest (20)")
        highest_20, lowest_20 = calculate_highest_lowest(high, self.lookback)
        features_df["highest_20"] = highest_20
        features_df["lowest_20"] = lowest_20

        print("    - Volume SMA (20)")
        volume_sma = calculate_sma(volume, self.lookback)
        features_df["volume_sma_20"] = volume_sma

        print("    - Volume ratio")
        volume_sma_safe = np.where(volume_sma > 0, volume_sma, 1.0)
        features_df["volume_ratio"] = volume / volume_sma_safe

        print("    - Returns and volatility")
        log_returns = calculate_returns(close)
        features_df["returns"] = log_returns
        features_df["returns_volatility"] = calculate_volatility(log_returns, period=self.lookback)

        print(f"  Features created: {features_df.shape}")
        print(f"  Feature columns: {self.feature_names}")

        return features_df

    def normalize_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Normalize technical features using StandardScaler (Z-score normalization).

        StandardScaler benefits:
        - Centers data around 0 with standard deviation of 1
        - Handles outliers better than MinMaxScaler
        - Preferred for neural networks like LSTM
        - Improves numerical stability during training

        Args:
            df (pd.DataFrame): DataFrame with features
            fit (bool): If True, fit scaler on this data. If False, use pre-fitted scaler.

        Returns:
            tuple: (Normalized DataFrame, fitted StandardScaler object)

        Raises:
            ValueError: If feature columns are missing from DataFrame
        """
        feature_columns = [col for col in self.feature_names if col in df.columns]
        missing = set(self.feature_names) - set(feature_columns)
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        if fit:
            print(f"  Fitting StandardScaler on {len(feature_columns)} features...")
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(df[feature_columns])
        else:
            print(f"  Applying pre-fitted StandardScaler...")
            scaled_data = self.scaler.transform(df[feature_columns])

        normalized_df = df.copy()
        normalized_df[feature_columns] = scaled_data

        print(f"  Normalization complete:")
        print(f"    Mean: {scaled_data.mean():.6f}")
        print(f"    Std: {scaled_data.std():.6f}")

        return normalized_df, self.scaler

    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive statistics for each feature.

        Statistics include:
        - Mean, standard deviation, median
        - Min, max values
        - Count of missing values

        Args:
            df (pd.DataFrame): DataFrame with features

        Returns:
            dict: Statistics for each feature in format:
                {
                    'feature_name': {
                        'mean': float,
                        'std': float,
                        'min': float,
                        'max': float,
                        'median': float,
                        'missing_count': int
                    },
                    ...
                }
        """
        stats = {}
        for feature in self.feature_names:
            if feature in df.columns:
                feature_data = df[feature]
                stats[feature] = {
                    "mean": float(feature_data.mean()),
                    "std": float(feature_data.std()),
                    "min": float(feature_data.min()),
                    "max": float(feature_data.max()),
                    "median": float(feature_data.median()),
                    "missing_count": int(feature_data.isna().sum()),
                }
        return stats

    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validate feature data integrity and consistency.

        Validations:
        1. All required feature columns present in DataFrame
        2. Check for excessive missing values (warn if >10%)
        3. Check for infinite values (NaN after normalization)
        4. Verify feature shapes are consistent

        Args:
            df (pd.DataFrame): DataFrame with features

        Returns:
            bool: True if all validations pass
        """
        print("  Validating features...")

        all_valid = True

        for feature in self.feature_names:
            if feature not in df.columns:
                print(f"    Error: Missing feature column '{feature}'")
                all_valid = False
                continue

            missing = df[feature].isna().sum()
            total = len(df[feature])
            missing_pct = (missing / total) * 100 if total > 0 else 0

            if missing > 0:
                if missing_pct > 10:
                    print(
                        f"    Warning: Feature '{feature}' has {missing} "
                        f"({missing_pct:.1f}%) missing values"
                    )
                else:
                    print(
                        f"    Info: Feature '{feature}' has {missing} "
                        f"({missing_pct:.1f}%) missing values"
                    )

            if np.isinf(df[feature]).any():
                inf_count = np.isinf(df[feature]).sum()
                print(f"    Warning: Feature '{feature}' has {inf_count} infinite values")

        if all_valid:
            print(f"  Feature validation passed: {len(df)} samples, {len(self.feature_names)} features")

        return all_valid

    def get_scaler(self) -> StandardScaler:
        """
        Get the fitted StandardScaler object.

        Returns:
            StandardScaler: Fitted scaler object for later use
        """
        return self.scaler

    def set_scaler(self, scaler: StandardScaler):
        """
        Set a pre-fitted StandardScaler object.

        Useful for applying the same normalization across train/test sets.

        Args:
            scaler (StandardScaler): Pre-fitted scaler object
        """
        self.scaler = scaler
