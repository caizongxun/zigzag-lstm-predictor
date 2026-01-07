"""
Data Cleaner Module: Clean and validate OHLCV data.
This module handles duplicate removal, missing value handling, OHLC relationship
validation, and timestamp continuity checks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict


logger = logging.getLogger(__name__)


class OHLCVCleaner:
    """
    Cleans OHLCV (Open-High-Low-Close-Volume) cryptocurrency data.
    
    Responsibilities:
    - Remove duplicate rows
    - Handle missing values (forward fill or interpolation)
    - Validate OHLC relationships
    - Check timestamp continuity
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize OHLCVCleaner.
        
        Args:
            config (dict): Configuration with keys:
                - fill_limit: Max consecutive NaN to fill (default: 5)
                - remove_duplicates: Whether to remove duplicates (default: True)
        """
        self.config = config or {}
        self.fill_limit = self.config.get('fill_limit', 5)
        self.remove_duplicates = self.config.get('remove_duplicates', True)
        self.cleaning_report = {}
    
    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute complete cleaning pipeline.
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Cleaned data and detailed report
        """
        df = df.copy()
        initial_rows = len(df)
        self.cleaning_report = {
            'initial_rows': initial_rows,
            'steps': {}
        }
        
        df = self._remove_duplicates(df)
        df = self._fix_missing_values(df)
        df = self._fix_ohlc_relationships(df)
        df = self._ensure_timestamp_order(df)
        df = self._remove_invalid_rows(df)
        
        final_rows = len(df)
        self.cleaning_report['final_rows'] = final_rows
        self.cleaning_report['rows_removed'] = initial_rows - final_rows
        self.cleaning_report['removal_percentage'] = round(
            (initial_rows - final_rows) / initial_rows * 100, 2
        ) if initial_rows > 0 else 0
        
        logger.info(f"Cleaning complete: {initial_rows} -> {final_rows} rows")
        logger.info(f"Report: {self.cleaning_report}")
        
        return df, self.cleaning_report
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows based on timestamp.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame without duplicates
        """
        if not self.remove_duplicates:
            self.cleaning_report['steps']['duplicates_removed'] = 0
            return df
        
        initial_rows = len(df)
        
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
        else:
            df = df.drop_duplicates(keep='last')
        
        duplicates_removed = initial_rows - len(df)
        self.cleaning_report['steps']['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df
    
    def _fix_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in OHLCV columns.
        Strategy: forward fill with limit, then interpolation for remaining.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        """
        df = df.copy()
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in ohlcv_cols if col in df.columns]
        
        initial_missing = df[missing_cols].isnull().sum().sum()
        
        for col in missing_cols:
            missing_in_col = df[col].isnull().sum()
            if missing_in_col > 0:
                df[col] = df[col].fillna(method='ffill', limit=self.fill_limit)
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                remaining_missing = df[col].isnull().sum()
                if remaining_missing > 0:
                    df[col] = df[col].fillna(df[col].mean())
        
        final_missing = df[missing_cols].isnull().sum().sum()
        
        self.cleaning_report['steps']['missing_values_fixed'] = {
            'initial_missing': initial_missing,
            'final_missing': final_missing
        }
        
        if initial_missing > 0:
            logger.info(f"Fixed {initial_missing} missing values ({final_missing} remain)")
        
        return df
    
    def _fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix OHLC relationships:
        - high >= open, close, low
        - low <= open, close, high
        - high >= low
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with corrected OHLC relationships
        """
        df = df.copy()
        invalid_rows = []
        
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            
            invalid_high = df['high'] < df[['open', 'close', 'low']].max(axis=1)
            invalid_low = df['low'] > df[['open', 'close', 'high']].min(axis=1)
            invalid_hl = df['high'] < df['low']
            
            invalid_mask = invalid_high | invalid_low | invalid_hl
            invalid_count = invalid_mask.sum()
            
            if invalid_count > 0:
                invalid_rows = df[invalid_mask].index.tolist()
                logger.warning(f"Found {invalid_count} rows with invalid OHLC relationships")
                
                for idx in invalid_rows:
                    o, h, l, c = df.loc[idx, ['open', 'high', 'low', 'close']]
                    new_h = max(o, h, l, c)
                    new_l = min(o, h, l, c)
                    df.loc[idx, 'high'] = new_h
                    df.loc[idx, 'low'] = new_l
        
        self.cleaning_report['steps']['ohlc_fixed'] = len(invalid_rows)
        
        return df
    
    def _ensure_timestamp_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure timestamps are in chronological order.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame sorted by timestamp
        """
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
            self.cleaning_report['steps']['timestamp_ordered'] = True
        
        return df
    
    def _remove_invalid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with zero or negative price values.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame without invalid rows
        """
        initial_rows = len(df)
        
        price_cols = ['open', 'high', 'low', 'close']
        existing_price_cols = [col for col in price_cols if col in df.columns]
        
        if existing_price_cols:
            df = df[(df[existing_price_cols] > 0).all(axis=1)]
        
        rows_removed = initial_rows - len(df)
        self.cleaning_report['steps']['invalid_rows_removed'] = rows_removed
        
        if rows_removed > 0:
            logger.info(f"Removed {rows_removed} rows with invalid prices")
        
        return df


def clean_ohlcv_data(df: pd.DataFrame, config: dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean OHLCV data using OHLCVCleaner.
    
    Args:
        df (pd.DataFrame): Raw OHLCV data
        config (dict): Cleaning configuration
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned data and report
    """
    cleaner = OHLCVCleaner(config)
    return cleaner.clean(df)
