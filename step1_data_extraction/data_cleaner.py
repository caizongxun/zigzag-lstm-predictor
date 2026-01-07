"""
Data Cleaner Module: Clean and validate OHLCV data.
Handles duplicate removal, missing value imputation, OHLC relationship validation,
and timestamp consistency checks.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, List
from datetime import timedelta

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Handles cleaning and validation of OHLCV data.
    
    Attributes:
        df (pd.DataFrame): Input DataFrame
        config (dict): Configuration parameters
        cleaning_report (dict): Report of cleaning operations
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize DataCleaner.
        
        Args:
            config (dict): Configuration dictionary with cleaning parameters
        """
        self.config = config or {}
        self.cleaning_report = {}
        self.df = None
    
    def clean_ohlcv_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform comprehensive OHLCV data cleaning.
        
        Args:
            df (pd.DataFrame): Raw OHLCV data
            
        Returns:
            Tuple[pd.DataFrame, Dict]: Cleaned DataFrame and cleaning report
        """
        self.df = df.copy()
        self.cleaning_report = {
            'initial_rows': len(self.df),
            'operations': []
        }
        
        logger.info(f"Starting data cleaning: {len(self.df)} rows")
        
        self.df = self._remove_duplicates()
        self.df = self._fix_missing_values()
        self.df = self._validate_ohlc_relationships()
        self.df = self._validate_timestamp_consistency()
        self.df = self._sort_by_timestamp()
        
        self.cleaning_report['final_rows'] = len(self.df)
        self.cleaning_report['rows_removed'] = self.cleaning_report['initial_rows'] - self.cleaning_report['final_rows']
        
        logger.info(f"Cleaning completed: {self.cleaning_report['final_rows']} rows remaining")
        
        return self.df, self.cleaning_report
    
    def _remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows based on timestamp.
        
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        """
        initial_count = len(self.df)
        
        self.df = self.df.drop_duplicates(subset=['timestamp'], keep='first')
        self.df = self.df.drop_duplicates(subset=['open', 'high', 'low', 'close', 'volume'], keep='first')
        
        duplicates_removed = initial_count - len(self.df)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
            self.cleaning_report['operations'].append({
                'step': 'remove_duplicates',
                'rows_removed': duplicates_removed
            })
        
        return self.df
    
    def _fix_missing_values(self) -> pd.DataFrame:
        """
        Handle missing values in OHLCV data.
        Uses forward fill followed by interpolation for remaining NaNs.
        
        Returns:
            pd.DataFrame: DataFrame with missing values imputed
        """
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        
        missing_before = self.df[ohlcv_cols].isna().sum().sum()
        
        for col in ohlcv_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(method='ffill', limit=5)
                self.df[col] = self.df[col].interpolate(method='linear', limit_direction='both')
                self.df[col] = self.df[col].fillna(self.df[col].mean())
        
        missing_after = self.df[ohlcv_cols].isna().sum().sum()
        
        if missing_before > 0:
            logger.info(f"Fixed missing values: {missing_before} -> {missing_after}")
            self.cleaning_report['operations'].append({
                'step': 'fix_missing_values',
                'missing_before': missing_before,
                'missing_after': missing_after
            })
        
        return self.df
    
    def _validate_ohlc_relationships(self) -> pd.DataFrame:
        """
        Validate OHLC relationships and fix anomalies.
        - high should be >= max(open, close)
        - low should be <= min(open, close)
        - open and close should be between high and low
        
        Returns:
            pd.DataFrame: DataFrame with invalid OHLC relationships fixed
        """
        anomalies_fixed = 0
        
        for idx in self.df.index:
            o, h, l, c = self.df.loc[idx, ['open', 'high', 'low', 'close']]
            
            if pd.isna([o, h, l, c]).any():
                continue
            
            original = (o, h, l, c)
            
            if h < max(o, c):
                self.df.loc[idx, 'high'] = max(o, c)
                anomalies_fixed += 1
            
            if l > min(o, c):
                self.df.loc[idx, 'low'] = min(o, c)
                anomalies_fixed += 1
            
            if o < self.df.loc[idx, 'low']:
                self.df.loc[idx, 'open'] = self.df.loc[idx, 'low']
                anomalies_fixed += 1
            
            if o > self.df.loc[idx, 'high']:
                self.df.loc[idx, 'open'] = self.df.loc[idx, 'high']
                anomalies_fixed += 1
            
            if c < self.df.loc[idx, 'low']:
                self.df.loc[idx, 'close'] = self.df.loc[idx, 'low']
                anomalies_fixed += 1
            
            if c > self.df.loc[idx, 'high']:
                self.df.loc[idx, 'close'] = self.df.loc[idx, 'high']
                anomalies_fixed += 1
        
        if anomalies_fixed > 0:
            logger.info(f"Fixed {anomalies_fixed} OHLC relationship anomalies")
            self.cleaning_report['operations'].append({
                'step': 'validate_ohlc_relationships',
                'anomalies_fixed': anomalies_fixed
            })
        
        return self.df
    
    def _validate_timestamp_consistency(self) -> pd.DataFrame:
        """
        Validate and fix timestamp consistency issues.
        - Ensures timestamps are sorted
        - Removes duplicates
        - Checks for backward timestamps
        
        Returns:
            pd.DataFrame: DataFrame with consistent timestamps
        """
        issues_found = 0
        
        self.df = self.df.sort_values('timestamp')
        
        timestamp_diffs = self.df['timestamp'].diff()
        backward_entries = (timestamp_diffs < timedelta(0)).sum()
        
        if backward_entries > 0:
            logger.warning(f"Found {backward_entries} backward timestamps")
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            issues_found += backward_entries
        
        duplicate_timestamps = self.df['timestamp'].duplicated().sum()
        
        if duplicate_timestamps > 0:
            logger.warning(f"Found {duplicate_timestamps} duplicate timestamps")
            self.df = self.df.drop_duplicates(subset=['timestamp'], keep='first')
            issues_found += duplicate_timestamps
        
        if issues_found > 0:
            self.cleaning_report['operations'].append({
                'step': 'validate_timestamp_consistency',
                'issues_fixed': issues_found
            })
        
        return self.df
    
    def _sort_by_timestamp(self) -> pd.DataFrame:
        """
        Sort DataFrame by timestamp in ascending order.
        
        Returns:
            pd.DataFrame: Sorted DataFrame
        """
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        return self.df


def clean_ohlcv_data(df: pd.DataFrame, config: dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to clean OHLCV data.
    
    Args:
        df (pd.DataFrame): Raw OHLCV data
        config (dict): Configuration dictionary
        
    Returns:
        Tuple[pd.DataFrame, Dict]: Cleaned data and report
    """
    cleaner = DataCleaner(config)
    return cleaner.clean_ohlcv_data(df)
