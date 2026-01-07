"""
Validator Module: Perform comprehensive data quality checks.
Validates timestamps, volume consistency, and overall data integrity.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from datetime import timedelta

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates OHLCV data quality and consistency.
    
    Attributes:
        df (pd.DataFrame): DataFrame to validate
        validation_report (dict): Comprehensive validation report
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataValidator.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
        """
        self.df = df.copy()
        self.validation_report = {}
    
    def validate_timestamps(self) -> Dict:
        """
        Validate timestamp consistency and continuity.
        
        Returns:
            Dict: Validation results
        """
        results = {
            'total_rows': len(self.df),
            'timestamp_issues': []
        }
        
        if 'timestamp' not in self.df.columns:
            results['timestamp_issues'].append('No timestamp column')
            return results
        
        timestamps = pd.to_datetime(self.df['timestamp'])
        
        if timestamps.isna().any():
            na_count = timestamps.isna().sum()
            results['timestamp_issues'].append(f'{na_count} NaT values found')
        
        if not (timestamps == timestamps.sort_values()).all():
            results['timestamp_issues'].append('Timestamps not in ascending order')
        
        duplicates = timestamps.duplicated().sum()
        if duplicates > 0:
            results['timestamp_issues'].append(f'{duplicates} duplicate timestamps')
        
        time_diffs = timestamps.diff()
        backward_diffs = (time_diffs < timedelta(0)).sum()
        if backward_diffs > 0:
            results['timestamp_issues'].append(f'{backward_diffs} backward time differences')
        
        zero_diffs = (time_diffs == timedelta(0)).sum()
        if zero_diffs > 0:
            results['timestamp_issues'].append(f'{zero_diffs} zero time differences')
        
        if len(time_diffs) > 1:
            valid_diffs = time_diffs.dropna()
            if len(valid_diffs) > 0:
                expected_diff = valid_diffs.mode()
                if len(expected_diff) > 0:
                    expected_interval = expected_diff[0]
                    unexpected_diffs = (valid_diffs != expected_interval).sum()
                    if unexpected_diffs > 0:
                        results['unexpected_intervals'] = unexpected_diffs
                        results['expected_interval'] = str(expected_interval)
        
        results['is_valid'] = len(results['timestamp_issues']) == 0
        
        logger.info(f"Timestamp validation: {results['is_valid']}")
        return results
    
    def validate_volume(self) -> Dict:
        """
        Validate volume data consistency and logic.
        
        Returns:
            Dict: Validation results
        """
        results = {
            'volume_issues': []
        }
        
        if 'volume' not in self.df.columns:
            results['volume_issues'].append('No volume column')
            results['is_valid'] = False
            return results
        
        volume = self.df['volume']
        
        if volume.isna().any():
            na_count = volume.isna().sum()
            results['volume_issues'].append(f'{na_count} NaN values')
        
        if (volume < 0).any():
            negative_count = (volume < 0).sum()
            results['volume_issues'].append(f'{negative_count} negative volumes')
        
        if (volume == 0).any():
            zero_count = (volume == 0).sum()
            zero_percentage = (zero_count / len(volume)) * 100
            results['zero_volume_percentage'] = round(zero_percentage, 2)
            if zero_percentage > 10:
                results['volume_issues'].append(f'{zero_percentage:.2f}% zero volumes (high)')
        
        non_zero_volume = volume[volume > 0]
        if len(non_zero_volume) > 0:
            results['volume_stats'] = {
                'mean': round(non_zero_volume.mean(), 2),
                'median': round(non_zero_volume.median(), 2),
                'std': round(non_zero_volume.std(), 2),
                'min': round(non_zero_volume.min(), 2),
                'max': round(non_zero_volume.max(), 2)
            }
            
            outlier_threshold = results['volume_stats']['mean'] + 5 * results['volume_stats']['std']
            outliers = (volume > outlier_threshold).sum()
            if outliers > 0:
                results['volume_outliers'] = outliers
        
        results['is_valid'] = len(results['volume_issues']) == 0
        
        logger.info(f"Volume validation: {results['is_valid']}")
        return results
    
    def check_data_quality(self) -> Dict:
        """
        Perform comprehensive data quality check.
        Returns a detailed quality report.
        
        Returns:
            Dict: Comprehensive quality report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns)
        }
        
        report['missing_data'] = self._check_missing_values()
        report['duplicate_data'] = self._check_duplicates()
        report['ohlc_anomalies'] = self._check_ohlc_anomalies()
        report['timestamp_validation'] = self.validate_timestamps()
        report['volume_validation'] = self.validate_volume()
        report['price_validation'] = self._check_price_consistency()
        
        report['overall_quality_score'] = self._calculate_quality_score(report)
        
        self.validation_report = report
        return report
    
    def _check_missing_values(self) -> Dict:
        """
        Check for missing values across all columns.
        
        Returns:
            Dict: Missing value statistics
        """
        missing = {}
        for col in self.df.columns:
            na_count = self.df[col].isna().sum()
            if na_count > 0:
                percentage = (na_count / len(self.df)) * 100
                missing[col] = {
                    'count': na_count,
                    'percentage': round(percentage, 2)
                }
        
        return {
            'total_missing': sum(self.df.isna().sum()),
            'columns_with_missing': missing,
            'has_missing': len(missing) > 0
        }
    
    def _check_duplicates(self) -> Dict:
        """
        Check for duplicate rows.
        
        Returns:
            Dict: Duplicate statistics
        """
        total_duplicates = self.df.duplicated().sum()
        
        timestamp_duplicates = 0
        if 'timestamp' in self.df.columns:
            timestamp_duplicates = self.df['timestamp'].duplicated().sum()
        
        return {
            'total_duplicates': total_duplicates,
            'timestamp_duplicates': timestamp_duplicates,
            'duplicate_percentage': round((total_duplicates / len(self.df)) * 100, 2) if len(self.df) > 0 else 0
        }
    
    def _check_ohlc_anomalies(self) -> Dict:
        """
        Check for OHLC relationship anomalies.
        
        Returns:
            Dict: OHLC anomaly statistics
        """
        anomalies = []
        
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.df.columns for col in required_cols):
            return {'error': 'Missing OHLC columns'}
        
        o, h, l, c = self.df['open'], self.df['high'], self.df['low'], self.df['close']
        
        high_below_oc = ((h < o) | (h < c)).sum()
        if high_below_oc > 0:
            anomalies.append(f'High below open/close: {high_below_oc} cases')
        
        low_above_oc = ((l > o) | (l > c)).sum()
        if low_above_oc > 0:
            anomalies.append(f'Low above open/close: {low_above_oc} cases')
        
        close_outside_hl = ((c < l) | (c > h)).sum()
        if close_outside_hl > 0:
            anomalies.append(f'Close outside H-L range: {close_outside_hl} cases')
        
        return {
            'total_anomalies': len(anomalies),
            'anomaly_details': anomalies,
            'has_anomalies': len(anomalies) > 0
        }
    
    def _check_price_consistency(self) -> Dict:
        """
        Check for price data consistency.
        
        Returns:
            Dict: Price consistency statistics
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in self.df.columns for col in required_cols):
            return {'error': 'Missing price columns'}
        
        prices = self.df[required_cols]
        
        has_negative = (prices < 0).any().any()
        has_zero = (prices == 0).any().any()
        
        returns = self.df['close'].pct_change()
        extreme_returns = (returns.abs() > 0.5).sum()
        
        return {
            'has_negative_prices': has_negative,
            'has_zero_prices': has_zero,
            'extreme_returns_count': extreme_returns if extreme_returns > 0 else 0,
            'extreme_return_percentage': round((extreme_returns / len(returns)) * 100, 2) if len(returns) > 0 else 0,
            'is_valid': not (has_negative or (extreme_returns > len(returns) * 0.1))
        }
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Args:
            report (Dict): Quality report
            
        Returns:
            float: Quality score
        """
        score = 100.0
        
        if report['missing_data']['has_missing']:
            score -= report['missing_data']['total_missing'] * 0.1
        
        score -= report['duplicate_data']['duplicate_percentage'] * 0.5
        
        if report['ohlc_anomalies']['has_anomalies']:
            score -= report['ohlc_anomalies']['total_anomalies'] * 0.5
        
        if not report['timestamp_validation']['is_valid']:
            score -= 20
        
        if not report['volume_validation']['is_valid']:
            score -= 10
        
        if not report['price_validation']['is_valid']:
            score -= 15
        
        return max(0, min(100, round(score, 2)))


def validate_timestamps(df: pd.DataFrame) -> Dict:
    """
    Convenience function to validate timestamps.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        Dict: Validation results
    """
    validator = DataValidator(df)
    return validator.validate_timestamps()


def validate_volume(df: pd.DataFrame) -> Dict:
    """
    Convenience function to validate volume.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        Dict: Validation results
    """
    validator = DataValidator(df)
    return validator.validate_volume()


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Convenience function to check overall data quality.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        Dict: Comprehensive quality report
    """
    validator = DataValidator(df)
    return validator.check_data_quality()


from datetime import datetime
