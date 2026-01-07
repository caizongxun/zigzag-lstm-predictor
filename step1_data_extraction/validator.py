"""
Validator Module: Comprehensive data quality checking and validation.
This module validates timestamps, volume, prices, and detects anomalies using
statistical methods.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from sklearn.ensemble import IsolationForest


logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates OHLCV cryptocurrency data quality.
    
    Checks performed:
    - Timestamp validation and continuity
    - Volume validation
    - Price range validation
    - Missing value percentage
    - Anomaly detection
    - Data statistics
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize DataValidator.
        
        Args:
            config (dict): Configuration with keys:
                - allow_missing_percent: Max missing % allowed (default: 5.0)
                - allow_zero_volume_percent: Max zero volume % (default: 20.0)
        """
        self.config = config or {}
        self.allow_missing_percent = self.config.get('allow_missing_percent', 5.0)
        self.allow_zero_volume_percent = self.config.get('allow_zero_volume_percent', 20.0)
    
    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Execute complete validation.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            Dict: Comprehensive quality report
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns)
        }
        
        report['timestamp_validation'] = self._validate_timestamps(df)
        report['volume_validation'] = self._validate_volume(df)
        report['price_validation'] = self._validate_prices(df)
        report['missing_values'] = self._check_missing_values(df)
        report['duplicates'] = self._check_duplicates(df)
        report['anomalies'] = self._detect_anomalies(df)
        report['statistics'] = self._calculate_statistics(df)
        
        report['overall_quality_score'] = self._calculate_quality_score(report)
        
        logger.info(f"Validation complete. Quality Score: {report['overall_quality_score']}/100")
        
        return report
    
    def _validate_timestamps(self, df: pd.DataFrame) -> Dict:
        """
        Validate timestamp continuity and format.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            Dict: Timestamp validation report
        """
        result = {
            'has_timestamp_column': 'timestamp' in df.columns,
            'timestamp_dtype': str(df['timestamp'].dtype) if 'timestamp' in df.columns else None,
            'unique_timestamps': df['timestamp'].nunique() if 'timestamp' in df.columns else 0,
            'is_sorted': True,
            'gaps_detected': 0,
            'status': 'pass'
        }
        
        if not result['has_timestamp_column']:
            result['status'] = 'fail'
            return result
        
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                result['status'] = 'fail'
                result['error'] = f"Cannot convert timestamp: {str(e)}"
                return result
        
        if not df['timestamp'].is_monotonic_increasing:
            result['is_sorted'] = False
            result['status'] = 'warning'
        
        if len(df) > 1:
            time_diffs = df['timestamp'].diff().dropna()
            mode_diff = time_diffs.mode()
            if len(mode_diff) > 0:
                expected_diff = mode_diff[0]
                gaps = (time_diffs != expected_diff).sum()
                result['gaps_detected'] = int(gaps)
                if gaps > 0:
                    result['status'] = 'warning'
                    logger.warning(f"Detected {gaps} timestamp gaps")
        
        return result
    
    def _validate_volume(self, df: pd.DataFrame) -> Dict:
        """
        Validate volume data.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            Dict: Volume validation report
        """
        result = {
            'has_volume': 'volume' in df.columns,
            'volume_dtype': str(df['volume'].dtype) if 'volume' in df.columns else None,
            'zero_volume_count': 0,
            'zero_volume_percent': 0.0,
            'negative_volume_count': 0,
            'status': 'pass'
        }
        
        if not result['has_volume']:
            return result
        
        zero_count = (df['volume'] == 0).sum()
        negative_count = (df['volume'] < 0).sum()
        
        result['zero_volume_count'] = int(zero_count)
        result['zero_volume_percent'] = round(
            (zero_count / len(df) * 100), 2
        ) if len(df) > 0 else 0
        result['negative_volume_count'] = int(negative_count)
        
        if result['zero_volume_percent'] > self.allow_zero_volume_percent:
            result['status'] = 'warning'
            logger.warning(
                f"High zero volume percentage: {result['zero_volume_percent']}%"
            )
        
        if negative_count > 0:
            result['status'] = 'warning'
            logger.warning(f"Detected {negative_count} negative volume entries")
        
        return result
    
    def _validate_prices(self, df: pd.DataFrame) -> Dict:
        """
        Validate price columns.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            Dict: Price validation report
        """
        result = {
            'price_columns_present': [],
            'negative_prices': 0,
            'zero_prices': 0,
            'ohlc_violations': 0,
            'status': 'pass'
        }
        
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                result['price_columns_present'].append(col)
                negative_count = (df[col] < 0).sum()
                zero_count = (df[col] == 0).sum()
                result['negative_prices'] += int(negative_count)
                result['zero_prices'] += int(zero_count)
        
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            ohlc_violations = (
                (df['high'] < df['low']).sum() +
                (df['high'] < df['open']).sum() +
                (df['high'] < df['close']).sum() +
                (df['low'] > df['open']).sum() +
                (df['low'] > df['close']).sum()
            )
            result['ohlc_violations'] = int(ohlc_violations)
            
            if ohlc_violations > 0:
                result['status'] = 'warning'
                logger.warning(f"Detected {ohlc_violations} OHLC violations")
        
        if result['negative_prices'] > 0 or result['zero_prices'] > 0:
            result['status'] = 'warning'
        
        return result
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Check for missing values.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            Dict: Missing values report
        """
        result = {
            'total_missing': int(df.isnull().sum().sum()),
            'missing_percent': round(
                (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100), 2
            ),
            'columns_with_missing': {},
            'status': 'pass'
        }
        
        for col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                percent = round((missing / len(df) * 100), 2)
                result['columns_with_missing'][col] = {
                    'count': int(missing),
                    'percent': percent
                }
        
        if result['missing_percent'] > self.allow_missing_percent:
            result['status'] = 'warning'
            logger.warning(f"High missing value percentage: {result['missing_percent']}%")
        
        return result
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """
        Check for duplicate rows.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            Dict: Duplicates report
        """
        result = {
            'total_duplicates': int(df.duplicated().sum()),
            'duplicate_percent': 0.0,
            'status': 'pass'
        }
        
        result['duplicate_percent'] = round(
            (result['total_duplicates'] / len(df) * 100), 2
        ) if len(df) > 0 else 0
        
        if result['total_duplicates'] > 0:
            result['status'] = 'warning'
            logger.warning(
                f"Detected {result['total_duplicates']} duplicate rows "
                f"({result['duplicate_percent']}%)"
            )
        
        return result
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            Dict: Anomaly detection report
        """
        result = {
            'method': 'isolation_forest',
            'anomalies_detected': 0,
            'anomaly_percent': 0.0,
            'status': 'pass'
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return result
        
        try:
            df_numeric = df[numeric_cols].dropna()
            
            if len(df_numeric) > 10:
                iso_forest = IsolationForest(
                    contamination=0.05,
                    random_state=42,
                    n_estimators=100
                )
                predictions = iso_forest.fit_predict(df_numeric)
                anomalies = (predictions == -1).sum()
                
                result['anomalies_detected'] = int(anomalies)
                result['anomaly_percent'] = round(
                    (anomalies / len(df_numeric) * 100), 2
                )
                
                if result['anomaly_percent'] > 10:
                    result['status'] = 'warning'
                    logger.warning(
                        f"High anomaly rate: {result['anomaly_percent']}%"
                    )
        
        except Exception as e:
            result['error'] = str(e)
            result['status'] = 'error'
            logger.error(f"Anomaly detection failed: {str(e)}")
        
        return result
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate basic statistics.
        
        Args:
            df (pd.DataFrame): Data to validate
            
        Returns:
            Dict: Statistics report
        """
        result = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols:
            result[col] = {
                'mean': round(df[col].mean(), 8),
                'std': round(df[col].std(), 8),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': round(df[col].median(), 8),
                'q25': round(df[col].quantile(0.25), 8),
                'q75': round(df[col].quantile(0.75), 8)
            }
        
        return result
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """
        Calculate overall data quality score (0-100).
        
        Args:
            report (Dict): Validation report
            
        Returns:
            float: Quality score
        """
        score = 100.0
        
        if report['missing_values']['status'] != 'pass':
            score -= 15
        
        if report['timestamp_validation']['status'] != 'pass':
            score -= 15
        
        if report['volume_validation']['status'] != 'pass':
            score -= 10
        
        if report['price_validation']['status'] != 'pass':
            score -= 15
        
        if report['duplicates']['status'] != 'pass':
            score -= 10
        
        if report['anomalies']['status'] == 'warning':
            anomaly_percent = report['anomalies']['anomaly_percent']
            score -= min(15, anomaly_percent)
        elif report['anomalies']['status'] == 'error':
            score -= 10
        
        return max(0, min(100, score))


def validate_timestamps(df: pd.DataFrame) -> Dict:
    """
    Validate timestamp continuity.
    
    Args:
        df (pd.DataFrame): Data to validate
        
    Returns:
        Dict: Timestamp validation result
    """
    validator = DataValidator()
    return validator._validate_timestamps(df)


def validate_volume(df: pd.DataFrame) -> Dict:
    """
    Validate volume logic.
    
    Args:
        df (pd.DataFrame): Data to validate
        
    Returns:
        Dict: Volume validation result
    """
    validator = DataValidator()
    return validator._validate_volume(df)


def check_data_quality(df: pd.DataFrame, config: dict = None) -> Dict:
    """
    Check overall data quality.
    
    Args:
        df (pd.DataFrame): Data to validate
        config (dict): Validation configuration
        
    Returns:
        Dict: Quality report with overall score
    """
    validator = DataValidator(config)
    return validator.validate(df)
