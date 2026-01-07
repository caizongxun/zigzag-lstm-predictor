"""
Data Loader Module: Download and load OHLCV data from HuggingFace Hub.
This module handles the downloading of cryptocurrency OHLCV data from the
v2-crypto-ohlcv-data dataset on HuggingFace, with retry logic and authentication.
Supports chunked reading for optimal memory usage with large Parquet files.
"""

import os
import pandas as pd
import logging
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import pyarrow.parquet as pq


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles downloading and loading OHLCV data from HuggingFace Hub.
    
    Features:
    - Retry logic with exponential backoff
    - HF_TOKEN authentication support
    - Chunked Parquet reading for large files
    - Memory-efficient processing
    
    Attributes:
        dataset_id (str): HuggingFace dataset identifier
        hf_token (str): HuggingFace API token for authentication
        cache_dir (str): Local cache directory for downloaded files
        symbols (list): List of cryptocurrency symbols
        timeframes (list): List of available timeframes
    """
    
    def __init__(self, config: dict):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config (dict): Configuration dictionary containing:
                - dataset_id: HF dataset identifier
                - symbols: List of symbols to download
                - timeframes: List of timeframes
                - cache_dir: Cache directory path
                - max_retries: Max download attempts
                - timeout: Request timeout in seconds
        """
        self.dataset_id = config.get('dataset_id', 'zongowo111/v2-crypto-ohlcv-data')
        self.hf_token = os.getenv('HF_TOKEN', None)
        self.cache_dir = config.get('cache_dir', './hf_cache')
        self.symbols = config.get('symbols', ['BTCUSDT'])
        self.timeframes = config.get('timeframes', ['15m', '1h'])
        self.max_retries = config.get('max_retries', 3)
        self.timeout = config.get('timeout', 300)
        self.chunk_size = config.get('chunk_size', 50000)
        
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"DataLoader initialized - Dataset: {self.dataset_id}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def _download_parquet(self, file_path: str) -> str:
        """
        Download a parquet file from HuggingFace Hub with retry logic.
        Uses exponential backoff for transient failures.
        
        Args:
            file_path (str): Path within dataset (e.g., 'klines/BTCUSDT/BTC_15m.parquet')
            
        Returns:
            str: Local path to the downloaded file
            
        Raises:
            Exception: If download fails after all retries
        """
        try:
            logger.info(f"Downloading: {file_path}")
            local_path = hf_hub_download(
                repo_id=self.dataset_id,
                filename=file_path,
                repo_type='dataset',
                token=self.hf_token,
                cache_dir=self.cache_dir,
                force_download=False,
                resume_download=True,
                local_dir_use_symlinks=False
            )
            logger.info(f"Successfully downloaded to: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Error downloading {file_path}: {str(e)}")
            raise
    
    def download_from_hf(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Download a specific symbol and timeframe data from HuggingFace.
        
        Args:
            symbol (str): Cryptocurrency symbol (e.g., 'BTCUSDT')
            timeframe (str): Timeframe (e.g., '15m', '1h')
            
        Returns:
            Optional[str]: Path to downloaded file, or None if failed
        """
        base_symbol = symbol.split('USDT')[0] if 'USDT' in symbol else symbol
        file_path = f'klines/{symbol}/{base_symbol}_{timeframe}.parquet'
        
        try:
            start_time = time.time()
            local_path = self._download_parquet(file_path)
            elapsed = time.time() - start_time
            
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(
                f"Downloaded {symbol} {timeframe}: "
                f"{file_size_mb:.2f}MB in {elapsed:.1f}s "
                f"({file_size_mb/elapsed:.1f}MB/s)"
            )
            
            return local_path
        except Exception as e:
            logger.error(f"Failed to download {symbol} {timeframe}: {str(e)}")
            return None
    
    def load_parquet(self, file_path: str, use_chunked: bool = False) -> Optional[pd.DataFrame]:
        """
        Load a parquet file into a pandas DataFrame.
        Supports chunked reading for memory efficiency with large files.
        
        Args:
            file_path (str): Path to the parquet file
            use_chunked (bool): Use chunked reading for large files
            
        Returns:
            Optional[pd.DataFrame]: DataFrame or None if failed
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            logger.info(f"Loading parquet file: {file_size_mb:.2f}MB")
            
            if use_chunked and file_size_mb > 100:
                logger.info("Using chunked reading for large file")
                return self._load_parquet_chunked(file_path)
            else:
                df = pd.read_parquet(
                    file_path,
                    engine='pyarrow',
                    use_nullable_dtypes=False,
                    memory_map=True
                )
                logger.info(
                    f"Loaded parquet file: "
                    f"{len(df)} rows, {len(df.columns)} columns"
                )
                return df
        except Exception as e:
            logger.error(f"Error loading parquet file {file_path}: {str(e)}")
            return None
    
    def _load_parquet_chunked(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Load a large parquet file in chunks for memory efficiency.
        
        Args:
            file_path (str): Path to the parquet file
            
        Returns:
            Optional[pd.DataFrame]: Concatenated DataFrame or None if failed
        """
        try:
            parquet_file = pq.ParquetFile(file_path)
            logger.info(
                f"Parquet file has {parquet_file.num_row_groups} row groups"
            )
            
            chunks = []
            for i in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(i)
                df_chunk = table.to_pandas()
                chunks.append(df_chunk)
                logger.info(
                    f"Loaded row group {i+1}/{parquet_file.num_row_groups} "
                    f"({len(df_chunk)} rows)"
                )
            
            df = pd.concat(chunks, ignore_index=True)
            logger.info(
                f"Loaded complete file: "
                f"{len(df)} rows, {len(df.columns)} columns"
            )
            return df
        except Exception as e:
            logger.error(f"Error in chunked loading: {str(e)}")
            return None
    
    def load_ohlcv_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Download and load OHLCV data for a specific symbol and timeframe.
        Automatically standardizes columns and ensures data quality.
        
        Args:
            symbol (str): Cryptocurrency symbol
            timeframe (str): Timeframe (15m or 1h)
            
        Returns:
            Optional[pd.DataFrame]: Loaded and standardized data or None if failed
        """
        file_path = self.download_from_hf(symbol, timeframe)
        if file_path is None:
            return None
        
        df = self.load_parquet(file_path)
        if df is None:
            return None
        
        if not self._validate_ohlcv_columns(df):
            return None
        
        df = self._standardize_columns(df)
        logger.info(
            f"Loaded OHLCV data for {symbol} {timeframe}: {len(df)} candles"
        )
        return df
    
    @staticmethod
    def _validate_ohlcv_columns(df: pd.DataFrame) -> bool:
        """
        Validate that required OHLCV columns exist.
        
        Args:
            df (pd.DataFrame): DataFrame to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        timestamp_cols = ['timestamp', 'time', 'index']
        
        missing_ohlcv = [col for col in required_cols if col not in df.columns]
        has_timestamp = any(col in df.columns for col in timestamp_cols)
        
        if missing_ohlcv:
            logger.error(f"Missing OHLCV columns: {missing_ohlcv}")
            return False
        
        if not has_timestamp:
            logger.warning(
                "No timestamp column detected, will use index as timestamp"
            )
        
        return True
    
    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and data types.
        Converts price columns to float, timestamp to datetime.
        
        Args:
            df (pd.DataFrame): DataFrame to standardize
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        df = df.copy()
        
        col_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'timestamp': 'timestamp',
            'time': 'timestamp'
        }
        
        for old_col, new_col in col_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df = df.rename(columns={old_col: new_col})
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            df = df.reset_index()
            if 'index' in df.columns:
                df = df.rename(columns={'index': 'timestamp'})
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Standardized columns: {list(df.columns[:5])}")
        return df
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get information about a downloaded file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            dict: File information (size, rows, columns, etc.)
        """
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return {}
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        try:
            parquet_file = pq.ParquetFile(file_path)
            df = parquet_file.read().to_pandas()
            
            info = {
                'file_path': file_path,
                'size_mb': round(file_size_mb, 2),
                'rows': len(df),
                'columns': list(df.columns),
                'row_groups': parquet_file.num_row_groups,
                'timestamp': datetime.now().isoformat()
            }
            
            return info
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return {'size_mb': file_size_mb, 'error': str(e)}


def create_data_loader(config: dict) -> DataLoader:
    """
    Factory function to create a DataLoader instance.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        DataLoader: Initialized DataLoader instance
    """
    return DataLoader(config)
