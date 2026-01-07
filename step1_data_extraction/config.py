"""
Configuration Module for Step 1 Data Extraction
"""

import os
from pathlib import Path


class Config:
    """
    Configuration class for data extraction pipeline.
    """
    
    def __init__(self):
        """Initialize configuration."""
        self.project_root = Path(__file__).parent.parent
        
        self.dataset_id = os.getenv('HF_DATASET_ID', 'zongowo111/v2-crypto-ohlcv-data')
        self.hf_token = os.getenv('HF_TOKEN', '')
        
        self.output_dir = './step1_output'
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.symbols = ['BTCUSDT']
        self.timeframes = ['15m', '1h']
        
        self.loader_config = {
            'dataset_id': self.dataset_id,
            'symbols': self.symbols,
            'timeframes': self.timeframes,
            'cache_dir': './hf_cache',
            'max_retries': 3,
            'timeout': 300
        }
        
        self.cleaner_config = {
            'fill_limit': 5,
            'remove_duplicates': True
        }
        
        self.validator_config = {
            'allow_missing_percent': 5.0,
            'allow_zero_volume_percent': 20.0
        }


DATASET_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'LINKUSDT', 'LTCUSDT', 'FILUSDT', 'MATICUSDT',
    'UNIUSDT', 'AVAXUSDT', 'SOLUSDT', 'OPUSDT', 'ARBUSDT',
    'NEARUSDT', 'ATOMUSDT', 'SUIUSDT', 'LUNCUSDT', 'GALAUSDT',
    'MANAUSDT', 'PEPEUSDT'
]

DATASET_TIMEFRAMES = ['15m', '1h']
