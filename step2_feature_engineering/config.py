"""
Configuration for Step 2 Feature Engineering
"""

ZIGZAG_CONFIG = {
    'threshold_percent': 5.0,
    'lookback': 20
}

TECHNICAL_INDICATORS = [
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

SEQUENCE_CONFIG = {
    'sequence_length': 30,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'normalize': True
}

OUTPUT_CONFIG = {
    'output_dir': 'step2_output',
    'save_format': 'npy',
    'compress': False
}
