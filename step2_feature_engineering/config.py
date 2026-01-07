STEP2_CONFIG = {
    'zigzag_threshold': 5,
    'lookback_period': 20,
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr_period': 14,
    'output_dir': './step2_output',
    'sequence_length': 30,
    'test_symbol': 'BTCUSDT',
    'test_timeframes': ['15m', '1h'],
}

TECHNICAL_INDICATORS = [
    'rsi',
    'macd',
    'macd_signal',
    'macd_histogram',
    'atr',
    'high_low_ratio',
    'close_range',
    'returns',
    'returns_volatility',
    'highest',
    'lowest',
    'volume_ratio',
    'price_acceleration'
]
