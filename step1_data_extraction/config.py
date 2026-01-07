import os

STEP1_CONFIG = {
    'hf_dataset_id': 'zongowo111/v2-crypto-ohlcv-data',
    'hf_token': os.getenv('HF_TOKEN', ''),
    'base_path': 'klines',
    'test_symbol': 'BTCUSDT',
    'test_timeframes': ['15m', '1h'],
    'output_dir': './step1_output',
    'output_format': 'csv',
    'chunk_size': 10000,
}

DATASET_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    'DOGEUSDT', 'LINKUSDT', 'LTCUSDT', 'FILUSDT', 'MATICUSDT',
    'UNIUSDT', 'AVAXUSDT', 'SOLUSDT', 'OPUSDT', 'ARBUSDT',
    'NEARUSDT', 'ATOMUSDT', 'SUIUSDT', 'LUNCUSDT', 'GALAUSDT',
    'MANAUSDT', 'PEPEUSDT'
]
