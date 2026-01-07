STEP3_CONFIG = {
    'sequence_length': 30,
    'lstm_layers': [64, 32],
    'dropout_rate': 0.2,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'early_stopping_patience': 15,
    'train_split': 0.7,
    'val_split': 0.15,
    'output_dir': './step3_output',
    'test_symbol': 'BTCUSDT',
    'test_timeframes': ['15m', '1h'],
}

MODEL_STRUCTURE = {
    'input_shape': (30, 13),
    'loss_weights': {
        'classification': 1.0,
        'regression': 0.5
    },
    'metrics': {
        'classification': ['accuracy'],
        'regression': ['mae']
    }
}

HF_UPLOAD_CONFIG = {
    'dataset_id': 'zongowo111/v2-crypto-ohlcv-data',
    'model_dir_pattern': 'v2_model/{symbol}/{timeframe}/',
    'files_to_upload': ['model.h5', 'scaler.pkl', 'config.json']
}
