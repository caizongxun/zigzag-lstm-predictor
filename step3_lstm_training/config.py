"""
Configuration for Step 3 LSTM Training
"""

MODEL_CONFIG = {
    'sequence_length': 30,
    'lstm_units_1': 64,
    'lstm_units_2': 32,
    'dense_units_1': 16,
    'dense_units_2': 8,
    'dropout_rate': 0.2,
    'activation': 'relu'
}

TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss_weights': {
        'classification': 1.0,
        'regression': 0.5
    },
    'early_stopping_patience': 15,
    'validation_split': 0.2
}

TEST_CONFIG = {
    'epochs': 5,
    'batch_size': 32,
    'data_fraction': 0.1,
    'sample': True
}

HF_CONFIG = {
    'dataset_id': 'zongowo111/v2-crypto-ohlcv-data',
    'model_dir': 'v2_model',
    'create_dir_if_missing': True
}

OUTPUT_CONFIG = {
    'output_dir': '../step3_output',
    'test_model_prefix': 'test_model',
    'scaler_suffix': 'scaler',
    'config_suffix': 'config',
    'save_format': 'h5'
}
