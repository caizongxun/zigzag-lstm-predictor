# Step 3: LSTM Training

Multi-task LSTM model training for Zigzag prediction. Includes test model generation and upload tools for HuggingFace.

## Status

Framework structure prepared. Ready for Colab integration and model training.

## Components

- **model_builder.py**: LSTM architecture definition
- **trainer.py**: Training logic and evaluation
- **test_trainer.py**: Quick test model generation
- **colab_upload.py**: Upload utility for Colab
- **local_upload.py**: Upload utility for local environment
- **colab_notebook_cells.py**: Colab-ready cell code
- **main.py**: Pipeline orchestration

## Input

Step 2 Output:
- `../step2_output/BTC_15m_X_sequences.npy`
- `../step2_output/BTC_15m_y_class.npy`
- `../step2_output/BTC_15m_y_reg.npy`
- `../step2_output/BTC_15m_scaler.pkl`

## Output

Test Model:
- `step3_output/test_model_BTC_15m.h5`
- `step3_output/test_scaler_BTC_15m.pkl`
- `step3_output/test_config_BTC_15m.json`

Colab Tools:
- `step3_output/colab_training_cells.py`
- `step3_output/complete_training_script.py`

HuggingFace Upload:
- Model destination: `v2_model/BTC/15m/`
- Files: `model.h5`, `scaler.pkl`, `config.json`

## Model Architecture

### Shared LSTM Layers
- LSTM(64, return_sequences=True)
- Dropout(0.2)
- LSTM(32, return_sequences=False)
- Dropout(0.2)

### Classification Branch (HH vs LL)
- Dense(16, relu)
- Dense(1, sigmoid)
- Loss: binary_crossentropy
- Weight: 1.0

### Regression Branch (Bars to Next)
- Dense(16, relu)
- Dense(8, relu)
- Dense(1, relu)
- Loss: mse
- Weight: 0.5

## Usage

### Generate Test Model
```bash
python main.py
```

### Colab Training
1. Copy cell code from `colab_training_cells.py`
2. Run in Google Colab notebook
3. Model uploads automatically to HuggingFace

### Local Training and Upload
```python
from complete_training_script import train_full_model
train_full_model('BTCUSDT', '15m')
```

## Next Steps

Implementation begins after Step 2 completion.
