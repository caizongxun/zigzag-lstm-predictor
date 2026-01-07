import os
import sys
from pathlib import Path

from .test_trainer import train_test_model
from .colab_notebook_cells import generate_colab_notebook_cells


def generate_full_training_script(timeframe: str = "15m") -> str:
    """
    Generate a full training script for local execution.
    Returns Python code as a string.
    """
    script = f"""
import os
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

from model_builder import build_multitask_lstm
from trainer import compile_and_train, evaluate_model
from config import MODEL_CONFIG, TRAINING_CONFIG

def load_data(data_dir: str, timeframe: str = "{timeframe}"):
    X_path = os.path.join(data_dir, f"{{timeframe}}_X_sequences.npy")
    y_class_path = os.path.join(data_dir, f"{{timeframe}}_y_class.npy")
    y_reg_path = os.path.join(data_dir, f"{{timeframe}}_y_reg.npy")
    scaler_path = os.path.join(data_dir, f"{{timeframe}}_scaler.pkl")

    X = np.load(X_path)
    y_class = np.load(y_class_path).astype(np.int32)
    y_reg = np.load(y_reg_path).astype(np.float32)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return X, y_class, y_reg, scaler

def log_transform_regression(y_reg: np.ndarray) -> np.ndarray:
    return np.log1p(y_reg)

def onehot_encode_classification(y_class: np.ndarray, num_classes: int = 4) -> np.ndarray:
    return tf.keras.utils.to_categorical(y_class, num_classes=num_classes)

def split_data_temporal(X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray):
    n = len(X)
    train_end = int(n * 0.7)
    val_end = train_end + int(n * 0.15)

    y_reg_transformed = log_transform_regression(y_reg)
    y_class_onehot = onehot_encode_classification(y_class, num_classes=4)

    train_data = (X[:train_end], y_class_onehot[:train_end], y_reg_transformed[:train_end])
    val_data = (X[train_end:val_end], y_class_onehot[train_end:val_end], y_reg_transformed[train_end:val_end])
    test_data = (X[val_end:], y_class_onehot[val_end:], y_reg_transformed[val_end:])

    return train_data, val_data, test_data

def build_config_json(timeframe: str, num_samples: int, history, results, scaler, start_time):
    end_time = datetime.now()
    training_time_sec = (end_time - start_time).total_seconds()

    config = {{
        "metadata": {{
            "symbol": "BTCUSDT",
            "timeframe": timeframe,
            "sequence_length": MODEL_CONFIG["sequence_length"],
            "num_features": 13,
            "num_classes": 4,
        }},
        "model_config": MODEL_CONFIG,
        "training_config": {{
            "epochs": len(history.history["loss"]),
            "batch_size": TRAINING_CONFIG["batch_size"],
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "class_weights": {{"0": 1.0, "1": 1.5, "2": 2.0, "3": 2.4}},
        }},
        "training_metadata": {{
            "training_date": start_time.isoformat(),
            "training_duration_seconds": training_time_sec,
            "total_samples": num_samples,
        }},
        "performance_metrics": results,
    }}
    return config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="{timeframe}", choices=["15m", "1h"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    args = parser.parse_args()

    if args.gpu:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"GPUs available: {{len(gpus)}}")
            for gpu in gpus:
                print(f"  {{gpu}}")
        else:
            print("No GPU found. Using CPU.")
    else:
        tf.config.set_visible_devices([], "GPU")
        print("GPU disabled. Using CPU.")

    start_time = datetime.now()
    print(f"Starting full training for {{args.timeframe}} timeframe...")

    data_dir = "../step2_feature_engineering/step2_output"
    output_dir = "../step3_output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {{data_dir}}...")
    X, y_class, y_reg, scaler = load_data(data_dir, args.timeframe)
    print(f"Loaded {{len(X)}} samples, X shape: {{X.shape}}")

    print("Splitting data...")
    train_data, val_data, test_data = split_data_temporal(X, y_class, y_reg)
    print(f"Train: {{len(train_data[0])}}, Val: {{len(val_data[0])}}, Test: {{len(test_data[0])}}")

    print("Building multi-task LSTM model...")
    model = build_multitask_lstm(
        sequence_length=MODEL_CONFIG["sequence_length"],
        num_features=13,
        num_classes=4,
    )
    print("Model architecture:")
    model.summary()

    print(f"Training model for {{args.epochs}} epochs...")
    train_config = {{
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": TRAINING_CONFIG["learning_rate"],
    }}
    model, history, results = compile_and_train(model, train_data, val_data, test_data, train_config)

    prefix = f"model_{{args.timeframe}}"
    model_path = os.path.join(output_dir, f"{{prefix}}.h5")
    scaler_path = os.path.join(output_dir, f"{{prefix}}_scaler.pkl")
    config_path = os.path.join(output_dir, f"{{prefix}}_config.json")

    print(f"Saving model to {{model_path}}...")
    model.save(model_path)

    print(f"Saving scaler to {{scaler_path}}...")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    config_json = build_config_json(args.timeframe, len(X), history, results, scaler, start_time)
    print(f"Saving config to {{config_path}}...")
    with open(config_path, "w") as f:
        json.dump(config_json, f, indent=2)

    print(f"\nFinal Results:")
    print(f"Classification Accuracy: {{results['classification']['accuracy']:.4f}}")
    print(f"AUC-ROC: {{results['classification']['auc_roc_ovr']:.4f}}")
    print(f"Regression MSE: {{results['regression']['mse']:.4f}}")
    print(f"Regression R2: {{results['regression']['r2']:.4f}}")
    print(f"Training duration: {{(datetime.now() - start_time).total_seconds():.2f}} seconds")
"""
    return script


def main():
    """
    Main orchestration function.
    1. Run test trainer to generate quick test model
    2. Generate Colab notebook cells
    3. Generate full training script for local
    4. Create README
    """
    print("[MAIN] Starting LSTM training setup...")

    # Step 1: Run test trainer
    print("\n[MAIN] Step 1: Training quick test model (10% data, 5 epochs)...")
    test_result = train_test_model(timeframe="15m")
    print(f"[MAIN] Test model saved to {test_result['model_path']}")
    print(f"[MAIN] Test model accuracy: {test_result['results']['classification']['accuracy']:.4f}")

    # Step 2: Generate Colab notebook cells
    print("\n[MAIN] Step 2: Generating Colab notebook cells...")
    colab_cells = generate_colab_notebook_cells()
    output_dir = Path("../step3_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colab_cells_path = output_dir / "colab_training_cells.py"
    with open(colab_cells_path, "w") as f:
        f.write(colab_cells)
    print(f"[MAIN] Colab cells saved to {colab_cells_path}")

    # Step 3: Generate full training script
    print("\n[MAIN] Step 3: Generating full training script for local execution...")
    full_script = generate_full_training_script(timeframe="15m")
    full_script_path = output_dir / "full_training_script.py"
    with open(full_script_path, "w") as f:
        f.write(full_script)
    print(f"[MAIN] Full training script saved to {full_script_path}")

    # Step 4: Generate README
    print("\n[MAIN] Step 4: Generating README...")
    readme_content = """
# LSTM Multi-Task Training Pipeline

This directory contains tools and scripts for training a multi-task LSTM model on BTCUSDT time-series data.

## Overview

The pipeline includes:
- **model_builder.py**: Multi-task LSTM architecture (shared LSTM + classification + regression branches)
- **trainer.py**: Training logic with class weights, early stopping, and learning rate reduction
- **test_trainer.py**: Quick test training on 10% data for validation
- **colab_upload.py / local_upload.py**: Upload trained models to Hugging Face Hub
- **colab_notebook_cells.py**: Pre-built Colab notebook cells (7 cells, ready to copy-paste)
- **full_training_script.py**: Local training script with GPU support

## Quick Start

### Option 1: Google Colab (Recommended for Quick Training)

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Read the content from `colab_training_cells.py`
4. Copy-paste each cell into your Colab notebook and run sequentially
5. Specify your Hugging Face token when prompted
6. Model will be automatically uploaded to HF Hub

**GPU Setup in Colab:**
- Go to Runtime > Change Runtime Type > GPU (T4 or higher recommended)
- Check GPU availability: `!nvidia-smi`

### Option 2: Local Machine (Full Control)

1. Ensure you have GPU drivers and CUDA installed (optional but recommended)
2. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas scikit-learn huggingface-hub
   ```

3. Run full training:
   ```bash
   python full_training_script.py --timeframe 15m --epochs 100 --batch_size 32 --gpu
   ```

4. Upload to Hugging Face:
   ```bash
   export HF_TOKEN="your_hf_token"
   python local_upload.py step3_output/model_15m.h5 step3_output/model_15m_scaler.pkl step3_output/model_15m_config.json BTCUSDT 15m
   ```

### Option 3: Quick Test

1. Run test trainer on 10% data:
   ```bash
   python test_trainer.py
   ```
   This will save test model to `step3_output/test_model_15m.h5`

## Data Specifications

**Input Data (from Step 2):**
- 15m: 219614 samples, shape (219614, 30, 13)
- 1h: 54886 samples, shape (54886, 30, 13)

**Classification Target (y_class):**
- Class 0 (HH): 41.89% - Higher High (trend continuation bullish)
- Class 1 (LL): 24.95% - Lower Low (trend continuation bearish)
- Class 2 (HL): 18.59% - Higher Low (potential reversal bullish)
- Class 3 (LH): 14.57% - Lower High (potential reversal bearish)
- Note: Class imbalance handled with weights {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.4}

**Regression Target (y_reg):**
- Distance to next zigzag turning point (bars)
- 15m: mean 391.5, range 1-4042
- 1h: mean 99.6, range 1-1010
- Applied log1p transformation during training

## Model Architecture

**Shared LSTM Layers:**
- LSTM(64, return_sequences=True)
- Dropout(0.2)
- LSTM(32, return_sequences=False)
- Dropout(0.2)

**Classification Branch:**
- Dense(16, relu)
- Dropout(0.2)
- Dense(4, softmax) -> 4 class outputs

**Regression Branch:**
- Dense(16, relu)
- Dropout(0.2)
- Dense(8, relu)
- Dense(1, relu) -> distance prediction

## Training Configuration

**Hyperparameters:**
- Optimizer: Adam (learning_rate=0.001)
- Classification loss: categorical_crossentropy
- Regression loss: mse
- Loss weights: classification=1.0, regression=0.5
- Batch size: 32 (adjustable)
- Epochs: 100 (adjustable)

**Callbacks:**
- EarlyStopping: patience=15, monitor=val_loss
- ReduceLROnPlateau: factor=0.5, patience=5

## Data Split Strategy

Temporal split (NO shuffle) to preserve time-series integrity:
- Training: 70% (first samples)
- Validation: 15% (middle samples)
- Testing: 15% (last samples)

## Output Files

After training, the following files are generated:
- `model_15m.h5`: Trained Keras model
- `model_15m_scaler.pkl`: StandardScaler from step2
- `model_15m_config.json`: Training metadata and metrics

## Hugging Face Upload

Models are uploaded to: `zongowo111/v2-crypto-ohlcv-data`

Directory structure:
```
v2_model/
  BTCUSDT/
    15m/
      model.h5
      scaler.pkl
      config.json
    1h/
      model.h5
      scaler.pkl
      config.json
```

## Performance Metrics

**Classification Metrics:**
- Accuracy
- Per-class precision, recall, F1-score
- AUC-ROC (one-vs-rest)

**Regression Metrics:**
- MSE, MAE, RMSE
- R2 score

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce batch_size: `--batch_size 16` or `--batch_size 8`
- Use smaller data fraction during test: modify `TEST_CONFIG['data_fraction']`

### GPU Not Detected
- Check CUDA installation: `nvidia-smi`
- In Colab, confirm Runtime > GPU is selected
- In local, verify TensorFlow GPU support: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

### Upload to HF Fails
- Verify HF token: `huggingface-cli login`
- Ensure repo `zongowo111/v2-crypto-ohlcv-data` exists and you have write access
- Check network connectivity

## References

- TensorFlow Keras: https://www.tensorflow.org/guide/keras
- Multi-task Learning: https://arxiv.org/abs/1595.06431
- Time Series LSTM: https://www.tensorflow.org/tutorials/structured_data/time_series
- Hugging Face Hub: https://huggingface.co/docs/huggingface_hub
"""
    readme_path = Path("README.md")
    with open(readme_path, "w") as f:
        f.write(readme_content)
    print(f"[MAIN] README saved to {readme_path}")

    # Step 5: Create EXECUTION_LOG
    print("\n[MAIN] Step 5: Creating EXECUTION_LOG...")
    execution_log = f"""
# LSTM Training Execution Log

Date: {datetime.now().isoformat()}

## Test Model Training

- Duration: [Automatically recorded]
- Data Fraction: 10%
- Epochs: 5
- Timeframe: 15m
- Accuracy: {test_result['results']['classification']['accuracy']:.4f}
- Regression MSE: {test_result['results']['regression']['mse']:.4f}

## Model Architecture Summary

- Shared LSTM: LSTM(64) -> LSTM(32) with Dropout(0.2)
- Classification Branch: Dense(16) -> Dense(4, softmax)
- Regression Branch: Dense(16) -> Dense(8) -> Dense(1, relu)
- Total Parameters: [Will be updated after full training]

## Generated Files

- test_model_15m.h5: {test_result['model_path']}
- colab_training_cells.py: {colab_cells_path}
- full_training_script.py: {full_script_path}
- README.md: {readme_path}

## Full Training Configuration (Recommended)

```bash
python full_training_script.py --timeframe 15m --epochs 100 --batch_size 32 --gpu
```

**Parameters:**
- timeframe: 15m (or 1h)
- epochs: 100 (recommended)
- batch_size: 32 (adjust based on GPU memory)
- gpu: Use GPU if available

## Colab Environment Notes

- GPU Runtime: Select T4 or higher
- Install: `!pip install tensorflow numpy pandas scikit-learn huggingface-hub`
- Mount Drive: `from google.colab import drive; drive.mount('/content/drive')`
- Check GPU: `!nvidia-smi`

## Local Environment Notes

- GPU Support: Install CUDA and cuDNN
- Recommended: RTX 3070 or higher for faster training
- CPU-only is supported but ~10-20x slower

## Next Steps

1. Execute test training locally to verify setup
2. Upload test model to HF (optional)
3. Use Colab for full training (recommended due to free GPU)
4. Or run full_training_script.py locally with GPU
5. Upload final model to HF Hub
6. Verify model files in v2_model/BTCUSDT/{timeframe}/ on HF
"""
    execution_log_path = Path("EXECUTION_LOG.md")
    with open(execution_log_path, "w") as f:
        f.write(execution_log)
    print(f"[MAIN] EXECUTION_LOG saved to {execution_log_path}")

    print("\n[MAIN] Setup complete!")
    print("[MAIN] Next steps:")
    print(f"  1. Review colab_training_cells.py for Colab training")
    print(f"  2. Or run: python full_training_script.py --gpu")
    print(f"  3. Upload model using: python local_upload.py <paths>")


if __name__ == "__main__":
    main()
