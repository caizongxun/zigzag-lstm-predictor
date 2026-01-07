import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf

from .config import TEST_CONFIG, OUTPUT_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from .model_builder import build_multitask_lstm
from .trainer import compile_and_train, evaluate_model


def _load_step2_data(
    data_dir: str, timeframe: str = "15m", fraction: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Load feature-engineered data from step2_output.

    Args:
        data_dir: Path to step2_output directory.
        timeframe: "15m" or "1h".
        fraction: Data fraction to load (0.0-1.0) for quick testing.

    Returns:
        (X_sequences, y_class, y_reg, scaler)
    """
    X_path = os.path.join(data_dir, f"{timeframe}_X_sequences.npy")
    y_class_path = os.path.join(data_dir, f"{timeframe}_y_class.npy")
    y_reg_path = os.path.join(data_dir, f"{timeframe}_y_reg.npy")
    scaler_path = os.path.join(data_dir, f"{timeframe}_scaler.pkl")

    X = np.load(X_path)
    y_class = np.load(y_class_path).astype(np.int32)
    y_reg = np.load(y_reg_path).astype(np.float32)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Sample fraction if needed
    if fraction < 1.0:
        n_samples = int(len(X) * fraction)
        X = X[:n_samples]
        y_class = y_class[:n_samples]
        y_reg = y_reg[:n_samples]

    return X, y_class, y_reg, scaler


def _log_transform_regression(y_reg: np.ndarray) -> np.ndarray:
    """Apply log1p transformation to regression targets."""
    return np.log1p(y_reg)


def _onehot_encode_classification(y_class: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Convert class labels to one-hot encoding."""
    return tf.keras.utils.to_categorical(y_class, num_classes=num_classes)


def _split_data_temporal(
    X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray, ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    Split data chronologically (no shuffle) into train/val/test.

    Args:
        X: Feature sequences.
        y_class: Classification labels (int, not one-hot yet).
        y_reg: Regression targets (will be log1p-transformed).
        ratios: (train, val, test) fractions.

    Returns:
        train_data, val_data, test_data
    """
    n = len(X)
    train_ratio, val_ratio, _ = ratios
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Log-transform regression targets
    y_reg_transformed = _log_transform_regression(y_reg)

    # Convert classification to one-hot
    y_class_onehot = _onehot_encode_classification(y_class, num_classes=4)

    # Temporal split
    train_data = (
        X[:train_end],
        y_class_onehot[:train_end],
        y_reg_transformed[:train_end],
    )
    val_data = (
        X[train_end:val_end],
        y_class_onehot[train_end:val_end],
        y_reg_transformed[train_end:val_end],
    )
    test_data = (
        X[val_end:],
        y_class_onehot[val_end:],
        y_reg_transformed[val_end:],
    )

    return train_data, val_data, test_data


def _build_config_json(
    timeframe: str,
    num_samples: int,
    history: tf.keras.callbacks.History,
    results: Dict[str, Any],
    scaler,
    start_time: datetime,
) -> Dict[str, Any]:
    """
    Build config.json metadata.

    Args:
        timeframe: "15m" or "1h".
        num_samples: Total samples used.
        history: Keras history object.
        results: Evaluation results from evaluate_model().
        scaler: Fitted StandardScaler from step2.
        start_time: Training start datetime.

    Returns:
        Configuration dictionary.
    """
    end_time = datetime.now()
    training_time_sec = (end_time - start_time).total_seconds()

    config = {
        "metadata": {
            "symbol": "BTCUSDT",
            "timeframe": timeframe,
            "sequence_length": MODEL_CONFIG["sequence_length"],
            "num_features": 13,
            "num_classes": 4,
        },
        "model_config": {
            "lstm_units_1": MODEL_CONFIG["lstm_units_1"],
            "lstm_units_2": MODEL_CONFIG["lstm_units_2"],
            "dense_units_1": MODEL_CONFIG["dense_units_1"],
            "dense_units_2": MODEL_CONFIG["dense_units_2"],
            "dropout_rate": MODEL_CONFIG["dropout_rate"],
        },
        "training_config": {
            "epochs": len(history.history["loss"]),
            "batch_size": TRAINING_CONFIG["batch_size"],
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "class_weights": {"0": 1.0, "1": 1.5, "2": 2.0, "3": 2.4},
            "early_stopping_patience": TRAINING_CONFIG["early_stopping_patience"],
        },
        "training_metadata": {
            "training_date": start_time.isoformat(),
            "training_duration_seconds": training_time_sec,
            "total_samples": num_samples,
        },
        "performance_metrics": results,
        "feature_names": [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "rsi",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
        ],
        "data_statistics": {
            "class_distribution": {
                "HH (0)": 0.4189,
                "LL (1)": 0.2495,
                "HL (2)": 0.1859,
                "LH (3)": 0.1457,
            },
            "regression_mean": float(np.log1p(391.5 if timeframe == "15m" else 99.6)),
            "regression_range": "log1p transformed",
        },
    }
    return config


def train_test_model(timeframe: str = "15m") -> Dict[str, Any]:
    """
    Train a quick test model using a fraction of data for validation.

    Args:
        timeframe: "15m" or "1h".

    Returns:
        Dictionary with model path, scaler path, config path, and metrics.
    """
    start_time = datetime.now()
    print(f"[TEST TRAINER] Starting test training for {timeframe} timeframe...")

    # Set up paths
    step2_output_dir = "../step2_feature_engineering/step2_output"
    output_dir = OUTPUT_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print(f"[TEST TRAINER] Loading data from {step2_output_dir}...")
    X, y_class, y_reg, scaler = _load_step2_data(
        step2_output_dir, timeframe, fraction=TEST_CONFIG["data_fraction"]
    )
    print(f"[TEST TRAINER] Loaded {len(X)} samples, shape: X {X.shape}")

    # Split data
    print("[TEST TRAINER] Splitting data (70% train, 15% val, 15% test)...")
    train_data, val_data, test_data = _split_data_temporal(X, y_class, y_reg)
    print(
        f"[TEST TRAINER] Train: {len(train_data[0])}, Val: {len(val_data[0])}, Test: {len(test_data[0])}"
    )

    # Build model
    print("[TEST TRAINER] Building multi-task LSTM model...")
    model = build_multitask_lstm(
        sequence_length=MODEL_CONFIG["sequence_length"],
        num_features=13,
        num_classes=4,
    )
    print("[TEST TRAINER] Model built. Summary:")
    model.summary()

    # Train model
    print(f"[TEST TRAINER] Training for {TEST_CONFIG['epochs']} epochs...")
    train_config = {
        "epochs": TEST_CONFIG["epochs"],
        "batch_size": TEST_CONFIG["batch_size"],
        "learning_rate": TRAINING_CONFIG["learning_rate"],
    }
    model, history, results = compile_and_train(model, train_data, val_data, test_data, train_config)
    print("[TEST TRAINER] Training complete.")

    # Save model, scaler, config
    prefix = f"{OUTPUT_CONFIG['test_model_prefix']}_{timeframe}"
    model_path = os.path.join(output_dir, f"{prefix}.{OUTPUT_CONFIG['save_format']}")
    scaler_path = os.path.join(output_dir, f"{prefix}_{OUTPUT_CONFIG['scaler_suffix']}.pkl")
    config_path = os.path.join(output_dir, f"{prefix}_{OUTPUT_CONFIG['config_suffix']}.json")

    print(f"[TEST TRAINER] Saving model to {model_path}...")
    model.save(model_path)

    print(f"[TEST TRAINER] Saving scaler to {scaler_path}...")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    config_json = _build_config_json(timeframe, len(X), history, results, scaler, start_time)
    print(f"[TEST TRAINER] Saving config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(config_json, f, indent=2)

    print(f"[TEST TRAINER] Test training complete.")
    print(f"[TEST TRAINER] Model accuracy: {results['classification']['accuracy']:.4f}")
    print(
        f"[TEST TRAINER] Model regression MSE: {results['regression']['mse']:.4f}"
    )

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "config_path": config_path,
        "results": results,
        "history": history,
        "model": model,
    }


if __name__ == "__main__":
    train_test_model(timeframe="15m")
