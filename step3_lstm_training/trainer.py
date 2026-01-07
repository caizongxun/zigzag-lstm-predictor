import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from .config import TRAINING_CONFIG


CLASS_WEIGHTS = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.4}


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    loss_weights: Dict[str, float]


def _build_optimizer(learning_rate: float) -> tf.keras.optimizers.Optimizer:
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def _compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    optimizer = _build_optimizer(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={
            "classification_output": "categorical_crossentropy",
            "regression_output": "mse",
        },
        loss_weights={
            "classification_output": TRAINING_CONFIG["loss_weights"]["classification"],
            "regression_output": TRAINING_CONFIG["loss_weights"]["regression"],
        },
        metrics={
            "classification_output": ["accuracy"],
            "regression_output": ["mse", "mae"],
        },
    )


def compile_and_train(
    model: tf.keras.Model,
    train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    config: Dict[str, Any],
):
    """Compile and train multi-task LSTM model.

    Args:
        model: Uncompiled Keras model from model_builder.build_multitask_lstm.
        train_data: (X_train, y_train_cls_onehot, y_train_reg_log1p)
        val_data: (X_val, y_val_cls_onehot, y_val_reg_log1p)
        test_data: (X_test, y_test_cls_onehot, y_test_reg_log1p)
        config: Training configuration dictionary.
    """
    X_train, y_train_cls, y_train_reg = train_data
    X_val, y_val_cls, y_val_reg = val_data

    train_cfg = TrainConfig(
        epochs=config.get("epochs", TRAINING_CONFIG["epochs"]),
        batch_size=config.get("batch_size", TRAINING_CONFIG["batch_size"]),
        learning_rate=config.get("learning_rate", TRAINING_CONFIG["learning_rate"]),
        loss_weights=TRAINING_CONFIG["loss_weights"],
    )

    _compile_model(model, train_cfg.learning_rate)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=TRAINING_CONFIG.get("early_stopping_patience", 15),
        restore_best_weights=True,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
    )

    # Map class weights to the categorical labels (expects integer labels for sample_weight).
    # For multi-output, we pass sample_weight as a dict. For simplicity we weight only the
    # classification head here.
    y_train_cls_int = np.argmax(y_train_cls, axis=1)
    class_weight_arr = np.vectorize(CLASS_WEIGHTS.get)(y_train_cls_int)

    history = model.fit(
        X_train,
        {"classification_output": y_train_cls, "regression_output": y_train_reg},
        validation_data=(
            X_val,
            {"classification_output": y_val_cls, "regression_output": y_val_reg},
        ),
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        verbose=1,
        callbacks=[early_stopping, reduce_lr],
        sample_weight={"classification_output": class_weight_arr},
    )

    # Evaluate on test set
    results = evaluate_model(model, test_data)
    return model, history, results


def evaluate_model(
    model: tf.keras.Model,
    test_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> Dict[str, Any]:
    """Evaluate trained model on test set.

    Args:
        model: Trained multitask model.
        test_data: (X_test, y_test_cls_onehot, y_test_reg_log1p)

    Returns:
        Dictionary of formatted classification and regression metrics.
    """
    X_test, y_test_cls, y_test_reg = test_data

    # Forward pass
    y_pred_cls_proba, y_pred_reg = model.predict(X_test, verbose=0)

    y_true_cls = np.argmax(y_test_cls, axis=1)
    y_pred_cls = np.argmax(y_pred_cls_proba, axis=1)

    # Classification metrics
    acc = accuracy_score(y_true_cls, y_pred_cls)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, labels=[0, 1, 2, 3], zero_division=0
    )

    # AUC-ROC using one-vs-rest
    try:
        auc_roc = roc_auc_score(y_test_cls, y_pred_cls_proba, multi_class="ovr")
    except ValueError:
        auc_roc = float("nan")

    # Regression metrics (on log1p scale, but still informative; caller can also inverse-transform)
    mse_val = mean_squared_error(y_test_reg, y_pred_reg)
    mae_val = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse_val = float(np.sqrt(mse_val))
    r2_val = r2_score(y_test_reg, y_pred_reg)

    report = {
        "classification": {
            "accuracy": float(acc),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "auc_roc_ovr": float(auc_roc),
        },
        "regression": {
            "mse": float(mse_val),
            "mae": float(mae_val),
            "rmse": float(rmse_val),
            "r2": float(r2_val),
        },
    }
    return report
