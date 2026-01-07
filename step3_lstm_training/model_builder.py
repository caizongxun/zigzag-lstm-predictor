import tensorflow as tf
from tensorflow.keras import layers, Model


def build_multitask_lstm(
    sequence_length: int = 30,
    num_features: int = 13,
    num_classes: int = 4,
) -> Model:
    """
    Build multi-task LSTM model for concurrent classification and regression.

    Architecture:
    - Shared LSTM layers: LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2)
    - Classification branch: Dense(16, relu) -> Dropout(0.2) -> Dense(num_classes, softmax)
    - Regression branch: Dense(16, relu) -> Dropout(0.2) -> Dense(8, relu) -> Dense(1, relu)

    Args:
        sequence_length: Length of input sequences (default: 30).
        num_features: Number of features per timestep (default: 13).
        num_classes: Number of classification classes (default: 4).

    Returns:
        Compiled Keras Model with two outputs: classification_output, regression_output.
    """
    # Input layer
    input_layer = layers.Input(shape=(sequence_length, num_features), name="input")

    # Shared LSTM layers
    x = layers.LSTM(64, return_sequences=True)(input_layer)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)

    # Classification branch
    clf_branch = layers.Dense(16, activation="relu")(x)
    clf_branch = layers.Dropout(0.2)(clf_branch)
    classification_output = layers.Dense(
        num_classes, activation="softmax", name="classification_output"
    )(clf_branch)

    # Regression branch
    reg_branch = layers.Dense(16, activation="relu")(x)
    reg_branch = layers.Dropout(0.2)(reg_branch)
    reg_branch = layers.Dense(8, activation="relu")(reg_branch)
    regression_output = layers.Dense(1, activation="relu", name="regression_output")(
        reg_branch
    )

    # Build model
    model = Model(
        inputs=input_layer,
        outputs=[classification_output, regression_output],
    )

    return model
