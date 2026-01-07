def generate_colab_notebook_cells() -> str:
    """
    Generate complete, independent Colab cell code for LSTM training.
    Returns a string containing 7 cells ready to copy-paste into Google Colab.
    """
    cells = """
# Cell 1: Install Dependencies and Import Libraries
!pip install tensorflow numpy pandas scikit-learn huggingface-hub -q

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from huggingface_hub import hf_hub_download
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Cell 2: Mount Google Drive and Download Data
from google.colab import drive
drive.mount('/content/drive')

print("Drive mounted. Downloading data from Hugging Face...")

data_dir = '/content/step2_data'
import os
os.makedirs(data_dir, exist_ok=True)

timeframe = '15m'  # Change to '1h' for hourly data

try:
    X_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename=f'{timeframe}_X_sequences.npy',
        repo_type='dataset',
        cache_dir=data_dir
    )
    y_class_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename=f'{timeframe}_y_class.npy',
        repo_type='dataset',
        cache_dir=data_dir
    )
    y_reg_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename=f'{timeframe}_y_reg.npy',
        repo_type='dataset',
        cache_dir=data_dir
    )
    scaler_path = hf_hub_download(
        repo_id='zongowo111/v2-crypto-ohlcv-data',
        filename=f'{timeframe}_scaler.pkl',
        repo_type='dataset',
        cache_dir=data_dir
    )
    print(f"Downloaded {timeframe} data successfully")
except Exception as e:
    print(f"Error downloading from HF: {e}")
    print("Attempting to use local data from drive...")

# Cell 3: Load and Preprocess Data
print("Loading data...")
X = np.load(X_path)
y_class = np.load(y_class_path).astype(np.int32)
y_reg = np.load(y_reg_path).astype(np.float32)

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print(f"X shape: {X.shape}")
print(f"y_class shape: {y_class.shape}")
print(f"y_reg shape: {y_reg.shape}")
print(f"y_class distribution: {np.bincount(y_class)}")

# Transform regression targets
y_reg_transformed = np.log1p(y_reg)

# One-hot encode classification labels
y_class_onehot = tf.keras.utils.to_categorical(y_class, num_classes=4)

print(f"y_reg_transformed mean: {y_reg_transformed.mean():.4f}, std: {y_reg_transformed.std():.4f}")
print(f"y_class_onehot shape: {y_class_onehot.shape}")

# Temporal split (70% train, 15% val, 15% test)
n = len(X)
train_end = int(n * 0.7)
val_end = train_end + int(n * 0.15)

X_train, y_class_train, y_reg_train = X[:train_end], y_class_onehot[:train_end], y_reg_transformed[:train_end]
X_val, y_class_val, y_reg_val = X[train_end:val_end], y_class_onehot[train_end:val_end], y_reg_transformed[train_end:val_end]
X_test, y_class_test, y_reg_test = X[val_end:], y_class_onehot[val_end:], y_reg_transformed[val_end:]

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# Cell 4: Build Multi-Task LSTM Model
def build_multitask_lstm(sequence_length=30, num_features=13, num_classes=4):
    input_layer = tf.keras.layers.Input(shape=(sequence_length, num_features), name='input')
    
    # Shared LSTM layers
    x = tf.keras.layers.LSTM(64, return_sequences=True)(input_layer)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Classification branch
    clf_branch = tf.keras.layers.Dense(16, activation='relu')(x)
    clf_branch = tf.keras.layers.Dropout(0.2)(clf_branch)
    classification_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification_output')(clf_branch)
    
    # Regression branch
    reg_branch = tf.keras.layers.Dense(16, activation='relu')(x)
    reg_branch = tf.keras.layers.Dropout(0.2)(reg_branch)
    reg_branch = tf.keras.layers.Dense(8, activation='relu')(reg_branch)
    regression_output = tf.keras.layers.Dense(1, activation='relu', name='regression_output')(reg_branch)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[classification_output, regression_output])
    return model

model = build_multitask_lstm(sequence_length=30, num_features=13, num_classes=4)
model.summary()

# Cell 5: Compile and Train Model
print("Compiling model...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'classification_output': 'categorical_crossentropy',
        'regression_output': 'mse'
    },
    loss_weights={
        'classification_output': 1.0,
        'regression_output': 0.5
    },
    metrics={
        'classification_output': ['accuracy'],
        'regression_output': ['mse', 'mae']
    }
)

print("Applying class weights for imbalanced data...")
class_weights = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.4}
y_class_int = np.argmax(y_class_train, axis=1)
class_weight_arr = np.vectorize(class_weights.get)(y_class_int)

print("Starting training...")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)

start_time = datetime.now()
history = model.fit(
    X_train,
    {'classification_output': y_class_train, 'regression_output': y_reg_train},
    validation_data=(
        X_val,
        {'classification_output': y_class_val, 'regression_output': y_reg_val}
    ),
    epochs=100,
    batch_size=32,
    verbose=1,
    callbacks=[early_stopping, reduce_lr],
    sample_weight={'classification_output': class_weight_arr}
)
training_time = (datetime.now() - start_time).total_seconds()
print(f"Training completed in {training_time:.2f} seconds")

# Cell 6: Evaluate Model and Visualize Results
print("Evaluating on test set...")
y_pred_class_proba, y_pred_reg = model.predict(X_test, verbose=0)
y_pred_class = np.argmax(y_pred_class_proba, axis=1)
y_true_class = np.argmax(y_class_test, axis=1)

accuracy = accuracy_score(y_true_class, y_pred_class)
precision, recall, f1, _ = precision_recall_fscore_support(y_true_class, y_pred_class, labels=[0, 1, 2, 3], zero_division=0)

try:
    auc_roc = roc_auc_score(y_class_test, y_pred_class_proba, multi_class='ovr')
except:
    auc_roc = float('nan')

mse_val = mean_squared_error(y_reg_test, y_pred_reg)
mae_val = mean_absolute_error(y_reg_test, y_pred_reg)
rmse_val = float(np.sqrt(mse_val))
r2_val = r2_score(y_reg_test, y_pred_reg)

print(f"Classification Accuracy: {accuracy:.4f}")
print(f"Precision per class: {precision}")
print(f"Recall per class: {recall}")
print(f"F1 per class: {f1}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"\nRegression MSE: {mse_val:.4f}")
print(f"Regression MAE: {mae_val:.4f}")
print(f"Regression RMSE: {rmse_val:.4f}")
print(f"Regression R2: {r2_val:.4f}")

# Plot training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Classification loss
axes[0, 0].plot(history.history['classification_output_loss'], label='train')
axes[0, 0].plot(history.history['val_classification_output_loss'], label='val')
axes[0, 0].set_title('Classification Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Classification accuracy
axes[0, 1].plot(history.history['classification_output_accuracy'], label='train')
axes[0, 1].plot(history.history['val_classification_output_accuracy'], label='val')
axes[0, 1].set_title('Classification Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Regression loss (MSE)
axes[1, 0].plot(history.history['regression_output_loss'], label='train')
axes[1, 0].plot(history.history['val_regression_output_loss'], label='val')
axes[1, 0].set_title('Regression Loss (MSE)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Total loss
axes[1, 1].plot(history.history['loss'], label='train')
axes[1, 1].plot(history.history['val_loss'], label='val')
axes[1, 1].set_title('Total Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()

# Cell 7: Save and Upload Model to Hugging Face
print("Saving model artifacts...")
output_dir = '/content/colab_models'
os.makedirs(output_dir, exist_ok=True)

model_path = f'{output_dir}/model_{timeframe}.h5'
scaler_path_out = f'{output_dir}/model_{timeframe}_scaler.pkl'
config_path = f'{output_dir}/model_{timeframe}_config.json'

model.save(model_path)
with open(scaler_path_out, 'wb') as f:
    pickle.dump(scaler, f)

config = {
    'metadata': {
        'symbol': 'BTCUSDT',
        'timeframe': timeframe,
        'sequence_length': 30,
        'num_features': 13,
        'num_classes': 4
    },
    'training_metadata': {
        'training_date': start_time.isoformat(),
        'training_duration_seconds': training_time,
        'total_samples': len(X)
    },
    'performance_metrics': {
        'classification': {
            'accuracy': float(accuracy),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'auc_roc_ovr': float(auc_roc)
        },
        'regression': {
            'mse': float(mse_val),
            'mae': float(mae_val),
            'rmse': float(rmse_val),
            'r2': float(r2_val)
        }
    }
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path_out}")
print(f"Config saved to {config_path}")

print("\nUploading to Hugging Face...")
from huggingface_hub import HfApi

api = HfApi()
hf_token = input("Enter your Hugging Face token: ")

repo_id = 'zongowo111/v2-crypto-ohlcv-data'
target_dir = f'v2_model/BTCUSDT/{timeframe}'

try:
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f'{target_dir}/model.h5',
        repo_id=repo_id,
        repo_type='dataset',
        token=hf_token
    )
    print(f"Uploaded model.h5 to {repo_id}/{target_dir}/")
    
    api.upload_file(
        path_or_fileobj=scaler_path_out,
        path_in_repo=f'{target_dir}/scaler.pkl',
        repo_id=repo_id,
        repo_type='dataset',
        token=hf_token
    )
    print(f"Uploaded scaler.pkl to {repo_id}/{target_dir}/")
    
    api.upload_file(
        path_or_fileobj=config_path,
        path_in_repo=f'{target_dir}/config.json',
        repo_id=repo_id,
        repo_type='dataset',
        token=hf_token
    )
    print(f"Uploaded config.json to {repo_id}/{target_dir}/")
    print("All files uploaded successfully!")
except Exception as e:
    print(f"Error uploading to HF: {e}")
    """
    return cells


if __name__ == "__main__":
    cells = generate_colab_notebook_cells()
    print(cells)
