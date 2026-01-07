def generate_colab_notebook_cells() -> str:
    """
    Generate complete, independent Colab cell code for LSTM training.
    Uses direct HTTP download instead of hf_hub_download for simplicity.
    Returns a string containing 7 cells ready to copy-paste into Google Colab.
    """
    cells = """
# Cell 1: Install Dependencies and Check GPU
!pip install tensorflow numpy pandas scikit-learn -q

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
import urllib.request
import os

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
if tf.config.list_physical_devices('GPU'):
    print("GPU is ready for training!")
else:
    print("WARNING: GPU not detected. Training will be slow. Recommend using GPU runtime.")

# Cell 2: Download Data from HF via Direct HTTP
print("Downloading preprocessed data from Hugging Face...")

data_dir = '/content/step2_data'
os.makedirs(data_dir, exist_ok=True)

timeframe = '15m'  # Change to '1h' for hourly data
hf_base_url = 'https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/resolve/main'

files_to_download = [
    f'{timeframe}_X_sequences.npy',
    f'{timeframe}_y_class.npy',
    f'{timeframe}_y_reg.npy',
    f'{timeframe}_scaler.pkl',
    f'{timeframe}_statistics.json'
]

def download_file(url, dest_path, timeout=300):
    try:
        print(f"  Downloading {os.path.basename(dest_path)}...", end=' ')
        urllib.request.urlretrieve(url, dest_path, timeout=timeout)
        file_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"OK ({file_size_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

print(f"Downloading {timeframe} dataset (this may take 2-5 minutes for 15m data)...")
for filename in files_to_download:
    url = f"{hf_base_url}/{filename}"
    dest_path = os.path.join(data_dir, filename)
    
    if not download_file(url, dest_path):
        print(f"WARNING: Failed to download {filename}")
        if 'statistics' in filename:
            print("  Proceeding without statistics file (optional)")
        else:
            print("  ERROR: Critical file missing. Cannot proceed.")
            raise RuntimeError(f"Failed to download {filename}")

print(f"Download complete. All files saved to {data_dir}")

# Cell 3: Load, Validate, and Preprocess Data
print("Loading and preprocessing data...")

try:
    X_path = os.path.join(data_dir, f'{timeframe}_X_sequences.npy')
    y_class_path = os.path.join(data_dir, f'{timeframe}_y_class.npy')
    y_reg_path = os.path.join(data_dir, f'{timeframe}_y_reg.npy')
    scaler_path = os.path.join(data_dir, f'{timeframe}_scaler.pkl')
    
    X = np.load(X_path)
    y_class = np.load(y_class_path).astype(np.int32)
    y_reg = np.load(y_reg_path).astype(np.float32)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"Data loaded successfully:")
    print(f"  X shape: {X.shape} (samples, sequence_length, features)")
    print(f"  y_class shape: {y_class.shape}")
    print(f"  y_reg shape: {y_reg.shape}")
    
    # Validate shapes
    assert X.shape[0] == len(y_class) == len(y_reg), "Sample count mismatch!"
    assert X.shape[1] == 30, "Sequence length should be 30"
    assert X.shape[2] == 13, "Features should be 13"
    print("  All validations passed!")
except Exception as e:
    print(f"ERROR loading data: {e}")
    raise

# Check class distribution
class_counts = np.bincount(y_class)
class_labels = ['HH (0)', 'LL (1)', 'HL (2)', 'LH (3)']
print(f"\nClass distribution (Class Imbalance Detected):")
for i, count in enumerate(class_counts):
    pct = 100 * count / len(y_class)
    print(f"  {class_labels[i]}: {count:7d} ({pct:5.2f}%)")

# Log-transform regression targets
print(f"\nApplying log1p transformation to regression targets...")
y_reg_transformed = np.log1p(y_reg)
print(f"  Original mean: {np.log1p(391.5 if timeframe == '15m' else 99.6):.4f} (reference)")
print(f"  Actual mean: {y_reg_transformed.mean():.4f}")
print(f"  Std dev: {y_reg_transformed.std():.4f}")

# One-hot encode classification labels
print(f"\nApplying one-hot encoding to classification targets...")
y_class_onehot = tf.keras.utils.to_categorical(y_class, num_classes=4)
print(f"  y_class_onehot shape: {y_class_onehot.shape}")

# Temporal split (70% train, 15% val, 15% test) - NO SHUFFLE
print(f"\nPerforming temporal data split (70% train, 15% val, 15% test)...")
n = len(X)
train_end = int(n * 0.7)
val_end = train_end + int(n * 0.15)

X_train = X[:train_end]
y_class_train = y_class_onehot[:train_end]
y_reg_train = y_reg_transformed[:train_end]

X_val = X[train_end:val_end]
y_class_val = y_class_onehot[train_end:val_end]
y_reg_val = y_reg_transformed[train_end:val_end]

X_test = X[val_end:]
y_class_test = y_class_onehot[val_end:]
y_reg_test = y_reg_transformed[val_end:]

print(f"  Training set: {len(X_train)} samples ({100*len(X_train)/n:.1f}%)")
print(f"  Validation set: {len(X_val)} samples ({100*len(X_val)/n:.1f}%)")
print(f"  Test set: {len(X_test)} samples ({100*len(X_test)/n:.1f}%)")

# Cell 4: Build and Summarize Multi-Task LSTM Model
print("Building multi-task LSTM model...")

def build_multitask_lstm(sequence_length=30, num_features=13, num_classes=4):
    input_layer = tf.keras.layers.Input(shape=(sequence_length, num_features), name='input')
    
    # Shared LSTM layers
    x = tf.keras.layers.LSTM(64, return_sequences=True)(input_layer)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Classification branch (4 classes)
    clf_branch = tf.keras.layers.Dense(16, activation='relu')(x)
    clf_branch = tf.keras.layers.Dropout(0.2)(clf_branch)
    classification_output = tf.keras.layers.Dense(num_classes, activation='softmax', name='classification_output')(clf_branch)
    
    # Regression branch (distance to next zigzag point)
    reg_branch = tf.keras.layers.Dense(16, activation='relu')(x)
    reg_branch = tf.keras.layers.Dropout(0.2)(reg_branch)
    reg_branch = tf.keras.layers.Dense(8, activation='relu')(reg_branch)
    regression_output = tf.keras.layers.Dense(1, activation='relu', name='regression_output')(reg_branch)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[classification_output, regression_output])
    return model

model = build_multitask_lstm(sequence_length=30, num_features=13, num_classes=4)
print("\nModel Architecture Summary:")
model.summary()

total_params = model.count_params()
print(f"\nTotal trainable parameters: {total_params:,}")

# Cell 5: Compile and Train Model with Class Weights
print("\nConfiguring model compilation...")

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

print("Applying class weights to handle imbalanced data...")
class_weights = {
    'classification_output': {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.4}
}

y_class_int = np.argmax(y_class_train, axis=1)
class_weight_dict = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.4}
class_weight_arr = np.vectorize(class_weight_dict.get)(y_class_int)

print(f"  Class 0 (HH): weight 1.0 (majority class)")
print(f"  Class 1 (LL): weight 1.5")
print(f"  Class 2 (HL): weight 2.0")
print(f"  Class 3 (LH): weight 2.4 (minority class)")

print("\nConfiguring callbacks...")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)

print("  EarlyStopping: patience=15, monitor=val_loss")
print("  ReduceLROnPlateau: factor=0.5, patience=5")

print("\nStarting training (this will take 30-60 minutes on GPU T4)...")
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
print(f"\nTraining completed in {training_time/60:.2f} minutes ({training_time:.0f} seconds)")
print(f"Total epochs trained: {len(history.history['loss'])}")

# Cell 6: Evaluate Model and Visualize Training Results
print("\nEvaluating model on test set...")

y_pred_class_proba, y_pred_reg = model.predict(X_test, verbose=0)
y_pred_class = np.argmax(y_pred_class_proba, axis=1)
y_true_class = np.argmax(y_class_test, axis=1)

# Classification metrics
accuracy = accuracy_score(y_true_class, y_pred_class)
precision, recall, f1, _ = precision_recall_fscore_support(y_true_class, y_pred_class, labels=[0, 1, 2, 3], zero_division=0)

try:
    auc_roc = roc_auc_score(y_class_test, y_pred_class_proba, multi_class='ovr')
except:
    auc_roc = float('nan')

# Regression metrics
mse_val = mean_squared_error(y_reg_test, y_pred_reg)
mae_val = mean_absolute_error(y_reg_test, y_pred_reg)
rmse_val = float(np.sqrt(mse_val))
r2_val = r2_score(y_reg_test, y_pred_reg)

print(f"\nClassification Metrics:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  AUC-ROC: {auc_roc:.4f}")
print(f"\n  Per-class metrics:")
for i in range(4):
    print(f"    Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")

print(f"\nRegression Metrics:")
print(f"  MSE: {mse_val:.4f}")
print(f"  MAE: {mae_val:.4f}")
print(f"  RMSE: {rmse_val:.4f}")
print(f"  R2 Score: {r2_val:.4f}")

# Plot training history
print("\nGenerating training history plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Classification loss
axes[0, 0].plot(history.history['classification_output_loss'], label='train', linewidth=2)
axes[0, 0].plot(history.history['val_classification_output_loss'], label='val', linewidth=2)
axes[0, 0].set_title('Classification Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Classification accuracy
axes[0, 1].plot(history.history['classification_output_accuracy'], label='train', linewidth=2)
axes[0, 1].plot(history.history['val_classification_output_accuracy'], label='val', linewidth=2)
axes[0, 1].set_title('Classification Accuracy', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Regression loss (MSE)
axes[1, 0].plot(history.history['regression_output_loss'], label='train', linewidth=2)
axes[1, 0].plot(history.history['val_regression_output_loss'], label='val', linewidth=2)
axes[1, 0].set_title('Regression Loss (MSE)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Total loss
axes[1, 1].plot(history.history['loss'], label='train', linewidth=2)
axes[1, 1].plot(history.history['val_loss'], label='val', linewidth=2)
axes[1, 1].set_title('Total Loss', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 7: Save Model and Upload to Hugging Face
print("\nSaving model artifacts to Colab storage...")

output_dir = '/content/colab_models'
os.makedirs(output_dir, exist_ok=True)

model_path = f'{output_dir}/model_{timeframe}.h5'
scaler_path_out = f'{output_dir}/model_{timeframe}_scaler.pkl'
config_path = f'{output_dir}/model_{timeframe}_config.json'

# Save model
print(f"  Saving model to {model_path}...")
model.save(model_path)
model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"    Model size: {model_size_mb:.2f} MB")

# Save scaler
print(f"  Saving scaler to {scaler_path_out}...")
with open(scaler_path_out, 'wb') as f:
    pickle.dump(scaler, f)
scaler_size_kb = os.path.getsize(scaler_path_out) / 1024
print(f"    Scaler size: {scaler_size_kb:.2f} KB")

# Build config
config = {
    'metadata': {
        'symbol': 'BTCUSDT',
        'timeframe': timeframe,
        'sequence_length': 30,
        'num_features': 13,
        'num_classes': 4,
        'model_params': total_params
    },
    'training_metadata': {
        'training_date': start_time.isoformat(),
        'training_duration_seconds': training_time,
        'total_samples': len(X),
        'epochs_trained': len(history.history['loss'])
    },
    'hyperparameters': {
        'learning_rate': 0.001,
        'batch_size': 32,
        'dropout_rate': 0.2,
        'class_weights': {'0': 1.0, '1': 1.5, '2': 2.0, '3': 2.4}
    },
    'performance_metrics': {
        'classification': {
            'accuracy': float(accuracy),
            'auc_roc': float(auc_roc),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist()
        },
        'regression': {
            'mse': float(mse_val),
            'mae': float(mae_val),
            'rmse': float(rmse_val),
            'r2': float(r2_val)
        }
    }
}

print(f"  Saving config to {config_path}...")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("\nAll artifacts saved successfully!")

# Upload to Hugging Face
print("\n" + "="*70)
print("UPLOADING TO HUGGING FACE HUB")
print("="*70)

print("\nYou have two options to upload:")
print("\nOption A: Using huggingface_hub library (requires HF token)")
print("  1. Run: !pip install huggingface-hub")
print("  2. Get token from: https://huggingface.co/settings/tokens")
print("  3. Run upload code below")

print("\nOption B: Manual upload via Hugging Face web interface")
print("  1. Download files from /content/colab_models/")
print("  2. Go to: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data")
print("  3. Upload to: v2_model/BTCUSDT/{}/".format(timeframe))

print("\n" + "-"*70)
print("OPTION A - Automatic Upload Code (uncomment to use):")
print("-"*70)

upload_code = f"""
# Uncomment and run these lines to upload to HF automatically

#!pip install huggingface-hub

from huggingface_hub import HfApi
import getpass

print("Uploading to Hugging Face...")
api = HfApi()
hf_token = getpass.getpass("Enter your HF token: ")

repo_id = 'zongowo111/v2-crypto-ohlcv-data'
target_dir = f'v2_model/BTCUSDT/{timeframe}'

try:
    print(f"Uploading model.h5...")
    api.upload_file(
        path_or_fileobj='{model_path}',
        path_in_repo=f'{{target_dir}}/model.h5',
        repo_id=repo_id,
        repo_type='dataset',
        token=hf_token
    )
    print(f"  OK: uploaded to {{repo_id}}/{{target_dir}}/model.h5")
    
    print(f"Uploading scaler.pkl...")
    api.upload_file(
        path_or_fileobj='{scaler_path_out}',
        path_in_repo=f'{{target_dir}}/scaler.pkl',
        repo_id=repo_id,
        repo_type='dataset',
        token=hf_token
    )
    print(f"  OK: uploaded to {{repo_id}}/{{target_dir}}/scaler.pkl")
    
    print(f"Uploading config.json...")
    api.upload_file(
        path_or_fileobj='{config_path}',
        path_in_repo=f'{{target_dir}}/config.json',
        repo_id=repo_id,
        repo_type='dataset',
        token=hf_token
    )
    print(f"  OK: uploaded to {{repo_id}}/{{target_dir}}/config.json")
    
    print("\nAll files uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{{repo_id}}/tree/main/{{target_dir}}")
except Exception as e:
    print(f"ERROR during upload: {{e}}")
"""

print(upload_code)

print("="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nModel saved at: {output_dir}")
print(f"Files ready for upload to HF")
    """
    return cells


if __name__ == "__main__":
    cells = generate_colab_notebook_cells()
    print(cells)
