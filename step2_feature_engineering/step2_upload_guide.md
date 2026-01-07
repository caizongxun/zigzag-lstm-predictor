# Step 2 Data Upload Guide - HuggingFace Dataset

Complete guide for uploading Step 2 processed data to HuggingFace dataset.

## Prerequisites

### 1. Install Required Packages

```bash
pip install huggingface-hub
```

### 2. Authenticate with HuggingFace

```bash
huggingface-cli login
```

You will be prompted to enter your HuggingFace token. Ensure the token has write permissions.

### 3. Verify HuggingFace Account

Check that you have access to the dataset repo:

```bash
huggingface-cli whoami
```

Expected output: Your HuggingFace username

## Directory Structure

The upload system organizes files as follows:

**Source** (from Step 2):
```
step2_output/
├── 15m_X_sequences.npy
├── 15m_y_class.npy
├── 15m_y_reg.npy
├── 15m_scaler.pkl
├── 15m_zigzag_points.json
├── 15m_statistics.json
├── 1h_X_sequences.npy
├── 1h_y_class.npy
├── 1h_y_reg.npy
├── 1h_scaler.pkl
├── 1h_zigzag_points.json
├── 1h_statistics.json
└── execution_summary.json
```

**Organized** (before upload):
```
organized_step2_output/
├── BTCUSDT/
│   ├── 15m/              <- Contains 1h_ prefixed files
│   │   ├── 1h_X_sequences.npy
│   │   ├── 1h_y_class.npy
│   │   ├── 1h_y_reg.npy
│   │   ├── 1h_scaler.pkl
│   │   ├── 1h_zigzag_points.json
│   │   └── 1h_statistics.json
│   └── 1h/               <- Contains 15m_ prefixed files
│       ├── 15m_X_sequences.npy
│       ├── 15m_y_class.npy
│       ├── 15m_y_reg.npy
│       ├── 15m_scaler.pkl
│       ├── 15m_zigzag_points.json
│       └── 15m_statistics.json
├── ETHUSDT/
│   ├── 15m/
│   └── 1h/
└── execution_summary.json
```

**Remote** (on HuggingFace):
```
klines/
├── BTCUSDT/
│   ├── 15m/
│   │   ├── 1h_X_sequences.npy
│   │   ├── 1h_y_class.npy
│   │   ├── 1h_y_reg.npy
│   │   ├── 1h_scaler.pkl
│   │   ├── 1h_zigzag_points.json
│   │   └── 1h_statistics.json
│   └── 1h/
│       ├── 15m_X_sequences.npy
│       ├── 15m_y_class.npy
│       ├── 15m_y_reg.npy
│       ├── 15m_scaler.pkl
│       ├── 15m_zigzag_points.json
│       └── 15m_statistics.json
├── ETHUSDT/
│   ├── 15m/
│   └── 1h/
execution_summary.json
```

## Key Rules

**Rule 1**: Files with prefix `1h_` go to `15m/` folder
- These are processing results from 15-minute data

**Rule 2**: Files with prefix `15m_` go to `1h/` folder
- These are processing results from 1-hour data

**Rule 3**: Only `BTCUSDT` and `ETHUSDT` are processed
- Other symbols are skipped

**Rule 4**: `execution_summary.json` goes to root directory

## Execution Steps

### Step 1: Prepare Source Data

Ensure `step2_output/` folder exists in the current directory with all processed files:

```python
import os
assert os.path.exists("./step2_output"), "step2_output folder not found"
files = os.listdir("./step2_output")
print(f"Found {len(files)} files in step2_output")
```

### Step 2: Run Upload Script

```bash
cd step2_feature_engineering
python step2_upload.py
```

The script will:
1. Verify prerequisites (login, dataset access)
2. Organize files from source folder
3. Upload by symbol (BTCUSDT first, then ETHUSDT)
4. Verify upload success

### Step 3: Monitor Output

The script provides detailed output:

```
========================================================================
STEP 2 DATA UPLOAD SYSTEM
========================================================================
Verifying prerequisites...
Success: Found source folder: ./step2_output
Success: Logged in to HuggingFace as: zongowo111
Success: Dataset accessible: zongowo111/v2-crypto-ohlcv-data

Organizing files...
Scanned 14 files
  Organize: BTCUSDT/15m/1h_X_sequences.npy
  Organize: BTCUSDT/1h/15m_X_sequences.npy
  ... (more files)

Success: Organized 13 files, skipped 0 files

Checking organized structure...
Symbols to upload: ['BTCUSDT', 'ETHUSDT']

Starting symbol-by-symbol upload...

[1/2] Uploading BTCUSDT...
Success: BTCUSDT uploaded successfully
Waiting 5 seconds (1/2 completed)...

[2/2] Uploading ETHUSDT...
Success: ETHUSDT uploaded successfully

Uploading execution summary...
Success: execution_summary.json uploaded

Verifying upload...
Success: Found klines/BTCUSDT/
Success: Found klines/ETHUSDT/
Success: Found execution_summary.json

========================================================================
SUCCESS: Upload completed
========================================================================
```

## Troubleshooting

### Issue 1: "step2_output not found"

**Cause**: Source folder does not exist

**Solution**:
1. Ensure you're in the correct directory
2. Run Step 2 to generate output: `python main.py`
3. Verify `step2_output/` folder exists

### Issue 2: "Not logged in to HuggingFace"

**Cause**: Token not configured

**Solution**:
```bash
huggingface-cli login
```

Enter your HuggingFace token (get from https://huggingface.co/settings/tokens)

### Issue 3: "Cannot access dataset"

**Cause**: Invalid dataset ID or permissions issue

**Solution**:
1. Verify you have write access: https://huggingface.co/zongowo111/v2-crypto-ohlcv-data
2. Check token has write permissions
3. Try re-authentication: `huggingface-cli logout` then `huggingface-cli login`

### Issue 4: "API rate limit exceeded"

**Cause**: Uploading too many files too quickly

**Solution**:
- The script already implements 5-second delays between symbols
- If still hitting limits, increase delay in code:
  ```python
  time.sleep(10)  # Increase from 5 to 10 seconds
  ```

### Issue 5: "Files organized incorrectly"

**Cause**: Symbol detection failed

**Solution**:
1. Check file names have correct prefixes: `15m_*` or `1h_*`
2. Manually verify organization in `organized_step2_output/`
3. Check logs for any errors during organization

### Issue 6: "Unauthorized symbols detected"

**Cause**: Files from ETHUSDT when only BTCUSDT expected

**Solution**:
- This is normal - system processes both BTCUSDT and ETHUSDT
- Both are authorized symbols
- If different symbol appears, check source folder contents

## Advanced: Manual Verification

Verify files after upload:

```python
from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    repo_type="dataset"
)

expected_files = [
    "klines/BTCUSDT/15m/1h_X_sequences.npy",
    "klines/BTCUSDT/1h/15m_X_sequences.npy",
    "klines/ETHUSDT/15m/1h_X_sequences.npy",
    "klines/ETHUSDT/1h/15m_X_sequences.npy",
    "execution_summary.json",
]

for expected in expected_files:
    if any(expected in f for f in files):
        print(f"OK: {expected}")
    else:
        print(f"MISSING: {expected}")
```

## Download in Step 3

Once uploaded, Step 3 (Colab training) can download data:

```python
from huggingface_hub import hf_hub_download
import numpy as np

X_sequences_path = hf_hub_download(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    filename="klines/BTCUSDT/15m/1h_X_sequences.npy",
    repo_type="dataset"
)
X_sequences = np.load(X_sequences_path)
print(f"Shape: {X_sequences.shape}")
```

## Rate Limiting Strategy

The upload system uses "symbol-by-symbol" approach:

1. Upload all files for BTCUSDT (multiple files, single commit)
2. Wait 5 seconds
3. Upload all files for ETHUSDT (multiple files, single commit)
4. Upload summary.json

This reduces API calls and respects rate limits.

## Important Notes

1. **No emoji**: Code follows no-emoji requirement
2. **Atomic uploads**: Each symbol uploads as single commit
3. **Automatic verification**: Script checks if files exist after upload
4. **Symbol whitelist**: Only BTCUSDT and ETHUSDT are processed
5. **File organization**: Automatic with correct prefix-based routing
6. **Delay strategy**: 5-second delay between symbol uploads

## Support

For issues:
1. Check this guide's troubleshooting section
2. Review script output for specific error messages
3. Verify HuggingFace dataset accessibility
4. Ensure source data exists from Step 2
