# File Mapping Reference - Step 2 Upload System

Complete reference for understanding how files are organized and uploaded.

## Core Rules

### Rule 1: File Prefix Mapping

| File Prefix | Source Data | Target Folder | Why |
|-------------|------------|---------------|---------|
| `1h_*` | 15-minute data processing | `15m/` | Output from 15m data pipeline |
| `15m_*` | 1-hour data processing | `1h/` | Output from 1h data pipeline |

**Example**:
- `1h_X_sequences.npy` comes from processing 15-minute data
- It goes to the `15m/` folder in the target structure
- This indicates it represents 1-hour-level features extracted from 15-minute data

### Rule 2: File Organization Path

```
Step 2 Output (Raw)
    |
    +-- 1h_X_sequences.npy
    +-- 1h_y_class.npy
    +-- 1h_y_reg.npy
    +-- 1h_scaler.pkl
    +-- 1h_zigzag_points.json
    +-- 1h_statistics.json
    +-- 15m_X_sequences.npy
    +-- 15m_y_class.npy
    +-- 15m_y_reg.npy
    +-- 15m_scaler.pkl
    +-- 15m_zigzag_points.json
    +-- 15m_statistics.json
    +-- execution_summary.json

         |
         | Organize by prefix
         v

Organized Structure (Before Upload)
    |
    +-- BTCUSDT/
    |   +-- 15m/                <- 1h_ prefixed files
    |   |   +-- 1h_X_sequences.npy
    |   |   +-- 1h_y_class.npy
    |   |   +-- 1h_y_reg.npy
    |   |   +-- 1h_scaler.pkl
    |   |   +-- 1h_zigzag_points.json
    |   |   +-- 1h_statistics.json
    |   |
    |   +-- 1h/                 <- 15m_ prefixed files
    |       +-- 15m_X_sequences.npy
    |       +-- 15m_y_class.npy
    |       +-- 15m_y_reg.npy
    |       +-- 15m_scaler.pkl
    |       +-- 15m_zigzag_points.json
    |       +-- 15m_statistics.json
    |
    +-- ETHUSDT/
    |   +-- 15m/
    |   +-- 1h/
    |
    +-- execution_summary.json

         |
         | Upload to HuggingFace
         v

Remote Structure (HuggingFace)
    |
    +-- klines/
        +-- BTCUSDT/
        |   +-- 15m/
        |   |   +-- 1h_X_sequences.npy
        |   |   +-- 1h_y_class.npy
        |   |   +-- 1h_y_reg.npy
        |   |   +-- 1h_scaler.pkl
        |   |   +-- 1h_zigzag_points.json
        |   |   +-- 1h_statistics.json
        |   |
        |   +-- 1h/
        |       +-- 15m_X_sequences.npy
        |       +-- 15m_y_class.npy
        |       +-- 15m_y_reg.npy
        |       +-- 15m_scaler.pkl
        |       +-- 15m_zigzag_points.json
        |       +-- 15m_statistics.json
        |
        +-- ETHUSDT/
        |   +-- 15m/
        |   +-- 1h/
        |
        +-- execution_summary.json (at root)
```

## File Type Reference

### For Each Timeframe (6 files per timeframe)

| File | Extension | Type | Purpose | Size (approx) |
|------|-----------|------|---------|---------------|
| `*_X_sequences.npy` | npy | NumPy array | Feature sequences (n_samples, 30, 13) | 300-350 MB (15m), 80-100 MB (1h) |
| `*_y_class.npy` | npy | NumPy array | Classification labels (HH, LL, HL, LH) | 0.8-1.0 MB |
| `*_y_reg.npy` | npy | NumPy array | Regression labels (bars to turning point) | 0.8-1.0 MB |
| `*_scaler.pkl` | pkl | Pickle | StandardScaler object for normalization | 1-2 KB |
| `*_zigzag_points.json` | json | JSON | Zigzag turning point details | 50-100 KB |
| `*_statistics.json` | json | JSON | Feature statistics and metadata | 5-10 KB |

### Summary File

| File | Type | Purpose | Location |
|------|------|---------|----------|
| `execution_summary.json` | JSON | Overall execution metadata | Root of dataset |

## Download Reference (for Step 3)

### Method 1: Individual File Download

```python
from huggingface_hub import hf_hub_download
import numpy as np

X_sequences = np.load(hf_hub_download(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    filename="klines/BTCUSDT/15m/1h_X_sequences.npy",
    repo_type="dataset"
))

y_class = np.load(hf_hub_download(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    filename="klines/BTCUSDT/15m/1h_y_class.npy",
    repo_type="dataset"
))
```

### Method 2: Complete Folder Download

```python
from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    repo_type="dataset",
    local_dir="./downloaded_data"
)
```

## Symbol Support

### Currently Supported

- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)

### Whitelisted in Code

```python
allowed_symbols = ["BTCUSDT", "ETHUSDT"]
```

Only these symbols are processed and uploaded. Other symbols are skipped with a message.

## Verification Checklist

### Before Upload

- [ ] `step2_output/` folder exists with all files
- [ ] Files are correctly named with `15m_` or `1h_` prefixes
- [ ] No other symbols are present
- [ ] `execution_summary.json` exists

### After Organization

- [ ] `organized_step2_output/` folder created
- [ ] Structure: `SYMBOL/TIMEFRAME/FILES`
- [ ] BTCUSDT has 15m/ and 1h/ folders
- [ ] ETHUSDT has 15m/ and 1h/ folders
- [ ] Each folder has 6 files
- [ ] `execution_summary.json` at root

### After Upload

- [ ] Can access dataset on HuggingFace
- [ ] `klines/BTCUSDT/15m/` exists with 6 files
- [ ] `klines/BTCUSDT/1h/` exists with 6 files
- [ ] `klines/ETHUSDT/15m/` exists with 6 files
- [ ] `klines/ETHUSDT/1h/` exists with 6 files
- [ ] `execution_summary.json` exists at root
- [ ] Total: 24 files + 1 summary = 25 files

## Common Questions

### Q1: Why does 1h_X_sequences.npy go to 15m/ folder?

A: The naming convention indicates the source data:
- `1h_*` = features extracted from 1-hour analysis, but they analyze 15-minute data
- `15m_*` = features extracted from 15-minute analysis, but they analyze 1-hour data

This is counterintuitive but follows the original naming convention.

### Q2: What if a file is from a different symbol?

A: The system automatically skips files from symbols not in the whitelist:
- BTCUSDT: Processed
- ETHUSDT: Processed
- XRPUSDT: Skipped with message
- Any other: Skipped

### Q3: Can I manually organize the files?

A: Yes, the script accepts any file structure as long as:
- File names have correct prefixes
- Files are in `step2_output/` folder
- Only two symbols present (or manually edit whitelist)

### Q4: What's the upload order?

A: Symbol-by-symbol with delays:
1. Verify prerequisites
2. Organize all files
3. Upload BTCUSDT (all 12 files as one commit)
4. Wait 5 seconds
5. Upload ETHUSDT (all 12 files as one commit)
6. Wait no delay
7. Upload execution_summary.json

This minimizes API calls while respecting rate limits.

### Q5: How long does upload take?

A: Depends on internet speed:
- BTCUSDT (~430 MB): 2-5 minutes
- ETHUSDT (~110 MB): 1-2 minutes
- Plus delays: ~10 seconds
- Total: ~5-10 minutes

## Technical Details

### Symbol Detection

```python
def extract_symbol(file_path, file_name):
    # Check if BTCUSDT or ETHUSDT in path
    for part in file_path.split(os.sep):
        if part in allowed_symbols:
            return part
    return None
```

### Prefix-based Routing

```python
if filename.startswith("1h_"):
    timeframe = "15m"
elif filename.startswith("15m_"):
    timeframe = "1h"
```

### Upload Command

```python
upload_folder(
    folder_path="./organized_step2_output/BTCUSDT",
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    path_in_repo="klines/BTCUSDT",
    repo_type="dataset"
)
```

This uploads the entire symbol folder with preserved subfolder structure.
