# Quick Start Guide - Step 2 Upload to HuggingFace

## In 3 Simple Steps

### Step 1: Login to HuggingFace

```bash
huggingface-cli login
```

Enter your HuggingFace token (get from https://huggingface.co/settings/tokens)

### Step 2: Run Upload

```bash
cd step2_feature_engineering
python step2_upload.py
```

The script will automatically:
- Verify your login and dataset access
- Organize files by symbol and timeframe
- Upload BTCUSDT data
- Wait 5 seconds
- Upload ETHUSDT data
- Verify all files uploaded successfully

### Step 3: Verify Upload (Optional)

```bash
python verify_upload.py
```

This generates a detailed report: `upload_verification_report.json`

## Expected Output

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

## What Gets Uploaded

For each symbol (BTCUSDT, ETHUSDT):

**15m/ folder** (1-hour data processing results):
- 1h_X_sequences.npy
- 1h_y_class.npy
- 1h_y_reg.npy
- 1h_scaler.pkl
- 1h_zigzag_points.json
- 1h_statistics.json

**1h/ folder** (15-minute data processing results):
- 15m_X_sequences.npy
- 15m_y_class.npy
- 15m_y_reg.npy
- 15m_scaler.pkl
- 15m_zigzag_points.json
- 15m_statistics.json

**Root**: execution_summary.json

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Not logged in" | Run `huggingface-cli login` |
| "Dataset not found" | Verify dataset exists: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data |
| "step2_output not found" | Run Step 2 first: `python main.py` |
| "API rate limit" | Already handled - script waits 5 seconds between uploads |

## Next Steps

Once uploaded successfully:

1. Step 3 (Colab training) can download data directly:

```python
from huggingface_hub import hf_hub_download
import numpy as np

X_sequences = np.load(hf_hub_download(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    filename="klines/BTCUSDT/15m/1h_X_sequences.npy",
    repo_type="dataset"
))
```

2. Verify on HuggingFace:
https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data

## Important Notes

- Only BTCUSDT and ETHUSDT are processed (other symbols are skipped)
- Files are organized by timeframe (15m/ for 1h_ files, 1h/ for 15m_ files)
- Upload respects API rate limits with 5-second delays
- All 13 files per symbol should upload successfully
- Total upload time: ~5-15 minutes depending on internet speed
