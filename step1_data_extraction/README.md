# Step 1: Data Extraction and Cleaning

This module handles the complete data pipeline for extracting cryptocurrency OHLCV data from HuggingFace Hub, cleaning it, and validating data quality.

## Overview

Step 1 performs the following operations:

1. Download OHLCV data from HuggingFace Hub (v2-crypto-ohlcv-data)
2. Clean and standardize data format
3. Validate data quality and integrity
4. Export clean data to CSV files
5. Generate comprehensive validation reports

## Architecture

The module is organized into four main components:

### data_loader.py
Handles downloading and loading OHLCV data from HuggingFace Hub.

**Key Classes:**
- `DataLoader`: Main class for HF Hub interactions
  - `download_from_hf(symbol, timeframe)`: Download specific symbol-timeframe
  - `load_ohlcv_data(symbol, timeframe)`: Complete download and load
  - `load_parquet(file_path)`: Load parquet files

**Features:**
- Automatic retry logic with exponential backoff
- HF token authentication support via HF_TOKEN env variable
- Column standardization
- File caching

### data_cleaner.py
Performs comprehensive data cleaning operations.

**Key Classes:**
- `DataCleaner`: Main cleaning class
  - `clean_ohlcv_data(df)`: Complete cleaning pipeline
  - `_remove_duplicates()`: Duplicate row removal
  - `_fix_missing_values()`: Forward fill and interpolation
  - `_validate_ohlc_relationships()`: OHLC constraint validation
  - `_validate_timestamp_consistency()`: Time series validation

**Cleaning Operations:**
- Duplicate removal (by timestamp and OHLCV values)
- Missing value imputation (forward fill + linear interpolation)
- OHLC relationship validation and fixing
- Timestamp consistency checks
- Data sorting by timestamp

### validator.py
Performs data quality validation and reporting.

**Key Classes:**
- `DataValidator`: Main validation class
  - `check_data_quality(df)`: Comprehensive quality report
  - `validate_timestamps(df)`: Timestamp validation
  - `validate_volume(df)`: Volume consistency checks

**Validation Checks:**
- Timestamp continuity and ordering
- Volume logic and outliers
- OHLC anomalies detection
- Missing data statistics
- Duplicate detection
- Price consistency
- Overall quality score (0-100)

### main.py
Orchestrates the complete pipeline execution.

**Key Classes:**
- `DataExtractionPipeline`: Main pipeline orchestrator
  - `run(symbols, timeframes)`: Execute complete pipeline
  - `_process_symbol_timeframe()`: Process single symbol-timeframe

**Output:**
- CSV files: `{SYMBOL}_{TIMEFRAME}.csv`
- Validation report: `validation_report.json`
- Execution log: `EXECUTION_LOG.md`

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the `step1_data_extraction` directory:

```
HF_TOKEN=your_huggingface_token
HF_DATASET_ID=zongowo111/v2-crypto-ohlcv-data
```

Or copy from template:
```bash
cp .env.example .env
```

### Config File

Edit `config.py` to modify:
- `symbols`: List of symbols to download (default: ['BTCUSDT'])
- `timeframes`: List of timeframes (default: ['15m', '1h'])
- `output_dir`: Output directory (default: './step1_output')

## Usage

### Run Complete Pipeline

```bash
cd step1_data_extraction
python main.py
```

### Run Specific Symbols

```python
from config import Config
from main import DataExtractionPipeline

config = Config()
pipeline = DataExtractionPipeline(config)
results = pipeline.run(
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['15m', '1h']
)
```

### Use Individual Modules

```python
from data_loader import DataLoader
from data_cleaner import clean_ohlcv_data
from validator import check_data_quality
from config import Config

config = Config()
loader = DataLoader(config.loader_config)

df = loader.load_ohlcv_data('BTCUSDT', '15m')
df_clean, report = clean_ohlcv_data(df)
quality = check_data_quality(df_clean)
```

## Output Files

### CSV Files
Location: `./step1_output/`

```
{SYMBOL}_{TIMEFRAME}.csv
```

Columns: `timestamp, open, high, low, close, volume`

### Validation Report
Location: `./step1_output/validation_report.json`

```json
{
  "execution_start": "2026-01-07T...",
  "execution_end": "2026-01-07T...",
  "duration_seconds": 45.2,
  "summary": {
    "total_processed": 2,
    "successful": 2,
    "failed": 0,
    "average_quality_score": 95.5
  }
}
```

### Execution Log
Location: `./step1_output/EXECUTION_LOG.md`

Timestamped events of pipeline execution.

## Data Quality Report

For each symbol-timeframe, the validation report includes:

- **Total Rows**: Number of candles
- **Columns**: Data fields present
- **Missing Data**: NaN statistics per column
- **Duplicates**: Duplicate row count and percentage
- **OHLC Anomalies**: OHLC constraint violations
- **Timestamp Validation**: Time series consistency
- **Volume Validation**: Volume consistency checks
- **Price Validation**: Price logic checks
- **Quality Score**: Overall 0-100 score

## Testing with BTC Data

```bash
python main.py
```

Default configuration processes BTC 15m and 1h data:
- Downloads from HuggingFace Hub
- Cleans approximately 100K+ candles each
- Generates quality scores (typically >90)
- Exports to CSV in `./step1_output/`

## Supported Symbols

All 22 symbols in the dataset:

```
BTCUSDT, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT,
DOGEUSDT, LINKUSDT, LTCUSDT, FILUSDT, MATICUSDT,
UNIUSDT, AVAXUSDT, SOLUSDT, OPUSDT, ARBUSDT,
NEARUSDT, ATOMUSDT, SUIUSDT, LUNCUSDT, GALAUSDT,
MANAUSDT, PEPEUSDT
```

Timeframes: `15m`, `1h`

## Error Handling

The pipeline includes:
- Automatic retry logic for network errors
- Graceful handling of missing data
- Detailed error logging
- Pipeline continues if individual symbol-timeframe fails

## Performance

Typical execution times:
- Download: 30-60 seconds per symbol-timeframe
- Cleaning: 5-10 seconds per symbol-timeframe
- Validation: 2-5 seconds per symbol-timeframe
- Total for BTC 15m+1h: ~3-5 minutes

## Next Steps

After completing Step 1, proceed to Step 2:
- Feature Engineering (Zigzag calculation, technical indicators)
- Sequence building for LSTM training
