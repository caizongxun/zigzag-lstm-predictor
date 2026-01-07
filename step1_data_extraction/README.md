# Step 1: OHLCV Data Extraction and Cleaning Pipeline

完整的加密貨幣OHLCV數據提取、清潔和驗證系統。

## 系統概述

本系統從HuggingFace Hub下載加密貨幣OHLCV數據,進行數據清潔和驗證,最後輸出高品質的CSV檔案供後續模型訓練使用。

### 核心功能

1. **數據下載** - 從HuggingFace Hub獲取Parquet格式的OHLCV數據
2. **數據清潔** - 移除重複、修復缺失值、驗證OHLC關係
3. **數據驗證** - 時間戳檢查、成交量驗證、異常檢測
4. **報告生成** - JSON驗證報告和CSV輸出

---

## 安裝和配置

### 前提要求

- Python 3.8+
- pip或conda包管理器
- HuggingFace Hub帳戶(可選,但推薦)

### 安裝步驟

```bash
cd step1_data_extraction

pip install -r requirements.txt
```

### 環境配置

#### 方法1: 使用.env檔案

```bash
cp .env.example .env

export HF_TOKEN=your_huggingface_token
```

編輯`.env`檔案:
```ini
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HF_DATASET_ID=zongowo111/v2-crypto-ohlcv-data
```

#### 方法2: 環境變數

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### 方法3: 直接配置 (config.py)

```python
self.hf_token = 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

---

## 使用指南

### 快速開始

#### 執行完整管道 (默認: BTC 15m和1h)

```bash
python main.py
```

#### 預期輸出

```
2026-01-07 09:05:23 - INFO - Configuration loaded successfully
2026-01-07 09:05:23 - INFO - Testing symbols: ['BTCUSDT']
2026-01-07 09:05:23 - INFO - Testing timeframes: ['15m', '1h']
2026-01-07 09:05:23 - INFO - Starting data extraction pipeline...
Processing data: 100%|#######| 2/2 [00:45<00:00, 22.50s/pair]

======================================================================
PIPELINE EXECUTION SUMMARY
======================================================================
Processed: 2 symbol-timeframe pairs
Successful: 2
Failed: 0
Total size: 45.32 MB
Total rows before cleaning: 2,547,283
Total rows after cleaning: 2,501,624
Rows removed: 45,659
Average quality score: 87.5/100
======================================================================
```

### 模組API文檔

#### 1. DataLoader (data_loader.py)

```python
from data_loader import DataLoader
from config import Config

config = Config()
loader = DataLoader(config.loader_config)

df = loader.load_ohlcv_data(symbol='BTCUSDT', timeframe='15m')
```

**主要方法:**

- `download_from_hf(symbol, timeframe)` -> Optional[str]
  - 從HF下載檔案,返回本地路徑
  - 自動重試機制(exponential backoff)
  
- `load_parquet(file_path, use_chunked=False)` -> Optional[pd.DataFrame]
  - 加載Parquet檔案
  - 自動選擇分塊讀取(>100MB)
  
- `load_ohlcv_data(symbol, timeframe)` -> Optional[pd.DataFrame]
  - 完整流程: 下載 -> 加載 -> 標準化
  - 自動驗證OHLCV列

**配置參數:**

```python
loader_config = {
    'dataset_id': 'zongowo111/v2-crypto-ohlcv-data',
    'cache_dir': './hf_cache',
    'max_retries': 3,
    'timeout': 300,
    'chunk_size': 50000
}
```

---

#### 2. OHLCVCleaner (data_cleaner.py)

```python
from data_cleaner import clean_ohlcv_data

df_clean, report = clean_ohlcv_data(df, config={'fill_limit': 5})
```

**清潔步驟:**

1. **去重** - 基於時間戳移除重複行
2. **缺失值修復** - 前向填充(limit=5) + 線性插值
3. **OHLC驗證** - high >= max(o,c,l), low <= min(o,c,l)
4. **時間戳排序** - 確保時間單調遞增
5. **無效行移除** - 清除零或負價格

**返回報告:**

```python
{
    'initial_rows': 1000000,
    'final_rows': 999500,
    'rows_removed': 500,
    'removal_percentage': 0.05,
    'steps': {
        'duplicates_removed': 50,
        'missing_values_fixed': {'initial_missing': 100, 'final_missing': 5},
        'ohlc_fixed': 10,
        'timestamp_ordered': True,
        'invalid_rows_removed': 335
    }
}
```

---

#### 3. DataValidator (validator.py)

```python
from validator import check_data_quality

report = check_data_quality(df, config={
    'allow_missing_percent': 5.0,
    'allow_zero_volume_percent': 20.0
})
```

**驗證項目:**

| 檢查項 | 滿分 | 條件 |
|-------|------|------|
| 缺失值 | 85 | < 5% |
| 時間戳 | 85 | 連續且遞增 |
| 成交量 | 90 | 正值且非零比例 > 80% |
| 價格範圍 | 85 | OHLC關係有效 |
| 重複 | 90 | 無重複 |
| 異常 | 85 | 異常率 < 5% |

**品質分數計算:**

```
Score = 100 - penalties
penalties = missing(max 15) + timestamp(max 15) + volume(max 10) 
          + price(max 15) + duplicates(max 10) + anomalies(max 15)
```

**返回報告結構:**

```python
{
    'overall_quality_score': 87.5,  # 0-100
    'total_rows': 1000000,
    'timestamp_validation': {...},
    'volume_validation': {...},
    'price_validation': {...},
    'missing_values': {...},
    'duplicates': {...},
    'anomalies': {...},
    'statistics': {
        'open': {'mean': 28500, 'std': 2000, ...},
        'volume': {'mean': 5000, 'std': 1500, ...},
        ...
    }
}
```

---

#### 4. DataExtractionPipeline (main.py)

```python
from main import DataExtractionPipeline
from config import Config

config = Config()
pipeline = DataExtractionPipeline(config)

summary = pipeline.run(
    symbols=['BTCUSDT', 'ETHUSDT'],
    timeframes=['15m', '1h']
)
```

**主要方法:**

- `run(symbols, timeframes)` - 執行完整管道
- 自動生成進度條(tqdm)
- 輸出CSV和JSON報告

---

## 數據集信息

### 源數據集

**HuggingFace:** [zongowo111/v2-crypto-ohlcv-data](https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data)

**統計信息:**
- 總文件: 46個
- 總數據點: 4,819,964
- 總大小: 110.57 MB
- 幣種: 23個
- 時間框架: 15m, 1h

**支持的幣種:**

```
BTCUST, ETHUSDT, BNBUSDT, XRPUSDT, ADAUSDT,
DOGEUST, LINKUSDT, LTCUSDT, FILUSDT, MATICUSDT,
UNIUST, AVAXUSDT, SOLUSDT, OPUSDT, ARBUSDT,
NEARUST, ATOMUSDT, SUIUSDT, LUNCUSDT, GALAUSDT,
MANAUST, PEPEUSDT
```

### Parquet文件位置

```
klines/{SYMBOL}/{SYMBOL_PREFIX}_{TIMEFRAME}.parquet

Example:
klines/BTCUSDT/BTC_15m.parquet
klines/ETHUSDT/ETH_1h.parquet
```

### 列結構

```
- timestamp: datetime - 開盤時間(UTC)
- open: float - 開盤價
- high: float - 最高價
- low: float - 最低價
- close: float - 收盤價
- volume: float - 成交量(USDT)
```

---

## 輸出文件

### CSV檔案

位置: `./step1_output/{SYMBOL}_{TIMEFRAME}.csv`

**示例: BTCUSDT_15m.csv**

```
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,42500.50,42650.25,42400.10,42600.75,15234567.89
2024-01-01 00:15:00,42600.75,42750.50,42550.25,42700.25,16345678.90
...
```

**行數:** 60,000-100,000+ (根據時間框架)
**大小:** 5-15 MB (未壓縮)

### 驗證報告

位置: `./step1_output/validation_report.json`

```json
{
  "execution_start": "2026-01-07T09:05:23.123456",
  "execution_end": "2026-01-07T09:06:08.654321",
  "duration_seconds": 45.53,
  "summary": {
    "total_processed": 2,
    "successful": 2,
    "failed": 0,
    "total_rows_before_cleaning": 2547283,
    "total_rows_after_cleaning": 2501624,
    "average_quality_score": 87.5
  },
  "details": {
    "BTCUSDT_15m": {
      "status": "success",
      "rows_before_cleaning": 1273641,
      "rows_after_cleaning": 1250812,
      "quality_report": {...}
    },
    "BTCUSDT_1h": {
      "status": "success",
      "rows_before_cleaning": 1273642,
      "rows_after_cleaning": 1250812,
      "quality_report": {...}
    }
  }
}
```

---

## 高級用法

### 自定義配置

```python
from config import Config

class CustomConfig(Config):
    def __init__(self):
        super().__init__()
        self.symbols = ['ETHUSDT', 'BNBUSDT']
        self.timeframes = ['1h']
        self.cleaner_config['fill_limit'] = 10
        self.validator_config['allow_missing_percent'] = 3.0
```

### 批量處理

```python
from config import DATASET_SYMBOLS

# 處理所有可用幣種
symbols = DATASET_SYMBOLS
timeframes = ['15m', '1h']

pipeline.run(symbols=symbols, timeframes=timeframes)
```

### 單個模組測試

```python
from data_loader import DataLoader

loader = DataLoader(config.loader_config)

df = loader.load_ohlcv_data('BTCUSDT', '15m')
print(f"Loaded {len(df)} rows")
print(df.head())
```

---

## 效能優化

### 記憶體使用

- 分塊讀取Parquet文件(>100MB自動啟用)
- PyArrow memory_map=True
- 優化數據類型(int32/float32)

### 下載速度

- 自動重試機制(指數退避)
- HuggingFace Hub快取
- 多線程Parquet讀取

### 執行時間

- 典型耗時: 2-5分鐘/2幣種
- 主要耗時: 網絡下載(50%) + 清潔驗證(30%) + I/O(20%)

---

## 故障排查

### 常見問題

| 問題 | 解決方案 |
|------|----------|
| `HF_TOKEN not found` | 設置環境變數或.env檔案 |
| `Network timeout` | 增加timeout參數或檢查網絡 |
| `Memory error` | 啟用分塊讀取或減少batch大小 |
| `Column not found` | 檢查Parquet檔案結構 |

### 調試

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 參考資源

- [HuggingFace Hub文檔](https://huggingface.co/docs/hub/)
- [PyArrow Parquet指南](https://arrow.apache.org/docs/python/parquet.html)
- [Pandas文檔](https://pandas.pydata.org/docs/)
- [Scikit-learn異常檢測](https://scikit-learn.org/stable/modules/outlier_detection.html)

---

## License

MIT License - 詳見LICENSE檔案

---

## 貢獻

歡迎提交Pull Request或報告Issue。

---

**最後更新:** 2026-01-07
**狀態:** Production Ready
