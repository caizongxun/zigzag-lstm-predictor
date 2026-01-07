# HuggingFace 特徵上傳指南

本文件說明如何在 Step 2 中將生成的特徵檔案上傳到 HuggingFace Dataset Repository。

## 目錄結構

特徵檔案將上傳到以下結構：

```
v2_model/
longrightarrow BTCUSDT/
    longrightarrow 15m/
        ├── 15m_X_sequences.npy (特徵序列)
        ├── 15m_y_class.npy (分類標籤)
        ├── 15m_y_reg.npy (迴歸標籤)
        ├── 15m_scaler.pkl (標準化器)
        ├── 15m_zigzag_points.json (轉折點詳情)
        ├── 15m_statistics.json (統計數據)
        ├── model.h5 (Step 3 訓練好的模型)
        ├── config.json (Step 3 配置)
        └── training_history.json (Step 3 訓練歷史)
    longrightarrow 1h/
        ├── 1h_X_sequences.npy
        ├── 1h_y_class.npy
        ├── 1h_y_reg.npy
        ├── 1h_scaler.pkl
        ├── 1h_zigzag_points.json
        ├── 1h_statistics.json
        ├── model.h5
        ├── config.json
        └── training_history.json
```

## 設置 HuggingFace Token

### 步驟 1: 獲取 HuggingFace Token

1. 登入 [HuggingFace](https://huggingface.co)
2. 進入 [Settings - Tokens](https://huggingface.co/settings/tokens)
3. 點擊 "New token"
4. 選擇 "write" 權限
5. 複製生成的 token

### 步驟 2: 配置環境變量

#### 方法 A: 使用 .env 檔案 (推薦)

1. 在 `step2_feature_engineering` 目錄下建立 `.env` 檔案
2. 添加以下內容：

```bash
HF_TOKEN=your_huggingface_token_here
```

3. 保存檔案

#### 方法 B: 使用環境變量

在命令行設定環境變量：

```bash
# Windows PowerShell
$env:HF_TOKEN="your_huggingface_token_here"
python main.py

# Linux/Mac
export HF_TOKEN="your_huggingface_token_here"
python main.py
```

#### 方法 C: HuggingFace CLI 認證

```bash
huggingface-cli login
# 按照提示輸入 token
```

## 執行特徵上傳

### 標準執行 (含上傳)

```bash
cd step2_feature_engineering
pip install -r requirements.txt
python main.py
```

執行完成後，你將看到：

```
==========================================================================================
STEP 7: UPLOADING FEATURES TO HUGGINGFACE
==========================================================================================

==========================================================================================
Uploading BTCUSDT 15m features to HuggingFace
==========================================================================================

  Uploading 6 files...
    Uploading 15m_X_sequences.npy (345.21 MB)... [OK]
    Uploading 15m_y_class.npy (868.62 KB)... [OK]
    Uploading 15m_y_reg.npy (868.62 KB)... [OK]
    Uploading 15m_scaler.pkl (4.23 KB)... [OK]
    Uploading 15m_zigzag_points.json (256.47 KB)... [OK]
    Uploading 15m_statistics.json (58.45 KB)... [OK]

  Successfully uploaded 6 files (347.26 MB)
  Remote directory: v2_model/BTCUSDT/15m

  Verifying BTCUSDT 15m upload...
    All 6 files verified successfully
```

### 禁用上傳

如果不想上傳到 HuggingFace，只需確保環境變量中沒有設定 `HF_TOKEN`。程序會自動跳過上傳步驟。

## Step 3 中下載特徵

在 Colab 或其他環境中，Step 3 可以直接從 HuggingFace 下載特徵：

### 方法 1: 使用便利函數

```python
from hf_feature_downloader import download_training_features

features = download_training_features(
    symbol='BTCUSDT',
    timeframes=['15m', '1h']
)

X_train_15m = features['15m']['X_sequences']
y_class_15m = features['15m']['y_class']
y_reg_15m = features['15m']['y_reg']
scaler_15m = features['15m']['scaler']
```

### 方法 2: 使用 HFFeatureDownloader 類

```python
from hf_feature_downloader import HFFeatureDownloader

downloader = HFFeatureDownloader()
features = downloader.download_training_features('BTCUSDT', ['15m', '1h'])
```

### 方法 3: 直接使用 HuggingFace Hub

```python
import numpy as np
from huggingface_hub import hf_hub_download

X_sequences_path = hf_hub_download(
    repo_id="zongowo111/zigzag-lstm-predictor",
    filename="v2_model/BTCUSDT/15m/15m_X_sequences.npy",
    repo_type="dataset"
)
X_sequences = np.load(X_sequences_path)
```

## 上傳統計

每次上傳完成後，`execution_summary.json` 將包含上傳結果：

```json
{
  "huggingface_upload": {
    "15m": {
      "symbol": "BTCUSDT",
      "timeframe": "15m",
      "success": true,
      "message": "Successfully uploaded 6 files (347.26 MB)",
      "files_uploaded": [
        {
          "filename": "15m_X_sequences.npy",
          "size": 362015360,
          "remote_path": "v2_model/BTCUSDT/15m/15m_X_sequences.npy"
        },
        ...
      ],
      "remote_dir": "v2_model/BTCUSDT/15m",
      "verification": {
        "all_present": true,
        "files_status": { ... }
      }
    },
    "1h": { ... }
  }
}
```

## 故障排除

### 錯誤 1: HF_TOKEN 未找到

```
WARNING: HF_TOKEN not found in environment variables.
Skipping HuggingFace upload.
```

**解決方法**：
1. 確認 `.env` 檔案存在且包含正確的 token
2. 確認 token 有 "write" 權限
3. 確認 HF_TOKEN 環境變量已正確設定

### 錯誤 2: 上傳權限不足

```
Error: You need to be authenticated to access this repository.
```

**解決方法**：
1. 確認 token 有 "write" 權限
2. 確認 token 未過期
3. 使用 `huggingface-cli login` 重新認證

### 錯誤 3: 網路連接問題

```
Error: Connection timeout while uploading.
```

**解決方法**：
1. 檢查網路連接
2. 重試上傳
3. 考慮使用 VPN 或代理

## 注意事項

1. **Token 安全**：
   - 不要在代碼中直接寫入 token
   - 不要將包含 token 的檔案提交到版本控制
   - `.env` 檔案已加入 `.gitignore`

2. **磁儲配額**：
   - 單個文件大小上限：50GB
   - 總倉庫大小：根據 HuggingFace 計畫而定

3. **上傳速度**：
   - 取決於網路速度和檔案大小
   - 15m 特徵 (~350MB) 通常需要 5-10 分鐘
   - 1h 特徵 (~90MB) 通常需要 1-3 分鐘

4. **驗證機制**：
   - 上傳完成後自動驗證所有檔案
   - 如果驗證失敗，將顯示警告
   - 可以手動重新驗證

## 支持的事物

- 多個幣種 (目前: BTCUSDT)
- 多個時間框架 (目前: 15m, 1h)
- 自動化上傳流程
- 驗證機制
- 人性化錯誤提示
