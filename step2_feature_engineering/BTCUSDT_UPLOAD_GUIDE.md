# BTCUSDT 專用上傳指南

**版本**: 1.0  
**更新日期**: 2026-01-07  
**狀態**: 生產就緒

---

## 概述

本指南專門用於將 BTCUSDT 特徵數據上傳到 Hugging Face。系統嚴格遵循以下規則:

- **只處理 BTCUSDT** - 不創建、假設或處理其他幣種
- **檔案組織規則** - 1h_ 前綴到 15m 資料夾,15m_ 前綴到 1h 資料夾
- **API 限制** - 上傳之間設置 5 秒延遲
- **驗證** - 上傳後自動驗證

---

## Hugging Face 數據集信息

| 項目 | 值 |
|------|----|
| **數據集 URL** | [https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data](https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data) |
| **數據集 ID** | zongowo111/v2-crypto-ohlcv-data |
| **數據集類型** | dataset |
| **分支** | main |

---

## 檔案組織規則

### Rule 1: 1h_ 前綴檔案 → 15m/ 資料夾

這些是從 **15 分鐘數據** 處理得到的特徵:

```
step2_output/
├── 1h_X_sequences.npy
├── 1h_y_class.npy
├── 1h_y_reg.npy
├── 1h_scaler.pkl
├── 1h_zigzag_points.json
└── 1h_statistics.json

↓ 組織到 ↓

klines/BTCUSDT/15m/
├── 1h_X_sequences.npy
├── 1h_y_class.npy
├── 1h_y_reg.npy
├── 1h_scaler.pkl
├── 1h_zigzag_points.json
└── 1h_statistics.json
```

**原因**: 1h_ 前綴表示這些特徵來自分析 1 小時間隔的數據,因此 1h 數據的結果被放在 15m/ 資料夾。

### Rule 2: 15m_ 前綴檔案 → 1h/ 資料夾

這些是從 **1 小時數據** 處理得到的特徵:

```
step2_output/
├── 15m_X_sequences.npy
├── 15m_y_class.npy
├── 15m_y_reg.npy
├── 15m_scaler.pkl
├── 15m_zigzag_points.json
└── 15m_statistics.json

↓ 組織到 ↓

klines/BTCUSDT/1h/
├── 15m_X_sequences.npy
├── 15m_y_class.npy
├── 15m_y_reg.npy
├── 15m_scaler.pkl
├── 15m_zigzag_points.json
└── 15m_statistics.json
```

**原因**: 15m_ 前綴表示這些特徵來自分析 15 分鐘間隔的數據,因此 15m 數據的結果被放在 1h/ 資料夾。

### Rule 3: execution_summary.json → 根目錄

```
step2_output/
└── execution_summary.json

↓ 上傳到 ↓

execution_summary.json (根目錄)
```

---

## 完整遠端結構

上傳後的最終結構:

```
Hugging Face Repository
zongowo111/v2-crypto-ohlcv-data
├── klines/
│   └── BTCUSDT/
│       ├── 15m/
│       │   ├── 1h_X_sequences.npy (326.7 MB)
│       │   ├── 1h_y_class.npy (857.9 KB)
│       │   ├── 1h_y_reg.npy (857.9 KB)
│       │   ├── 1h_scaler.pkl (1.1 KB)
│       │   ├── 1h_zigzag_points.json (70.96 KB)
│       │   └── 1h_statistics.json (5.1 KB)
│       └── 1h/
│           ├── 15m_X_sequences.npy (81.66 MB)
│           ├── 15m_y_class.npy (214.5 KB)
│           ├── 15m_y_reg.npy (214.5 KB)
│           ├── 15m_scaler.pkl (1.1 KB)
│           ├── 15m_zigzag_points.json (68.81 KB)
│           └── 15m_statistics.json (5.08 KB)
└── execution_summary.json
```

---

## 前置需求

### 1. 安裝套件

```bash
pip install huggingface-hub
```

### 2. HuggingFace 登入

```bash
huggingface-cli login
```

**操作步驟**:
1. 按 Enter 鍵打開登入頁面
2. 使用 zongowo111 帳號登入
3. 生成或使用現有 token (Settings → Tokens → New token)
4. 確保 token 有 "write" 權限
5. 複製 token 並貼入終端
6. 按 Enter 確認

**預期輸出**:
```
Token is valid (saved at /home/user/.cache/huggingface/token)
```

### 3. 驗證前置條件

```python
import os
from huggingface_hub import whoami, repo_info

# 檢查 step2_output 資料夾
assert os.path.exists("./step2_output"), "step2_output 資料夾不存在"
print("OK: step2_output 資料夾存在")

# 檢查 HuggingFace 登入
user = whoami()
print(f"OK: 已登入為 {user['name']}")

# 檢查數據集存在
repo = repo_info(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    repo_type="dataset"
)
print(f"OK: 數據集存在: {repo.id}")
```

---

## 執行步驟

### Step 1: 特徵工程 (10-20 秒)

```bash
cd step2_feature_engineering
python main.py
```

**預期輸出**: `step2_output/` 資料夾包含 13 個檔案

### Step 2: HuggingFace 登入 (1-2 秒)

```bash
huggingface-cli login
```

### Step 3: 上傳 BTCUSDT (3-5 分鐘)

```bash
python step2_upload_btcusdt_only.py
```

**預期輸出**:
```
======================================================================
STEP 2 DATA UPLOAD SYSTEM - BTCUSDT ONLY
======================================================================

[VERIFICATION] Checking prerequisites...
  OK: Source folder exists: ./step2_output
  OK: Logged in to HuggingFace as: zongowo111
  OK: Dataset accessible: zongowo111/v2-crypto-ohlcv-data

[ORGANIZATION] Organizing files for BTCUSDT...
  Found 13 files in source folder
    Organized: BTCUSDT/15m/1h_X_sequences.npy
    Organized: BTCUSDT/15m/1h_y_class.npy
    Organized: BTCUSDT/15m/1h_y_reg.npy
    Organized: BTCUSDT/15m/1h_scaler.pkl
    Organized: BTCUSDT/15m/1h_zigzag_points.json
    Organized: BTCUSDT/15m/1h_statistics.json
    Organized: BTCUSDT/1h/15m_X_sequences.npy
    Organized: BTCUSDT/1h/15m_y_class.npy
    Organized: BTCUSDT/1h/15m_y_reg.npy
    Organized: BTCUSDT/1h/15m_scaler.pkl
    Organized: BTCUSDT/1h/15m_zigzag_points.json
    Organized: BTCUSDT/1h/15m_statistics.json
    Organized: execution_summary.json
  Summary: 13 organized, 0 skipped

[REMOTE CHECK] Checking remote structure...
  Target symbol: BTCUSDT
  Target paths:
    - klines/BTCUSDT/15m/
    - klines/BTCUSDT/1h/

[UPLOAD] Uploading BTCUSDT...
  OK: BTCUSDT uploaded successfully

[RATE LIMIT] Waiting 5 seconds...

[UPLOAD] Uploading execution_summary.json...
  OK: execution_summary.json uploaded successfully

======================================================================
SUCCESS: Upload completed
======================================================================

Verify at: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
```

### Step 4: 驗證上傳 (1-2 秒)

```bash
python verify_btcusdt_upload.py
```

**預期輸出**:
```
======================================================================
BTCUSBT UPLOAD VERIFICATION REPORT
======================================================================

[REPOSITORY INFO]
  Repo ID: zongowo111/v2-crypto-ohlcv-data
  Total files: 14+

[STRUCTURE CHECK]
  BTCUSDT folder found: True
  15m folder found: True
  1h folder found: True
  execution_summary.json found: True

[FILE DETAILS]
  Found files: 13/13
    OK: klines/BTCUSDT/15m/1h_X_sequences.npy
    OK: klines/BTCUSDT/15m/1h_y_class.npy
    OK: klines/BTCUSDT/15m/1h_y_reg.npy
    OK: klines/BTCUSDT/15m/1h_scaler.pkl
    OK: klines/BTCUSDT/15m/1h_zigzag_points.json
    OK: klines/BTCUSDT/15m/1h_statistics.json
    OK: klines/BTCUSDT/1h/15m_X_sequences.npy
    OK: klines/BTCUSDT/1h/15m_y_class.npy
    OK: klines/BTCUSDT/1h/15m_y_reg.npy
    OK: klines/BTCUSDT/1h/15m_scaler.pkl
    OK: klines/BTCUSDT/1h/15m_zigzag_points.json
    OK: klines/BTCUSDT/1h/15m_statistics.json
    OK: execution_summary.json

[VERIFICATION RESULT]
  SUCCESS: All files uploaded correctly

======================================================================

Report saved to: btcusdt_verification_report.json
```

---

## 快速命令

### 一鍵上傳

```bash
python main.py && huggingface-cli login && python step2_upload_btcusdt_only.py && python verify_btcusdt_upload.py
```

### 分步執行

```bash
# 步驟 1: 特徵工程
python main.py

# 步驟 2: 登入
huggingface-cli login

# 步驟 3: 上傳
python step2_upload_btcusdt_only.py

# 步驟 4: 驗證
python verify_btcusdt_upload.py
```

---

## 故障排除

### 問題 1: "step2_output 資料夾不存在"

**原因**: 未執行 `main.py`

**解決方案**:
```bash
python main.py
```

### 問題 2: "未登入 Hugging Face"

**原因**: 未執行 `huggingface-cli login` 或 token 過期

**解決方案**:
```bash
huggingface-cli login
```

### 問題 3: "無上傳權限"

**原因**: Token 沒有 write 權限

**解決方案**:
1. 訪問 https://huggingface.co/settings/tokens
2. 建立新 token,確保有 write 權限
3. 重新登入: `huggingface-cli login`

### 問題 4: "API 速率限制"

**原因**: 上傳請求太頻繁

**解決方案**: 系統已內置 5 秒延遲,一般不會出現此問題。如果仍出現,請等待 10 分鐘後重試。

### 問題 5: "檔案組織不正確"

**症狀**: 1h_ 檔案出現在 1h/ 資料夾,而不是 15m/ 資料夾

**原因**: 檔案前綴規則反了

**解決方案**:
- 1h_ 前綴 → 15m/ 資料夾 (來自 15 分鐘數據)
- 15m_ 前綴 → 1h/ 資料夾 (來自 1 小時數據)

---

## 驗證清單

### 執行前
- [ ] Step 1 已完成 (step2_output 存在)
- [ ] HuggingFace 帳號已建立
- [ ] Token 已生成且有 write 權限
- [ ] 已執行 `huggingface-cli login`

### 執行期間
- [ ] main.py 執行成功 (2/2 files processed)
- [ ] step2_output 包含 13 個檔案
- [ ] step2_upload_btcusdt_only.py 執行成功 (SUCCESS: Upload completed)

### 執行後
- [ ] btcusdt_verification_report.json 已生成
- [ ] 所有 13 個檔案都已驗證
- [ ] 可在 HuggingFace 上查看檔案
- [ ] 準備進入 Step 3 (訓練)

---

## 性能指標

| 指標 | 數值 |
|------|------|
| BTCUSDT 15m 大小 | ~327 MB |
| BTCUSDT 1h 大小 | ~82 MB |
| 特徵工程時間 | 10-20 秒 |
| 上傳時間 | 3-5 分鐘 |
| 驗證時間 | 1-2 秒 |
| **總耗時** | **3-5 分鐘** |

---

## 禁止事項

- ❌ 不要使用 emoji
- ❌ 不要上傳整個專案資料夾
- ❌ 不要重複上傳已上傳的檔案
- ❌ 不要在短時間內發起大量上傳請求
- ❌ 不要處理或假設除 BTCUSDT 之外的任何幣種
- ❌ 不要創建 BTCUSDT 以外的資料夾
- ❌ 不要修改檔案組織規則
- ❌ 不要跳過 API 延遲

---

## 下一步 (Step 3)

上傳完成後,可在 Step 3 中下載並使用特徵進行模型訓練:

```python
from huggingface_hub import hf_hub_download
import numpy as np

# 下載 15m 數據特徵
X_15m = np.load(hf_hub_download(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    filename="klines/BTCUSDT/15m/1h_X_sequences.npy",
    repo_type="dataset"
))

# 下載 1h 數據特徵
X_1h = np.load(hf_hub_download(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    filename="klines/BTCUSDT/1h/15m_X_sequences.npy",
    repo_type="dataset"
))

print(f"X_15m shape: {X_15m.shape}")
print(f"X_1h shape: {X_1h.shape}")
```

---

## 支援資源

- **Hugging Face 官方文檔**: https://huggingface.co/docs/hub/
- **GitHub 專案**: https://github.com/caizongxun/zigzag-lstm-predictor
- **數據集首頁**: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data

---

**系統已完全就緒。開始上傳 BTCUSDT 數據!**
