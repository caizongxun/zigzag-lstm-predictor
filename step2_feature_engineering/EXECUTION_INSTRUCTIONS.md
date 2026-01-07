# Step 2 完整執行指南

**最後更新**: 2026-01-07
**Status**: 已修正並就緒執行

## 核心變更

主要問題已修正:
- 舊上傳邏輯指向錯誤的 repo: `zongowo111/zigzag-lstm-predictor` (404 失敗)
- 新系統使用正確的 repo: `zongowo111/v2-crypto-ohlcv-data`
- `main.py` 現在只進行特徵工程,不執行上傳
- 上傳是獨立的步驟,透過 `step2_upload.py` 執行

## 完整執行流程

### Phase 1: 特徵工程處理 (main.py)

```bash
cd step2_feature_engineering
python main.py
```

**預期時間**: 10-20 秒

**預期輸出**:
```
ZIGZAG LSTM PREDICTOR - STEP 2: FEATURE ENGINEERING

[STEP 1] Loading and validating data...
[STEP 2] Calculating Zigzag turning points...
[STEP 3] Extracting 13 technical features...
[STEP 4] Normalizing features...
[STEP 5] Preparing training sequences...
[STEP 6] Saving outputs to disk...

EXECUTION SUMMARY
15m: SUCCESS
1h: SUCCESS

Final Result: 2/2 files processed successfully
```

**生成的輸出**: `step2_output/` 資料夾包含 13 個檔案

---

### Phase 2: HuggingFace 登入

```bash
huggingface-cli login
```

**操作步驟**:
1. 按 Enter 鍵開啟登入頁面
2. 登入 HuggingFace 帳號 (zongowo111)
3. 建立或使用現有 token (Settings → Tokens → New token)
4. 確保 token 有 "write" 權限
5. 複製 token 並貼入終端
6. 按 Enter 確認

**預期輸出**:
```
Token is valid (saved at /home/user/.cache/huggingface/token)
```

---

### Phase 3: 檔案上傳 (step2_upload.py)

```bash
python step2_upload.py
```

**預期時間**: 5-15 分鐘 (取決於網路速度)

**執行過程**:
1. 驗證前提條件 (登入、資料集訪問)
2. 組織檔案結構 (按幣種和時間框架)
3. 上傳 BTCUSDT (~430 MB, 約 3-5 分鐘)
4. 等待 5 秒
5. 上傳 ETHUSDT (~110 MB, 約 1-2 分鐘)
6. 上傳 execution_summary.json
7. 驗證所有檔案

**預期輸出**:
```
STEP 2 DATA UPLOAD SYSTEM
Verifying prerequisites...
Success: Found source folder: ./step2_output
Success: Logged in to HuggingFace as: zongowo111
Success: Dataset accessible: zongowo111/v2-crypto-ohlcv-data

Organizing files...
Success: Organized 13 files, skipped 0 files

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

SUCCESS: Upload completed
```

---

### Phase 4: 驗證上傳 (可選)

```bash
python verify_upload.py
```

**預期輸出**: `upload_verification_report.json` (包含詳細驗證結果)

---

## 快速參考

### 3 行代碼即可完成

```bash
python main.py                    # 特徵工程
python step2_upload.py             # 上傳
python verify_upload.py            # 驗證
```

### 手動驗證

```python
from huggingface_hub import list_repo_files

files = list_repo_files(
    repo_id="zongowo111/v2-crypto-ohlcv-data",
    repo_type="dataset"
)

for symbol in ["BTCUSDT", "ETHUSDT"]:
    for timeframe in ["15m", "1h"]:
        path = f"klines/{symbol}/{timeframe}/"
        found = any(path in f for f in files)
        print(f"{symbol}/{timeframe}: {'OK' if found else 'MISSING'}")
```

---

## 故障排除

### 問題 1: FileNotFoundError (step2_output 不存在)

**原因**: 未執行 `main.py`

**解決方案**:
```bash
python main.py
```

---

### 問題 2: 404 Repository Not Found

**原因**: 使用錯誤的 repo ID

**檢查**:
- 舊版本 (錯誤): `zongowo111/zigzag-lstm-predictor` 
- 新版本 (正確): `zongowo111/v2-crypto-ohlcv-data`

**解決方案**: 確認使用新版本 `step2_upload.py`

---

### 問題 3: "Not logged in to HuggingFace"

**原因**: Token 未配置

**解決方案**:
```bash
huggingface-cli login
```

---

### 問題 4: "Cannot access dataset"

**原因**: 無上傳權限或 token 無效

**檢查清單**:
- [ ] Token 有 "write" 權限
- [ ] 帳號是 zongowo111
- [ ] Token 未過期
- [ ] 網路連線正常

**解決方案**:
```bash
huggingface-cli logout
huggingface-cli login  # 重新登入
```

---

### 問題 5: "API rate limit exceeded"

**原因**: 上傳請求太頻繁

**自動處理**: 系統已內置 5 秒延遲

**手動調整** (如需要): 編輯 `step2_upload.py` 增加延遲
```python
time.sleep(10)  # 改為 10 秒
```

---

## 檔案清單

### 核心執行檔案
- `main.py` - 特徵工程主程序 (已更新)
- `step2_upload.py` - HuggingFace 上傳系統
- `verify_upload.py` - 上傳驗證工具

### 文檔
- `QUICK_START.md` - 3 步快速指南
- `step2_upload_guide.md` - 詳細使用指南
- `FILE_MAPPING_REFERENCE.md` - 檔案對應規則
- `EXECUTION_INSTRUCTIONS.md` - 本檔案
- `STEP2_UPDATE_NOTES.md` - 更新說明

### 配置
- `config.py` - 參數配置
- `.env` - 環境變數 (需要手動建立)

---

## 環境變數配置 (可選)

### 方式 1: .env 檔案 (推薦)

在 `step2_feature_engineering/` 中建立 `.env` 檔案:

```bash
HF_TOKEN=your_huggingface_token_here
```

### 方式 2: 系統環境變數

Windows PowerShell:
```powershell
$env:HF_TOKEN="your_token"
```

Linux/Mac:
```bash
export HF_TOKEN="your_token"
```

### 方式 3: HuggingFace CLI

```bash
huggingface-cli login
```

---

## 性能指標

| 階段 | 時間 | 大小 | 備註 |
|------|------|------|------|
| main.py | 10-20 秒 | 430 MB | 特徵工程處理 |
| 登入 | 1-2 秒 | - | HuggingFace 認證 |
| BTCUSDT 上傳 | 3-5 分鐘 | 430 MB | 包含 12 個檔案 |
| 等待 | 5 秒 | - | 速率限制延遲 |
| ETHUSDT 上傳 | 1-2 分鐘 | 110 MB | 包含 12 個檔案 |
| 驗證 | 1-2 秒 | - | 遠端檔案檢查 |
| **總計** | **5-15 分鐘** | **540 MB** | **完整流程** |

---

## 驗證清單

### 執行前
- [ ] 已完成 Step 1 (step1_output 包含 BTCUSDT_15m.csv 和 BTCUSDT_1h.csv)
- [ ] 已安裝 huggingface-hub: `pip install huggingface-hub`
- [ ] HuggingFace 帳號已建立
- [ ] Token 已生成並有 write 權限

### 執行期間
- [ ] main.py 執行成功 (2/2 files processed successfully)
- [ ] step2_output 包含 13 個檔案
- [ ] HuggingFace 登入成功
- [ ] step2_upload.py 執行成功 (SUCCESS: Upload completed)

### 執行後
- [ ] 驗證報告生成: upload_verification_report.json
- [ ] 所有檔案驗證通過
- [ ] 可在 HuggingFace 上查看上傳的檔案
- [ ] Step 3 可以從 HuggingFace 下載特徵

---

## 下一步

1. **Step 3: Colab 訓練**
   - 使用線上特徵進行 LSTM 模型訓練
   - 可直接從 HuggingFace 下載: `hf_hub_download(...)`

2. **更新記錄** (可選)
   - 查看 `execution_summary.json` 了解處理詳情
   - 查看 `upload_verification_report.json` 了解驗證結果

3. **監控上傳** (可選)
   - 訪問: https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data
   - 確認檔案已上傳

---

**所有步驟已準備完成。開始執行!**
