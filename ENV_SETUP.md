# TCAI 環境建立與執行步驟

這份文件把 **一般使用環境** 與 **Llama 3.1 微調環境** 分開，避免你只想跑 RAG/評估時，卻被 Unsloth / CUDA 依賴卡住。

---

## 1. 你要先決定要建哪一種環境

### A. 一般使用環境（推薦先做這個）
適合你要做這些事：
- 啟動 Streamlit RAG 介面
- 跑問答測試
- 跑 `所有指標.py` 評估

### B. 微調訓練環境
適合你要做這些事：
- 用自己的資料集重新訓練 TCAI
- 使用 `turtle_llama3_1_(8b).py`
- 輸出 LoRA adapter

> 微調環境需要 **NVIDIA GPU + CUDA**。如果你只有 CPU，請不要先裝 training 環境，先使用一般環境即可。

---

## 2. 一般使用環境建立步驟

### macOS / Linux
```bash
bash install.sh
```

### Windows
```bat
install.bat
```

### 手動版本
```bash
python -m venv .venv
source .venv/bin/activate   # Windows 改成 .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
ollama pull mxbai-embed-large
ollama pull llama3.1:latest
```

---

## 3. 微調訓練環境建立步驟

### macOS / Linux
```bash
bash install.sh training
```

### Windows
```bat
install.bat training
```

### 手動版本
```bash
python -m venv .venv
source .venv/bin/activate   # Windows 改成 .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-train.txt
```

---

## 4. 確認環境是否建好

```bash
python --version
python -m pip --version
python -m pytest tests/test_metrics_cli.py tests/test_training_cli.py
```

如果你要微調，再確認 GPU：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果輸出是 `True`，才建議直接開始微調。

---

## 5. 執行 TCAI 的方式

### 啟動 RAG Web UI
```bash
python -m streamlit run temp.py
```

### 執行評估
```bash
python 所有指標.py --input turtle1QA.xlsx --output model_scores.xlsx
```

### 用你的資料重新訓練 TCAI
```bash
python 'turtle_llama3_1_(8b).py' \
  --dataset turtleQA_R2.csv \
  --question-column Question \
  --answer-column Response \
  --reasoning-column Complex_CoT \
  --max-steps 200 \
  --output-dir outputs/llama3_1_tcai \
  --save-adapter-dir lora_model
```

---

## 6. 建議的實際操作順序

1. 先建一般環境。
2. 先確認 `temp.py` 與 `所有指標.py` 都能跑。
3. 再確認你是否真的有 NVIDIA GPU + CUDA。
4. 有 GPU 再安裝 training 環境。
5. 最後才執行 `turtle_llama3_1_(8b).py` 微調。

---

## 7. 常見問題

### `ollama: command not found`
代表你還沒安裝 Ollama，或 PATH 沒設好。

### `torch.cuda.is_available()` 是 `False`
代表目前環境無法做 Llama 3.1 微調。

### `unsloth` / `bitsandbytes` 安裝失敗
通常是因為系統、CUDA 或 Python 版本不匹配；這時請先保留一般環境，等 GPU 環境確認完成後再裝 training 依賴。
