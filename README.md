# TCAI: A Domain-Specific AI Assistant for Turtle Care

TCAI（Turtle Care AI）是面向烏龜飼養與照護領域的 AI 問答專案，結合：
- **領域資料集**（烏龜 QA）
- **RAG 檢索增強生成**（Streamlit + ChromaDB + Ollama）
- **自動化評估工具**（Accuracy / BLEU / ROUGE-L / F1 / Recall / SemSim / BERTScore）

---

## ✨ 專案亮點

- 以烏龜照護知識為核心，降低通用模型在專業領域的幻覺風險。
- 提供可互動的 Streamlit 問答介面（含 Top-K、距離閾值、檢索分數、歷史對話）。
- 提供可重現的評估 CLI 腳本，支援 `xlsx/csv` 輸入與結果匯出。

---

## 🧱 系統架構

```mermaid
flowchart LR
    A[使用者問題] --> B[Streamlit temp.py]
    B --> C[Ollama Embedding mxbai-embed-large]
    C --> D[ChromaDB Persistent Collection]
    D --> E[取回 Top-K 相關片段]
    E --> F[距離閾值過濾]
    F --> G[Ollama LLM llama3.1]
    G --> H[中文回答 + 來源片段 + 分數顯示]
```

---

## 📁 專案結構

- `temp.py`：RAG 問答介面（Streamlit）
- `所有指標.py`：模型回答評估 CLI
- `turtle_llama3_1_(8b).py`：Llama 3.1 微調流程（Unsloth）
- `requirements.txt`：執行依賴
- `turtleQA_R2.csv` / `turtle1QA.xlsx` / `烏龜問題集測試rag2.xlsx`：資料檔

---

## 🚀 快速開始（推薦）

### 0) 一鍵安裝（可選）

- macOS / Linux：
```bash
bash install.sh
```
- Windows：
```bat
install.bat
```

### 1) 建立環境與安裝

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2) 準備 Ollama 模型

```bash
ollama pull mxbai-embed-large
ollama pull llama3.1:latest
```

### 3) 啟動 RAG Web 介面

```bash
python -m streamlit run temp.py
```

### 4) 執行評估腳本

```bash
python 所有指標.py --input turtle1QA.xlsx --output model_scores.xlsx
```

---

## 🖥️ RAG 介面功能

在 `temp.py` 介面中，你可以：
- 調整 **Top-K 文件數**
- 調整 **初步召回數（重排前）**（先擴大召回再重排）
- 調整 **最大距離閾值（越小越嚴格）**
- 檢視 **檢索片段與混合分數（語意+關鍵字）**
- 設定 **最低混合分數門檻** 並查看問題關鍵字高亮
- 檢視與清除 **歷史對話**

---

## 📊 評估指標說明

`所有指標.py` 會輸出以下指標：
- **Accuracy**：完全比對正確率
- **BLEU**：字元級重疊品質
- **ROUGE-L**：長序列重疊品質
- **F1 / Recall**：字元集合層級評估
- **SemSim**：SBERT 語意相似度
- **BERTScore**：語意層級比對

輸出檔案預設為：`model_scores.xlsx`。

---

## 🧪 微調（選用）

若你要訓練領域模型，可參考：
- `turtle_llama3_1_(8b).py`

此腳本為 Unsloth/Colab 風格流程，建議在具 GPU 環境執行。

---

## ⚠️ 常見問題

1. **`streamlit: command not found`**
   - 請先啟用虛擬環境並安裝 `requirements.txt`。

2. **Ollama 呼叫失敗 / 模型找不到**
   - 先確認 Ollama 服務已啟動，且 `ollama pull` 完成模型下載。
   - 可先測試：`ollama list` 是否能看到 `mxbai-embed-large` 與 `llama3.1:latest`。

3. **首次啟動很慢**
   - 系統會先把知識庫建立 embedding 並寫入本地 ChromaDB；之後啟動會快很多。

---

## 🔗 參考連結

- Fine-tuning Colab: <https://colab.research.google.com/drive/1ObUhsEMFp2aHvy-fSGOtUY84A9JCBVH->
- Metrics Colab: <https://colab.research.google.com/drive/1IuupF9GvOvMlW0m2Eq3U08EP-ZtQMotB>

---

## 📌 License

若你有授權策略需求（研究/商用），建議在此補上明確 License（例如 MIT / Apache-2.0）。
