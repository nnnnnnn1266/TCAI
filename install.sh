#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

echo "[1/5] 建立虛擬環境..."
python -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/5] 升級 pip..."
python -m pip install --upgrade pip

echo "[3/5] 安裝依賴 requirements.txt..."
python -m pip install -r requirements.txt

echo "[4/5] 檢查 Ollama 是否可用..."
if command -v ollama >/dev/null 2>&1; then
  echo "Ollama 已安裝，準備下載模型..."
  echo "[5/5] 下載模型（若已存在會自動略過）..."
  ollama pull mxbai-embed-large
  ollama pull llama3.1:latest
else
  echo "警告：找不到 ollama 指令，略過模型下載。"
  echo "請先安裝 Ollama，再手動執行："
  echo "  ollama pull mxbai-embed-large"
  echo "  ollama pull llama3.1:latest"
fi

echo
echo "✅ 安裝完成！"
echo "啟用環境：source $VENV_DIR/bin/activate"
echo "啟動 UI：streamlit run temp.py"
