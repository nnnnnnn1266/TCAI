#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"
INSTALL_MODE="${1:-app}"

show_help() {
  cat <<'HELP'
用法：
  bash install.sh            # 安裝 RAG / 評估環境
  bash install.sh app        # 同上
  bash install.sh training   # 安裝 Llama 3.1 微調環境（含 Unsloth）
  bash install.sh --help
HELP
}

if [[ "$INSTALL_MODE" == "--help" || "$INSTALL_MODE" == "-h" ]]; then
  show_help
  exit 0
fi

if [[ "$INSTALL_MODE" == "training" ]]; then
  REQUIREMENTS_FILE="requirements-train.txt"
  TOTAL_STEPS=6
else
  INSTALL_MODE="app"
  REQUIREMENTS_FILE="requirements.txt"
  TOTAL_STEPS=5
fi

echo "[$INSTALL_MODE] 使用 $REQUIREMENTS_FILE"
echo "[1/$TOTAL_STEPS] 建立虛擬環境..."
python -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[2/$TOTAL_STEPS] 升級 pip / setuptools / wheel..."
if ! python -m pip install --upgrade pip setuptools wheel; then
  echo "警告：無法從網路升級 pip / setuptools / wheel，將繼續使用現有版本。"
fi

echo "[3/$TOTAL_STEPS] 安裝依賴 $REQUIREMENTS_FILE..."
python -m pip install -r "$REQUIREMENTS_FILE"

if [[ "$INSTALL_MODE" == "training" ]]; then
  echo "[4/$TOTAL_STEPS] 提醒：微調需要 NVIDIA GPU + CUDA。"
  python - <<'PY'
import sys
print(f"Python: {sys.version.split()[0]}")
PY
fi

echo "[$((TOTAL_STEPS-1))/$TOTAL_STEPS] 檢查 Ollama 是否可用..."
if command -v ollama >/dev/null 2>&1; then
  echo "Ollama 已安裝，準備下載模型..."
  echo "[$TOTAL_STEPS/$TOTAL_STEPS] 下載模型（若已存在會自動略過）..."
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
if [[ "$INSTALL_MODE" == "training" ]]; then
  echo "開始微調：python 'turtle_llama3_1_(8b).py' --dataset turtleQA_R2.csv --question-column Question --answer-column Response --reasoning-column Complex_CoT"
else
  echo "啟動 UI：python -m streamlit run temp.py"
fi
