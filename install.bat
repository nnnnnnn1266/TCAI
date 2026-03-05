@echo off
setlocal enabledelayedexpansion

set VENV_DIR=.venv

echo [1/5] 建立虛擬環境...
python -m venv %VENV_DIR%
if errorlevel 1 goto :error

echo [2/5] 升級 pip...
call %VENV_DIR%\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [3/5] 安裝依賴 requirements.txt...
call %VENV_DIR%\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo [4/5] 檢查 Ollama 是否可用...
where ollama >nul 2>&1
if errorlevel 1 (
  echo 警告：找不到 ollama 指令，略過模型下載。
  echo 請先安裝 Ollama，再手動執行：
  echo   ollama pull mxbai-embed-large
  echo   ollama pull llama3.1:latest
) else (
  echo [5/5] 下載模型（若已存在會自動略過）...
  ollama pull mxbai-embed-large
  if errorlevel 1 goto :error
  ollama pull llama3.1:latest
  if errorlevel 1 goto :error
)

echo.
echo ✅ 安裝完成！
echo 啟用環境：%VENV_DIR%\Scripts\activate
echo 啟動 UI：streamlit run temp.py
goto :eof

:error
echo.
echo ❌ 安裝失敗，請檢查上方錯誤訊息。
exit /b 1
