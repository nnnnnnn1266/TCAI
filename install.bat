@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

set VENV_DIR=.venv
set INSTALL_MODE=%1
if "%INSTALL_MODE%"=="" set INSTALL_MODE=app

if "%INSTALL_MODE%"=="--help" goto :help
if "%INSTALL_MODE%"=="-h" goto :help

if /I "%INSTALL_MODE%"=="training" (
  set REQUIREMENTS_FILE=requirements-train.txt
  set TOTAL_STEPS=6
) else (
  set INSTALL_MODE=app
  set REQUIREMENTS_FILE=requirements.txt
  set TOTAL_STEPS=5
)

echo [%INSTALL_MODE%] 使用 %REQUIREMENTS_FILE%
echo [1/%TOTAL_STEPS%] 建立虛擬環境...
python -m venv %VENV_DIR%
if errorlevel 1 goto :error

echo [2/%TOTAL_STEPS%] 升級 pip / setuptools / wheel...
call %VENV_DIR%\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo 警告：無法從網路升級 pip / setuptools / wheel，將繼續使用現有版本。
)

echo [3/%TOTAL_STEPS%] 安裝依賴 %REQUIREMENTS_FILE%...
call %VENV_DIR%\Scripts\python.exe -m pip install -r %REQUIREMENTS_FILE%
if errorlevel 1 goto :error

if /I "%INSTALL_MODE%"=="training" (
  echo [4/%TOTAL_STEPS%] 提醒：微調需要 NVIDIA GPU + CUDA。
  call %VENV_DIR%\Scripts\python.exe -c "import sys; print('Python:', sys.version.split()[0])"
  if errorlevel 1 goto :error
)

set /a CHECK_STEP=%TOTAL_STEPS%-1
echo [!CHECK_STEP!/%TOTAL_STEPS%] 檢查 Ollama 是否可用...
where ollama >nul 2>&1
if errorlevel 1 (
  echo 警告：找不到 ollama 指令，略過模型下載。
  echo 請先安裝 Ollama，再手動執行：
  echo   ollama pull mxbai-embed-large
  echo   ollama pull llama3.1:latest
) else (
  echo [%TOTAL_STEPS%/%TOTAL_STEPS%] 下載模型（若已存在會自動略過）...
  ollama pull mxbai-embed-large
  if errorlevel 1 goto :error
  ollama pull llama3.1:latest
  if errorlevel 1 goto :error
)

echo.
echo [OK] 安裝完成！
echo 啟用環境：%VENV_DIR%\Scripts\activate
if /I "%INSTALL_MODE%"=="training" (
  echo 開始微調：%VENV_DIR%\Scripts\python.exe "turtle_llama3_1_(8b).py" --dataset turtleQA_R2.csv --question-column Question --answer-column Response --reasoning-column Complex_CoT
) else (
  echo 啟動 UI：%VENV_DIR%\Scripts\python.exe -m streamlit run temp.py
)
goto :eof

:help
echo 用法：
echo   install.bat            ^(安裝 RAG / 評估環境^)
echo   install.bat app        ^(同上^)
echo   install.bat training   ^(安裝 Llama 3.1 微調環境，含 Unsloth^)
echo   install.bat --help
goto :eof

:error
echo.
echo [ERROR] 安裝失敗，請檢查上方錯誤訊息。
exit /b 1
