@echo off
title Case 1 - Intelligent Customer Service and Enterprise Private Knowledge Base (Local RAG)
cd /d "%~dp0"

set "RUNTIME_DIR=%~dp0..\runtime"
if exist "%RUNTIME_DIR%\Scripts\python.exe" (
    set "PY_EXE=%RUNTIME_DIR%\Scripts\python.exe"
    set "ENV_NAME=Portable Python Environment"
) else (
    set "PY_EXE=python"
    set "ENV_NAME=System Python Environment"
)
set "DOTENV_PATH=%RUNTIME_DIR%\.env"

cls
echo ===============================================================================
echo   [Case 1] Intelligent Customer Service and Enterprise Private Knowledge Base System [Local RAG]
echo ===============================================================================
echo   - Corresponds to slides 10-37 in the course presentation
echo   - Architecture: LangChain LCEL + FAISS vector database
echo   - Model configuration: Embedding BAAI/bge-m3 ^| Chat Qwen/Qwen3-8B (cloud)
echo   - Runtime environment: %ENV_NAME%
echo ===============================================================================
echo.
echo   Starting the Streamlit Web interface. Please wait...
echo   The browser will open http://localhost:8501 automatically.
echo.
"%PY_EXE%" -m streamlit run app_en.py
echo.
echo ===============================================================================
echo   Press any key to stop the service and close this window...
echo ===============================================================================
pause >nul
