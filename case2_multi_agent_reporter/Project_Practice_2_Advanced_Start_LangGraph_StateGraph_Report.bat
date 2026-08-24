@echo off
title Project Practice 2 Advanced: LangGraph StateGraph Financial Research Report System
cd /d "%~dp0"

set "RUNTIME_DIR=%~dp0..\runtime"
if exist "%RUNTIME_DIR%\Scripts\python.exe" (
    set "PY_EXE=%RUNTIME_DIR%\Scripts\python.exe"
    set "ENV_NAME=Portable Standalone Runtime"
) else (
    set "PY_EXE=python"
    set "ENV_NAME=System Python Environment"
)
set "DOTENV_PATH=%RUNTIME_DIR%\.env"

cls
echo ===============================================================================
echo   [Project Practice 2 Advanced] LangGraph StateGraph Financial Research Report System [State-Machine Engine]
echo ===============================================================================
echo   - Course slides: corresponds to PPT Slides 58-59
echo   - Core architecture: LangGraph StateGraph + shared ReportState state bus
echo   - Model configuration: SiliconFlow Qwen/Qwen3-8B (free)
echo   - Runtime environment: %ENV_NAME%
echo ===============================================================================
echo.
echo   Starting LangGraph graph-based state-machine node orchestration...
echo.
"%PY_EXE%" generator_langgraph_en.py
echo.
echo ===============================================================================
echo   LangGraph research report generation completed! See the output_reports directory.
echo ===============================================================================
pause
