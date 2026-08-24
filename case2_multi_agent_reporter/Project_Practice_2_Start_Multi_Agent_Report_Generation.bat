@echo off
title Project Practice 2: Multi-Agent Multimodal Financial Research Report System (Manager-Workers)
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
echo   [Project Practice 2] Multi-Agent Multimodal Financial Research Report System [Classic Pipeline]
echo ===============================================================================
echo   - Course slides: corresponds to PPT Slides 38-57
echo   - Agents: Planner / DataEngineer / Analyst
echo   - Model configuration: SiliconFlow Qwen/Qwen3-8B (free)
echo   - Runtime environment: %ENV_NAME%
echo ===============================================================================
echo.
echo   Starting the multi-agent collaborative pipeline to analyze market data and generate an in-depth research report...
echo.
"%PY_EXE%" generator_en.py
echo.
echo ===============================================================================
echo   Research report generation completed!
echo   - Word deliverable saved to: output_reports\
echo   - High-resolution trend chart saved to: output_charts\
echo ===============================================================================
pause
