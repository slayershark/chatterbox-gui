@echo off
title Chatterbox TTS Studio
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Python venv missing. Run:
    echo    py -3.11 -m venv .venv
    echo    .venv\Scripts\python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
    echo    .venv\Scripts\python -m pip install -r requirements.txt
    echo then run start.bat again.
    pause
    exit /b 1
)

if not exist "data\voices\presets\meta.json" (
    echo [setup] Creating voice presets...
    call .venv\Scripts\python scripts\make_presets.py
)

echo.
echo  -------------------------------------------
echo   Chatterbox TTS Studio  ^>  http://127.0.0.1:8488
echo  -------------------------------------------
echo.

.venv\Scripts\python server.py
