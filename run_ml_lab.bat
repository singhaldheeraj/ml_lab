@echo off
rem Double-click to start ML Lab.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH. Install Python 3 and tick "Add python.exe to PATH".
    pause
    exit /b 1
)

python -c "import numpy, matplotlib" >nul 2>nul
if errorlevel 1 (
    echo Installing required packages...
    python -m pip install -r requirements.txt
)

python run_ml_lab.py
if errorlevel 1 (
    echo.
    echo ML Lab exited with an error - see the messages above.
    pause
)
