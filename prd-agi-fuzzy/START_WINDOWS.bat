@echo off
title PRD-AGI Phase 6 + Fuzzy
echo.
echo  ============================================================
echo   PRD-AGI Phase 6 + Fuzzy ^| The Nameless Intelligence
echo  ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit
)

if not exist "prd-env\Scripts\activate.bat" (
    echo  Creating virtual environment...
    python -m venv prd-env
)

call prd-env\Scripts\activate.bat
echo  Installing dependencies...
pip install -r requirements.txt -q
echo  Starting PRD-AGI...
python launch.py
pause
