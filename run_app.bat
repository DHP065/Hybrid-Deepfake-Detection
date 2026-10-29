@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Change to script directory
cd /d "%~dp0"

REM Create virtual environment if missing (prefer Python 3.11 if available)
if not exist .venv\Scripts\python.exe (
  echo [INFO] Creating virtual environment...
  py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate

REM Upgrade pip (quietly) and install requirements
echo [INFO] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Launch Streamlit app
echo [INFO] Starting web app...
streamlit run app.py

endlocal
