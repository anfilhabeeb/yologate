@echo off
REM YOLOgate - run on Windows without Docker (lite: YOLO via ONNX + OpenCV).
REM Requires Python 3.10+ from https://www.python.org/downloads/ ("Add python.exe to PATH" ticked).
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
    echo Creating Python environment...
    py -3 -m venv venv 2>nul || python -m venv venv
    if errorlevel 1 ( echo Python not found. Install it from https://www.python.org/downloads/ & pause & exit /b 1 )
)
venv\Scripts\python.exe -m pip install --disable-pip-version-check --timeout 120 --retries 10 -q -r requirements-lite.txt
if errorlevel 1 ( echo Installing dependencies failed - check your internet connection and run again. & pause & exit /b 1 )
echo.
echo  YOLOgate is starting: open http://localhost:8000 in your browser
echo  Press Ctrl+C in this window to stop.
echo.
venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
