@echo off
REM Walmart Sales NN - Environment Setup Script
REM Windows Batch Script

echo ========================================
echo Walmart Sales NN - Environment Setup
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10.19 first
    pause
    exit /b 1
)

echo.
echo [1/4] Creating virtual environment: walmart_nn_env
python -m venv walmart_nn_env

echo.
echo [2/4] Activating virtual environment...
call walmart_nn_env\Scripts\activate.bat

echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/4] Installing requirements...
pip install -r requirements.txt

echo.
echo ========================================
echo ✓ Setup Complete!
echo ========================================
echo.
echo To activate the environment later, run:
echo   walmart_nn_env\Scripts\activate.bat
echo.
echo Next steps:
echo 1. Download Walmart.csv from Kaggle
echo 2. Run: python train_model.py
echo 3. Run: python app.py
echo ========================================
pause
