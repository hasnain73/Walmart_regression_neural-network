@echo off
echo ========================================
echo Fixing TensorFlow for Python 3.12
echo ========================================

echo.
echo Uninstalling old TensorFlow...
pip uninstall -y tensorflow

echo.
echo Installing TensorFlow 2.16.1 (Python 3.12 compatible)...
pip install tensorflow==2.16.1

echo.
echo Installing other updated dependencies...
pip install numpy==1.26.4 flask==3.0.0

echo.
echo ========================================
echo ✓ Fix Complete!
echo ========================================
echo.
echo Now run: python app_flask.py
echo ========================================
pause
