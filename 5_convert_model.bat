@echo off
echo ===================================
echo TensorRT Model Conversion (FP16)
echo ===================================
echo.

REM Run the Python script with any arguments passed to this batch file
python model_utils.py %*

echo.
pause 