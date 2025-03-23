@echo off
echo Starting model training workflow...
echo.

REM Set default values
set MODEL=yolov8n.pt
set EPOCHS=100
set BATCH=16
set IMG_SIZE=640
set DEVICE=0

REM Check if arguments are provided
if not "%1"=="" set MODEL=%1
if not "%2"=="" set EPOCHS=%2
if not "%3"=="" set BATCH=%3
if not "%4"=="" set IMG_SIZE=%4
if not "%5"=="" set DEVICE=%5

echo Training configuration:
echo Model: %MODEL%
echo Epochs: %EPOCHS%
echo Batch size: %BATCH%
echo Image size: %IMG_SIZE%
echo Device: %DEVICE%
echo.

REM Check if YAML config exists
if not exist dataset\training_config.yaml (
    echo Error: training_config.yaml not found!
    echo Please run 2_prepare_dataset.bat first.
    pause
    exit /b 1
)

REM Run YOLOv8 training
echo Starting training...
yolo detect train model=%MODEL% data=dataset\training_config.yaml epochs=%EPOCHS% batch=%BATCH% imgsz=%IMG_SIZE% device=%DEVICE%

echo.
echo Training complete!
echo Next steps:
echo 1. Check the results in runs/detect/train
echo 2. Convert the best model to TensorRT using model_convert_fp16.py
echo 3. Test the model using aimbot_core.py
echo.
pause 