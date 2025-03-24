@echo off
echo Starting data collection workflow...
echo.

REM Set default values
set COUNT=100
set INTERVAL=0.5
set FOV=640
set CLASSES=head

REM Debug info - show exactly what parameters were received
echo Parameters received:
echo 1: [%1]
echo 2: [%2]
echo 3: [%3]
echo 4: [%4]

REM Check if arguments are provided, handling potential quotes
if not "%~1"=="" set COUNT=%~1
if not "%~2"=="" set INTERVAL=%~2
if not "%~3"=="" set FOV=%~3
if not "%~4"=="" set CLASSES=%~4

echo After processing parameters:
echo COUNT: [%COUNT%]
echo INTERVAL: [%INTERVAL%]
echo FOV: [%FOV%]
echo CLASSES: [%CLASSES%]
echo.

echo Collecting %COUNT% images with %INTERVAL%s interval...
echo FOV size: %FOV%x%FOV%
echo Classes: %CLASSES%
echo.

REM Run the dataset manager with capture flag
python dataset_manager.py --capture --count %COUNT% --interval %INTERVAL% --fov %FOV% --classes %CLASSES%

echo.
echo ===================================
echo Data Collection Complete!
echo ===================================
echo.
echo Next steps:
echo 1. Run 2_label_data.bat to label your images
echo 2. Run 3_prepare_dataset.bat to convert and split the dataset 
echo 3. Run 4_train_model.bat to train the model
echo.
pause 