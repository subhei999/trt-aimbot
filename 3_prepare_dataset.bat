@echo off
echo Starting dataset preparation workflow...
echo.

REM Set default values
set TRAIN_RATIO=0.8
set VAL_RATIO=0.1
set TEST_RATIO=0.1
set CLASSES=head

REM Check if arguments are provided
if not "%1"=="" set TRAIN_RATIO=%1
if not "%2"=="" set VAL_RATIO=%2
if not "%3"=="" set TEST_RATIO=%3
if not "%4"=="" set CLASSES=%4

echo Preparing dataset with split ratios:
echo Train: %TRAIN_RATIO%
echo Val: %VAL_RATIO%
echo Test: %TEST_RATIO%
echo Classes: %CLASSES%
echo.

REM Find the most recent annotation file
for /f "delims=" %%a in ('dir /b /o-d dataset\annotations\annotations_*.json') do (
    set LATEST_JSON=%%a
    goto :found
)
:found

if not defined LATEST_JSON (
    echo Error: No annotation files found in dataset/annotations/
    echo Please run 1_collect_data.bat first and annotate the images.
    pause
    exit /b 1
)

echo Using annotation file: %LATEST_JSON%
echo.

REM Convert annotations to YOLO format
echo Converting annotations to YOLO format...
python dataset_manager.py --convert-json dataset\annotations\%LATEST_JSON% --classes %CLASSES%

REM Split the dataset
echo.
echo Splitting dataset...
python dataset_manager.py --split-data --train-ratio %TRAIN_RATIO% --val-ratio %VAL_RATIO% --test-ratio %TEST_RATIO%

REM Generate YAML configuration
echo.
echo Generating YAML configuration...
python dataset_manager.py --generate-yaml --classes %CLASSES%

echo.
echo ===================================
echo Dataset Preparation Complete!
echo ===================================
echo.
echo Next steps:
echo 1. Run 4_train_model.bat to train the YOLOv8 model
echo.
pause 