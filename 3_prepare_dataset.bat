@echo off
echo Starting dataset preparation workflow...
echo.

REM Set default values
set TRAIN_RATIO=0.8
set VAL_RATIO=0.1
set TEST_RATIO=0.1
set CLASSES=head

REM Check if arguments are provided, handling potential quotes
if not "%~1"=="" set TRAIN_RATIO=%~1
if not "%~2"=="" set VAL_RATIO=%~2
if not "%~3"=="" set TEST_RATIO=%~3
if not "%~4"=="" set CLASSES=%~4

echo Preparing dataset with split ratios:
echo Train: %TRAIN_RATIO%
echo Val: %VAL_RATIO%
echo Test: %TEST_RATIO%
echo Classes: %CLASSES%
echo.

REM Find the most recent session folder with images
echo Looking for the most recent session folder with images...
set SOURCE_FOLDER=
for /f "delims=" %%a in ('dir /b /ad /o-d dataset\raw 2^>nul') do (
    if exist "dataset\raw\%%a\*.jpg" (
        set SOURCE_FOLDER=%%a
        goto :found_folder
    )
)
:found_folder

if "%SOURCE_FOLDER%"=="" (
    echo Error: No folders with images found in dataset\raw\
    echo Please run 1_collect_data.bat and 2_label_data.bat first.
    pause
    exit /b 1
)

echo Found session folder: %SOURCE_FOLDER%
echo.

REM Split the dataset
echo Splitting dataset...
python dataset_manager.py --split-data --session %SOURCE_FOLDER% --train-ratio %TRAIN_RATIO% --val-ratio %VAL_RATIO% --test-ratio %TEST_RATIO%

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