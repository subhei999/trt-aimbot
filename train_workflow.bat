@echo off
title Complete YOLOv8 Training Workflow
color 0A

echo ===============================================
echo        COMPLETE YOLOV8 TRAINING WORKFLOW
echo ===============================================
echo.
echo This script will run the entire training workflow:
echo  1. Collect training images
echo  2. Label the collected images
echo  3. Prepare and split the dataset
echo  4. Train the YOLOv8 model
echo  5. Convert model to TensorRT engine
echo.
echo You can choose to run all steps or skip any step.
echo.
echo ===============================================
echo.

REM Set default values
set COUNT=100
set INTERVAL=0.5
set FOV=640
set CLASSES=head
set TRAIN_RATIO=0.8
set VAL_RATIO=0.1
set TEST_RATIO=0.1
set MODEL=yolov8n.pt
set EPOCHS=100
set BATCH=16
set IMG_SIZE=640
set DEVICE=0
set INPUT_MODEL=
set OUTPUT_MODEL=

:STEP1
echo STEP 1: DATA COLLECTION
echo --------------------------------------
choice /C YN /M "Do you want to collect training images"
if errorlevel 2 goto STEP2
if errorlevel 1 (
    cls
    echo STEP 1: DATA COLLECTION
    echo --------------------------------------
    echo.
    set /p COUNT="Number of images to collect [%COUNT%]: "
    set /p INTERVAL="Capture interval in seconds [%INTERVAL%]: "
    set /p FOV="FOV size [%FOV%]: "
    set /p CLASSES="Class name [%CLASSES%]: "
    
    echo.
    echo Running data collection with parameters:
    echo - Images: %COUNT%
    echo - Interval: %INTERVAL%s
    echo - FOV: %FOV%x%FOV%
    echo - Class: %CLASSES%
    echo.
    pause
    
    call 1_collect_data.bat %COUNT% %INTERVAL% %FOV% %CLASSES%
)

:STEP2
cls
echo STEP 2: IMAGE LABELING
echo --------------------------------------
choice /C YN /M "Do you want to label the collected images"
if errorlevel 2 goto STEP3
if errorlevel 1 (
    cls
    echo STEP 2: IMAGE LABELING
    echo --------------------------------------
    echo.
    echo This will open the image labeler.
    echo Remember the controls:
    echo - Left click + drag: Draw bounding box
    echo - Right click: Delete last box
    echo - C: Clear all boxes
    echo - A/D: Previous/Next image
    echo - O: Open directory
    echo - Q: Quit
    echo.
    pause
    
    call 2_label_data.bat
)

:STEP3
cls
echo STEP 3: DATASET PREPARATION
echo --------------------------------------
choice /C YN /M "Do you want to prepare and split the dataset"
if errorlevel 2 goto STEP4
if errorlevel 1 (
    cls
    echo STEP 3: DATASET PREPARATION
    echo --------------------------------------
    echo.
    set /p TRAIN_RATIO="Train ratio [%TRAIN_RATIO%]: "
    set /p VAL_RATIO="Validation ratio [%VAL_RATIO%]: "
    set /p TEST_RATIO="Test ratio [%TEST_RATIO%]: "
    set /p CLASSES="Class name [%CLASSES%]: "
    
    echo.
    echo Running dataset preparation with parameters:
    echo - Train: %TRAIN_RATIO%
    echo - Validation: %VAL_RATIO%
    echo - Test: %TEST_RATIO%
    echo - Class: %CLASSES%
    echo.
    pause
    
    call 3_prepare_dataset.bat %TRAIN_RATIO% %VAL_RATIO% %TEST_RATIO% %CLASSES%
)

:STEP4
cls
echo STEP 4: MODEL TRAINING
echo --------------------------------------
choice /C YN /M "Do you want to train the YOLOv8 model"
if errorlevel 2 goto STEP5
if errorlevel 1 (
    cls
    echo STEP 4: MODEL TRAINING
    echo --------------------------------------
    echo.
    set /p MODEL="Base model [%MODEL%]: "
    set /p EPOCHS="Number of epochs [%EPOCHS%]: "
    set /p BATCH="Batch size [%BATCH%]: "
    set /p IMG_SIZE="Image size [%IMG_SIZE%]: "
    set /p DEVICE="Device (0=GPU, cpu=CPU) [%DEVICE%]: "
    
    echo.
    echo Running model training with parameters:
    echo - Model: %MODEL%
    echo - Epochs: %EPOCHS%
    echo - Batch size: %BATCH%
    echo - Image size: %IMG_SIZE%
    echo - Device: %DEVICE%
    echo.
    pause
    
    call 4_train_model.bat %MODEL% %EPOCHS% %BATCH% %IMG_SIZE% %DEVICE%
)

:STEP5
cls
echo STEP 5: MODEL CONVERSION (TensorRT)
echo --------------------------------------
choice /C YN /M "Do you want to convert the model to TensorRT engine"
if errorlevel 2 goto END
if errorlevel 1 (
    cls
    echo STEP 5: MODEL CONVERSION (TensorRT)
    echo --------------------------------------
    echo.
    echo The script will automatically find the latest trained model.
    echo You can optionally specify custom input and output paths.
    echo Leave blank to use automatic detection and timestamp naming.
    echo.
    set /p INPUT_MODEL="Input model path (leave blank for auto-detect): "
    set /p OUTPUT_MODEL="Output engine path (leave blank for auto naming): "
    
    echo.
    echo Running model conversion...
    
    if not "%INPUT_MODEL%"=="" (
        if not "%OUTPUT_MODEL%"=="" (
            call 5_convert_model.bat --input_model "%INPUT_MODEL%" --output_model "%OUTPUT_MODEL%"
        ) else (
            call 5_convert_model.bat --input_model "%INPUT_MODEL%"
        )
    ) else (
        if not "%OUTPUT_MODEL%"=="" (
            call 5_convert_model.bat --output_model "%OUTPUT_MODEL%"
        ) else (
            call 5_convert_model.bat
        )
    )
)

:END
cls
echo ===============================================
echo      WORKFLOW COMPLETED SUCCESSFULLY!
echo ===============================================
echo.
echo Thank you for using the YOLOv8 Training Workflow.
echo Your trained model is saved in:
echo runs/detect/train/ (or trainX/ for multiple runs)
echo.
echo Your TensorRT engine (if converted) is saved in:
echo the models/ directory with a timestamp filename
echo.
echo You can now use your model with aimbot_core.py
echo (The aimbot will automatically use the latest engine)
echo.
pause 