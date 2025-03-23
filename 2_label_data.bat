@echo off
title YOLO Data Labeler
echo ===================================
echo Image Labeling Tool
echo ===================================
echo.
echo This tool will open the simple image labeler to annotate your collected images.
echo Controls:
echo - Left click + drag: Draw bounding box
echo - Right click: Delete last box
echo - C: Clear all boxes for current image
echo - A/D: Previous/Next image
echo - O: Open new directory
echo - Q: Quit
echo.
echo Press any key to start labeling the most recent data collection session...
pause > nul

:: Run the simple labeler script
python simple_labeler.py --class-name head

echo.
echo ===================================
echo Labeling Complete!
echo ===================================
echo.
echo Next steps:
echo 1. Run 3_prepare_dataset.bat to convert and split the dataset
echo 2. Run 4_train_model.bat to train the model
echo.
pause 