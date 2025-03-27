# TRT-Aimbot
![logo](assets/IMG_5686.png)

A high-performance, customizable aimbot system powered by YOLOv8 and TensorRT acceleration.

## Overview

This project combines computer vision with real-time detection to create a flexible aimbot system that can be trained on custom datasets. The workflow includes data collection, annotation, dataset preparation, and model training - all streamlined through batch scripts.

## Demo

![TRT-Aimbot in action](assets/trt-aimbot-test-gif.gif)

## Features

- **Complete training workflow** - Collect data, label images, prepare datasets, and train models
- **Simple UI-based labeling tool** - Quick bounding box annotation tool with direct YOLO format output
- **TensorRT acceleration** - High-performance inference using TensorRT optimization
- **Highly customizable** - Train on your own datasets and tweak performance parameters
- **Real-time detection** - Fast frame capture and processing with dxcam
- **PID-based movement control** - Smooth, controllable aim with configurable parameters
- **Strafe detection** - Accounts for player movement during target tracking
- **Performance optimizations** - Frame skipping, toggleable visualization, and more

## Requirements

### Hardware
- NVIDIA GPU (for TensorRT acceleration)
- 8GB+ RAM recommended
- Windows 10/11

### Software
- Python 3.8+ (recommended: Python 3.10)
- CUDA Toolkit 11.7+
- cuDNN 8.6.0+
- TensorRT 8.5.1+
- Visual Studio 2019+ with C++ build tools

## Development Environment

This project was developed and tested with the following specific setup:

- **Operating System**: Windows 10 (version 10.0.22631)
- **Python**: 3.11.7
- **CUDA**: 12.7
- **TensorRT**: 8.6.1
- **GPU**: NVIDIA GeForce RTX 4080
- **RAM**: 32GB (31,891 MB)
- **PyTorch**: 2.1.2+cu118
- **Ultralytics**: 8.1.9

This specific configuration is known to work well with the provided code. Your setup may vary, but staying close to these versions will minimize compatibility issues.

## Installation

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/trt-aimbot.git
   cd trt-aimbot
   ```

2. **Create a virtual environment**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

4. **Install PyTorch with CUDA**
   ```
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. **Install Ultralytics**
   ```
   pip install ultralytics
   ```

6. **Install TensorRT (if not already installed)**
   Follow NVIDIA's instructions for installing TensorRT appropriate for your CUDA version

## Usage

### Complete Workflow

The easiest way to use the system is through the master workflow script:

```
train_workflow.bat
```

This interactive script guides you through all steps of the process:

1. **Data Collection** - Captures training images from your screen
2. **Image Labeling** - Opens the simple labeler to annotate your images
3. **Dataset Preparation** - Splits the dataset and creates YAML configuration
4. **Model Training** - Trains the YOLOv8 model on your dataset
5. **Model Conversion** - Converts the trained model to a TensorRT engine

### Individual Steps

#### 1. Data Collection
```
1_collect_data.bat [count] [interval] [fov] [class]
```
- `count`: Number of images to collect (default: 100)
- `interval`: Seconds between captures (default: 0.5)
- `fov`: Size of capture area (default: 640)
- `class`: Object class name (default: head)

#### 2. Image Labeling
```
2_label_data.bat
```
Use the simple labeler to annotate your images:
- Left click + drag: Draw bounding box
- Right click: Delete last box
- C: Clear all boxes for current image
- A/D: Previous/Next image
- O: Open new directory
- Q: Quit

#### 3. Dataset Preparation
```
3_prepare_dataset.bat [train_ratio] [val_ratio] [test_ratio] [class]
```
- `train_ratio`: Proportion for training (default: 0.8)
- `val_ratio`: Proportion for validation (default: 0.1)
- `test_ratio`: Proportion for testing (default: 0.1)
- `class`: Object class name (default: head)

This script now works directly with the YOLO format files created by the simple labeler, splitting your dataset and generating the YAML configuration needed for training.

#### 4. Model Training
```
4_train_model.bat [model] [epochs] [batch] [img_size] [device]
```
- `model`: Base model (default: yolov8n.pt)
- `epochs`: Training epochs (default: 100)
- `batch`: Batch size (default: 16)
- `img_size`: Training image size (default: 640)
- `device`: Training device (default: 0 for GPU)

#### 5. Model Conversion (TensorRT)
```
5_convert_model.bat [--input_model INPUT_MODEL] [--output_model OUTPUT_MODEL]
```
- `--input_model`: Path to the PyTorch model (default: automatically finds latest train folder)
- `--output_model`: Path for the output TensorRT engine (default: models/[timestamp]_fp16.engine)

This final step automatically finds the latest training run (train, train2, train3, etc.) and converts the PyTorch model to a TensorRT engine with FP16 precision. The engine is named with a timestamp for easy tracking of different versions.

## Complete Training Pipeline

1. Collect data with `1_collect_data.bat`
2. Label images with `simple_labeler.py` 
3. Prepare dataset with `3_prepare_dataset.bat`
4. Train model with `4_train_model.bat`
5. Convert to TensorRT with `5_convert_model.bat`

### Using the Aimbot

After training, run the aimbot:
```
python aimbot_core.py
```

Keyboard controls:
- P: Toggle prediction
- I: Toggle PID control
- V: Toggle visualization
- F: Toggle FPS display
- +/-: Adjust mouse sensitivity
- C: Toggle aimbot active/inactive
- W: Window selection mode (choose which game window to track)

### Window Capture Features

The aimbot can now capture from any selected window:

1. When started, select your game window from the list
2. The aimbot will capture a 640x640 square from the center of the window
3. If your window moves, the aimbot will track it automatically
4. Press 'W' during operation to select a different window

#### Troubleshooting Capture Issues

If you experience freezing or static frames:

- Press 'D' to open the debug view showing the actual capture region
- Make sure the game is running in windowed mode, not fullscreen
- Try running the aimbot with administrator privileges
- Some games with anti-cheat may block screen capture tools

## Customization Options

### Model Size
Choose different YOLOv8 model sizes based on your needs:
- `yolov8n.pt` - Nano (fastest, less accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - XLarge (slowest, most accurate)

### Performance Tuning
- **Frame skipping**: Increase performance by processing every N frames
- **Visualization toggle**: Disable visual elements for higher performance
- **Mouse sensitivity**: Fine-tune aiming precision
- **PID parameters**: Adjust the PID controller constants in aimbot_core.py

## Technical Details

### Architecture
- **Detection**: YOLOv8 optimized with TensorRT for faster inference
- **Screen Capture**: DXcam for efficient DirectX-based screen capture
- **Aim Control**: PID controller for smooth mouse movement
- **Target Prediction**: Velocity-based prediction for moving targets

### File Structure
- `aimbot_core.py` - Main aimbot implementation
- `simple_labeler.py` - Image annotation tool
- `dataset_manager.py` - Data collection and management
- `modules/` - Support modules and utilities
- `*.bat` - Batch scripts for the workflow

## Disclaimer

This project is for educational purposes only. Use of aimbots may violate terms of service for many games. The authors do not endorse using this software in competitive online games. Use responsibly and at your own risk. 
