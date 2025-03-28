import os
import time
import json
import random
import shutil
import argparse
import cv2
import dxcam
import win32api
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Configuration
DEFAULT_CAPTURE_INTERVAL = 0.5  # seconds between captures
DEFAULT_CAPTURE_COUNT = 100     # how many images to capture
DEFAULT_FOV_SIZE = 640         # window capture dimension
DEFAULT_ROOT_FOLDER = 'dataset'
DEFAULT_IMAGES_FOLDER = 'images'
DEFAULT_LABELS_FOLDER = 'labels'
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.1
DEFAULT_TEST_RATIO = 0.1

def setup_folders(root_folder, session_id=None):
    """Setup folder structure for dataset with optional session identifier"""
    if session_id is None:
        # Generate a timestamp-based session ID if none provided
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main dataset folders
    os.makedirs(root_folder, exist_ok=True)
    
    # Create a session subfolder for raw collected images
    raw_folder = os.path.join(root_folder, 'raw', session_id)
    os.makedirs(raw_folder, exist_ok=True)
    
    # Create folders for the final dataset structure
    images_folder = os.path.join(root_folder, DEFAULT_IMAGES_FOLDER)
    labels_folder = os.path.join(root_folder, DEFAULT_LABELS_FOLDER)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(images_folder, split), exist_ok=True)
        os.makedirs(os.path.join(labels_folder, split), exist_ok=True)
    
    # Additional folder for annotations (if using a tool that exports JSON)
    annotations_folder = os.path.join(root_folder, 'annotations')
    os.makedirs(annotations_folder, exist_ok=True)
    
    return {
        'root': root_folder,
        'raw': raw_folder,
        'images': images_folder,
        'labels': labels_folder,
        'annotations': annotations_folder,
        'session_id': session_id
    }

def capture_images(folders, count, interval, fov_size):
    """Capture images from screen and save to raw folder"""
    # Automatically detect primary monitor resolution
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)

    # Center a square region on the detected screen
    left = (screen_width - fov_size) // 2
    top = (screen_height - fov_size) // 2
    right = left + fov_size
    bottom = top + fov_size
    region = (left, top, right, bottom)

    # Initialize camera
    camera = dxcam.create(region=region, output_color="BGR")
    camera.start(target_fps=0, video_mode=False)

    print(f"Capturing {count} images every {interval}s...")
    captured_files = []
    
    # Use tqdm for a progress bar
    for i in tqdm(range(count)):
        frame = camera.get_latest_frame()
        if frame is not None:
            # Use zero-padded file names for proper sorting
            file_name = f'img_{i:06d}.jpg'
            file_path = os.path.join(folders['raw'], file_name)
            cv2.imwrite(file_path, frame)
            captured_files.append(file_path)
        else:
            print("Warning: No frame captured...")

        time.sleep(interval)

    camera.stop()
    print(f"Captured {len(captured_files)} images to {folders['raw']}")
    
    # Return list of captured file paths
    return captured_files

def convert_json_to_yolo(json_file, class_map, output_folder):
    """
    Convert JSON annotations to YOLO format txt files
    Expected JSON format: 
    {
        "img_1.jpg": [
            {"class": "person", "bbox": [x1, y1, x2, y2]},
            ...
        ],
        ...
    }
    """
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    
    converted_files = []
    
    for image_file, objects in annotations.items():
        # Get image dimensions to normalize bounding box
        img_path = os.path.join(os.path.dirname(json_file), image_file)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_path} not found, skipping annotation")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping annotation")
            continue
            
        height, width = img.shape[:2]
        
        # Create txt file with same base name as image
        base_name = os.path.splitext(image_file)[0]
        txt_file = os.path.join(output_folder, f"{base_name}.txt")
        
        with open(txt_file, 'w') as f:
            for obj in objects:
                # Get class index from map
                class_name = obj["class"]
                if class_name not in class_map:
                    print(f"Warning: Class '{class_name}' not in class map, skipping")
                    continue
                    
                class_idx = class_map[class_name]
                
                # Convert bbox from [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]
                x1, y1, x2, y2 = obj["bbox"]
                
                # Normalize to 0-1
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                bbox_width = (x2 - x1) / width
                bbox_height = (y2 - y1) / height
                
                # Write YOLO format line: class x_center y_center width height
                f.write(f"{class_idx} {x_center} {y_center} {bbox_width} {bbox_height}\n")
        
        converted_files.append(txt_file)
    
    return converted_files

def split_dataset(folders, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, session=None):
    """Split dataset into train/val/test sets and organize files"""
    # If a specific session is provided, use that instead of the one in folders
    if session:
        # Correctly construct the raw folder path with raw/session
        raw_folder = os.path.join(folders['root'], 'raw', session)
        if not os.path.exists(raw_folder):
            print(f"Error: Session folder {raw_folder} not found")
            return
    else:
        raw_folder = folders['raw']
    
    print(f"Using source folder: {raw_folder}")
    
    # Get all image files from the source folder
    image_files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No image files found in the source folder!")
        return
    
    # Check for labels folder within the session folder
    labels_folder = os.path.join(raw_folder, 'labels')
    if not os.path.exists(labels_folder):
        print(f"Warning: No labels folder found at {labels_folder}")
        print("Make sure you have labeled your images.")
    
    # Shuffle files for random split
    random.shuffle(image_files)
    
    # Calculate split indices
    num_files = len(image_files)
    train_end = int(num_files * train_ratio)
    val_end = train_end + int(num_files * val_ratio)
    
    # Split into train/val/test
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]
    
    print(f"Splitting dataset: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Track copied files for verification
    train_imgs_copied = 0
    val_imgs_copied = 0
    test_imgs_copied = 0
    train_labels_copied = 0
    val_labels_copied = 0
    test_labels_copied = 0
    
    # Copy files to their respective folders
    for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        for file in files:
            # Copy image
            src_path = os.path.join(raw_folder, file)
            dst_path = os.path.join(folders['images'], split, file)
            shutil.copy2(src_path, dst_path)
            
            # Track copied images
            if split == 'train':
                train_imgs_copied += 1
            elif split == 'val':
                val_imgs_copied += 1
            else:
                test_imgs_copied += 1
            
            # If there's a corresponding label file in the session/labels folder, copy it
            base_name = os.path.splitext(file)[0]
            label_src = os.path.join(labels_folder, f"{base_name}.txt")
            if os.path.exists(label_src):
                label_dst = os.path.join(folders['labels'], split, f"{base_name}.txt")
                shutil.copy2(label_src, label_dst)
                
                # Track copied labels
                if split == 'train':
                    train_labels_copied += 1
                elif split == 'val':
                    val_labels_copied += 1
                else:
                    test_labels_copied += 1
    
    # Print summary of copied files
    print(f"Train: {train_imgs_copied} images, {train_labels_copied} labels")
    print(f"Val: {val_imgs_copied} images, {val_labels_copied} labels")
    print(f"Test: {test_imgs_copied} images, {test_labels_copied} labels")
    
    if train_labels_copied == 0 and val_labels_copied == 0 and test_labels_copied == 0:
        print("Warning: No label files were copied. Check that your labels are in the correct format and location.")
    
    # Return the split information
    return {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }

def generate_yaml(folders, class_names, output_file='dataset.yaml'):
    """Generate YAML configuration file for training"""
    yaml_path = os.path.join(folders['root'], output_file)
    
    # Create class name to index mapping
    class_dict = {name: idx for idx, name in enumerate(class_names)}
    
    # Get absolute paths for dataset
    root_path = os.path.abspath(folders['root'])
    
    # Create YAML content
    yaml_content = f"""# YOLO dataset configuration
path: {root_path}  # dataset root
train: {os.path.join(DEFAULT_IMAGES_FOLDER, 'train')}  # train images
val: {os.path.join(DEFAULT_IMAGES_FOLDER, 'val')}  # val images
test: {os.path.join(DEFAULT_IMAGES_FOLDER, 'test')}  # test images (optional)

# Classes
names:
"""
    
    # Add class names
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"
    
    # Write YAML file
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Generated YAML configuration at {yaml_path}")
    return yaml_path

def create_empty_annotations(folders, captured_files):
    """Create an empty JSON annotation file for the captured images"""
    annotations = {}
    
    for file_path in captured_files:
        # Get just the filename without path
        file_name = os.path.basename(file_path)
        # Add empty annotation entry
        annotations[file_name] = []
    
    json_file = os.path.join(folders['annotations'], f"annotations_{folders['session_id']}.json")
    with open(json_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created empty annotation template at {json_file}")
    print("Please fill in annotations using your preferred tool, then run:")
    print(f"python collect_training_images.py --convert-json {json_file} --split-data")
    
    return json_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Collect and process training images for YOLO object detection')
    
    # Capture options
    parser.add_argument('--capture', action='store_true', help='Capture new images')
    parser.add_argument('--count', type=int, default=DEFAULT_CAPTURE_COUNT, help='Number of images to capture')
    
    # Convert interval from string to float to handle Windows batch file parameter passing issues
    parser.add_argument('--interval', type=str, default=str(DEFAULT_CAPTURE_INTERVAL), 
                        help='Interval between captures in seconds')
    
    parser.add_argument('--fov', type=int, default=DEFAULT_FOV_SIZE, help='Size of capture region (square)')
    
    # Dataset options
    parser.add_argument('--root', type=str, default=DEFAULT_ROOT_FOLDER, help='Root folder for dataset')
    parser.add_argument('--session', type=str, default=None, help='Session identifier for a specific data collection session')
    
    # Processing options
    parser.add_argument('--convert-json', type=str, help='Convert JSON annotations to YOLO format')
    parser.add_argument('--split-data', action='store_true', help='Split dataset into train/val/test sets')
    
    # Convert ratios from string to float to handle Windows batch file parameter passing issues
    parser.add_argument('--train-ratio', type=str, default=str(DEFAULT_TRAIN_RATIO), help='Training set ratio')
    parser.add_argument('--val-ratio', type=str, default=str(DEFAULT_VAL_RATIO), help='Validation set ratio')
    parser.add_argument('--test-ratio', type=str, default=str(DEFAULT_TEST_RATIO), help='Test set ratio')
    
    # YAML generation
    parser.add_argument('--classes', type=str, nargs='+', default=['person'], help='Class names for dataset')
    parser.add_argument('--generate-yaml', action='store_true', help='Generate YAML configuration file')
    
    args = parser.parse_args()
    
    # Convert interval from string to float
    try:
        interval = float(args.interval.replace(',', '.'))  # Handle both comma and period as decimal separator
    except ValueError:
        print(f"Warning: Could not convert interval '{args.interval}' to float, using default: {DEFAULT_CAPTURE_INTERVAL}")
        interval = DEFAULT_CAPTURE_INTERVAL
    
    # Convert ratio parameters from string to float
    try:
        train_ratio = float(args.train_ratio.replace(',', '.'))
    except ValueError:
        print(f"Warning: Could not convert train ratio '{args.train_ratio}' to float, using default: {DEFAULT_TRAIN_RATIO}")
        train_ratio = DEFAULT_TRAIN_RATIO
    
    try:
        val_ratio = float(args.val_ratio.replace(',', '.'))
    except ValueError:
        print(f"Warning: Could not convert validation ratio '{args.val_ratio}' to float, using default: {DEFAULT_VAL_RATIO}")
        val_ratio = DEFAULT_VAL_RATIO
    
    try:
        test_ratio = float(args.test_ratio.replace(',', '.'))
    except ValueError:
        print(f"Warning: Could not convert test ratio '{args.test_ratio}' to float, using default: {DEFAULT_TEST_RATIO}")
        test_ratio = DEFAULT_TEST_RATIO
    
    # Normalize ratios to sum to 1.0
    total = train_ratio + val_ratio + test_ratio
    if total != 1.0:
        print(f"Warning: Ratios sum to {total}, normalizing to 1.0")
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    
    # Only create a new session folder if we're capturing new data
    if args.capture:
        # When capturing, we always want a new session
        folders = setup_folders(args.root, args.session)
        captured_files = capture_images(folders, args.count, interval, args.fov)
        # Create template annotation file
        create_empty_annotations(folders, captured_files)
    else:
        # For other operations, just ensure the base directories exist
        # without creating a new session folder
        folders = {
            'root': args.root,
            'raw': os.path.join(args.root, 'raw'),
            'images': os.path.join(args.root, DEFAULT_IMAGES_FOLDER),
            'labels': os.path.join(args.root, DEFAULT_LABELS_FOLDER),
            'annotations': os.path.join(args.root, 'annotations'),
            'session_id': args.session or datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Ensure only the necessary base directories exist
        os.makedirs(folders['root'], exist_ok=True)
        os.makedirs(folders['raw'], exist_ok=True)
        os.makedirs(folders['images'], exist_ok=True)
        os.makedirs(folders['labels'], exist_ok=True)
        os.makedirs(folders['annotations'], exist_ok=True)
        
        # Create split directories if needed for dataset splitting
        if args.split_data:
            for split in ['train', 'val', 'test']:
                os.makedirs(os.path.join(folders['images'], split), exist_ok=True)
                os.makedirs(os.path.join(folders['labels'], split), exist_ok=True)
    
    if args.convert_json:
        if not os.path.exists(args.convert_json):
            print(f"Error: JSON file {args.convert_json} not found")
            return
        
        # Create class map (name to index)
        class_map = {name: idx for idx, name in enumerate(args.classes)}
        
        # Convert JSON to YOLO format
        convert_json_to_yolo(args.convert_json, class_map, folders['labels'])
        print(f"Converted annotations to YOLO format in {folders['labels']}")
    
    if args.split_data:
        splits = split_dataset(folders, train_ratio, val_ratio, test_ratio, args.session)
        print("Dataset split complete")
    
    if args.generate_yaml:
        yaml_path = generate_yaml(folders, args.classes)
        print(f"YAML configuration file generated at {yaml_path}")
    
    # If no specific action was requested, show help
    if not (args.capture or args.convert_json or args.split_data or args.generate_yaml):
        parser.print_help()
        print("\nNo action specified. Please use one or more of the following flags:")
        print("  --capture: Capture new images")
        print("  --convert-json: Convert JSON annotations to YOLO format")
        print("  --split-data: Split dataset into train/val/test sets")
        print("  --generate-yaml: Generate YAML configuration file")

if __name__ == "__main__":
    main()
