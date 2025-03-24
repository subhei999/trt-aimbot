import os
import glob
import time
import logging
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_latest_train_folder(base_dir="runs/detect/"):
    """
    Find the latest train folder (train, train2, train3, etc.) 
    that contains a weights/best.pt file.
    
    Args:
        base_dir (str): Base directory to search in

    Returns:
        str: Path to the latest train folder or None if not found
    """
    if not os.path.exists(base_dir):
        logger.error(f"Base directory {base_dir} does not exist")
        return None
    
    # Find all train folders
    train_folders = glob.glob(os.path.join(base_dir, "train*"))
    
    if not train_folders:
        logger.error(f"No train folders found in {base_dir}")
        return None
    
    valid_folders = []
    
    # Check each folder for a weights/best.pt file
    for folder in train_folders:
        model_path = os.path.join(folder, "weights", "best.pt")
        if os.path.exists(model_path):
            valid_folders.append(folder)
    
    if not valid_folders:
        logger.error("No train folders with weights/best.pt found")
        return None
    
    # Sort by modification time (newest first)
    valid_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    latest_folder = valid_folders[0]
    logger.info(f"Found latest train folder: {latest_folder}")
    
    return latest_folder

def get_timestamp_filename(prefix="", suffix=""):
    """
    Generate a filename with a timestamp.
    
    Args:
        prefix (str): Prefix for the filename
        suffix (str): Suffix for the filename (e.g., file extension)
    
    Returns:
        str: Timestamped filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{timestamp}{suffix}"

def convert_model(input_model=None, output_model=None):
    """
    Convert a YOLOv8 model to TensorRT FP16 format.
    
    If input_model is None, find the latest trained model.
    If output_model is None, generate a timestamped output path.
    
    Args:
        input_model (str): Path to input model (.pt file)
        output_model (str): Path to output TensorRT model (.engine file)
    
    Returns:
        tuple: (bool, str, str) success, input_model_path, output_model_path
    """
    # If no input model specified, find the latest train folder
    if input_model is None:
        latest_train_folder = find_latest_train_folder()
        if latest_train_folder is None:
            return False, None, None
        
        input_model = os.path.join(latest_train_folder, "weights", "best.pt")
    
    # If input model doesn't exist, return failure
    if not os.path.exists(input_model):
        logger.error(f"Input model not found: {input_model}")
        return False, input_model, None
    
    # If no output model specified, generate a timestamped filename
    if output_model is None:
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        output_model = os.path.join("models", get_timestamp_filename("", "_fp16.engine"))
    
    # Import here to avoid circular imports
    from model_convert_fp16 import convert_to_fp16
    
    # Convert the model
    success = convert_to_fp16(input_model, output_model)
    
    if success:
        logger.info(f"Successfully converted {input_model} to {output_model}")
    else:
        logger.error(f"Failed to convert {input_model} to {output_model}")
    
    return success, input_model, output_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find latest train folder and convert model")
    parser.add_argument("--input_model", help="Path to input model (.pt file)", default=None)
    parser.add_argument("--output_model", help="Path to output TensorRT model (.engine file)", default=None)
    
    args = parser.parse_args()
    
    success, input_path, output_path = convert_model(args.input_model, args.output_model)
    
    if success:
        print("\nConversion successful!")
        print(f"Input model: {input_path}")
        print(f"Output model: {output_path}")
        exit(0)
    else:
        print("\nConversion failed. Check the logs for details.")
        exit(1) 