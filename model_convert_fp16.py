import os
import logging
import tensorrt as trt
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_fp16(model_path='models/best.pt', output_path='models/best_fp16.engine'):
    """
    Convert a YOLOv8 model to TensorRT FP16 precision format
    
    FP16 is still a significant improvement over FP32 and is widely supported.
    FP8 requires specific model structure with quantization layers which are 
    typically found in transformer models, not directly in YOLOv8.
    """
    logger.info(f"Converting {model_path} to FP16 precision")
    
    # Make sure models directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Export command with FP16 precision
    export_cmd = f"yolo export model={model_path} format=engine device=0 half=True workspace=8"
    
    logger.info(f"Running export command: {export_cmd}")
    result = os.system(export_cmd)
    
    if result != 0:
        logger.error("Export command failed!")
        return False
    
    # The export creates the engine file in the same location as the .pt file
    # with the extension changed to .engine
    engine_path = model_path.replace('.pt', '.engine')
    
    # Move the generated engine file to the desired output path if they're different
    if engine_path != output_path and os.path.exists(engine_path):
        import shutil
        shutil.move(engine_path, output_path)
        logger.info(f"Moved engine from {engine_path} to {output_path}")
    elif engine_path == output_path:
        logger.info(f"Engine created at the correct location: {output_path}")
    
    if os.path.exists(output_path):
        logger.info(f"Successfully created FP16 TensorRT engine at {output_path}")
        return True
    else:
        logger.error(f"Failed to create FP16 TensorRT engine at {output_path}")
        return False

if __name__ == "__main__":
    # Convert to FP16
    logger.info("Note: While true FP8 precision requires quantization layers typically found in transformer models,")
    logger.info("we can still get improved performance using FP16 precision with TensorRT.")
    convert_to_fp16() 