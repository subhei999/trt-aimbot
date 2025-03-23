from ultralytics import YOLO
import ultralytics
import torch
import tensorrt
print(tensorrt.__version__)
assert tensorrt.Builder(tensorrt.Logger())
ultralytics.checks()
# Check if a GPU is available and set as the default device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and PyTorch is using GPU.")
else:
    device = torch.device("cpu")
    print("No GPU available, PyTorch is using CPU.")
# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.engine')

# Specify the source image
source = './Counter_Strike_2.jpg'

# Make predictions
results = model.predict(source, save=True, imgsz=640, conf=0.5)

# Extract bounding box dimensions
boxes = results[0].boxes.xywh.cpu()
for box in boxes:
    x, y, w, h = box
    print(f"Width of Box: {w}, Height of Box: {h}")