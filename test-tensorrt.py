import cv2
import numpy as np
import random
import time
import torch
import dxcam
import pathlib
from ultralytics import YOLO
import ultralytics
import modules.utils as utils
from modules.autobackend import AutoBackend

# Load the standard image
image_path = 'bus.jpg'
image = cv2.imread(image_path)
# Crop the image to 640x640
crop_size = 640
height, width, _ = image.shape
crop_x = (width - crop_size) // 2
crop_y = (height - crop_size) // 2
image = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

# Load the TensorRT model
tensorrt_model_path = 'yolov8n.engine'  # Replace with the actual TensorRT model path
tensorrt_model = YOLO("yolov8n.engine")#AutoBackend(tensorrt_model_path, device=torch.device('cuda:0'), fp16=True)
# Warmup
#tensorrt_model.warmup()

# Load the YOLOv8 model
yolov8_model_path = 'yolov8n.pt'  # Replace with the actual YOLOv8 model path
yolov8_model = YOLO(yolov8_model_path)

# Class Name and Colors
label_map = yolov8_model.names  # Assuming both models have the same class names
COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in label_map]

# TensorRT detection function
def tensorrt_detection(model, image):
    # Preprocess
    im = utils.preprocess(image)

    # Inference
    preds = model(im)

    # Post Process
    results = utils.postprocess(preds, im, image, model.names, "")
    d = results[0].boxes

    # Get information from result
    tensor_size = d.cls.size()[0]
    if tensor_size > 1:
        cls, conf, box = d.cls.squeeze(), d.conf.squeeze(), d.xyxy.squeeze()
    else:
        cls, conf, box = d.cls, d.conf, d.xyxy

    return cls, conf, box

# YOLOv8 detection function
def yolov8_detection(model, image):
    # Update object localizer
    results = model.predict(image, imgsz=640, conf=0.5, verbose=False)
    result = results[0].cpu()

    # Get information from result
    box = result.boxes.xyxy.numpy()
    conf = result.boxes.conf.numpy()
    cls = result.boxes.cls.numpy().astype(int)

    return cls, conf, box


inftime_i = time.perf_counter()
# Perform TensorRT detection
tensorrt_cls, tensorrt_conf, tensorrt_box = yolov8_detection(tensorrt_model, image)
inftime_f = time.perf_counter()
print(f'.engine:{1000*(inftime_f-inftime_i):.2f} ms')
# Perform YOLOv8 detection
inftime_i = time.perf_counter()
yolov8_cls, yolov8_conf, yolov8_box = yolov8_detection(yolov8_model, image)
inftime_f = time.perf_counter()
print(f'.pt:{1000*(inftime_f-inftime_i):.2f} ms')

# Visualize the results (draw bounding boxes)
tensorrt_output,closest_human_center = utils.draw_box(image.copy(), list(zip(tensorrt_cls, tensorrt_conf, tensorrt_box)), label_map, COLORS,0,0)
yolov8_output,closest_human_center = utils.draw_box(image.copy(), list(zip(yolov8_cls, yolov8_conf, yolov8_box)), label_map, COLORS,0,0)

# Display the results
cv2.imshow("YOLOv8 Detection", yolov8_output)
cv2.imshow("TensorRT Detection", tensorrt_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
