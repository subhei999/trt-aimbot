import os
import cv2
import math
import time
import ctypes
import random
import torch
import pathlib
import win32api
import win32con
import numpy as np
import dxcam

from ctypes import windll, Structure, c_long, byref
from ultralytics import YOLO

import modules.utils as utils
from modules.autobackend import AutoBackend

# -------------------------------
# Constants
# -------------------------------
OUTPUT_DIRECTORY = 'frames'
VK_LSHIFT = 0xA2
VK_LEFT = 0x25      # Left arrow key
VK_RIGHT = 0x27     # Right arrow key
VK_TILDE = 0xC0     # Tilde key (~)

# -------------------------------
# Utility Classes & Functions
# -------------------------------
class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

class PIDController:
    """Basic PID controller for mouse movement."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        """Update PID values given the current error."""
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

def query_mouse_position():
    """Return the current mouse position as a dict with keys 'x' and 'y'."""
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return {"x": pt.x, "y": pt.y}

def is_key_pressed(key_code):
    """Check if a given key is currently pressed."""
    return ctypes.windll.user32.GetAsyncKeyState(key_code) != 0

def wait_for_time(seconds):
    """
    Busy-wait for a specified amount of time in seconds.
    Note: original code used ms, but it’s simpler to pass in seconds and convert if needed.
    """
    start_time = time.perf_counter()
    target_time = start_time + seconds
    while time.perf_counter() < target_time:
        pass

def move_cursor_if_key_pressed(dx, dy, key_code):
    """Move the cursor by (dx, dy) if a specified key is pressed."""
    if is_key_pressed(key_code):
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
        print('Moving mouse by:', int(dx), int(dy))

def tensorrt_detection(model, image):
    """
    Perform detection using a TensorRT-based model.
    (You had some commented-out lines for warmup; left them here if needed.)
    """
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

def yolov8_detection(model, image):
    """
    Perform detection using a YOLOv8 model.
    """
    results = model.predict(image, imgsz=640, conf=0.5, verbose=False)
    result = results[0].cpu()

    # Get information from result
    box = result.boxes.xyxy.numpy()
    conf = result.boxes.conf.numpy()
    cls = result.boxes.cls.numpy().astype(int)

    return cls, conf, box

def detect_targets(model, frame, file_extension):
    """
    Single entry-point for detection, deciding which detection function to use
    based on the file extension (".engine" -> YOLOv8 or tensorrt if needed).
    """
    if frame is None:
        return None, None, None

    if file_extension == ".engine":
        # You can switch to tensorrt_detection if you prefer:
        # cls, conf, box = tensorrt_detection(model, frame)
        cls, conf, box = yolov8_detection(model, frame)
    else:
        cls, conf, box = yolov8_detection(model, frame)

    return cls, conf, box

# -------------------------------
# Main Execution
# -------------------------------
def main():
    # Prepare output directory
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Model Selection & Loading
    model_path = 'best.engine'  # Replace with your actual path
    file_extension = pathlib.Path(model_path).suffix

    if file_extension == ".engine":
        # Using YOLO for label map extraction and then loading the engine
        tmp_model = YOLO('best.pt')
        label_map = tmp_model.names
        model = YOLO(model_path)
        # If you need TensorRT warmup:
        # model = AutoBackend(model_path, device=torch.device('cuda:0'), fp16=True)
        # model.warmup()
    else:
        model = YOLO(model_path)
        label_map = model.names

    # For random bounding-box colors (if needed)
    # COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in label_map]
    # We use "" in your code to disable color usage in draw_box
    COLORS = ""

    # Automatically detect the primary monitor resolution
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)

    # Center a 640×640 region on the detected screen
    left = (screen_width - 640) // 2
    top = (screen_height - 640) // 2
    right = left + 640
    bottom = top + 640
    region = (left, top, right, bottom)

    print(f'Using region: {region} (centered FOV 640x640 on {screen_width}x{screen_height})')

    camera = dxcam.create(region=region, output_color="BGR")
    camera.start(target_fps=0, video_mode=False)
    # PID controller for aim adjustment
    controller = PIDController(kp=0.4, ki=0.0, kd=0.1)

    # Variables for FPS measurement
    frame_count = 0
    total_fps = 0
    init_time = time.perf_counter()

    while True:
        # Start time for calculating per-frame FPS
        start_time = time.perf_counter()

        # Capture frame
        frame_start = time.perf_counter()
        frame = camera.get_latest_frame()
        frame_end = time.perf_counter()
        print(f'Frame capture time: {1000 * (frame_end - frame_start):.2f} ms')

        # Detection
        inf_start = time.perf_counter()
        cls, conf, box = detect_targets(model, frame, file_extension)
        inf_end = time.perf_counter()
        print(f'Inference time: {1000 * (inf_end - inf_start):.2f} ms')

        if frame is not None:
            detection_output = list(zip(cls, conf, box)) if cls is not None else []
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2

            # Draw bounding boxes and get the center of the closest human
            image_output, closest_human_center = utils.draw_box(
                frame,
                detection_output,
                label_map,
                COLORS,
                center_x,
                center_y
            )

            # Mouse movement logic
            if closest_human_center is not None:
                dx = closest_human_center[0] - center_x
                dy = closest_human_center[1] - center_y
                print(f'Closest human center: {closest_human_center}')

                # If Tilde is pressed, apply PID correction and move mouse
                #if is_key_pressed(VK_TILDE):
                control_output_x = controller.update(dx)
                control_output_y = controller.update(dy)
                inputtime_i = time.perf_counter()
                win32api.mouse_event(
                    win32con.MOUSEEVENTF_MOVE,
                    int(control_output_x),
                    int(control_output_y),
                    0,
                    0
                )
                print(f'Moving: {int(dx)}, {int(dy)}')
                inputtime_f = time.perf_counter()
                print(f'Input time: {1000 * (inputtime_f - inputtime_i):.2f} ms')

            # Draw FPS & total runtime
            end_time = time.perf_counter()
            frame_count += 1
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
            total_fps += fps
            avg_fps = total_fps / frame_count

            image_output = utils.draw_fps(avg_fps, image_output)
            image_output = utils.draw_time(time.perf_counter() - init_time, image_output)

            # Show frame
            if image_output is not None:
                cv2.imshow("Detection Output", image_output)

        # Exit condition
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        # Limit loop speed if desired (10 ms wait = 0.01 s)
        #wait_for_time(0.01)

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()

# Standard Python entry point
if __name__ == '__main__':
    main()
