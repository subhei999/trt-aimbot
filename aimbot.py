import dxcam
import cv2
import numpy as np
import win32api, win32con
import threading
import tkinter as tk
from tkinter import ttk
import time
import onnxruntime as ort

# Screen Capture Setup
camera = dxcam.create(output_color="BGR")

# YOLO ONNX Setup
session = ort.InferenceSession('yolov8n.onnx', providers=['CUDAExecutionProvider'])
input_size = (640, 640)

# GUI Setup
root = tk.Tk()
root.title("YOLO Aimbot GUI")
root.geometry("400x200")

# Parameters
deadzone_threshold = tk.IntVar(value=10)
smoothing_steps = tk.IntVar(value=5)
confidence_threshold = tk.DoubleVar(value=0.6)

# Capture Region
region = (0, 0, 1920, 1080)  # Adjust as necessary

# Functions
def detect(frame):
    resized_frame = cv2.resize(frame, input_size)
    blob = resized_frame.transpose(2, 0, 1).astype('float32') / 255.0
    blob = blob[np.newaxis, ...]
    outputs = session.run(None, {'images': blob})[0]
    return outputs

def select_target(predictions):
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        if conf > confidence_threshold.get():
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            return int(cx * region[2] / input_size[0]), int(cy * region[3] / input_size[1])
    return None

def smooth_move(start, end, steps):
    x_step = (end[0] - start[0]) / steps
    y_step = (end[1] - start[1]) / steps
    for i in range(1, steps + 1):
        win32api.SetCursorPos((int(start[0] + x_step * i), int(start[1] + y_step * i)))
        time.sleep(0.001)

def within_deadzone(cursor_pos, target_pos, threshold):
    distance = ((cursor_pos[0] - target_pos[0]) ** 2 + (cursor_pos[1] - target_pos[1]) ** 2) ** 0.5
    return distance < threshold

def aim_loop():
    camera.start(target_fps=144, region=region)
    while True:
        frame = camera.get_latest_frame()
        if frame is not None:
            predictions = detect(frame)
            target = select_target(predictions)
            if target:
                current_pos = win32api.GetCursorPos()
                if not within_deadzone(current_pos, target, deadzone_threshold.get()):
                    smooth_move(current_pos, target, smoothing_steps.get())

def start_aimbot():
    threading.Thread(target=aim_loop, daemon=True).start()

# GUI Widgets
start_button = ttk.Button(root, text="Start Aimbot", command=start_aimbot)
start_button.pack(pady=10)

deadzone_slider = ttk.Scale(root, from_=1, to=50, variable=deadzone_threshold, orient='horizontal')
ttk.Label(root, text="Deadzone Threshold").pack()
deadzone_slider.pack()

smooth_slider = ttk.Scale(root, from_=1, to=20, variable=smoothing_steps, orient='horizontal')
ttk.Label(root, text="Smoothing Steps").pack()
smooth_slider.pack()

conf_slider = ttk.Scale(root, from_=0.1, to=1.0, variable=confidence_threshold, orient='horizontal')
ttk.Label(root, text="Confidence Threshold").pack()
conf_slider.pack()

# Non-Focus Capture Window (Preview Only)
def update_preview():
    preview_frame = camera.get_latest_frame()
    if preview_frame is not None:
        preview_frame = cv2.resize(preview_frame, (400, 225))
        cv2.imshow('Aimbot Preview (No Focus)', preview_frame)
    root.after(5, update_preview)

root.after(5, update_preview)

# Run GUI
root.mainloop()
