import os
import cv2
import math
import time
import ctypes
import pathlib
import win32api
import win32con
import win32gui
import win32ui
import numpy as np
import dxcam

from ctypes import windll, Structure, c_long, byref, wintypes
from ultralytics import YOLO

import modules.utils as utils

# -------------------------------
# Constants
# -------------------------------
VK_TILDE = 0xC0     # Tilde key (~)
VK_A = 0x41         # A key for strafe left
VK_D = 0x44         # D key for strafe right
VK_P = 0x50         # P key to toggle prediction
VK_I = 0x49         # I key to toggle PID controller
VK_PLUS = 0x6B      # Numpad + key to increase sensitivity
VK_MINUS = 0x6D     # Numpad - key to decrease sensitivity
VK_V = 0x56         # V key to toggle visualization
VK_F = 0x46         # F key to toggle FPS display
VK_C = 0x43         # C key to toggle aimbot active state
VK_W = 0x57         # W key to toggle window selection mode
WINDOW_NAME = "Detection Output"  # Define window name as a constant for reuse
KEY_TOGGLE_TOPMOST = ord('t')     # Press 't' to toggle always-on-top mode
KEY_QUIT = ord('q')              # Press 'q' to quit
KEY_TOGGLE_PREDICTION = ord('p')  # Press 'p' to toggle prediction
KEY_TOGGLE_PID = ord('i')        # Press 'i' to toggle PID controller
KEY_INCREASE_SENS = ord('+')     # Press '+' to increase sensitivity
KEY_DECREASE_SENS = ord('-')     # Press '-' to decrease sensitivity
KEY_TOGGLE_VISUAL = ord('v')     # Press 'v' to toggle visualization
KEY_TOGGLE_FPS = ord('f')        # Press 'f' to toggle FPS display
KEY_TOGGLE_ACTIVE = ord('c')     # Press 'c' to toggle aimbot active state
KEY_WINDOW_SELECT = ord('w')     # Press 'w' to toggle window selection mode
KEY_DEBUG_REGION = ord('d')      # Press 'd' to toggle capture region debug window

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

class TargetPredictor:
    """Predicts future position of targets based on movement history."""
    def __init__(self, history_size=5, prediction_time=0.05):
        self.history_size = history_size          # How many positions to keep in history
        self.prediction_time = prediction_time    # How far into the future to predict (in seconds)
        self.position_history = []                # List of (position, timestamp) tuples
        self.last_prediction = None               # Last predicted position
        self.strafe_detected = False              # Flag to indicate strafing
        self.strafe_direction = 0                 # Direction of strafing (-1 left, 1 right)
        self.strafe_compensation = 1.5            # Amplify prediction during strafing
    
    def add_position(self, position, timestamp):
        """Add a new position observation with timestamp."""
        if position is None:
            return
        
        self.position_history.append((position, timestamp))
        
        # Keep only the most recent observations
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
    
    def update_strafe_state(self):
        """Update strafe state based on key presses instead of velocity analysis."""
        # Check if A or D keys are pressed
        a_pressed = is_key_pressed(VK_A)
        d_pressed = is_key_pressed(VK_D)
        
        if a_pressed and not d_pressed:
            self.strafe_detected = True
            self.strafe_direction = -1  # Left
        elif d_pressed and not a_pressed:
            self.strafe_detected = True
            self.strafe_direction = 1   # Right
        else:
            self.strafe_detected = False
            self.strafe_direction = 0
    
    def predict_position(self, current_time):
        """Predict future position based on movement history."""
        if len(self.position_history) < 2:
            # Need at least two points to calculate velocity
            return self.position_history[-1][0] if self.position_history else None
        
        # Calculate velocity from the last few positions
        positions = [p[0] for p in self.position_history]
        times = [p[1] for p in self.position_history]
        
        # Use more recent points for rapidly changing trajectories
        recent_idx = min(2, len(positions) - 1)
        
        # Simple linear velocity calculation from the most recent positions
        dx = positions[-1][0] - positions[-1-recent_idx][0]
        dy = positions[-1][1] - positions[-1-recent_idx][1]
        dt = times[-1] - times[-1-recent_idx]
        
        if dt < 0.001:  # Avoid division by zero
            return positions[-1]
        
        # Calculate velocity
        vx = dx / dt
        vy = dy / dt
        
        # Apply strafe compensation if detected
        if self.strafe_detected:
            # Based on which key is pressed, apply compensation in the appropriate direction
            horizontal_multiplier = 1.0
            
            # Add extra compensation in the strafe direction
            if self.strafe_direction != 0:
                # If strafing right (D key), increase compensation for right-moving targets
                # If strafing left (A key), increase compensation for left-moving targets
                # This makes the prediction lead the target more when the player is strafing
                same_direction = (self.strafe_direction > 0 and vx > 0) or (self.strafe_direction < 0 and vx < 0)
                opposite_direction = (self.strafe_direction > 0 and vx < 0) or (self.strafe_direction < 0 and vx > 0)
                
                if same_direction:
                    # When target moves in same direction as player strafe, need more lead
                    horizontal_multiplier = self.strafe_compensation * 1.5
                elif opposite_direction:
                    # When target moves opposite to player strafe, need less lead
                    horizontal_multiplier = self.strafe_compensation * 0.8
                else:
                    # Default compensation
                    horizontal_multiplier = self.strafe_compensation
                
                # Apply the multiplier
                vx = vx * horizontal_multiplier
        
        # Predict future position
        adaptive_time = self.prediction_time
        pred_x = positions[-1][0] + vx * adaptive_time
        pred_y = positions[-1][1] + vy * adaptive_time
        
        # Store this prediction
        self.last_prediction = (pred_x, pred_y)
        
        return (pred_x, pred_y)
    
    def get_strafe_info(self):
        """Return information about detected strafing for visualization."""
        if self.strafe_detected:
            direction = "Right" if self.strafe_direction > 0 else "Left"
            return f"Player Strafe: {direction}"
        return None
    
    def reset(self):
        """Reset the predictor state."""
        self.position_history = []
        self.last_prediction = None
        self.strafe_detected = False
        self.strafe_direction = 0

def query_mouse_position():
    """Return the current mouse position as a dict with keys 'x' and 'y'."""
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return {"x": pt.x, "y": pt.y}

def is_key_pressed(key_code):
    """Check if a given key is currently pressed using GetAsyncKeyState."""
    return (ctypes.windll.user32.GetAsyncKeyState(key_code) & 0x8000) != 0

def enum_windows_callback(hwnd, windows):
    """Callback function for enumerating windows."""
    if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
        windows.append((hwnd, win32gui.GetWindowText(hwnd)))
    return True

def get_visible_windows():
    """Get a list of all visible windows with titles."""
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    # Sort windows by title for easier selection
    return sorted(windows, key=lambda x: x[1].lower())

def get_window_rect(hwnd):
    """Get the rectangle of a window."""
    rect = win32gui.GetWindowRect(hwnd)
    # rect is (left, top, right, bottom)
    return rect

def select_window(windows):
    """Display a list of windows and let user select one."""
    print("\nAvailable windows:")
    for i, (_, title) in enumerate(windows):
        print(f"{i}: {title}")
    
    while True:
        try:
            selection = input("\nEnter window number (or part of title to search): ")
            # Try to parse as a number first
            try:
                index = int(selection)
                if 0 <= index < len(windows):
                    return windows[index][0]  # Return the window handle
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                # Try to find by partial title match
                matches = [(i, title) for i, (_, title) in enumerate(windows) 
                           if selection.lower() in title.lower()]
                if len(matches) == 1:
                    return windows[matches[0][0]][0]  # Return the window handle
                elif len(matches) > 1:
                    print("\nMultiple matches found:")
                    for i, title in matches:
                        print(f"{i}: {title}")
                    continue
                else:
                    print("No matches found. Try again.")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None

def highlight_window(hwnd, duration=2.0):
    """Highlight the selected window by drawing a red border around it."""
    if not hwnd:
        return
    
    # Save original window position and style
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    rect = win32gui.GetWindowRect(hwnd)
    
    # Draw a border by creating a borderless, topmost window
    border_width = 5
    border_color = 0x0000FF  # Red in BGR format
    
    # Create windows for each side of the border
    borders = []
    for i in range(4):
        border_hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED,
            "STATIC",
            None,
            win32con.WS_POPUP,
            0, 0, 0, 0,
            None, None, None, None
        )
        borders.append(border_hwnd)
        
        # Set border color
        win32gui.SetLayeredWindowAttributes(
            border_hwnd, 
            border_color, 
            255,  # Alpha (0-255)
            win32con.LWA_COLORKEY | win32con.LWA_ALPHA
        )
    
    # Position the border windows
    left, top, right, bottom = rect
    # Top border
    win32gui.SetWindowPos(
        borders[0], win32con.HWND_TOPMOST,
        left, top, right - left, border_width,
        win32con.SWP_SHOWWINDOW
    )
    # Right border
    win32gui.SetWindowPos(
        borders[1], win32con.HWND_TOPMOST,
        right - border_width, top, border_width, bottom - top,
        win32con.SWP_SHOWWINDOW
    )
    # Bottom border
    win32gui.SetWindowPos(
        borders[2], win32con.HWND_TOPMOST,
        left, bottom - border_width, right - left, border_width,
        win32con.SWP_SHOWWINDOW
    )
    # Left border
    win32gui.SetWindowPos(
        borders[3], win32con.HWND_TOPMOST,
        left, top, border_width, bottom - top,
        win32con.SWP_SHOWWINDOW
    )
    
    # Wait for the specified duration
    time.sleep(duration)
    
    # Remove the border windows
    for border_hwnd in borders:
        win32gui.DestroyWindow(border_hwnd)

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

    cls, conf, box = yolov8_detection(model, frame)
    return cls, conf, box

def draw_model_info(image, model_path, fps, prediction_enabled=True, pid_enabled=True, mouse_sensitivity=0.5, aimbot_active=True):
    """Draw model information including precision on the image"""
    if image is None:
        return image
    
    # Check model precision
    precision = "FP16" if "fp16" in model_path.lower() else "FP32"
    
    # Draw text on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    info_text = f"Model: {os.path.basename(model_path)} ({precision})"
    fps_text = f"FPS: {fps:.1f}"
    controls_text = "Keys: Q-Quit, T-Toggle Pin, P-Prediction, I-PID, C-Toggle Aimbot"
    sens_text = f"Sensitivity: {mouse_sensitivity:.2f} (+/- to adjust)"
    
    # Mode indicators
    prediction_status = "ON" if prediction_enabled else "OFF"
    pid_status = "ON" if pid_enabled else "OFF"
    aimbot_status = "ON" if aimbot_active else "OFF"
    mode_text = f"Prediction: {prediction_status}  |  PID: {pid_status}  |  Aimbot: {aimbot_status}"
    
    # Background for better readability
    cv2.rectangle(image, (10, 10), (400, 120), (0, 0, 0), -1)
    
    # Add text
    cv2.putText(image, info_text, (20, 30), font, 0.6, (0, 255, 0), 2)
    cv2.putText(image, fps_text, (20, 50), font, 0.6, (0, 255, 0), 2)
    cv2.putText(image, controls_text, (20, 70), font, 0.5, (0, 255, 255), 1)
    cv2.putText(image, mode_text, (20, 90), font, 0.5, (0, 255, 255), 1)
    cv2.putText(image, sens_text, (20, 110), font, 0.5, (0, 255, 255), 1)
    
    return image

def position_window(window_name, x, y, width, height):
    """Position and resize an OpenCV window."""
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd == 0:
        print(f"Window '{window_name}' not found")
        return False
    
    # Set window position and size
    ctypes.windll.user32.SetWindowPos(
        hwnd, 
        0, 
        x, y, width, height, 
        0
    )
    
    print(f"Window '{window_name}' positioned at ({x}, {y}) with size {width}x{height}")
    return True

def mouse_move(dx, dy):
    """Alternative mouse movement function using ctypes directly."""
    # Constants from winuser.h
    MOUSEEVENTF_MOVE = 0x0001
    
    # Use ctypes directly
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, ctypes.c_long(dx), ctypes.c_long(dy), 0, 0)
    print(f"Direct ctypes mouse move: ({dx}, {dy})")

def capture_window_screenshot(hwnd, width=640, height=640):
    """Capture a screenshot of a window using win32gui and win32ui."""
    # Get window rectangle
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    window_width = right - left
    window_height = bottom - top
    
    # Find the center point
    center_x = window_width // 2
    center_y = window_height // 2
    
    # Calculate capture region (centered)
    half_width = width // 2
    half_height = height // 2
    
    # Create device context
    try:
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        
        # Create bitmap
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, window_width, window_height)
        cDC.SelectObject(dataBitMap)
        
        # Copy screen to bitmap
        cDC.BitBlt((0, 0), (window_width, window_height), dcObj, (0, 0), win32con.SRCCOPY)
        
        # Convert bitmap to numpy array
        bmpinfo = dataBitMap.GetInfo()
        bmpstr = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
        
        # Remove alpha channel
        img = img[:, :, :3]
        
        # Crop to center region
        start_x = max(0, center_x - half_width)
        start_y = max(0, center_y - half_height)
        end_x = min(window_width, start_x + width)
        end_y = min(window_height, start_y + height)
        
        # If window is too small, pad with black
        if end_x - start_x < width or end_y - start_y < height:
            # Create black image of target size
            result = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate offset for centered placement
            offset_x = (width - (end_x - start_x)) // 2
            offset_y = (height - (end_y - start_y)) // 2
            
            # Place the captured portion in the center
            actual_width = min(end_x - start_x, width)
            actual_height = min(end_y - start_y, height)
            result[offset_y:offset_y+actual_height, offset_x:offset_x+actual_width] = img[start_y:end_y, start_x:end_x]
        else:
            # Just crop the center if window is large enough
            result = img[start_y:end_y, start_x:end_x]
        
        # Clean up
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        
        # Convert to BGR for OpenCV
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        return result
    
    except Exception as e:
        print(f"Error in window capture: {e}")
        return None

# -------------------------------
# Main Execution
# -------------------------------
def main():
    # Find the latest TensorRT engine model
    models_dir = 'models'
    engine_models = [f for f in os.listdir(models_dir) if f.endswith('_fp16.engine') or f.endswith('.engine')]
    
    if not engine_models:
        print("No TensorRT engine models found in models directory.")
        print("Please run 5_convert_model.bat to convert a model first.")
        return
    
    # Sort by modification time, newest first
    engine_models.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    latest_model = engine_models[0]
    model_path = os.path.join(models_dir, latest_model)
    
    print(f"Using latest TensorRT engine: {model_path}")
    
    file_extension = pathlib.Path(model_path).suffix

    if file_extension == ".engine":
        # Find a corresponding .pt model for label mapping
        pt_models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
        if pt_models:
            # Sort by modification time, newest first
            pt_models.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
            pt_model_path = os.path.join(models_dir, pt_models[0])
            
            # Using YOLO for label map extraction and then loading the engine
            tmp_model = YOLO(pt_model_path)
            label_map = tmp_model.names
            model = YOLO(model_path)
            print(f"Using label map from: {pt_model_path}")
        else:
            # Fallback if no .pt model is found
            model = YOLO(model_path)
            label_map = model.names
            print("No .pt model found for label mapping, using engine's built-in labels")
    else:
        model = YOLO(model_path)
        label_map = model.names
        print(f"Using YOLO model: {model_path}")

    # We use "" to disable color usage in draw_box
    COLORS = ""

    # Automatically detect the primary monitor resolution
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)
    
    # Initialize window selection
    target_window = None
    window_title = "Unknown"
    
    # Get available windows and let user select one
    print("Please select the game window to capture from:")
    windows = get_visible_windows()
    target_window = select_window(windows)
    
    if target_window:
        window_rect = get_window_rect(target_window)
        window_title = win32gui.GetWindowText(target_window)
        print(f"Selected window: {window_title}")
        print(f"Window rectangle: {window_rect}")
        highlight_window(target_window)
        
        # Use window dimensions for capture region
        left, top, right, bottom = window_rect
        
        # Get window dimensions
        window_width = right - left
        window_height = bottom - top
        
        # Find the center point of the window
        center_x = left + window_width // 2
        center_y = top + window_height // 2
        
        # Create a 640x640 square region centered on the window
        capture_size = 640
        half_size = capture_size // 2
        
        # Calculate the square region
        region_left = center_x - half_size
        region_top = center_y - half_size
        region_right = center_x + half_size
        region_bottom = center_y + half_size
        
        # Create region tuple for DXcam
        region = (region_left, region_top, region_right, region_bottom)
        
        print(f"Using center of window: {window_title}")
        print(f"Window size: {window_width}x{window_height}")
        print(f"Capturing 640x640 region from center: {region}")
        
        # Highlight the capture region
        try:
            # Create a transparent window that shows just the capture area
            highlight_hwnd = win32gui.CreateWindowEx(
                win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT,
                "STATIC",
                None,
                win32con.WS_POPUP,
                region_left, region_top, capture_size, capture_size,
                None, None, None, None
            )
            
            # Set border color to green
            win32gui.SetLayeredWindowAttributes(
                highlight_hwnd, 
                0x00FF00,  # Green in BGR format
                128,  # Alpha (0-255)
                win32con.LWA_COLORKEY | win32con.LWA_ALPHA
            )
            
            # Show the highlight
            win32gui.ShowWindow(highlight_hwnd, win32con.SW_SHOW)
            time.sleep(2.0)
            
            # Remove the highlight
            win32gui.DestroyWindow(highlight_hwnd)
        except Exception as e:
            print(f"Error highlighting capture region: {e}")
        
        # Create camera and start capturing
        camera = dxcam.create(region=region, output_color="BGR")
        camera.start(target_fps=120, video_mode=True)  # Changed to video_mode=True for continuous capture
        
        # Give the camera a moment to initialize
        time.sleep(0.5)
    else:
        # No window selected, use center of screen as fallback
        print("No window selected, using center of screen.")
        left = (screen_width - 640) // 2
        top = (screen_height - 640) // 2
        right = left + 640
        bottom = top + 640
        region = (left, top, right, bottom)

    print(f'Using capture region: {region}')

    # Create a named window before showing any frames
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    
    # Calculate a good position for the preview window (top-right corner by default)
    preview_width = 640  # Full original size
    preview_height = 640
    preview_x = screen_width - preview_width - 20  # 20px padding from right edge
    preview_y = 40  # 40px padding from top
    
    # PID controller for aim adjustment - reduce kp value to decrease overshooting
    controller = PIDController(kp=0.3, ki=0.0, kd=0.1)
    
    # Add target predictor with reduced prediction time and scaling
    predictor = TargetPredictor(history_size=5, prediction_time=0.05)
    
    # Prediction scaling factor to reduce overshooting (0.0-1.0)
    prediction_scale = 0.3
    
    # Mouse sensitivity calibration (pixels to mouse movement ratio)
    # This is critical for preventing oscillations when PID is disabled
    mouse_sensitivity = 0.5  # Starting value (can be adjusted with + and - keys)
    
    # Mode flags
    prediction_enabled = True
    pid_enabled = True
    aimbot_active = True     # Toggle with 'c' key
    visualization_enabled = True  # Toggle with 'v' key
    fps_display_enabled = True    # Toggle with 'f' key
    window_selection_mode = False # Toggle with 'w' key
    debug_capture_region = False  # Toggle with 'd' key - shows the exact capture area
    
    # Performance optimization options
    process_every_n_frames = 1    # Process every frame by default, increase to skip frames
    visualization_interval = 1    # Update visualization every N frames

    # Variables for FPS measurement
    frame_count = 0
    total_fps = 0
    init_time = time.perf_counter()
    display_fps = 0  # Store calculated FPS for display
    
    # Flag to track if window has been configured
    window_configured = False
    
    # Flags to track if sensitivity keys are pressed
    sens_up_pressed = False
    sens_down_pressed = False
    
    # Store latest detection and image for visualization
    last_image_output = None
    last_closest_human_center = None
    
    print("\nAimbot started. Press 'W' to change target window, 'Q' to quit.")
    
    while True:
        # Start time for calculating per-frame FPS
        start_time = time.perf_counter()

        # Check if we should reposition the capture region (if window moves)
        if target_window and win32gui.IsWindow(target_window):
            try:
                current_rect = get_window_rect(target_window)
                if current_rect != window_rect:
                    print(f"Window moved. Updating capture region.")
                    left, top, right, bottom = current_rect
                    window_rect = current_rect
                    
                    # Get window dimensions
                    window_width = right - left
                    window_height = bottom - top
                    
                    # Find the center point of the window
                    center_x = left + window_width // 2
                    center_y = top + window_height // 2
                    
                    # Create a 640x640 square region centered on the window
                    capture_size = 640
                    half_size = capture_size // 2
                    
                    # Calculate the square region
                    region_left = center_x - half_size
                    region_top = center_y - half_size
                    region_right = center_x + half_size
                    region_bottom = center_y + half_size
                    
                    # Update region
                    region = (region_left, region_top, region_right, region_bottom)
                    
                    print(f"Window moved. Updated capture region: {region}")
                    
                    # Restart camera with new region
                    camera.stop()
                    camera = dxcam.create(region=region, output_color="BGR")
                    camera.start(target_fps=120, video_mode=True)  # Changed to video_mode=True for continuous capture
                    
                    # Give the camera a moment to initialize
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error updating window position: {e}")
        
        # Capture frame
        frame = camera.get_latest_frame()
        
        # If frame capture fails, try an alternative capture method
        if frame is None:
            # Print warning only occasionally to avoid spam
            if frame_count % 30 == 0:
                print("Warning: Primary frame capture failed, trying backup method...")
            try:
                # Alternative capture using grab method
                frame = camera.grab()
                if frame is not None and frame_count % 60 == 0:
                    print(f"Successfully captured frame with backup method: {frame.shape}")
            except Exception as e:
                if frame_count % 60 == 0:
                    print(f"Backup capture also failed: {e}")
            
            # If both dxcam methods failed and we have a valid window, try direct window capture
            if frame is None and target_window and win32gui.IsWindow(target_window):
                try:
                    if frame_count % 30 == 0:
                        print("Attempting direct window capture as last resort...")
                    frame = capture_window_screenshot(target_window)
                    if frame is not None and frame_count % 60 == 0:
                        print(f"Successfully captured frame with direct window capture: {frame.shape}")
                except Exception as e:
                    if frame_count % 60 == 0:
                        print(f"Direct window capture failed: {e}")
        
        # Add debug information for frame capture
        if frame is None:
            if frame_count % 30 == 0:  # Reduce log frequency
                print("Warning: All frame capture methods failed")
        elif frame_count % 60 == 0:  # Only log every 60 frames to avoid spam
            print(f"Frame captured: {frame.shape}")
        
        # Display the raw capture region if debug is enabled
        if debug_capture_region and frame is not None:
            # Draw a border around the frame to make it easier to see
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (0, 0), (debug_frame.shape[1]-1, debug_frame.shape[0]-1), (0, 255, 0), 2)
            
            # Add text showing current captured area
            cv2.putText(debug_frame, f"Raw Capture: {region}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show the debug window
            cv2.imshow("Capture Region Debug", debug_frame)
        
        # Skip processing for performance if needed
        should_process = (frame_count % process_every_n_frames == 0)
        should_visualize = visualization_enabled and (frame_count % visualization_interval == 0)
        
        # Process frame (detection and aiming)
        if should_process and frame is not None:
            # Detection
            cls, conf, box = detect_targets(model, frame, file_extension)
            
            detection_output = list(zip(cls, conf, box)) if cls is not None else []
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2

            # Draw bounding boxes and get the center of the closest human
            # Only generate visual output if visualization is enabled
            if should_visualize:
                image_output, closest_human_center = utils.draw_box(
                    frame,
                    detection_output,
                    label_map,
                    COLORS,
                    center_x,
                    center_y
                )
                last_image_output = image_output
            else:
                # Still get the closest human center for aiming, but skip visualization
                _, closest_human_center = utils.draw_box(
                    frame,
                    detection_output,
                    label_map,
                    COLORS,
                    center_x,
                    center_y,
                    draw_boxes=False  # Skip drawing for performance
                )
            
            last_closest_human_center = closest_human_center

            # Mouse movement logic
            if closest_human_center is not None:
                # Add the position to our predictor (always track even if prediction is disabled)
                current_time = time.perf_counter()
                predictor.add_position(closest_human_center, current_time)
                
                # Check for strafe key presses (A/D) if prediction is enabled
                if prediction_enabled:
                    predictor.update_strafe_state()
                
                # Determine target position based on prediction mode
                if prediction_enabled:
                    # Get predicted position
                    predicted_position = predictor.predict_position(current_time)
                    
                    if predicted_position is not None:
                        # Use predicted position for aiming
                        pred_x, pred_y = predicted_position
                        
                        # Apply prediction scaling to reduce overshooting
                        # Scale between current position and predicted position
                        scaled_x = closest_human_center[0] + prediction_scale * (pred_x - closest_human_center[0])
                        scaled_y = closest_human_center[1] + prediction_scale * (pred_y - closest_human_center[1])
                        
                        dx = scaled_x - center_x
                        dy = scaled_y - center_y
                        
                        # Draw predicted position (for visualization)
                        if should_visualize:
                            cv2.circle(image_output, (int(pred_x), int(pred_y)), 5, (0, 255, 255), -1)  # Full prediction (yellow)
                            cv2.circle(image_output, (int(scaled_x), int(scaled_y)), 5, (0, 255, 0), -1)  # Scaled prediction (green)
                            
                            # Ensure both points are tuples of integers for cv2.line
                            start_point = (int(closest_human_center[0]), int(closest_human_center[1]))
                            end_point = (int(scaled_x), int(scaled_y))
                            cv2.line(image_output, start_point, end_point, (0, 255, 0), 2)
                            
                            # Show strafe detection info
                            strafe_info = predictor.get_strafe_info()
                            if strafe_info:
                                # Draw strafe indicator near the target with blue background
                                text_size = cv2.getTextSize(strafe_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                text_x = int(closest_human_center[0] - text_size[0] // 2)
                                text_y = int(closest_human_center[1] - 30)  # 30 pixels above target
                                cv2.rectangle(image_output, 
                                            (text_x - 5, text_y - 20), 
                                            (text_x + text_size[0] + 5, text_y + 5), 
                                            (255, 0, 0), -1)
                                cv2.putText(image_output, strafe_info, (text_x, text_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        # Fall back to current position if prediction isn't available
                        dx = closest_human_center[0] - center_x
                        dy = closest_human_center[1] - center_y
                else:
                    # No prediction, use current position
                    dx = closest_human_center[0] - center_x
                    dy = closest_human_center[1] - center_y
                
                # Apply aim movement based on PID mode
                if pid_enabled:
                    # Use PID controller for smooth movement
                    control_output_x = controller.update(dx)
                    control_output_y = controller.update(dy)
                else:
                    # Direct movement with calibrated sensitivity
                    # This prevents the oscillation problems when PID is disabled
                    control_output_x = dx * mouse_sensitivity
                    control_output_y = dy * mouse_sensitivity
                
                # Apply the mouse movement only if aimbot is active
                if aimbot_active:
                    mouse_move(int(control_output_x), int(control_output_y))
                
                # Debug output to check if mouse movement values are reasonable
                print(f"Moving mouse by: ({int(control_output_x)}, {int(control_output_y)}), Active: {aimbot_active}")
                print(f"Raw target offset: ({dx}, {dy})")
                print(f"PID: {pid_enabled}, Sensitivity: {mouse_sensitivity}")
            else:
                # No target detected, reset the predictor
                predictor.reset()
        
        # Calculate FPS only if processing was done
        if should_process:
            # Calculate FPS
            end_time = time.perf_counter()
            frame_time = end_time - start_time
            current_fps = 1 / frame_time if frame_time > 0 else 0
            # Smooth FPS display with moving average (70% previous, 30% current)
            display_fps = display_fps * 0.7 + current_fps * 0.3 if display_fps > 0 else current_fps
            
            total_fps += current_fps
        
        # Always increment frame count, regardless of processing
        frame_count += 1

        # Visual display - only update visuals at the specified interval
        if visualization_enabled and last_image_output is not None:
            # Add model information and FPS only if enabled
            if fps_display_enabled:
                # Draw FPS & model info
                image_display = draw_model_info(
                    last_image_output.copy(), 
                    model_path, 
                    display_fps, 
                    prediction_enabled, 
                    pid_enabled, 
                    mouse_sensitivity,
                    aimbot_active
                )
                
                # Add total runtime and window info
                image_display = utils.draw_time(time.perf_counter() - init_time, image_display)
                
                # Add window title info
                if target_window:
                    win_info_text = f"Target: {window_title}"
                    cv2.putText(image_display, win_info_text, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 255), 1)
            else:
                image_display = last_image_output.copy()
            
            # Show frame
            cv2.imshow(WINDOW_NAME, image_display)
            
            # Position window only once
            if not window_configured:
                # Allow some time for the window to be created
                time.sleep(0.2)
                position_window(WINDOW_NAME, preview_x, preview_y, preview_width, preview_height)
                window_configured = True
        elif not visualization_enabled and not window_configured:
            # If visualization is disabled from the start, still create a small window
            # to allow key capture
            dummy_frame = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, "Visualization disabled (press V to enable)", 
                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(WINDOW_NAME, dummy_frame)
            position_window(WINDOW_NAME, preview_x, preview_y, 400, 100)
            window_configured = True

        # Check for sensitivity adjustment using keyboard
        # Using GetAsyncKeyState for continuous checks
        if is_key_pressed(VK_PLUS) and not sens_up_pressed:
            mouse_sensitivity += 0.05
            sens_up_pressed = True
            print(f"Sensitivity increased to {mouse_sensitivity:.2f}")
        elif not is_key_pressed(VK_PLUS) and sens_up_pressed:
            sens_up_pressed = False
            
        if is_key_pressed(VK_MINUS) and not sens_down_pressed:
            mouse_sensitivity = max(0.05, mouse_sensitivity - 0.05)  # Prevent going below 0.05
            sens_down_pressed = True
            print(f"Sensitivity decreased to {mouse_sensitivity:.2f}")
        elif not is_key_pressed(VK_MINUS) and sens_down_pressed:
            sens_down_pressed = False

        # Key handling
        key = cv2.waitKey(1)
        
        # Check for toggle and quit commands
        if key == KEY_QUIT:
            break
        elif key == KEY_TOGGLE_PREDICTION:
            prediction_enabled = not prediction_enabled
            print(f"Prediction {'enabled' if prediction_enabled else 'disabled'}")
            # Reset controller when changing modes
            controller.prev_error = 0
            controller.integral = 0
        elif key == KEY_TOGGLE_PID:
            pid_enabled = not pid_enabled
            print(f"PID control {'enabled' if pid_enabled else 'disabled'}")
            # Reset controller when changing modes
            controller.prev_error = 0
            controller.integral = 0
        elif key == KEY_INCREASE_SENS:
            mouse_sensitivity += 0.05
            print(f"Sensitivity increased to {mouse_sensitivity:.2f}")
        elif key == KEY_DECREASE_SENS:
            mouse_sensitivity = max(0.05, mouse_sensitivity - 0.05)  # Prevent going below 0.05
            print(f"Sensitivity decreased to {mouse_sensitivity:.2f}")
        elif key == KEY_TOGGLE_VISUAL:
            visualization_enabled = not visualization_enabled
            print(f"Visualization {'enabled' if visualization_enabled else 'disabled'}")
        elif key == KEY_TOGGLE_FPS:
            fps_display_enabled = not fps_display_enabled
            print(f"FPS display {'enabled' if fps_display_enabled else 'disabled'}")
        elif key == KEY_TOGGLE_ACTIVE:
            # Toggle aimbot active state
            aimbot_active = not aimbot_active
            print(f"Aimbot {'enabled' if aimbot_active else 'disabled'}")
            pid_enabled = aimbot_active
        elif key == KEY_WINDOW_SELECT:
            # Toggle window selection mode
            print("Entering window selection mode...")
            
            # Stop the camera while selecting
            camera.stop()
            
            # Get available windows and let user select one
            windows = get_visible_windows()
            new_window = select_window(windows)
            
            if new_window:
                target_window = new_window
                window_rect = get_window_rect(target_window)
                window_title = win32gui.GetWindowText(target_window)
                print(f"Selected window: {window_title}")
                print(f"Window rectangle: {window_rect}")
                highlight_window(target_window)
                
                # Use window dimensions for capture region
                left, top, right, bottom = window_rect
                
                # Get window dimensions
                window_width = right - left
                window_height = bottom - top
                
                # Find the center point of the window
                center_x = left + window_width // 2
                center_y = top + window_height // 2
                
                # Create a 640x640 square region centered on the window
                capture_size = 640
                half_size = capture_size // 2
                
                # Calculate the square region
                region_left = center_x - half_size
                region_top = center_y - half_size
                region_right = center_x + half_size
                region_bottom = center_y + half_size
                
                # Create region tuple for DXcam
                region = (region_left, region_top, region_right, region_bottom)
                
                print(f"Using center of window: {window_title}")
                print(f"Window size: {window_width}x{window_height}")
                print(f"Capturing 640x640 region from center: {region}")
                
                # Highlight the capture region
                try:
                    # Create a transparent window that shows just the capture area
                    highlight_hwnd = win32gui.CreateWindowEx(
                        win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT,
                        "STATIC",
                        None,
                        win32con.WS_POPUP,
                        region_left, region_top, capture_size, capture_size,
                        None, None, None, None
                    )
                    
                    # Set border color to green
                    win32gui.SetLayeredWindowAttributes(
                        highlight_hwnd, 
                        0x00FF00,  # Green in BGR format
                        128,  # Alpha (0-255)
                        win32con.LWA_COLORKEY | win32con.LWA_ALPHA
                    )
                    
                    # Show the highlight
                    win32gui.ShowWindow(highlight_hwnd, win32con.SW_SHOW)
                    time.sleep(2.0)
                    
                    # Remove the highlight
                    win32gui.DestroyWindow(highlight_hwnd)
                except Exception as e:
                    print(f"Error highlighting capture region: {e}")
                
                # Restart camera with new region
                camera = dxcam.create(region=region, output_color="BGR")
                camera.start(target_fps=120, video_mode=True)  # Changed to video_mode=True for continuous capture
                
                # Give the camera a moment to initialize
                time.sleep(0.5)
            else:
                print("Window selection cancelled")
                # Restart camera with existing region
                camera = dxcam.create(region=region, output_color="BGR")
                camera.start(target_fps=120, video_mode=True)  # Changed to video_mode=True for continuous capture
                
                # Give the camera a moment to initialize
                time.sleep(0.5)
        elif key == KEY_DEBUG_REGION:
            debug_capture_region = not debug_capture_region
            print(f"Capture region debug {'enabled' if debug_capture_region else 'disabled'}")
        # Number keys 1-9 control process_every_n_frames for performance tuning
        elif key >= ord('1') and key <= ord('9'):
            process_every_n_frames = key - ord('0')  # Convert ASCII to number
            print(f"Processing every {process_every_n_frames} frame(s)")

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()

# Standard Python entry point
if __name__ == '__main__':
    main()
