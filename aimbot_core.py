import os
import cv2
import math
import time
import pathlib
import win32api
import win32gui
import win32con
import numpy as np
import dxcam

from ultralytics import YOLO

# Import modules
from constants import *
import modules.utils as utils
from modules.input_utils import mouse_move, is_key_pressed
from modules.window_utils import (get_visible_windows, select_window, 
                                highlight_window, get_window_rect, 
                                position_window, capture_window_screenshot)
from modules.controllers import PIDController, TargetPredictor
from modules.detection import detect_targets
from modules.visualization import draw_model_info

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
            highlight_hwnd = win32api.CreateWindowEx(
                win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT,
                "STATIC",
                None,
                win32con.WS_POPUP,
                region_left, region_top, capture_size, capture_size,
                None, None, None, None
            )
            
            # Set border color to green
            win32api.SetLayeredWindowAttributes(
                highlight_hwnd, 
                0x00FF00,  # Green in BGR format
                128,  # Alpha (0-255)
                win32con.LWA_COLORKEY | win32con.LWA_ALPHA
            )
            
            # Show the highlight
            win32api.ShowWindow(highlight_hwnd, win32con.SW_SHOW)
            time.sleep(2.0)
            
            # Remove the highlight
            win32api.DestroyWindow(highlight_hwnd)
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
    mouse_sensitivity = 0.45  # Starting value (can be adjusted with + and - keys)
    
    # Target persistence settings
    persistent_target_enabled = True  # Toggle with 't' key
    max_target_dropouts = 10         # Maximum consecutive frames to maintain target without detection
    target_switch_threshold = 100    # Pixel distance threshold for considering a new target
    current_target_id = None         # ID for the current target (None = no target)
    target_dropout_counter = 0       # Count of consecutive frames with target lost
    last_target_position = None      # Last known position of the target
    
    # Mode flags
    prediction_enabled = False
    pid_enabled = False
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

            # Handle target persistence
            closest_human_center = None
            all_targets = []
            
            # Process all detected targets
            if detection_output:
                # Extract all potential targets with their positions
                for i, (cls_id, conf_val, bbox) in enumerate(detection_output):
                    if cls_id == 0:  # Person class
                        # Calculate center of the bounding box
                        x1, y1, x2, y2 = bbox
                        target_center_x = (x1 + x2) / 2
                        target_center_y = (y1 + y2) / 2
                        
                        # Calculate distance from screen center
                        dist_from_center = math.sqrt((target_center_x - center_x)**2 + (target_center_y - center_y)**2)
                        
                        # Add to targets list with unique ID
                        all_targets.append({
                            'id': i,
                            'position': (target_center_x, target_center_y),
                            'confidence': conf_val,
                            'bbox': bbox,
                            'distance': dist_from_center
                        })
            
            # Apply target persistence logic
            if persistent_target_enabled and all_targets:
                if current_target_id is None:
                    # No current target, select the closest one to center
                    all_targets.sort(key=lambda x: x['distance'])
                    closest_target = all_targets[0]
                    closest_human_center = closest_target['position']
                    current_target_id = closest_target['id']
                    last_target_position = closest_human_center
                    target_dropout_counter = 0
                    
                    # Debug output for target acquisition
                    print(f"Acquired new target ID: {current_target_id} at position {closest_human_center}")
                else:
                    # We have a current target, try to find it or the closest match
                    if last_target_position:
                        # Find the closest target to our last known position
                        min_dist = float('inf')
                        best_match = None
                        
                        for target in all_targets:
                            tx, ty = target['position']
                            lx, ly = last_target_position
                            
                            # Calculate distance to last known position
                            dist_to_last = math.sqrt((tx - lx)**2 + (ty - ly)**2)
                            
                            if dist_to_last < min_dist:
                                min_dist = dist_to_last
                                best_match = target
                        
                        # Check if the closest target is within our threshold
                        if min_dist < target_switch_threshold:
                            # Target found or close enough match
                            closest_human_center = best_match['position']
                            current_target_id = best_match['id']
                            last_target_position = closest_human_center
                            target_dropout_counter = 0
                            
                            # Debug output for target tracking
                            print(f"Tracking target ID: {current_target_id}, distance from last: {min_dist:.2f}px")
                        else:
                            # No close match found, increment dropout counter
                            target_dropout_counter += 1
                            
                            # Use the last known position for a few frames
                            if target_dropout_counter <= max_target_dropouts:
                                closest_human_center = last_target_position
                                print(f"Target lost, using last position. Dropout counter: {target_dropout_counter}/{max_target_dropouts}")
                            else:
                                # We've exceeded our dropout threshold, acquire new target
                                all_targets.sort(key=lambda x: x['distance'])
                                closest_target = all_targets[0]
                                closest_human_center = closest_target['position']
                                current_target_id = closest_target['id']
                                last_target_position = closest_human_center
                                target_dropout_counter = 0
                                
                                print(f"Target dropout limit reached, switching to new target ID: {current_target_id}")
            elif persistent_target_enabled and not all_targets:
                # No targets detected at all
                if current_target_id is not None:
                    # We had a target but lost it, increment dropout counter
                    target_dropout_counter += 1
                    
                    # Use the last known position for a few frames
                    if target_dropout_counter <= max_target_dropouts and last_target_position:
                        closest_human_center = last_target_position
                        print(f"All targets lost, using last position. Dropout counter: {target_dropout_counter}/{max_target_dropouts}")
                    else:
                        # We've exceeded our dropout threshold, clear targeting
                        current_target_id = None
                        last_target_position = None
                        print("Target dropout limit reached, clearing target")
            else:
                # Persistent targeting disabled, use standard closest target logic
                # Draw bounding boxes and get the center of the closest human
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
            
            # Handle visualization with persistent targeting
            if persistent_target_enabled and should_visualize:
                # Create a copy of the frame for drawing
                image_output = frame.copy()
                
                # Draw all detected boxes
                for target in all_targets:
                    x1, y1, x2, y2 = target['bbox']
                    conf_val = target['confidence']
                    
                    # Get a class name (assuming 0 = person)
                    class_name = label_map.get(0, "person")
                    
                    # Determine color based on whether this is our current target
                    if target['id'] == current_target_id:
                        # Current target - use red
                        color = (0, 0, 255)
                        thickness = 2
                    else:
                        # Other targets - use blue
                        color = (255, 0, 0)
                        thickness = 1
                    
                    # Draw the bounding box
                    cv2.rectangle(image_output, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    
                    # Draw the label
                    label = f"{class_name} {conf_val:.2f}"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(image_output, (int(x1), int(y1) - 20), (int(x1) + text_size[0], int(y1)), color, -1)
                    cv2.putText(image_output, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw a special indicator for the current target
                if closest_human_center:
                    cv2.circle(image_output, (int(closest_human_center[0]), int(closest_human_center[1])), 5, (0, 255, 0), -1)
                    
                    # Draw target ID and dropout counter if using persistence
                    status_text = f"Target ID: {current_target_id}, Dropouts: {target_dropout_counter}/{max_target_dropouts}"
                    cv2.putText(image_output, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw crosshair at center
                cv2.line(image_output, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 255), 1)
                cv2.line(image_output, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 255), 1)
                
                last_image_output = image_output
            
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
                    aimbot_active,
                    persistent_target_enabled
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
            controller.reset()
        elif key == KEY_TOGGLE_PID:
            pid_enabled = not pid_enabled
            print(f"PID control {'enabled' if pid_enabled else 'disabled'}")
            # Reset controller when changing modes
            controller.reset()
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
        elif key == ord('t'):  # Add toggle for target persistence
            persistent_target_enabled = not persistent_target_enabled
            # Reset targeting data when toggling
            current_target_id = None
            last_target_position = None
            target_dropout_counter = 0
            print(f"Target persistence {'enabled' if persistent_target_enabled else 'disabled'}")
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
                    highlight_hwnd = win32api.CreateWindowEx(
                        win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT,
                        "STATIC",
                        None,
                        win32con.WS_POPUP,
                        region_left, region_top, capture_size, capture_size,
                        None, None, None, None
                    )
                    
                    # Set border color to green
                    win32api.SetLayeredWindowAttributes(
                        highlight_hwnd, 
                        0x00FF00,  # Green in BGR format
                        128,  # Alpha (0-255)
                        win32con.LWA_COLORKEY | win32con.LWA_ALPHA
                    )
                    
                    # Show the highlight
                    win32api.ShowWindow(highlight_hwnd, win32con.SW_SHOW)
                    time.sleep(2.0)
                    
                    # Remove the highlight
                    win32api.DestroyWindow(highlight_hwnd)
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
