"""
Visualization utilities for rendering detection results and interface elements.
"""
import os
import cv2

def draw_model_info(image, model_path, fps, prediction_enabled=True, pid_enabled=True, 
                    mouse_sensitivity=0.5, aimbot_active=True):
    """Draw model information including precision on the image"""
    if image is None:
        return image
    
    # Check model precision
    precision = "FP16" if "fp16" in model_path.lower() else "FP32"
    
    # Draw text on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    info_text = f"Model: {os.path.basename(model_path)} ({precision})"
    fps_text = f"FPS: {fps:.1f}"
    controls_text = "Keys: Q-Quit, P-Prediction, I-PID, C-Toggle Aimbot"
    sens_text = f"Sensitivity: {mouse_sensitivity:.2f} (+/- to adjust)"
    
    # Mode indicators
    prediction_status = "ON" if prediction_enabled else "OFF"
    pid_status = "ON" if pid_enabled else "OFF"
    aimbot_status = "ON" if aimbot_active else "OFF"
    mode_text = f"Prediction: {prediction_status}  |  PID: {pid_status}  |  Aimbot: {aimbot_status}"
    
    # Background for better readability
    cv2.rectangle(image, (10, 10), (600, 120), (0, 0, 0), -1)
    
    # Add text
    cv2.putText(image, info_text, (20, 30), font, 0.6, (0, 255, 0), 2)
    cv2.putText(image, fps_text, (20, 50), font, 0.6, (0, 255, 0), 2)
    cv2.putText(image, controls_text, (20, 70), font, 0.5, (0, 255, 255), 1)
    cv2.putText(image, mode_text, (20, 90), font, 0.5, (0, 255, 255), 1)
    cv2.putText(image, sens_text, (20, 110), font, 0.5, (0, 255, 255), 1)
    
    return image 