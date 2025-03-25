"""
Visualization utilities for rendering detection results and interface elements.
"""
import os
import cv2
import numpy as np

def draw_circular_mask(image, center_x, center_y, radius):
    """
    Draw a circular mask on the image to visualize the target exclusion zone.
    """
    if image is None:
        return image
    
    # Create a copy of the image to avoid modifying the original
    output = image.copy()
    
    # Draw the circle with a semi-transparent overlay
    overlay = output.copy()
    cv2.circle(overlay, (center_x, center_y), radius, (0, 255, 255), 2)  # Yellow circle
    
    # Add some transparency
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    # Add a label for the mask
    cv2.putText(output, f"Mask: {radius}px", (center_x - 50, center_y + radius + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    return output

def draw_model_info(image, model_path, fps, prediction_enabled=True, pid_enabled=True, 
                    mouse_sensitivity=0.5, aimbot_active=True, mask_enabled=False, mask_radius=240,
                    vertical_offset=0.0, horizontal_offset=0.0, adjust_vertical=True):
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
    
    # Format offset values as percentages
    v_offset_pct = vertical_offset * 100
    h_offset_pct = horizontal_offset * 100
    
    # Mode indicators
    prediction_status = "ON" if prediction_enabled else "OFF"
    pid_status = "ON" if pid_enabled else "OFF"
    aimbot_status = "ON" if aimbot_active else "OFF"
    mask_status = f"Mask: {mask_radius}px" if mask_enabled else "Mask: OFF"
    
    # Show which offset is being adjusted
    offset_adjust_mode = "Vertical" if adjust_vertical else "Horizontal"
    
    mode_text = f"Prediction: {prediction_status}  |  PID: {pid_status}  |  Aimbot: {aimbot_status}"
    mask_text = f"{mask_status} (M to toggle, ./,  to adjust)"
    
    # Create offset text with direction indicators for easier understanding
    v_direction = "↑" if vertical_offset > 0 else "↓" if vertical_offset < 0 else "-"
    h_direction = "→" if horizontal_offset > 0 else "←" if horizontal_offset < 0 else "-"
    
    offset_text = f"Aim Offset: V:{v_direction}{abs(v_offset_pct):.1f}% | H:{h_direction}{abs(h_offset_pct):.1f}%"
    offset_help = f"Adjusting: {offset_adjust_mode} (O to toggle, =/- to adjust)"
    
    # Background for better readability - make it larger to fit new info
    cv2.rectangle(image, (10, 10), (600, 170), (0, 0, 0), -1)
    
    # Add text
    cv2.putText(image, info_text, (20, 30), font, 0.6, (0, 255, 0), 2)
    cv2.putText(image, fps_text, (20, 50), font, 0.6, (0, 255, 0), 2)
    cv2.putText(image, controls_text, (20, 70), font, 0.5, (0, 255, 255), 1)
    cv2.putText(image, mode_text, (20, 90), font, 0.5, (0, 255, 255), 1)
    cv2.putText(image, sens_text, (20, 110), font, 0.5, (0, 255, 255), 1)
    cv2.putText(image, mask_text, (20, 130), font, 0.5, (0, 255, 255), 1)
    
    # Add offset information
    # Use magenta color to match the aim point color
    cv2.putText(image, offset_text, (20, 150), font, 0.5, (255, 0, 255), 1)
    cv2.putText(image, offset_help, (20, 170), font, 0.5, (255, 0, 255), 1)
    
    return image 