import cv2
import os
import torch
import numpy as np

from ultralytics.utils import ops
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results

def get_text_color(box_color):
    text_color = (255,255,255)

    brightness = box_color[2]*0.299 + box_color[1]*0.587 + box_color[0]*0.114

    if(brightness > 180):
        text_color = (0, 0, 0)

    return text_color

def draw_box(
    image, 
    detection_output, 
    label_map, 
    color_list, 
    center_x=None, 
    center_y=None, 
    draw_boxes=True
):
    """
    Draw the detected boxes, showing the nearest to crosshair.
    If draw_boxes=False, only calculates the closest center without drawing anything.
    """
    # Create a copy of the image for drawing
    if draw_boxes:
        image_output = image.copy()
    else:
        image_output = image  # Just use the original reference, won't modify it
        
    # Whether to show center crosshair
    show_center = (center_x is not None and center_y is not None)

    # Calculate closest center to crosshair
    closest_center = None
    closest_distance = float('inf')

    # For each object detected
    for idx, (cls, conf, box) in enumerate(detection_output):
        cls_name = label_map[cls]

        # Get box coordinates
        xb1, yb1, xb2, yb2 = box.astype(int)
        
        # Calculate center of the box
        center = (int((xb1 + xb2) / 2), int((yb1 + yb2) / 2))
        
        # If center is provided, calculate the distance to this box
        if show_center:
            distance = ((center[0] - center_x) ** 2 + (center[1] - center_y) ** 2) ** 0.5
            
            # Check if this is the closest human to center
            if distance < closest_distance and cls_name.lower() == "head":
                closest_distance = distance
                closest_center = center

        # Draw boxes and labels only if requested (for performance)
        if draw_boxes:
            # Draw box
            cv2.rectangle(image_output, (xb1, yb1), (xb2, yb2), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(image_output, center, 5, (0, 0, 255), -1)
            
            # Draw class name and confidence
            text = f"{cls_name}: {conf:.2f}"
            cv2.putText(image_output, text, (xb1, yb1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw center crosshair if center is provided and drawing is enabled
    if show_center and draw_boxes:
        cv2.line(image_output, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
        cv2.line(image_output, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)
        
        # Draw line to closest human
        if closest_center is not None:
            cv2.line(image_output, (center_x, center_y), closest_center, (255, 0, 0), 2)

    return image_output, closest_center

def draw_fps(avg_fps, combined_img):        
    avg_fps_str = float("{:.2f}".format(avg_fps))
    
    cv2.rectangle(combined_img, (10,2), (660,110), (255,255,255), -1)
    cv2.putText(combined_img, "FPS: "+str(avg_fps_str), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0,255,0), thickness=6)

    return combined_img

def draw_time(time, combined_img):        
    time_str = "{:.3f}".format(time)
    
    # Draw the time below the FPS
    cv2.putText(combined_img, "Time: " + time_str, (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    return combined_img


def get_name(source_path):    
    name_idx = 0
    file_pos = (source_path).rfind('\\')

    if(file_pos == -1):
        file_pos = (source_path).rfind('/')

        if(file_pos == -1):
            file_pos = 0
    
    name_idx = file_pos + 1

    name = source_path[name_idx:]

    return name

def get_save_path(file_name, folder_name):
    path = "result"
    save_path = os.path.join(path, folder_name)

    exists = os.path.exists(save_path) 

    if(not exists):
        os.makedirs(save_path)

    save_path = os.path.join(save_path, file_name)    

    return save_path

# def preprocess(img):
#     # Ensure that img is a valid NumPy array or PyTorch tensor
#     if isinstance(img, np.ndarray):
#         # Convert NumPy array to PyTorch tensor
#         img = torch.from_numpy(img).to(torch.device('cuda:0'))
#     elif not isinstance(img, torch.Tensor):
#         raise ValueError("Input 'img' should be a NumPy array or a PyTorch tensor.")

#     # Perform preprocessing
#     img = img.half()  # uint8 to fp16/32
#     img /= 255.0  # Normalize from 0-255 to 0.0-1.0

#     if len(img.shape) == 3:
#         img = img[None]  # Expand for batch dimension if not present
#     #img = img.permute(0, 3, 1, 2)  # Transpose dimensions to (batch, channels, height, width)
#     return img

def preprocess(img):   
    # LetterBox
    im = LetterBox((640, 640), False)(image=img)
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    img = torch.from_numpy(im).to(torch.device('cuda:0'))
    img = img.half()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0

    if len(img.shape) == 3:
        img = img[None]  # expand for batch dim

    return img

def postprocess(preds, img, orig_img, names, source_path):
    preds = ops.non_max_suppression(preds,
                                    0.5,
                                    0.7,
                                    agnostic=False,
                                    max_det=300,
                                    classes=None)

    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
        shape = orig_img.shape
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        path = source_path
        img_path = path[i] if isinstance(path, list) else path
        results.append(Results(orig_img=orig_img, path=img_path, names=names, boxes=pred))

    return results

# def letterbox(img):
    

#     return im

