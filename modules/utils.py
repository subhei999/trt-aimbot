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

def draw_box(img, detection_output, class_list, colors, center_x, center_y):
    # Copy image, in case that we need the original image for something
    out_image = img

    closest_distance = float('inf')
    closest_human_box = None
    closest_human_center = None  # To store the center of the closest human

    for run_output in detection_output:
        # Unpack
        label, con, box = run_output

        # Choose color
        # Blue color in BGR
        box_color = (255, 0, 0)
        # box_color = colors[int(label.item())]
        # text_color = (255,255,255)
        text_color = get_text_color(box_color)
        # Get Class Name
        label = class_list[int(label.item())]
        # Draw object box
        first_half_box = (int(box[0].item()), int(box[1].item()))
        second_half_box = (int(box[2].item()), int(box[3].item()))
        cv2.rectangle(out_image, first_half_box, second_half_box, box_color, 2)
        # Create text
        text_print = '{label} {con:.2f}'.format(label=label, con=con.item())
        # Locate text position
        text_location = (int(box[0]), int(box[1] - 10))
        # Get size and baseline
        labelSize, baseLine = cv2.getTextSize(text_print, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)

        # Draw text's background
        cv2.rectangle(out_image,
                      (int(box[0]), int(box[1] - labelSize[1] - 10)),
                      (int(box[0]) + labelSize[0], int(box[1] + baseLine - 10)),
                      box_color, cv2.FILLED)
        # Put text
        cv2.putText(out_image, text_print, text_location,
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    text_color, 2, cv2.LINE_AA)

        if label == 'head':  # Check if the label is 'person'
            # Calculate the center of the bounding box
            box_center_x = (box[0] + box[2]) / 2
            box_center_y = (box[1] + box[3]) / 2

            # Calculate the distance from the center of the frame
            distance = ((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2) ** 0.5

            # Check if this is the closest human so far
            if distance < closest_distance:
                closest_distance = distance
                closest_human_box = box
                closest_human_center = (box_center_x, box_center_y)

    if closest_human_box is not None:
        # Draw a small box at the center of the closest human bounding box
        center_box_color = (0, 255, 0)  # Green
        center_box_thickness = 2
        center_x, center_y = closest_human_center  # Use the center of the closest human
        cv2.rectangle(out_image,
                      (int(center_x) - 5, int(center_y) - 5),  # Top-left corner of the small box
                      (int(center_x) + 5, int(center_y) + 5),  # Bottom-right corner of the small box
                      center_box_color, center_box_thickness)

    return out_image, closest_human_center



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

