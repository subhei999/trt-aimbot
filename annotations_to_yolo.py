import os
import json
import glob
import cv2

# Directory containing your labeled JSON files (Labelme output)
JSON_DIR = "collected_images"
# Where to put the YOLO txt labels
OUTPUT_LABELS_DIR = "labels_yolo"
# The single class name. YOLO class index = 0
CLASS_NAME = "head"

def main():
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

    # Find all .json files in the JSON_DIR
    json_files = glob.glob(os.path.join(JSON_DIR, "*.json"))

    for json_file in json_files:
        # Read JSON
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # If you have the labeled image in the same folder:
        # data["imagePath"] is the file name of the image
        image_file = os.path.join(JSON_DIR, data["imagePath"])

        # Load the image to get width & height
        if not os.path.isfile(image_file):
            print(f"Warning: Image file {image_file} not found, skipping.")
            continue

        img = cv2.imread(image_file)
        if img is None:
            print(f"Warning: Unable to open {image_file}, skipping.")
            continue

        img_height, img_width = img.shape[:2]

        # Build YOLO label lines
        yolo_lines = []
        for shape in data["shapes"]:
            # shape["label"] should match your single class "head"
            # If you have multiple classes, handle that logic here
            if shape["label"] != CLASS_NAME:
                # If your dataset is truly single-class, you might skip or treat as class 0
                continue

            # shape could be polygon or rectangle; either way we find min/max x,y
            points = shape["points"]
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            # YOLO expects normalized centerX, centerY, width, height
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            center_x = x_min + bbox_width / 2
            center_y = y_min + bbox_height / 2

            # Normalize [0,1]
            center_x /= img_width
            center_y /= img_height
            bbox_width /= img_width
            bbox_height /= img_height

            # class_index x_center y_center width height
            yolo_line = f"0 {center_x} {center_y} {bbox_width} {bbox_height}\n"
            yolo_lines.append(yolo_line)

        # Write YOLO .txt annotation
        # Use the same base name as the original image/json, but with .txt
        base_name = os.path.splitext(os.path.basename(data["imagePath"]))[0]
        txt_path = os.path.join(OUTPUT_LABELS_DIR, f"{base_name}.txt")

        # If there are no shapes, we create an empty txt, meaning "no objects"
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in yolo_lines:
                f.write(line)

        print(f"Converted {json_file} -> {txt_path}")

if __name__ == "__main__":
    main()
