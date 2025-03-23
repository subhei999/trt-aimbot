import os
import shutil
import random

# 1) Input folders
IMAGES_DIR = "collected_images"
LABELS_DIR = "labels_yolo"

# 2) Output root (splits & data.yaml go inside here)
OUTPUT_ROOT = "dataset"

# 3) Split ratios (train, val, test)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 4) Class names for YOLO (one class: "head")
CLASSES = ["head"]

def main():
    # Gather all image files
    image_files = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    # Shuffle for random split
    random.shuffle(image_files)

    total_count = len(image_files)
    train_count = int(total_count * TRAIN_RATIO)
    val_count = int(total_count * VAL_RATIO)
    test_count = total_count - train_count - val_count

    print(f"Total images: {total_count}")
    print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")

    # Prepare output directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

    # Split subsets
    train_images = image_files[:train_count]
    val_images = image_files[train_count:train_count + val_count]
    test_images = image_files[train_count + val_count:]

    def copy_to_split(img_name, split):
        # Copy image
        src_image_path = os.path.join(IMAGES_DIR, img_name)
        dst_image_path = os.path.join(OUTPUT_ROOT, "images", split, img_name)
        shutil.copy2(src_image_path, dst_image_path)

        # Copy matching label (same base name, .txt extension)
        base_name, _ = os.path.splitext(img_name)
        label_name = base_name + ".txt"
        src_label_path = os.path.join(LABELS_DIR, label_name)
        if os.path.exists(src_label_path):
            dst_label_path = os.path.join(OUTPUT_ROOT, "labels", split, label_name)
            shutil.copy2(src_label_path, dst_label_path)

    # Copy each subset
    for img in train_images:
        copy_to_split(img, "train")
    for img in val_images:
        copy_to_split(img, "val")
    for img in test_images:
        copy_to_split(img, "test")

    # Auto-generate data.yaml
    data_yaml_path = os.path.join(OUTPUT_ROOT, "data.yaml")
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write("train: " + os.path.join(OUTPUT_ROOT, "images", "train") + "\n")
        f.write("val: " + os.path.join(OUTPUT_ROOT, "images", "val") + "\n")
        f.write("test: " + os.path.join(OUTPUT_ROOT, "images", "test") + "\n\n")
        f.write(f"nc: {len(CLASSES)}\n")
        f.write(f"names: {CLASSES}\n")

    print("Data split completed!")
    print(f"data.yaml created at: {data_yaml_path}")

if __name__ == "__main__":
    main()
