import os
import glob

# Path to the combined dataset
dataset_path = "dataset/raw/20250327_000216"

# Count images
image_files = glob.glob(os.path.join(dataset_path, "*.jpg"))
print(f"Number of image files: {len(image_files)}")

# Count labels
label_files = glob.glob(os.path.join(dataset_path, "labels", "*.txt"))
print(f"Number of label files: {len(label_files)}")

# List some of the highest numbered images
highest_images = sorted(image_files, reverse=True)[:5]
print("\nHighest numbered images:")
for img in highest_images:
    print(f" - {os.path.basename(img)}")

# List some of the highest numbered labels
highest_labels = sorted(label_files, reverse=True)[:5]
print("\nHighest numbered labels:")
for lbl in highest_labels:
    print(f" - {os.path.basename(lbl)}") 