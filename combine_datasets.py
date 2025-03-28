import os
import shutil
import datetime
import glob
import re

# Define source directories
src_dir1 = "dataset/raw/20250324_185641"
src_dir2 = "dataset/raw/20250325_000257"

# Create timestamp in the same format
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")

# Define destination directory
dst_dir = f"dataset/raw/{timestamp}"

# Create destination directory and labels subdirectory
os.makedirs(dst_dir, exist_ok=True)
os.makedirs(os.path.join(dst_dir, "labels"), exist_ok=True)

# Get all image files from both directories
all_images = []
for src_dir in [src_dir1, src_dir2]:
    images = glob.glob(os.path.join(src_dir, "*.jpg"))
    all_images.extend([(img, src_dir) for img in images])

# Function to extract number from filename
def get_file_number(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

# Sort images by their source directory and then by number
all_images.sort(key=lambda x: (x[1], get_file_number(x[0])))

# Copy and rename images sequentially
for i, (img_path, src_dir) in enumerate(all_images):
    # Generate new sequential filename
    new_img_name = f"img_{i:06d}.jpg"
    new_img_path = os.path.join(dst_dir, new_img_name)
    
    # Copy image with new name
    shutil.copy2(img_path, new_img_path)
    
    # Get original filename without extension
    original_basename = os.path.splitext(os.path.basename(img_path))[0]
    
    # Copy corresponding label if it exists
    label_path = os.path.join(src_dir, "labels", f"{original_basename}.txt")
    if os.path.exists(label_path):
        new_label_name = f"img_{i:06d}.txt"
        new_label_path = os.path.join(dst_dir, "labels", new_label_name)
        shutil.copy2(label_path, new_label_path)

print(f"Successfully combined datasets into {dst_dir}")
print(f"All {len(all_images)} images and their labels have been preserved.")
print(f"This folder will be used as the most recent training data") 