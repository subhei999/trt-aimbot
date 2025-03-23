import cv2
import os
import numpy as np
from pathlib import Path
import argparse
import tkinter as tk
from tkinter import filedialog
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleLabeler:
    def __init__(self, image_dir, label_dir, class_name="head"):
        try:
            self.image_dir = Path(image_dir)
            self.label_dir = Path(label_dir)
            self.class_name = class_name
            self.class_id = 0  # Since we only have one class
            
            logger.info(f"Initializing labeler with image_dir: {image_dir}")
            logger.info(f"Label directory: {label_dir}")
            
            # Create label directory if it doesn't exist
            self.label_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all image files
            self.image_files = sorted([f for f in self.image_dir.glob("*") 
                                     if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            
            if not self.image_files:
                raise ValueError(f"No images found in {image_dir}")
            
            logger.info(f"Found {len(self.image_files)} images")
            
            self.current_idx = 0
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.current_boxes = []
            
            # Create window with WINDOW_NORMAL flag and set it to full screen
            cv2.namedWindow('Image Labeler', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Image Labeler', self.mouse_callback)
            
            # Get screen resolution
            screen_width = 1920  # Default to common resolution
            screen_height = 1080
            
            # Load first image to get its dimensions
            first_img = cv2.imread(str(self.image_files[0]))
            if first_img is None:
                raise ValueError(f"Failed to load first image: {self.image_files[0]}")
                
            img_height, img_width = first_img.shape[:2]
            
            # Calculate scale to fit image on screen while maintaining aspect ratio
            scale = min(screen_width/img_width, screen_height/img_height) * 0.9  # 90% of screen
            window_width = int(img_width * scale)
            window_height = int(img_height * scale)
            
            # Set window size
            cv2.resizeWindow('Image Labeler', window_width, window_height)
            
            # Show first image
            self.show_current_image()
            
            print("\nControls:")
            print("Left click + drag: Draw bounding box")
            print("Right click: Delete last box")
            print("C: Clear all boxes for current image")
            print("A/D: Previous/Next image")
            print("O: Open new directory")
            print("Q: Quit")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Quit
                    self.save_annotations()  # Save before quitting
                    break
                elif key == ord('a'):  # Previous image
                    self.save_annotations()  # Save current annotations
                    self.current_idx = (self.current_idx - 1) % len(self.image_files)
                    self.current_boxes = []  # Clear boxes for new image
                    self.load_annotations()  # Load annotations for new image
                    self.show_current_image()
                elif key == ord('d'):  # Next image
                    self.save_annotations()  # Save current annotations
                    self.current_idx = (self.current_idx + 1) % len(self.image_files)
                    self.current_boxes = []  # Clear boxes for new image
                    self.load_annotations()  # Load annotations for new image
                    self.show_current_image()
                elif key == ord('c'):  # Clear all boxes
                    self.current_boxes = []
                    self.show_current_image()
                    print("Cleared all boxes for current image")
                elif key == ord('o'):  # Open new directory
                    self.open_new_directory()
            
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Error in SimpleLabeler initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def open_new_directory(self):
        """Open a new directory using tkinter file dialog"""
        try:
            # Create and hide the root window
            root = tk.Tk()
            root.withdraw()
            
            # Open directory dialog
            new_dir = filedialog.askdirectory(
                title="Select Directory with Images",
                initialdir=str(self.image_dir.parent)
            )
            
            if new_dir:
                new_dir = Path(new_dir)
                logger.info(f"Selected new directory: {new_dir}")
                
                # Check if directory contains images
                image_files = sorted([f for f in new_dir.glob("*") 
                                    if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                
                if not image_files:
                    logger.warning(f"No images found in {new_dir}")
                    print(f"No images found in {new_dir}")
                    return
                
                # Save current annotations before switching
                self.save_annotations()
                
                # Update directories
                self.image_dir = new_dir
                self.label_dir = new_dir / "labels"
                self.label_dir.mkdir(parents=True, exist_ok=True)
                
                # Update image files and reset state
                self.image_files = image_files
                self.current_idx = 0
                self.current_boxes = []
                
                # Load annotations for first image
                self.load_annotations()
                
                # Update display
                self.show_current_image()
                print(f"Switched to directory: {new_dir}")
                
        except Exception as e:
            logger.error(f"Error opening new directory: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"Error opening new directory: {str(e)}")
    
    def load_annotations(self):
        """Load existing annotations for current image"""
        try:
            current_label_file = self.get_label_file(self.image_files[self.current_idx])
            if current_label_file.exists():
                with open(current_label_file, 'r') as f:
                    for line in f:
                        try:
                            class_id, x_center, y_center, width, height = map(float, line.strip().split())
                            # Convert YOLO format to pixel coordinates
                            img = cv2.imread(str(self.image_files[self.current_idx]))
                            if img is None:
                                logger.error(f"Failed to load image: {self.image_files[self.current_idx]}")
                                continue
                                
                            h, w = img.shape[:2]
                            x1 = int((x_center - width/2) * w)
                            y1 = int((y_center - height/2) * h)
                            x2 = int((x_center + width/2) * w)
                            y2 = int((y_center + height/2) * h)
                            self.current_boxes.append((x1, y1, x2, y2))
                        except ValueError as e:
                            logger.error(f"Error parsing annotation line: {line.strip()}")
                            logger.error(f"Error details: {str(e)}")
                            continue
        except Exception as e:
            logger.error(f"Error loading annotations: {str(e)}")
            logger.error(traceback.format_exc())
    
    def get_label_file(self, image_file):
        """Get corresponding label file path"""
        return self.label_dir / f"{image_file.stem}.txt"
    
    def mouse_callback(self, event, x, y, flags, param):
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.end_point = (x, y)
                    self.show_current_image()
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                if self.start_point and self.end_point:
                    # Ensure coordinates are in correct order
                    x1, x2 = sorted([self.start_point[0], self.end_point[0]])
                    y1, y2 = sorted([self.start_point[1], self.end_point[1]])
                    self.current_boxes.append((x1, y1, x2, y2))
                    self.start_point = None
                    self.end_point = None
                    self.show_current_image()
                    self.save_annotations()  # Autosave after adding a box
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.current_boxes:
                    self.current_boxes.pop()
                    self.show_current_image()
                    self.save_annotations()  # Autosave after deleting a box
        except Exception as e:
            logger.error(f"Error in mouse callback: {str(e)}")
            logger.error(traceback.format_exc())
    
    def show_current_image(self):
        """Display current image with boxes"""
        try:
            img = cv2.imread(str(self.image_files[self.current_idx]))
            if img is None:
                logger.error(f"Failed to load image: {self.image_files[self.current_idx]}")
                return
                
            display_img = img.copy()
            
            # Draw existing boxes
            for box in self.current_boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw current box being drawn
            if self.drawing and self.start_point and self.end_point:
                cv2.rectangle(display_img, self.start_point, self.end_point, (0, 0, 255), 2)
            
            # Add image counter and box count
            counter_text = f"Image {self.current_idx + 1}/{len(self.image_files)}"
            box_text = f"Boxes: {len(self.current_boxes)}"
            cv2.putText(display_img, counter_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, box_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Image Labeler', display_img)
        except Exception as e:
            logger.error(f"Error showing current image: {str(e)}")
            logger.error(traceback.format_exc())
    
    def save_annotations(self):
        """Save current annotations in YOLO format"""
        try:
            img = cv2.imread(str(self.image_files[self.current_idx]))
            if img is None:
                logger.error(f"Failed to load image for saving annotations: {self.image_files[self.current_idx]}")
                return
                
            h, w = img.shape[:2]
            
            label_file = self.get_label_file(self.image_files[self.current_idx])
            with open(label_file, 'w') as f:
                for box in self.current_boxes:
                    x1, y1, x2, y2 = box
                    # Convert to YOLO format (normalized coordinates)
                    x_center = ((x1 + x2) / 2) / w
                    y_center = ((y1 + y2) / 2) / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    f.write(f"{self.class_id} {x_center} {y_center} {width} {height}\n")
        except Exception as e:
            logger.error(f"Error saving annotations: {str(e)}")
            logger.error(traceback.format_exc())

def main():
    try:
        parser = argparse.ArgumentParser(description='Simple YOLO format image labeler')
        parser.add_argument('--dir', type=str, help='Custom directory containing images to label')
        parser.add_argument('--class-name', type=str, default='head', help='Class name (default: head)')
        args = parser.parse_args()
        
        if args.dir:
            # Use custom directory
            image_dir = Path(args.dir)
            label_dir = image_dir / "labels"
            print(f"Using custom directory: {image_dir}")
        else:
            # Use most recent session
            dataset_dir = Path("dataset")
            raw_dir = dataset_dir / "raw"
            
            # Find the most recent session folder
            session_folders = sorted(raw_dir.glob("*"), reverse=True)
            if not session_folders:
                print("No session folders found. Please run 1_collect_data.bat first or use --dir to specify a custom directory.")
                return
            
            latest_session = session_folders[0]
            image_dir = latest_session
            label_dir = latest_session / "labels"
            print(f"Using session folder: {latest_session.name}")
        
        # Create labels directory if it doesn't exist
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Start the labeler
        labeler = SimpleLabeler(
            image_dir=image_dir,
            label_dir=label_dir,
            class_name=args.class_name
        )
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Error: {str(e)}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main() 