import os
import time
import cv2
import dxcam
import win32api

# Configuration
CAPTURE_INTERVAL = 0.5  # seconds between captures
CAPTURE_COUNT = 500      # how many images to capture
OUTPUT_FOLDER = 'collected_images'
FOV_SIZE = 640          # window capture dimension

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Automatically detect primary monitor resolution
    screen_width = win32api.GetSystemMetrics(0)
    screen_height = win32api.GetSystemMetrics(1)

    # Center a 640×640 region on the detected screen
    left = (screen_width - FOV_SIZE) // 2
    top = (screen_height - FOV_SIZE) // 2
    right = left + FOV_SIZE
    bottom = top + FOV_SIZE
    region = (left, top, right, bottom)

    # Initialize camera
    camera = dxcam.create(region=region, output_color="BGR")
    camera.start(target_fps=0, video_mode=False)

    print(f"Capturing {CAPTURE_COUNT} images every {CAPTURE_INTERVAL}s...")
    for i in range(CAPTURE_COUNT):
        frame = camera.get_latest_frame()
        if frame is not None:
            file_path = os.path.join(OUTPUT_FOLDER, f'image_{i:03d}.jpg')
            cv2.imwrite(file_path, frame)
            print(f"Saved {file_path}")
        else:
            print("No frame captured...")

        time.sleep(CAPTURE_INTERVAL)

    camera.stop()
    print("Done capturing.")

if __name__ == "__main__":
    main()
