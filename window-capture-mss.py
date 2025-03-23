import cv2
import mss
import numpy as np
import time
from PIL import Image

#latency over 100 ms
# Define the region to capture (adjust the coordinates and size accordingly)
capture_region = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}

# Initialize variables for measuring latency
prev_time = time.time()

with mss.mss() as sct:
    while True:
        # Capture the entire screen
        screenshot = sct.shot()
        # Calculate and print the latency
        curr_time = time.time()
        latency = curr_time - prev_time
        prev_time = curr_time
        print(f'Latency: {latency * 1000:.2f} ms')

        # You can manipulate the frame in memory here as needed
        # Example: Apply image processing or analysis

        # Break the loop on a key press (e.g., press 'q' to quit)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close the OpenCV window
cv2.destroyAllWindows()
