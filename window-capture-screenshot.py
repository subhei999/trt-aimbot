import time
import cv2
import pygetwindow as gw
import pyautogui
import numpy as np


#measured latency = 60-70 ms / frame
# Define the window title of the application you want to record
window_title = "Steam"

# Get the window object
window = gw.getWindowsWithTitle(window_title)[0]

# Define the output video file
output_filename = "output.mp4"

# Create a VideoWriter object to save the frames as a video
frame_width = int(window.width)
frame_height = int(window.height)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))

# Initialize variables for measuring latency
start_time = time.time()
frame_count = 0

try:
    while True:
        # Capture the screen content of the specified window
        screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        # Calculate and print the latency
        end_time = time.time()
        latency = end_time - start_time
        # Write the frame to the output video
        out.write(frame)
        frame_count += 1


        print(f"Frame {frame_count}, Latency: {latency*1000:.2f} ms")

        # Update the start time for the next frame
        start_time = end_time

except KeyboardInterrupt:
    # User interrupted the recording
    pass

# Release the VideoWriter and close the window
out.release()
cv2.destroyAllWindows()
