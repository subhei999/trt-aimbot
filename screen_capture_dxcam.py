import dxcam
import time
from PIL import Image

# Create a DXCam camera instance
camera = dxcam.create()
left, top = (1920 - 640) // 2, (1080 - 640) // 2
right, bottom = left + 640, top + 640
region = (left, top, right, bottom)
# Number of frames to capture for latency measurement
num_frames = 1000

# Initialize a list to store latency values
latency_values = []

try:
    for _ in range(num_frames):
        # Record the timestamp before grabbing a frame
        start_time = time.perf_counter()

        # Grab a frame from the camera
        frame = camera.grab(region=region)
        #Image.fromarray(frame).show()
        # Record the timestamp immediately after grabbing the frame
        end_time = time.perf_counter()

        # Calculate and store the latency
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(latency)
        latency_values.append(latency)

except KeyboardInterrupt:
    pass  # Allow the user to stop the loop using Ctrl+C

finally:
    # Release the camera when done
    camera.release()

# Plot latency distribution as a histogram
import matplotlib.pyplot as plt

plt.hist(latency_values, bins=30, edgecolor='k')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.title('Latency Distribution (dxcam)')
plt.show()
