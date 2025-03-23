import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

# Specify the full path to the FFmpeg executable
ffmpeg_executable = r'ffmpeg'  # Replace with the actual path

# Replace "Your Window Title" with the title of your windowed application
window_title = "Battle.net"

# Number of frames to capture for latency measurement
num_frames = 100

# Initialize a list to store latency values
latency_values = []

try:
    for _ in range(num_frames):
        # Record the timestamp before capturing a frame
        start_time = time.time()

        # Capture a frame using FFmpeg
        subprocess.run([
            ffmpeg_executable,
            "-f", "gdigrab",
            "-framerate", "60",  # Adjust frame rate as needed
            "-i", f"title={window_title}",
            "-vf", "fps=60,scale=1920:1080",  # Adjust video filters as needed
            "-c:v", "libx264", "-crf", "0",
            "output.mp4"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Record the timestamp after capturing the frame
        end_time = time.time()

        # Calculate and store the latency
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        latency_values.append(latency)

except KeyboardInterrupt:
    pass  # Allow the user to stop the loop using Ctrl+C

# Plot latency distribution as a histogram
plt.hist(latency_values, bins=30, edgecolor='k')
plt.xlabel('Latency (ms)')
plt.ylabel('Frequency')
plt.title('Latency Distribution (FFmpeg)')
plt.show()
