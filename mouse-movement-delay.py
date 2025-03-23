import ctypes
import time
import random
import matplotlib.pyplot as plt
from ctypes import windll, Structure, c_long, byref

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]



def queryMousePosition():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return { "x": pt.x, "y": pt.y}

# Number of random moves
num_moves = 1000

# Initialize a list to store the time taken for each move
move_times = []

for _ in range(num_moves):
    # Generate random relative pixel offsets (dx, dy) for each move
    dx = random.randint(-1000, 1000)  # Random value between -100 and 100
    dy = random.randint(-1000, 1000)  # Random value between -100 and 100

    # Get the current time before the mouse movement
    start_time = time.time()

    # Get the current mouse cursor position
    pos = queryMousePosition()
    # Calculate the new position
    new_x = pos['x'] + dx
    new_y = pos['y'] + dy

    # Use ctypes to call the Windows API function SetCursorPos to move the mouse
    ctypes.windll.user32.SetCursorPos(new_x, new_y)

    # Get the current time after the mouse movement
    end_time = time.time()

    # Calculate the elapsed time in milliseconds
    elapsed_time_ms = (end_time - start_time) * 1000

    move_times.append(elapsed_time_ms)

# Plot a histogram of the time taken for the random moves
plt.hist(move_times, bins=50, edgecolor='k')
plt.xlabel('Time (ms)')
plt.ylabel('Frequency')
plt.title(f'Histogram of {num_moves} Random Mouse Moves')
plt.grid(True)
plt.show()
