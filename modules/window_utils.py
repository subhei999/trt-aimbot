"""
Window management utilities for handling game window selection and capture.
"""
import time
import ctypes
import win32gui
import win32con
import win32ui
import numpy as np
import cv2

def enum_windows_callback(hwnd, windows):
    """Callback function for enumerating windows."""
    if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
        windows.append((hwnd, win32gui.GetWindowText(hwnd)))
    return True

def get_visible_windows():
    """Get a list of all visible windows with titles."""
    windows = []
    win32gui.EnumWindows(enum_windows_callback, windows)
    # Sort windows by title for easier selection
    return sorted(windows, key=lambda x: x[1].lower())

def get_window_rect(hwnd):
    """Get the rectangle of a window."""
    rect = win32gui.GetWindowRect(hwnd)
    # rect is (left, top, right, bottom)
    return rect

def select_window(windows):
    """Display a list of windows and let user select one."""
    print("\nAvailable windows:")
    for i, (_, title) in enumerate(windows):
        print(f"{i}: {title}")
    
    while True:
        try:
            selection = input("\nEnter window number (or part of title to search): ")
            # Try to parse as a number first
            try:
                index = int(selection)
                if 0 <= index < len(windows):
                    return windows[index][0]  # Return the window handle
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                # Try to find by partial title match
                matches = [(i, title) for i, (_, title) in enumerate(windows) 
                           if selection.lower() in title.lower()]
                if len(matches) == 1:
                    return windows[matches[0][0]][0]  # Return the window handle
                elif len(matches) > 1:
                    print("\nMultiple matches found:")
                    for i, title in matches:
                        print(f"{i}: {title}")
                    continue
                else:
                    print("No matches found. Try again.")
        except KeyboardInterrupt:
            print("\nSelection cancelled.")
            return None

def highlight_window(hwnd, duration=2.0):
    """Highlight the selected window by drawing a red border around it."""
    if not hwnd:
        return
    
    # Save original window position and style
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    rect = win32gui.GetWindowRect(hwnd)
    
    # Draw a border by creating a borderless, topmost window
    border_width = 5
    border_color = 0x0000FF  # Red in BGR format
    
    # Create windows for each side of the border
    borders = []
    for i in range(4):
        border_hwnd = win32gui.CreateWindowEx(
            win32con.WS_EX_TOPMOST | win32con.WS_EX_LAYERED,
            "STATIC",
            None,
            win32con.WS_POPUP,
            0, 0, 0, 0,
            None, None, None, None
        )
        borders.append(border_hwnd)
        
        # Set border color
        win32gui.SetLayeredWindowAttributes(
            border_hwnd, 
            border_color, 
            255,  # Alpha (0-255)
            win32con.LWA_COLORKEY | win32con.LWA_ALPHA
        )
    
    # Position the border windows
    left, top, right, bottom = rect
    # Top border
    win32gui.SetWindowPos(
        borders[0], win32con.HWND_TOPMOST,
        left, top, right - left, border_width,
        win32con.SWP_SHOWWINDOW
    )
    # Right border
    win32gui.SetWindowPos(
        borders[1], win32con.HWND_TOPMOST,
        right - border_width, top, border_width, bottom - top,
        win32con.SWP_SHOWWINDOW
    )
    # Bottom border
    win32gui.SetWindowPos(
        borders[2], win32con.HWND_TOPMOST,
        left, bottom - border_width, right - left, border_width,
        win32con.SWP_SHOWWINDOW
    )
    # Left border
    win32gui.SetWindowPos(
        borders[3], win32con.HWND_TOPMOST,
        left, top, border_width, bottom - top,
        win32con.SWP_SHOWWINDOW
    )
    
    # Wait for the specified duration
    time.sleep(duration)
    
    # Remove the border windows
    for border_hwnd in borders:
        win32gui.DestroyWindow(border_hwnd)

def position_window(window_name, x, y, width, height):
    """Position and resize an OpenCV window."""
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd == 0:
        print(f"Window '{window_name}' not found")
        return False
    
    # Set window position and size
    ctypes.windll.user32.SetWindowPos(
        hwnd, 
        0, 
        x, y, width, height, 
        0
    )
    
    print(f"Window '{window_name}' positioned at ({x}, {y}) with size {width}x{height}")
    return True

def capture_window_screenshot(hwnd, width=640, height=640):
    """Capture a screenshot of a window using win32gui and win32ui."""
    # Get window rectangle
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    window_width = right - left
    window_height = bottom - top
    
    # Find the center point
    center_x = window_width // 2
    center_y = window_height // 2
    
    # Calculate capture region (centered)
    half_width = width // 2
    half_height = height // 2
    
    # Create device context
    try:
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        
        # Create bitmap
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, window_width, window_height)
        cDC.SelectObject(dataBitMap)
        
        # Copy screen to bitmap
        cDC.BitBlt((0, 0), (window_width, window_height), dcObj, (0, 0), win32con.SRCCOPY)
        
        # Convert bitmap to numpy array
        bmpinfo = dataBitMap.GetInfo()
        bmpstr = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
        
        # Remove alpha channel
        img = img[:, :, :3]
        
        # Crop to center region
        start_x = max(0, center_x - half_width)
        start_y = max(0, center_y - half_height)
        end_x = min(window_width, start_x + width)
        end_y = min(window_height, start_y + height)
        
        # If window is too small, pad with black
        if end_x - start_x < width or end_y - start_y < height:
            # Create black image of target size
            result = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Calculate offset for centered placement
            offset_x = (width - (end_x - start_x)) // 2
            offset_y = (height - (end_y - start_y)) // 2
            
            # Place the captured portion in the center
            actual_width = min(end_x - start_x, width)
            actual_height = min(end_y - start_y, height)
            result[offset_y:offset_y+actual_height, offset_x:offset_x+actual_width] = img[start_y:end_y, start_x:end_x]
        else:
            # Just crop the center if window is large enough
            result = img[start_y:end_y, start_x:end_x]
        
        # Clean up
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())
        
        # Convert to BGR for OpenCV
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        return result
    
    except Exception as e:
        print(f"Error in window capture: {e}")
        return None 