"""
Input utilities for keyboard and mouse control.
"""
import ctypes
from ctypes import windll, Structure, c_long, byref

class POINT(Structure):
    """Structure for mouse position."""
    _fields_ = [("x", c_long), ("y", c_long)]

def query_mouse_position():
    """Return the current mouse position as a dict with keys 'x' and 'y'."""
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return {"x": pt.x, "y": pt.y}

def is_key_pressed(key_code):
    """Check if a given key is currently pressed using GetAsyncKeyState."""
    return (ctypes.windll.user32.GetAsyncKeyState(key_code) & 0x8000) != 0

def mouse_move(dx, dy):
    """Alternative mouse movement function using ctypes directly."""
    # Constants from winuser.h
    MOUSEEVENTF_MOVE = 0x0001
    
    # Use ctypes directly
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, ctypes.c_long(dx), ctypes.c_long(dy), 0, 0) 