"""
Constants used across the aimbot modules.
"""

# Key codes
VK_TILDE = 0xC0     # Tilde key (~)
VK_A = 0x41         # A key for strafe left
VK_D = 0x44         # D key for strafe right
VK_P = 0x50         # P key to toggle prediction
VK_I = 0x49         # I key to toggle PID controller
VK_PLUS = 0x6B      # Numpad + key to increase sensitivity
VK_MINUS = 0x6D     # Numpad - key to decrease sensitivity
VK_V = 0x56         # V key to toggle visualization
VK_F = 0x46         # F key to toggle FPS display
VK_C = 0x43         # C key to toggle aimbot active state
VK_W = 0x57         # W key to toggle window selection mode

# OpenCV key constants
KEY_TOGGLE_TOPMOST = ord('t')     # Press 't' to toggle always-on-top mode
KEY_QUIT = ord('q')              # Press 'q' to quit
KEY_TOGGLE_PREDICTION = ord('p')  # Press 'p' to toggle prediction
KEY_TOGGLE_PID = ord('i')        # Press 'i' to toggle PID controller
KEY_INCREASE_SENS = ord('+')     # Press '+' to increase sensitivity
KEY_DECREASE_SENS = ord('-')     # Press '-' to decrease sensitivity
KEY_TOGGLE_VISUAL = ord('v')     # Press 'v' to toggle visualization
KEY_TOGGLE_FPS = ord('f')        # Press 'f' to toggle FPS display
KEY_TOGGLE_ACTIVE = ord('c')     # Press 'c' to toggle aimbot active state
KEY_WINDOW_SELECT = ord('w')     # Press 'w' to toggle window selection mode
KEY_DEBUG_REGION = ord('d')      # Press 'd' to toggle capture region debug window
KEY_TOGGLE_MASK = ord('m')       # Press 'm' to toggle target mask
KEY_INCREASE_MASK = ord('.')     # Press '.' to increase mask radius
KEY_DECREASE_MASK = ord(',')     # Press ',' to decrease mask radius

# Window settings
WINDOW_NAME = "Detection Output"  # OpenCV window name 

# Mask settings
DEFAULT_MASK_ENABLED = False     # Whether the mask is enabled by default
DEFAULT_MASK_RADIUS = 240        # Default radius (in pixels) for the circular mask
MIN_MASK_RADIUS = 100            # Minimum mask radius
MAX_MASK_RADIUS = 320            # Maximum mask radius (half of 640x640)
MASK_RADIUS_STEP = 20            # How much to increase/decrease radius per keypress 

# Aim offset settings
DEFAULT_VERTICAL_OFFSET = 0.0    # Default vertical offset (0.0 = center, positive = higher, negative = lower)
DEFAULT_HORIZONTAL_OFFSET = 0.0  # Default horizontal offset (0.0 = center, positive = right, negative = left)
OFFSET_STEP = 0.05               # How much to increase/decrease offset per keypress (5% of box height/width)
MIN_OFFSET = -0.5                # Minimum offset (-50% from center, i.e., bottom or left edge)
MAX_OFFSET = 0.5                 # Maximum offset (+50% from center, i.e., top or right edge)

# Keys for adjusting aim offset
KEY_TOGGLE_OFFSET = ord('o')     # Press 'o' to toggle between vertical and horizontal offset adjustment
KEY_INCREASE_OFFSET = ord('=')   # Press '=' to increase current offset
KEY_DECREASE_OFFSET = ord('-')   # Press '-' to decrease current offset