"""
Controller classes for aim control and target prediction.
"""
import time
from modules.input_utils import is_key_pressed
from constants import VK_A, VK_D

class PIDController:
    """Basic PID controller for mouse movement."""
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error):
        """Update PID values given the current error."""
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
    
    def reset(self):
        """Reset the controller state."""
        self.prev_error = 0
        self.integral = 0

class TargetPredictor:
    """Predicts future position of targets based on movement history."""
    def __init__(self, history_size=10, prediction_time=0.05, sample_interval=5):
        self.history_size = history_size          # Increased history size for wider sampling
        self.prediction_time = prediction_time    # How far into the future to predict (in seconds)
        self.sample_interval = sample_interval    # Number of frames to skip for velocity calculation
        self.position_history = []                # List of (position, timestamp) tuples
        self.last_prediction = None               # Last predicted position
        self.strafe_detected = False              # Flag to indicate strafing
        self.strafe_direction = 0                 # Direction of strafing (-1 left, 1 right)
        self.strafe_compensation = 1.5            # Amplify prediction during strafing
        
        # Velocity smoothing parameters
        self.smoothed_vx = 0
        self.smoothed_vy = 0
        self.smoothing_factor = 0.2               # Lower = more smoothing (0.2 is a good balance)
    
    def add_position(self, position, timestamp):
        """Add a new position observation with timestamp."""
        if position is None:
            return
        
        self.position_history.append((position, timestamp))
        
        # Keep only the most recent observations
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
    
    def update_strafe_state(self):
        """Update strafe state based on key presses instead of velocity analysis."""
        # Check if A or D keys are pressed
        a_pressed = is_key_pressed(VK_A)
        d_pressed = is_key_pressed(VK_D)
        
        if a_pressed and not d_pressed:
            self.strafe_detected = True
            self.strafe_direction = -1  # Left
        elif d_pressed and not a_pressed:
            self.strafe_detected = True
            self.strafe_direction = 1   # Right
        else:
            self.strafe_detected = False
            self.strafe_direction = 0
    
    def predict_position(self, current_time):
        """Predict future position based on movement history with wider sampling."""
        # Need enough history for the sampling interval
        required_samples = self.sample_interval + 1
        if len(self.position_history) < required_samples:
            return self.position_history[-1][0] if self.position_history else None
        
        # Use current position and time
        current_pos = self.position_history[-1][0]
        current_time = self.position_history[-1][1]
        
        # Implement dynamic sampling interval for better stability
        min_interval = 2
        max_interval = min(self.sample_interval, len(self.position_history) - 1)
        
        # Find the best sampling interval based on movement magnitude
        best_interval = min_interval
        for interval in range(min_interval, max_interval + 1):
            older_pos = self.position_history[-interval-1][0]
            # Calculate displacement
            dx = current_pos[0] - older_pos[0]
            dy = current_pos[1] - older_pos[1]
            dist = (dx**2 + dy**2)**0.5
            
            # If we've moved at least 5 pixels, this is enough displacement to measure
            if dist >= 5:
                best_interval = interval
                break
        
        # Use the best interval for calculation
        older_pos = self.position_history[-best_interval-1][0]
        older_time = self.position_history[-best_interval-1][1]
        
        # Calculate time difference between samples
        dt = current_time - older_time
        
        if dt < 0.001:  # Avoid division by zero
            return current_pos
        
        # Calculate displacement between sampled points
        dx = current_pos[0] - older_pos[0]
        dy = current_pos[1] - older_pos[1]
        
        # Calculate velocity
        vx = dx / dt
        vy = dy / dt
        
        # Apply exponential smoothing to velocity
        self.smoothed_vx = self.smoothing_factor * vx + (1 - self.smoothing_factor) * self.smoothed_vx
        self.smoothed_vy = self.smoothing_factor * vy + (1 - self.smoothing_factor) * self.smoothed_vy
        
        # Use smoothed velocity for calculations
        vx = self.smoothed_vx
        vy = self.smoothed_vy
        
        # Debug output
        if len(self.position_history) % 30 == 0:  # Limit logging frequency
            print(f"Using sample interval: {best_interval}, dt: {dt:.3f}s")
            print(f"Raw velocity: ({dx/dt:.1f}, {dy/dt:.1f}), Smoothed: ({vx:.1f}, {vy:.1f})")
        
        # Apply strafe compensation if detected
        if self.strafe_detected:
            # Based on which key is pressed, apply compensation in the appropriate direction
            horizontal_multiplier = 1.0
            
            # Add extra compensation in the strafe direction
            if self.strafe_direction != 0:
                # If strafing right (D key), increase compensation for right-moving targets
                # If strafing left (A key), increase compensation for left-moving targets
                # This makes the prediction lead the target more when the player is strafing
                same_direction = (self.strafe_direction > 0 and vx > 0) or (self.strafe_direction < 0 and vx < 0)
                opposite_direction = (self.strafe_direction > 0 and vx < 0) or (self.strafe_direction < 0 and vx > 0)
                
                if same_direction:
                    # When target moves in same direction as player strafe, need more lead
                    horizontal_multiplier = self.strafe_compensation * 1.5
                elif opposite_direction:
                    # When target moves opposite to player strafe, need less lead
                    horizontal_multiplier = self.strafe_compensation * 0.8
                else:
                    # Default compensation
                    horizontal_multiplier = self.strafe_compensation
                
                # Apply the multiplier
                vx = vx * horizontal_multiplier
        
        # Predict future position
        pred_x = current_pos[0] + vx * self.prediction_time
        pred_y = current_pos[1] + vy * self.prediction_time
        
        # Store this prediction
        self.last_prediction = (pred_x, pred_y)
        
        return (pred_x, pred_y)
    
    def get_strafe_info(self):
        """Return information about detected strafing for visualization."""
        if self.strafe_detected:
            direction = "Right" if self.strafe_direction > 0 else "Left"
            return f"Player Strafe: {direction}"
        return None
    
    def get_velocity_info(self):
        """Return information about current velocity for debugging."""
        if self.smoothed_vx == 0 and self.smoothed_vy == 0:
            return None
        
        speed = (self.smoothed_vx**2 + self.smoothed_vy**2)**0.5
        return f"Velocity: {speed:.1f}px/s"
    
    def reset(self):
        """Reset the predictor state."""
        self.position_history = []
        self.last_prediction = None
        self.strafe_detected = False
        self.strafe_direction = 0
        self.smoothed_vx = 0
        self.smoothed_vy = 0 