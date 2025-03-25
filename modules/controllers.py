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
    def __init__(self, history_size=5, prediction_time=0.05):
        self.history_size = history_size          # How many positions to keep in history
        self.prediction_time = prediction_time    # How far into the future to predict (in seconds)
        self.position_history = []                # List of (position, timestamp) tuples
        self.last_prediction = None               # Last predicted position
        self.strafe_detected = False              # Flag to indicate strafing
        self.strafe_direction = 0                 # Direction of strafing (-1 left, 1 right)
        self.strafe_compensation = 1.5            # Amplify prediction during strafing
    
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
        """Predict future position based on movement history."""
        if len(self.position_history) < 2:
            # Need at least two points to calculate velocity
            return self.position_history[-1][0] if self.position_history else None
        
        # Calculate velocity from the last few positions
        positions = [p[0] for p in self.position_history]
        times = [p[1] for p in self.position_history]
        
        # Use more recent points for rapidly changing trajectories
        recent_idx = min(2, len(positions) - 1)
        
        # Simple linear velocity calculation from the most recent positions
        dx = positions[-1][0] - positions[-1-recent_idx][0]
        dy = positions[-1][1] - positions[-1-recent_idx][1]
        dt = times[-1] - times[-1-recent_idx]
        
        if dt < 0.001:  # Avoid division by zero
            return positions[-1]
        
        # Calculate velocity
        vx = dx / dt
        vy = dy / dt
        
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
        adaptive_time = self.prediction_time
        pred_x = positions[-1][0] + vx * adaptive_time
        pred_y = positions[-1][1] + vy * adaptive_time
        
        # Store this prediction
        self.last_prediction = (pred_x, pred_y)
        
        return (pred_x, pred_y)
    
    def get_strafe_info(self):
        """Return information about detected strafing for visualization."""
        if self.strafe_detected:
            direction = "Right" if self.strafe_direction > 0 else "Left"
            return f"Player Strafe: {direction}"
        return None
    
    def reset(self):
        """Reset the predictor state."""
        self.position_history = []
        self.last_prediction = None
        self.strafe_detected = False
        self.strafe_direction = 0 