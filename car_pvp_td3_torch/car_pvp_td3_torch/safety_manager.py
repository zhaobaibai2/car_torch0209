import numpy as np


class SafetyManager:
    def __init__(self, config: dict):
        self.max_steering_angle = config.get('max_steering_angle', 200)
        self.emergency_brake_threshold = config.get('emergency_brake_threshold', 5.0)
        self.min_safe_distance = config.get('min_safe_distance', 2.0)
        self.emergency_stop = False

        self.emergency_brakes = 0
        self.steer_violations = 0

    def check_action_safety(self, action: np.ndarray, state: np.ndarray) -> bool:
        if len(action) >= 2:
            steer_angle_deg = float(action[1]) * 200.0
            if abs(steer_angle_deg) > self.max_steering_angle:
                self.steer_violations += 1
                return False

        if len(state) >= 3:
            norm_error_distance = float(state[1])
            carspeed_norm = float(state[2])

            raw_error_distance = norm_error_distance * 5.0
            raw_carspeed = carspeed_norm * 15.0

            if raw_error_distance < -self.min_safe_distance and raw_carspeed > self.emergency_brake_threshold:
                self.emergency_brakes += 1
                return False

        return True

    def get_safety_stats(self) -> dict:
        return {
            'emergency_brakes': self.emergency_brakes,
            'steer_violations': self.steer_violations,
            'emergency_stop': self.emergency_stop,
        }
