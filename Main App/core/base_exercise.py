import math 
from abc import ABC, abstractmethod


class BaseExercise(ABC):
    def __init__(self):
        self.reps = 0                     # Backwards-compatible; synchronized with valid_reps
        self.valid_reps = 0               # Only increments when rep strictly passes validation
        self.rejected_attempts = 0        # Increments when movement cycle had form errors or incomplete ROM
        self.stage = None                 # Backwards-compatible stage ('ready', 'up', 'down')
        self.current_phase = "READY"      # Exercise-specific movement phase
        self.current_attempt_errors = []  # Accumulated form errors during active repetition window
        self.rejection_reason = None      # Reason for most recent rep rejection
        self.last_rep_valid = None        # True if last completed rep was valid, False if rejected

    def calculate_angle(self, a, b, c):
        ax, ay = a[0] - b[0], a[1] - b[1]
        cx, cy = c[0] - b[0], c[1] - b[1]

        dot = ax * cx + ay * cy

        mag_a = math.sqrt(ax ** 2 + ay ** 2)
        mag_c = math.sqrt(cx ** 2 + cy ** 2)

        if mag_a * mag_c == 0:
            return 0.0

        cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))

        return math.degrees(math.acos(cos_angle))

    def get_point(self, landmarks, idx):
        p = landmarks[idx]
        return (p.x, p.y)

    def record_attempt_error(self, error_message: str):
        """Records a form error during the active repetition attempt."""
        if error_message and error_message not in self.current_attempt_errors:
            self.current_attempt_errors.append(error_message)

    def complete_attempt(self, reached_target_rom: bool, rom_failure_reason: str = "Incomplete range of motion."):
        """
        Validates the completed repetition cycle.
        If target ROM was reached and NO critical form errors occurred -> valid_rep.
        Otherwise -> rejected_attempt with explanation.
        """
        if not reached_target_rom:
            self.record_attempt_error(rom_failure_reason)

        if len(self.current_attempt_errors) == 0:
            self.valid_reps += 1
            self.reps = self.valid_reps
            self.last_rep_valid = True
            self.rejection_reason = None
        else:
            self.rejected_attempts += 1
            self.last_rep_valid = False
            self.rejection_reason = self.current_attempt_errors[0]

        # Reset active attempt tracking
        self.current_attempt_errors = []

    @abstractmethod
    def process(self, landmarks):
        pass

    @abstractmethod
    def reset(self):
        pass
