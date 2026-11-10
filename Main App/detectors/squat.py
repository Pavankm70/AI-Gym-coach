from core.base_exercise import BaseExercise


class SquatDetector(BaseExercise):
    DOWN_THRESHOLD = 100   
    UP_THRESHOLD = 160     
    MIN_VISIBILITY = 0.45

    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12

    def __init__(self):
        super().__init__()
        self._reached_bottom = False
        self._rep_in_progress = False
        self.current_phase = "READY"

    def reset(self):
        self.reps = 0
        self.valid_reps = 0
        self.rejected_attempts = 0
        self.stage = None
        self.current_phase = "READY"
        self._reached_bottom = False
        self._rep_in_progress = False
        self.current_attempt_errors = []
        self.rejection_reason = None
        self.last_rep_valid = None

    def process(self, landmarks):
        left_knee_angle = self.calculate_angle(
            self.get_point(landmarks, self.LEFT_HIP),
            self.get_point(landmarks, self.LEFT_KNEE),
            self.get_point(landmarks, self.LEFT_ANKLE)
        )

        right_knee_angle = self.calculate_angle(
            self.get_point(landmarks, self.RIGHT_HIP),
            self.get_point(landmarks, self.RIGHT_KNEE),
            self.get_point(landmarks, self.RIGHT_ANKLE)
        )

        left_vis = landmarks[self.LEFT_KNEE].visibility
        right_vis = landmarks[self.RIGHT_KNEE].visibility

        if left_vis >= right_vis:
            knee_angle = left_knee_angle
            hip_idx, knee_idx, ankle_idx, shoulder_idx = (
                self.LEFT_HIP, self.LEFT_KNEE, self.LEFT_ANKLE, self.LEFT_SHOULDER
            )
        else:
            knee_angle = right_knee_angle
            hip_idx, knee_idx, ankle_idx, shoulder_idx = (
                self.RIGHT_HIP, self.RIGHT_KNEE, self.RIGHT_ANKLE, self.RIGHT_SHOULDER
            )

        back_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx),
            self.get_point(landmarks, hip_idx),
            self.get_point(landmarks, knee_idx)
        )

        key_landmark_visible = (
            landmarks[hip_idx].visibility >= self.MIN_VISIBILITY
            and landmarks[knee_idx].visibility >= self.MIN_VISIBILITY
            and landmarks[ankle_idx].visibility >= self.MIN_VISIBILITY
        )

        if key_landmark_visible:
            # 1. Active descent / bottom / rising
            if knee_angle < 150:
                self._rep_in_progress = True
                if knee_angle <= self.DOWN_THRESHOLD:
                    self._reached_bottom = True
                    self.current_phase = "BOTTOM"
                elif self._reached_bottom:
                    self.current_phase = "RISING"
                else:
                    self.current_phase = "LOWERING"
                self.stage = "down"

            # Form error evaluation during active repetition attempt
            if self._rep_in_progress:
                if back_angle < 75:
                    self.record_attempt_error("Excessive forward torso lean. Keep your chest up.")
                if not key_landmark_visible:
                    self.record_attempt_error("Key leg landmarks obscured during movement.")

            # 2. Return to standing lockout (knee_angle >= 160)
            if knee_angle >= self.UP_THRESHOLD:
                if self._rep_in_progress:
                    # Finalize repetition validation
                    self.complete_attempt(
                        reached_target_rom=self._reached_bottom,
                        rom_failure_reason="Squat depth was too shallow. Thighs must reach parallel."
                    )
                    self._rep_in_progress = False
                    self._reached_bottom = False
                    self.current_phase = "COMPLETE"
                    self.stage = "up"
                else:
                    self.current_phase = "READY"
                    if self.stage is None:
                        self.stage = "ready"

        # Depth status calculation
        if self.stage == "down":
            depth_status = "GOOD DEPTH" if knee_angle <= self.DOWN_THRESHOLD else "TOO HIGH"
        elif self.stage == "up" or self.current_phase in ["READY", "COMPLETE"]:
            depth_status = "STANDING"
        else:
            depth_status = "N/A"

        return {
            "reps": self.reps,
            "valid_reps": self.valid_reps,
            "rejected_attempts": self.rejected_attempts,
            "current_phase": self.current_phase,
            "stage": self.stage,
            "knee_angle": int(knee_angle),
            "back_angle": int(back_angle),
            "depth_status": depth_status,
            "rejection_reason": self.rejection_reason,
            "last_rep_valid": self.last_rep_valid
        }
