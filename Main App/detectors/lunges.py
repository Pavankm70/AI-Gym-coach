from core.base_exercise import BaseExercise


class LungesDetector(BaseExercise):
    DOWN_THRESHOLD = 100
    UP_THRESHOLD = 160
    MIN_VISIBILITY = 0.45
    BALANCE_TOLERANCE = 0.10

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
        self._rep_in_progress = False
        self._reached_bottom = False
        self.current_phase = "READY"

    def reset(self) -> None:
        self.reps = 0
        self.valid_reps = 0
        self.rejected_attempts = 0
        self.stage = None
        self.current_phase = "READY"
        self._rep_in_progress = False
        self._reached_bottom = False
        self.current_attempt_errors = []
        self.rejection_reason = None
        self.last_rep_valid = None

    def process(self, landmarks) -> dict:
        left_knee_angle = self.calculate_angle(
            self.get_point(landmarks, self.LEFT_HIP),
            self.get_point(landmarks, self.LEFT_KNEE),
            self.get_point(landmarks, self.LEFT_ANKLE),
        )

        right_knee_angle = self.calculate_angle(
            self.get_point(landmarks, self.RIGHT_HIP),
            self.get_point(landmarks, self.RIGHT_KNEE),
            self.get_point(landmarks, self.RIGHT_ANKLE),
        )

        if left_knee_angle <= right_knee_angle:
            front_knee_angle = left_knee_angle
            front_hip_idx = self.LEFT_HIP
            front_knee_idx = self.LEFT_KNEE
            front_ankle_idx = self.LEFT_ANKLE
            shoulder_idx_for_torso = self.LEFT_SHOULDER
        else:
            front_knee_angle = right_knee_angle
            front_hip_idx = self.RIGHT_HIP
            front_knee_idx = self.RIGHT_KNEE
            front_ankle_idx = self.RIGHT_ANKLE
            shoulder_idx_for_torso = self.RIGHT_SHOULDER

        torso_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx_for_torso),
            self.get_point(landmarks, front_hip_idx),
            self.get_point(landmarks, front_knee_idx),
        )

        shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2
        hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2
        lateral_offset = abs(shoulder_mid_x - hip_mid_x)

        if lateral_offset <= self.BALANCE_TOLERANCE:
            balance_status = "BALANCED"
        else:
            balance_status = "OFF BALANCE"

        key_landmarks_visible = (
            landmarks[front_hip_idx].visibility > self.MIN_VISIBILITY
            and landmarks[front_knee_idx].visibility > self.MIN_VISIBILITY
            and landmarks[front_ankle_idx].visibility > self.MIN_VISIBILITY
        )

        if key_landmarks_visible:
            # 1. Lowering / Bottom / Rising
            if front_knee_angle < 150:
                self._rep_in_progress = True
                if front_knee_angle <= self.DOWN_THRESHOLD:
                    self._reached_bottom = True
                    self.current_phase = "BOTTOM"
                elif self._reached_bottom:
                    self.current_phase = "RISING"
                else:
                    self.current_phase = "LOWERING"
                self.stage = "down"

            # Form error evaluation during repetition attempt
            if self._rep_in_progress:
                if balance_status == "OFF BALANCE":
                    self.record_attempt_error("Loss of lateral balance. Keep feet hip-width apart.")
                if not key_landmarks_visible:
                    self.record_attempt_error("Key leg landmarks obscured during lunge.")

            # 2. Return to upright standing
            if front_knee_angle >= self.UP_THRESHOLD:
                if self._rep_in_progress:
                    self.complete_attempt(
                        reached_target_rom=self._reached_bottom,
                        rom_failure_reason="Incomplete lunge depth. Lower front thigh until knee reaches 90-100 degrees."
                    )
                    self._rep_in_progress = False
                    self._reached_bottom = False
                    self.current_phase = "COMPLETE"
                    self.stage = "up"
                else:
                    self.current_phase = "READY"
                    if self.stage is None:
                        self.stage = "ready"

        return {
            "reps": self.reps,
            "valid_reps": self.valid_reps,
            "rejected_attempts": self.rejected_attempts,
            "current_phase": self.current_phase,
            "stage": self.stage,
            "front_knee_angle": int(front_knee_angle),
            "torso_angle": int(torso_angle),
            "balance_status": balance_status,
            "rejection_reason": self.rejection_reason,
            "last_rep_valid": self.last_rep_valid
        }
