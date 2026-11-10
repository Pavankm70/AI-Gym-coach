import math
from core.base_exercise import BaseExercise


class BicepsCurlDetector(BaseExercise):
    UP_THRESHOLD = 50
    DOWN_THRESHOLD = 160
    MIN_VISIBILITY = 0.45
    ELBOW_DRIFT_TOLERANCE = 0.06
    SWING_THRESHOLD = 15

    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self):
        super().__init__()
        self._shoulder_x_baseline = None
        self._rep_in_progress = False
        self._reached_peak = False
        self.current_phase = "START"

    def reset(self) -> None:
        self.reps = 0
        self.valid_reps = 0
        self.rejected_attempts = 0
        self.stage = None
        self.current_phase = "START"
        self._shoulder_x_baseline = None
        self._rep_in_progress = False
        self._reached_peak = False
        self.current_attempt_errors = []
        self.rejection_reason = None
        self.last_rep_valid = None

    def process(self, landmarks) -> dict:
        left_vis = landmarks[self.LEFT_ELBOW].visibility
        right_vis = landmarks[self.RIGHT_ELBOW].visibility

        if left_vis >= right_vis:
            shoulder_idx = self.LEFT_SHOULDER
            elbow_idx = self.LEFT_ELBOW
            wrist_idx = self.LEFT_WRIST
            hip_idx = self.LEFT_HIP
        else:
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST
            hip_idx = self.RIGHT_HIP

        elbow_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx),
            self.get_point(landmarks, elbow_idx),
            self.get_point(landmarks, wrist_idx),
        )

        shoulder_x = landmarks[shoulder_idx].x
        elbow_x = landmarks[elbow_idx].x
        elbow_drift = abs(elbow_x - shoulder_x)

        if elbow_drift <= self.ELBOW_DRIFT_TOLERANCE:
            shoulder_status = "STABLE"
        else:
            shoulder_status = "ELBOW DRIFTING"

        if landmarks[self.LEFT_SHOULDER].visibility > 0.5 and landmarks[self.RIGHT_SHOULDER].visibility > 0.5 and abs(landmarks[self.LEFT_SHOULDER].x - landmarks[self.RIGHT_SHOULDER].x) > 0.05:
            shoulder_mid_x = (landmarks[self.LEFT_SHOULDER].x + landmarks[self.RIGHT_SHOULDER].x) / 2
            shoulder_mid_y = (landmarks[self.LEFT_SHOULDER].y + landmarks[self.RIGHT_SHOULDER].y) / 2
            hip_mid_x = (landmarks[self.LEFT_HIP].x + landmarks[self.RIGHT_HIP].x) / 2
            hip_mid_y = (landmarks[self.LEFT_HIP].y + landmarks[self.RIGHT_HIP].y) / 2
        else:
            shoulder_mid_x = landmarks[shoulder_idx].x
            shoulder_mid_y = landmarks[shoulder_idx].y
            hip_mid_x = landmarks[hip_idx].x
            hip_mid_y = landmarks[hip_idx].y

        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y

        torso_angle_from_vertical = self._safe_angle(dx, dy)

        if torso_angle_from_vertical <= self.SWING_THRESHOLD:
            swing_status = "NO SWING"
        else:
            swing_status = "SWINGING"

        key_landmarks_visible = (
            landmarks[shoulder_idx].visibility > self.MIN_VISIBILITY
            and landmarks[elbow_idx].visibility > self.MIN_VISIBILITY
            and landmarks[wrist_idx].visibility > self.MIN_VISIBILITY
        )

        if key_landmarks_visible:
            # 1. Curling / Flexed / Lowering
            if elbow_angle < 145:
                self._rep_in_progress = True
                if elbow_angle < self.UP_THRESHOLD:
                    self._reached_peak = True
                    self.current_phase = "FLEXED"
                elif self._reached_peak:
                    self.current_phase = "LOWERING"
                else:
                    self.current_phase = "CURLING"
                self.stage = "up"

            # Form error evaluation during repetition attempt
            if self._rep_in_progress:
                if elbow_drift > self.ELBOW_DRIFT_TOLERANCE:
                    self.record_attempt_error("Elbow drifted forward/backward. Keep elbows pinned to your sides.")
                if torso_angle_from_vertical > self.SWING_THRESHOLD:
                    self.record_attempt_error("Excessive torso swing. Do not use body momentum to swing weights.")
                if not key_landmarks_visible:
                    self.record_attempt_error("Arm landmarks obscured during repetition.")

            # 2. Return to extension
            if elbow_angle >= 155:
                if self._rep_in_progress:
                    self.complete_attempt(
                        reached_target_rom=self._reached_peak,
                        rom_failure_reason="Incomplete curl range. Flex fully until dumbbell reaches shoulder level."
                    )
                    self._rep_in_progress = False
                    self._reached_peak = False
                    self.current_phase = "COMPLETE"
                    self.stage = "down"
                else:
                    self.current_phase = "START"
                    if self.stage is None:
                        self.stage = "ready"

        return {
            "reps": self.reps,
            "valid_reps": self.valid_reps,
            "rejected_attempts": self.rejected_attempts,
            "current_phase": self.current_phase,
            "stage": self.stage,
            "elbow_angle": int(elbow_angle),
            "shoulder_status": shoulder_status,
            "swing_status": swing_status,
            "rejection_reason": self.rejection_reason,
            "last_rep_valid": self.last_rep_valid
        }

    def _safe_angle(self, dx, dy):
        return math.degrees(math.atan2(abs(dx), abs(dy))) if dy != 0 else 0.0
