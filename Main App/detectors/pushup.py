from core.base_exercise import BaseExercise


class PushUpDetector(BaseExercise):
    DOWN_THRESHOLD = 90
    UP_THRESHOLD = 160
    MIN_VISIBILITY = 0.45
    HIP_SAG_TOLERANCE = 0.08

    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

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
        left_vis = landmarks[self.LEFT_ELBOW].visibility
        right_vis = landmarks[self.RIGHT_ELBOW].visibility

        if left_vis >= right_vis:
            shoulder_idx = self.LEFT_SHOULDER
            elbow_idx = self.LEFT_ELBOW
            wrist_idx = self.LEFT_WRIST
            hip_idx = self.LEFT_HIP
            ankle_idx = self.LEFT_ANKLE
        else:
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST
            hip_idx = self.RIGHT_HIP
            ankle_idx = self.RIGHT_ANKLE

        elbow_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx),
            self.get_point(landmarks, elbow_idx),
            self.get_point(landmarks, wrist_idx),
        )

        body_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx),
            self.get_point(landmarks, hip_idx),
            self.get_point(landmarks, ankle_idx),
        )

        shoulder_y = landmarks[shoulder_idx].y
        ankle_y = landmarks[ankle_idx].y
        hip_y = landmarks[hip_idx].y

        expected_hip_y = (shoulder_y + ankle_y) / 2
        hip_deviation = hip_y - expected_hip_y

        if body_angle > 160:
            body_alignment = "Straight"
        elif body_angle > 140:
            body_alignment = "Slight Bend"
        else:
            body_alignment = "Poor Form"

        if abs(hip_deviation) <= self.HIP_SAG_TOLERANCE:
            hip_status = "LEVEL"
        elif hip_deviation > self.HIP_SAG_TOLERANCE:
            hip_status = "SAGGING"
        else:
            hip_status = "PIKED UP"

        key_landmarks_visible = (
            landmarks[shoulder_idx].visibility > self.MIN_VISIBILITY
            and landmarks[elbow_idx].visibility > self.MIN_VISIBILITY
            and landmarks[wrist_idx].visibility > self.MIN_VISIBILITY
            and landmarks[hip_idx].visibility > self.MIN_VISIBILITY
        )

        if key_landmarks_visible:
            # 1. Lowering / Bottom / Pushing
            if elbow_angle < 145:
                self._rep_in_progress = True
                if elbow_angle <= self.DOWN_THRESHOLD:
                    self._reached_bottom = True
                    self.current_phase = "BOTTOM"
                elif self._reached_bottom:
                    self.current_phase = "PUSHING"
                else:
                    self.current_phase = "LOWERING"
                self.stage = "down"

            # Form error evaluation during repetition attempt
            if self._rep_in_progress:
                if hip_status == "SAGGING":
                    self.record_attempt_error("Hips sagging. Maintain a flat plank from shoulders to heels.")
                elif hip_status == "PIKED UP":
                    self.record_attempt_error("Hips piked up too high. Lower your hips into a straight line.")
                elif body_alignment == "Poor Form":
                    self.record_attempt_error("Body is not straight. Engage your core to stabilize spine.")
                if not key_landmarks_visible:
                    self.record_attempt_error("Key body landmarks obscured during push-up.")

            # 2. Return to lockout
            if elbow_angle >= self.UP_THRESHOLD:
                if self._rep_in_progress:
                    self.complete_attempt(
                        reached_target_rom=self._reached_bottom,
                        rom_failure_reason="Incomplete push-up depth. Lower chest until elbows reach 90 degrees."
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
            "elbow_angle": int(elbow_angle),
            "body_alignment": body_alignment,
            "hip_status": hip_status,
            "rejection_reason": self.rejection_reason,
            "last_rep_valid": self.last_rep_valid
        }
