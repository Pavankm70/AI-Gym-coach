from core.base_exercise import BaseExercise


class ShoulderPressDetector(BaseExercise):
    """
    Robust Shoulder Press Detector with overhead geometry verification,
    bilateral joint tracking, validated repetition state transitions, and
    exercise-specific phases: RESTING, RACK, PRESSING, OVERHEAD, RETURNING, COMPLETE.
    """
    UP_THRESHOLD = 155      # Angle for full overhead extension
    DOWN_THRESHOLD = 95     # Angle for lowered / starting rack position
    MIN_VISIBILITY = 0.45    # Minimum visibility required for key joints

    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    def __init__(self):
        super().__init__()
        self._reached_overhead = False
        self._rep_in_progress = False
        self.stage = "ready"
        self.current_phase = "RESTING"

    def reset(self) -> None:
        self.reps = 0
        self.valid_reps = 0
        self.rejected_attempts = 0
        self.stage = "ready"
        self.current_phase = "RESTING"
        self._reached_overhead = False
        self._rep_in_progress = False
        self.current_attempt_errors = []
        self.rejection_reason = None
        self.last_rep_valid = None

    def process(self, landmarks) -> dict:
        left_shoulder = self.get_point(landmarks, self.LEFT_SHOULDER)
        left_elbow = self.get_point(landmarks, self.LEFT_ELBOW)
        left_wrist = self.get_point(landmarks, self.LEFT_WRIST)

        right_shoulder = self.get_point(landmarks, self.RIGHT_SHOULDER)
        right_elbow = self.get_point(landmarks, self.RIGHT_ELBOW)
        right_wrist = self.get_point(landmarks, self.RIGHT_WRIST)

        left_elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

        left_vis = (
            landmarks[self.LEFT_SHOULDER].visibility
            + landmarks[self.LEFT_ELBOW].visibility
            + landmarks[self.LEFT_WRIST].visibility
        ) / 3.0

        right_vis = (
            landmarks[self.RIGHT_SHOULDER].visibility
            + landmarks[self.RIGHT_ELBOW].visibility
            + landmarks[self.RIGHT_WRIST].visibility
        ) / 3.0

        if left_vis >= right_vis:
            selected_side = "left"
            shoulder_idx = self.LEFT_SHOULDER
            elbow_idx = self.LEFT_ELBOW
            wrist_idx = self.LEFT_WRIST
            hip_idx = self.LEFT_HIP
            knee_idx = self.LEFT_KNEE
            elbow_angle = left_elbow_angle
            wrist_point = left_wrist
            shoulder_point = left_shoulder
        else:
            selected_side = "right"
            shoulder_idx = self.RIGHT_SHOULDER
            elbow_idx = self.RIGHT_ELBOW
            wrist_idx = self.RIGHT_WRIST
            hip_idx = self.RIGHT_HIP
            knee_idx = self.RIGHT_KNEE
            elbow_angle = right_elbow_angle
            wrist_point = right_wrist
            shoulder_point = right_shoulder

        key_landmarks_visible = (
            landmarks[shoulder_idx].visibility >= self.MIN_VISIBILITY
            and landmarks[elbow_idx].visibility >= self.MIN_VISIBILITY
            and landmarks[wrist_idx].visibility >= self.MIN_VISIBILITY
        )

        arms_hanging_down = (wrist_point[1] > shoulder_point[1] + 0.08)
        hands_overhead = (wrist_point[1] < shoulder_point[1])
        hands_at_rack = (wrist_point[1] <= shoulder_point[1] + 0.12)

        # Back arch analysis
        back_angle = self.calculate_angle(
            self.get_point(landmarks, shoulder_idx),
            self.get_point(landmarks, hip_idx),
            self.get_point(landmarks, knee_idx),
        )

        if back_angle >= 160:
            back_arch_status = "Neutral"
        elif back_angle >= 140:
            back_arch_status = "Slight Arch"
        else:
            back_arch_status = "Excessive Arch"

        if key_landmarks_visible:
            if arms_hanging_down:
                self.stage = "ready"
                self.current_phase = "RESTING"
                self._rep_in_progress = False
                self._reached_overhead = False
            else:
                # 1. Starting Rack Position
                if elbow_angle <= self.DOWN_THRESHOLD and hands_at_rack and not self._rep_in_progress:
                    self.stage = "down"
                    self.current_phase = "RACK"

                # 2. Concentric Pressing Upward
                elif elbow_angle > self.DOWN_THRESHOLD and not self._reached_overhead and hands_overhead:
                    self._rep_in_progress = True
                    self.stage = "up"
                    self.current_phase = "PRESSING"

                # 3. Peak Overhead Extension reached (elbow >= 155 and hands above shoulders)
                if elbow_angle >= self.UP_THRESHOLD and hands_overhead:
                    self._reached_overhead = True
                    self._rep_in_progress = True
                    self.stage = "up"
                    self.current_phase = "OVERHEAD"

                # 4. Eccentric Returning Down towards Rack
                elif self._reached_overhead and elbow_angle < self.UP_THRESHOLD and elbow_angle > self.DOWN_THRESHOLD:
                    self.stage = "lowering"
                    self.current_phase = "RETURNING"

                # Form error evaluation during active repetition attempt
                if self._rep_in_progress:
                    if back_arch_status == "Excessive Arch":
                        self.record_attempt_error("Excessive lower back arch. Brace your abdominal core to protect lumbar spine.")
                    if not key_landmarks_visible:
                        self.record_attempt_error("Arm landmarks obscured during press.")

                # 5. Return to Rack: Finalize Repetition Validation
                if elbow_angle <= self.DOWN_THRESHOLD and hands_at_rack:
                    if self._rep_in_progress:
                        self.complete_attempt(
                            reached_target_rom=self._reached_overhead,
                            rom_failure_reason="Incomplete overhead extension. Press fully overhead until arms are straight."
                        )
                        self._rep_in_progress = False
                        self._reached_overhead = False
                        self.current_phase = "COMPLETE"
                        self.stage = "down"
                    else:
                        self.current_phase = "RACK"
                        self.stage = "down"

        # Extension status
        if arms_hanging_down:
            extension_status = "RESTING"
        elif elbow_angle >= self.UP_THRESHOLD and hands_overhead:
            extension_status = "FULL EXTENSION"
        elif elbow_angle >= 130 and hands_overhead:
            extension_status = "NEARLY EXTENDED"
        elif elbow_angle >= self.DOWN_THRESHOLD:
            extension_status = "PRESSING"
        else:
            extension_status = "START POSITION"

        return {
            "reps": self.reps,
            "valid_reps": self.valid_reps,
            "rejected_attempts": self.rejected_attempts,
            "current_phase": self.current_phase,
            "stage": self.stage,
            "elbow_angle": int(elbow_angle),
            "left_elbow_angle": int(left_elbow_angle),
            "right_elbow_angle": int(right_elbow_angle),
            "selected_side": selected_side,
            "extension_status": extension_status,
            "back_arch_status": back_arch_status,
            "back_angle": int(back_angle),
            "hands_overhead": hands_overhead,
            "rejection_reason": self.rejection_reason,
            "last_rep_valid": self.last_rep_valid
        }
