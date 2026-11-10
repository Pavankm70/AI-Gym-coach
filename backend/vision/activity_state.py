"""
Activity-State Management Module
================================
Tracks the 10-state workout lifecycle and provides exercise-specific starting posture
guidance, intelligent rest handling, and debounced coaching cues.

The 10 State Flow:
1. EXERCISE_SELECTED: Exercise chosen, awaiting user setup.
2. CAMERA_CHECK: Verifying camera view and framing.
3. READY_IDLE: Athlete in view, standing idle before setup.
4. WAITING_FOR_START_POSITION: Prompting exercise-specific starting posture.
5. READY_TO_EXERCISE: Starting position confirmed; prompt to begin first rep.
6. EXERCISING: Active movement through biomechanical phases.
7. SET_COMPLETE: Target repetitions for current set achieved.
8. RESTING: Intentional recovery interval between sets.
9. NEXT_SET_OR_EXERCISE: Rest complete; guiding to next set or exercise.
10. WORKOUT_COMPLETE: All target sets completed.

Preserves compatibility aliases for all historical coaching states.
"""

import math
import time
from typing import Dict, Any, Optional, List, Tuple


def _calculate_angle_2d(a, b, c) -> float:
    """Calculate 2D angle ABC in degrees."""
    try:
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.hypot(ba[0], ba[1])
        mag_bc = math.hypot(bc[0], bc[1])
        if mag_ba * mag_bc == 0:
            return 0.0
        cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cosine))
    except Exception:
        return 0.0


# Exercise-specific starting guidance templates
EXERCISE_START_INSTRUCTIONS = {
    "Squats": "Stand upright and keep your feet shoulder-width apart.",
    "Push-ups": "Move into the push-up starting position and align your body in a plank.",
    "Biceps Curls (Dumbbell)": "Stand upright, keep your elbows close to your body, and prepare to curl.",
    "Shoulder Press": "Stand upright and bring both hands to shoulder level in the rack position.",
    "Lunges": "Stand upright and prepare your starting stance with feet together."
}


class ActivityStateManager:
    # 13 Workout Flow States (Authoritative Lifecycle)
    EXERCISE_SELECTED = "EXERCISE_SELECTED"
    CAMERA_CHECK = "CAMERA_CHECK"
    READY_IDLE = "READY_IDLE"
    WAITING_FOR_START_POSITION = "WAITING_FOR_START_POSITION"
    READY_TO_EXERCISE = "READY_TO_EXERCISE"
    EXERCISING = "EXERCISING"
    SET_COMPLETE = "SET_COMPLETE"
    GEMINI_REVIEW_PENDING = "GEMINI_REVIEW_PENDING"
    GEMINI_REVIEW_READY = "GEMINI_REVIEW_READY"
    RESTING = "RESTING"
    NEXT_SET = "NEXT_SET"
    NEXT_EXERCISE = "NEXT_EXERCISE"
    WORKOUT_COMPLETE = "WORKOUT_COMPLETE"

    # Compatibility Aliases
    NEXT_SET_OR_EXERCISE = "NEXT_SET_OR_EXERCISE"
    NO_EXERCISE_SELECTED = "NO_EXERCISE_SELECTED"
    CAMERA_NOT_READY = "CAMERA_NOT_READY"
    WAITING_FOR_USER = "WAITING_FOR_USER"
    START_POSITION_REQUIRED = "START_POSITION_REQUIRED"
    READY_TO_BEGIN = "READY_TO_BEGIN"
    EXERCISE_STARTED = "EXERCISE_STARTED"
    MOVEMENT_IN_PROGRESS = "MOVEMENT_IN_PROGRESS"
    EXERCISE_IN_PROGRESS = "EXERCISE_IN_PROGRESS"
    FORM_ERROR_DETECTED = "FORM_ERROR_DETECTED"
    VALID_REPETITION_COMPLETED = "VALID_REPETITION_COMPLETED"
    INVALID_REPETITION_REJECTED = "INVALID_REPETITION_REJECTED"
    SET_COMPLETED = "SET_COMPLETED"
    EXERCISE_COMPLETED = "EXERCISE_COMPLETED"

    def __init__(
        self,
        idle_start_timeout: float = 4.5,
        rest_prompt_timeout: float = 15.0,
        inactivity_timeout: float = 30.0,
        message_cooldown: float = 4.0
    ):
        self.idle_start_timeout = idle_start_timeout
        self.rest_prompt_timeout = rest_prompt_timeout
        self.inactivity_timeout = inactivity_timeout
        self.message_cooldown = message_cooldown

        self.current_state = self.WAITING_FOR_USER
        self.state_enter_time = time.time()
        self.last_movement_time = time.time()
        self.last_rep_completed_time = 0.0
        self.last_rep_count = 0
        self.last_rejected_count = 0
        self.last_prompt_time = 0.0
        self.last_prompt_message = ""
        self._start_confirmed_for_exercise = None

        self._prev_landmarks: Optional[List[Any]] = None

    def reset(self):
        """Resets the state manager for a new workout."""
        self.current_state = self.WAITING_FOR_USER
        self.state_enter_time = time.time()
        self.last_movement_time = time.time()
        self.last_rep_completed_time = 0.0
        self.last_rep_count = 0
        self.last_rejected_count = 0
        self.last_prompt_time = 0.0
        self.last_prompt_message = ""
        self._start_confirmed_for_exercise = None
        self._prev_landmarks = None

    def _estimate_movement(self, landmarks: Optional[List[Any]]) -> float:
        """Calculates average spatial displacement of key anchor joints between frames."""
        if not landmarks or len(landmarks) < 25 or not self._prev_landmarks:
            self._prev_landmarks = landmarks
            return 0.0

        joint_indices = [11, 12, 13, 14, 23, 24]
        displacements = []
        for idx in joint_indices:
            if idx < len(landmarks) and idx < len(self._prev_landmarks):
                curr = landmarks[idx]
                prev = self._prev_landmarks[idx]
                displacements.append(math.hypot(curr.x - prev.x, curr.y - prev.y))

        self._prev_landmarks = landmarks
        return sum(displacements) / len(displacements) if displacements else 0.0

    def verify_starting_position(self, exercise_name: str, landmarks: Optional[List[Any]]) -> Tuple[bool, str]:
        """
        Exercise-specific starting position detection.
        Returns (is_ready, posture_instruction).
        """
        ex = (exercise_name or "").strip()
        default_instruction = EXERCISE_START_INSTRUCTIONS.get(ex, "Stand in frame and prepare to begin.")

        if not landmarks or len(landmarks) < 25:
            return False, default_instruction

        ls, rs = landmarks[11], landmarks[12]
        lh, rh = landmarks[23], landmarks[24]
        le, re = landmarks[13], landmarks[14]
        lw, rw = landmarks[15], landmarks[16]

        if ex == "Squats":
            # Upright standing posture: knee angle >= 155
            if len(landmarks) > 28:
                lk, rk = landmarks[25], landmarks[26]
                la, ra = landmarks[27], landmarks[28]
                left_knee_ang = _calculate_angle_2d(lh, lk, la)
                right_knee_ang = _calculate_angle_2d(rh, rk, ra)
                avg_knee = (left_knee_ang + right_knee_ang) / 2.0 if (left_knee_ang and right_knee_ang) else (left_knee_ang or right_knee_ang)
                if avg_knee < 155.0:
                    return False, EXERCISE_START_INSTRUCTIONS["Squats"]
            return True, "Squat starting posture confirmed."

        elif ex == "Push-ups":
            # High plank posture: horizontal body alignment and arms extended >= 140 deg
            torso_vertical = ((lh.y + rh.y) / 2.0) - ((ls.y + rs.y) / 2.0)
            if torso_vertical > 0.22:
                # Still standing upright, not down in a horizontal plank position
                return False, EXERCISE_START_INSTRUCTIONS["Push-ups"]
            left_elbow_ang = _calculate_angle_2d(ls, le, lw)
            right_elbow_ang = _calculate_angle_2d(rs, re, rw)
            avg_elbow = (left_elbow_ang + right_elbow_ang) / 2.0
            if avg_elbow < 140.0:
                return False, EXERCISE_START_INSTRUCTIONS["Push-ups"]
            return True, "Push-up starting posture confirmed."

        elif ex == "Biceps Curls (Dumbbell)":
            # Arms extended downward at sides: elbow angle >= 140 deg
            left_elbow_ang = _calculate_angle_2d(ls, le, lw)
            right_elbow_ang = _calculate_angle_2d(rs, re, rw)
            if left_elbow_ang < 140.0 and right_elbow_ang < 140.0:
                return False, EXERCISE_START_INSTRUCTIONS["Biceps Curls (Dumbbell)"]
            return True, "Biceps curl starting posture confirmed."

        elif ex == "Shoulder Press":
            # Dumbbells at shoulder level: elbows bent 60-140 deg, wrists near shoulder height
            left_elbow_ang = _calculate_angle_2d(ls, le, lw)
            right_elbow_ang = _calculate_angle_2d(rs, re, rw)
            wrist_shoulder_dy = abs(((lw.y + rw.y) / 2.0) - ((ls.y + rs.y) / 2.0))
            if wrist_shoulder_dy > 0.22 or (left_elbow_ang > 140 and right_elbow_ang > 140):
                return False, EXERCISE_START_INSTRUCTIONS["Shoulder Press"]
            return True, "Shoulder press starting posture confirmed (rack position at shoulder height)."

        elif ex == "Lunges":
            if len(landmarks) > 28:
                lk, rk = landmarks[25], landmarks[26]
                la, ra = landmarks[27], landmarks[28]
                left_knee_ang = _calculate_angle_2d(lh, lk, la)
                right_knee_ang = _calculate_angle_2d(rh, rk, ra)
                avg_knee = (left_knee_ang + right_knee_ang) / 2.0
                if avg_knee < 150.0:
                    return False, EXERCISE_START_INSTRUCTIONS["Lunges"]
            return True, "Lunge starting posture confirmed."

        return True, "Starting posture ready."

    def update(
        self,
        camera_connected: bool,
        pose_detected: bool,
        camera_status: str,
        pose_reliability: str,
        exercise_name: Optional[str],
        current_phase: str,
        valid_reps: int,
        rejected_attempts: int,
        workout_completed: bool,
        landmarks: Optional[List[Any]] = None,
        form_error: Optional[str] = None,
        now: Optional[float] = None,
        set_completed: bool = False
    ) -> Dict[str, Any]:
        """
        Updates coaching activity state based on incoming frame telemetry.
        Returns activity state, contextual guidance message, and debounced prompt trigger.
        """
        if now is None:
            now = time.time()

        if self.last_movement_time > now:
            self.last_movement_time = now
        if self.state_enter_time > now:
            self.state_enter_time = now

        total_reps = valid_reps + rejected_attempts
        rep_completed_just_now = False
        rep_rejected_just_now = False

        if valid_reps > self.last_rep_count:
            self.last_rep_completed_time = now
            self.last_rep_count = valid_reps
            rep_completed_just_now = True

        if rejected_attempts > self.last_rejected_count:
            self.last_rejected_count = rejected_attempts
            rep_rejected_just_now = True

        # Landmark movement estimation
        movement_magnitude = self._estimate_movement(landmarks)
        is_moving = movement_magnitude > 0.012

        if is_moving:
            self.last_movement_time = now

        time_since_last_movement = now - self.last_movement_time
        time_since_last_rep = (now - self.last_rep_completed_time) if self.last_rep_completed_time > 0 else 0.0

        prev_state = self.current_state

        # Genuinely blocked camera status check
        is_camera_blocked = (
            camera_status in ["NOT_READY", "BLOCKED"]
            or pose_reliability == "UNRELIABLE"
        )

        # -------------------------------------------------------------
        # State Determination
        # -------------------------------------------------------------
        if workout_completed:
            new_state = self.EXERCISE_COMPLETED
        elif set_completed:
            new_state = self.SET_COMPLETED
        elif not camera_connected or not pose_detected:
            new_state = self.WAITING_FOR_USER
        elif is_camera_blocked:
            new_state = self.CAMERA_NOT_READY
        elif not exercise_name or exercise_name.strip() == "":
            new_state = self.NO_EXERCISE_SELECTED
        else:
            # Camera is ready/warning and person is in frame
            if total_reps == 0:
                start_ok, _ = self.verify_starting_position(exercise_name, landmarks)
                active_phases = ["LOWERING", "BOTTOM", "RISING", "CURLING", "FLEXED", "PRESSING", "OVERHEAD", "down", "up", "moving"]

                if current_phase in active_phases:
                    new_state = self.MOVEMENT_IN_PROGRESS
                elif not start_ok and landmarks and len(landmarks) >= 25:
                    new_state = self.START_POSITION_REQUIRED
                else:
                    new_state = self.READY_TO_EXERCISE
            else:
                # Active workout in progress
                if rep_completed_just_now:
                    new_state = self.VALID_REPETITION_COMPLETED
                elif rep_rejected_just_now:
                    new_state = self.INVALID_REPETITION_REJECTED
                elif form_error and "warning" not in form_error.lower() and "camera" not in form_error.lower() and form_error.strip():
                    new_state = self.FORM_ERROR_DETECTED
                else:
                    active_phases = ["LOWERING", "BOTTOM", "RISING", "CURLING", "FLEXED", "PRESSING", "OVERHEAD", "down", "up", "moving"]
                    if current_phase in active_phases:
                        new_state = self.MOVEMENT_IN_PROGRESS
                    elif time_since_last_movement >= 4.0 or time_since_last_rep >= 4.0:
                        new_state = self.RESTING
                    else:
                        new_state = self.MOVEMENT_IN_PROGRESS

        if new_state != prev_state:
            self.current_state = new_state
            self.state_enter_time = now

        time_in_state = now - self.state_enter_time

        # -------------------------------------------------------------
        # Contextual Guidance Message Generation
        # -------------------------------------------------------------
        guidance_message = ""
        should_prompt = False

        if self.current_state == self.WAITING_FOR_USER:
            guidance_message = "Person not detected. Please stand in front of the camera."
            if time_in_state >= 1.0:
                should_prompt = True

        elif self.current_state == self.CAMERA_NOT_READY:
            guidance_message = "Your full body is not visible. Move the camera backwards or adjust its position."
            if time_in_state >= 2.0:
                should_prompt = True

        elif self.current_state == self.NO_EXERCISE_SELECTED:
            guidance_message = "Please select an exercise to begin your workout."
            should_prompt = True

        elif self.current_state in [self.WAITING_FOR_START_POSITION, self.START_POSITION_REQUIRED]:
            instruction = EXERCISE_START_INSTRUCTIONS.get(exercise_name or "", "Get into your starting position.")
            guidance_message = f"Starting position needed: {instruction}"
            if time_in_state >= 1.0:
                should_prompt = True

        elif self.current_state in [self.READY_TO_EXERCISE, self.READY_TO_BEGIN]:
            if time_in_state >= self.idle_start_timeout:
                guidance_message = "Your camera is ready. Stand in the correct starting position and begin your first repetition when ready."
                should_prompt = True
            else:
                guidance_message = "Good, you are in the starting position and ready. Begin your first repetition."
                should_prompt = True

        elif self.current_state in [self.EXERCISING, self.MOVEMENT_IN_PROGRESS]:
            guidance_message = f"Exercise in progress: {exercise_name} (Phase: {current_phase})."

        elif self.current_state == self.FORM_ERROR_DETECTED:
            guidance_message = f"Form issue detected: {form_error}. Adjust your technique."
            should_prompt = True

        elif self.current_state == self.VALID_REPETITION_COMPLETED:
            guidance_message = f"Repetition {valid_reps} completed with great form!"
            should_prompt = True

        elif self.current_state == self.INVALID_REPETITION_REJECTED:
            guidance_message = "Repetition rejected. Ensure full range of motion without form collapse."
            should_prompt = True

        elif self.current_state == self.RESTING:
            if time_since_last_rep >= self.inactivity_timeout:
                guidance_message = "You have been resting for a while. Begin your next repetition when ready."
                should_prompt = True
            elif time_since_last_rep >= self.rest_prompt_timeout:
                guidance_message = "Good rest interval. Catch your breath and begin your next repetition when ready."
                should_prompt = True
            else:
                rest_secs = int(time_since_last_rep)
                guidance_message = f"Resting between repetitions ({rest_secs}s elapsed)."

        elif self.current_state in [self.SET_COMPLETE, self.SET_COMPLETED]:
            guidance_message = "Set completed! Take a short rest before your next set."
            should_prompt = True

        elif self.current_state in [self.WORKOUT_COMPLETE, self.EXERCISE_COMPLETED]:
            guidance_message = "Workout target completed! Outstanding effort!"
            should_prompt = True

        # Debounce and Cooldown Enforcement
        dispatch_prompt = False
        if should_prompt and guidance_message:
            is_new_message = (guidance_message != self.last_prompt_message)
            cooldown_passed = (now - self.last_prompt_time) >= self.message_cooldown
            if is_new_message or cooldown_passed:
                self.last_prompt_time = now
                self.last_prompt_message = guidance_message
                dispatch_prompt = True

        return {
            "activity_state": self.current_state,
            "guidance_message": guidance_message,
            "is_idle": self.current_state in [
                self.WAITING_FOR_USER,
                self.CAMERA_NOT_READY,
                self.START_POSITION_REQUIRED,
                self.WAITING_FOR_START_POSITION,
                self.READY_TO_EXERCISE,
                self.READY_TO_BEGIN,
                self.RESTING
            ],
            "time_in_state": round(time_in_state, 1),
            "time_since_last_movement": round(time_since_last_movement, 1),
            "time_since_last_rep": round(time_since_last_rep, 1),
            "dispatch_prompt": dispatch_prompt
        }
