"""
Natural Coaching Feedback Service
=================================
Generates concise, supportive natural coaching feedback based on
activity state, repetition validity, detected form errors, and workout milestones.
All speech output is governed by the centralized BackendAudioArbiter.
"""


import time
from typing import Dict, Any, Optional
from ai_services.audio_arbiter import get_audio_arbiter, AudioPriority, CoachingIntensity, BackendAudioArbiter


class NaturalFeedbackService:
    def __init__(
        self,
        cooldown: float = 3.5,
        arbiter: Optional[BackendAudioArbiter] = None,
        speech_service: Optional[Any] = None
    ):
        if arbiter is not None:
            self.arbiter = arbiter
        else:
            self.arbiter = BackendAudioArbiter(speech_service=speech_service, silence_window_seconds=cooldown)
            self.arbiter.set_coaching_intensity(CoachingIntensity.STANDARD)

        if speech_service:
            self.arbiter.speech_service = speech_service
        self.arbiter.silence_window = cooldown
        self.cooldown = cooldown
        self.last_valid_reps = -1
        self.last_rejected_attempts = 0
        self.last_form_error = ""

        # Continuous camera-problem episode tracking (Max 2 spoken warnings)
        self.max_camera_spoken_warnings: int = 2
        self.camera_warning_reminder_interval: float = 10.0
        self.camera_episode_active: bool = False
        self.camera_warnings_spoken_in_episode: int = 0
        self.last_camera_warning_time: float = 0.0
        self.consecutive_recovered_frames: int = 0
        self.recovery_frames_threshold: int = 5

    def reset(self, reset_camera_episode: bool = False):
        self.last_valid_reps = -1
        self.last_rejected_attempts = 0
        self.last_form_error = ""
        if reset_camera_episode:
            self.camera_episode_active = False
            self.camera_warnings_spoken_in_episode = 0
            self.consecutive_recovered_frames = 0
            self.last_camera_warning_time = 0.0
        self.arbiter.reset()

    def configure_camera_warning_limits(self, max_warnings: int = 2, reminder_interval: float = 10.0):
        """Allows configuring the maximum spoken warnings per camera episode and reminder interval."""
        self.max_camera_spoken_warnings = max(1, max_warnings)
        self.camera_warning_reminder_interval = max(1.0, reminder_interval)

    def set_coaching_intensity(self, mode: str):
        self.arbiter.set_coaching_intensity(mode)

    def evaluate_feedback(
        self,
        exercise: str,
        metrics: Dict[str, Any],
        camera_status: str,
        pose_reliability: str,
        now: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates whether an exercise or camera event warrants spoken coaching feedback,
        applying 5-tier priority arbitration, episode limits, and post-speech silence enforcement.
        """
        if now is None:
            now = time.time()

        # Update coaching intensity mode if requested in metrics
        if "coaching_intensity" in metrics:
            self.set_coaching_intensity(metrics["coaching_intensity"])

        form_error = metrics.get("form_error")
        form_instruction = metrics.get("form_instruction")
        rejection_reason = metrics.get("rejection_reason")
        valid_reps = metrics.get("valid_reps", metrics.get("reps", 0))
        rejected_attempts = metrics.get("rejected_attempts", 0)
        workout_completed = metrics.get("workout_completed", False)
        sets_completed = metrics.get("sets_completed", 0)
        guidance_message = metrics.get("guidance_message", "")
        dispatch_prompt = metrics.get("dispatch_prompt", False)

        # -------------------------------------------------------------
        # 1. PRIORITY 1: Critical Safety & Genuine Blocking Camera Failures
        # -------------------------------------------------------------
        # Only genuinely blocked camera views (NOT_READY / BLOCKED) trigger P1 audio.
        # CAMERA_WARNING never triggers spoken camera audio.
        # Rules:
        # - First genuine blocking event: speak one short camera warning.
        # - Second reminder permitted after meaningful interval (10.0s).
        # - Max 2 spoken warnings per continuous episode; then silent until sustained recovery.
        # - Does NOT reset when a new set begins.
        is_camera_blocked = (
            camera_status in ["NOT_READY", "BLOCKED"]
            or pose_reliability == "UNRELIABLE"
        )

        if is_camera_blocked:
            self.consecutive_recovered_frames = 0
            if not self.camera_episode_active:
                self.camera_episode_active = True
                self.camera_warnings_spoken_in_episode = 0

            # Rule: Max 2 spoken warnings per continuous episode
            if self.camera_warnings_spoken_in_episode >= self.max_camera_spoken_warnings:
                # Remain visually red and silent until recovery
                return None

            # Rule: Meaningful interval before 2nd reminder
            if self.camera_warnings_spoken_in_episode > 0:
                if (now - self.last_camera_warning_time) < self.camera_warning_reminder_interval:
                    return None

            rel_info = metrics.get("reliability", {})
            sugg = rel_info.get("suggestions", [])
            msg = sugg[0] if sugg else (guidance_message or "Camera view is blocked. Please move back and make sure your required body joints are visible.")
            
            cue = self.arbiter.evaluate_and_synthesize(
                event="camera_warning",
                text=msg,
                priority=AudioPriority.CRITICAL_SAFETY,
                now=now
            )
            if cue is not None:
                self.camera_warnings_spoken_in_episode += 1
                self.last_camera_warning_time = now
            return cue
        else:
            # Camera is READY or in non-blocking WARNING:
            # Require sustained recovery (5 frames) before resetting episode counter
            if self.camera_episode_active:
                self.consecutive_recovered_frames += 1
                if self.consecutive_recovered_frames >= self.recovery_frames_threshold:
                    self.camera_episode_active = False
                    self.camera_warnings_spoken_in_episode = 0
                    self.consecutive_recovered_frames = 0

        # Update arbiter with current activity state to clear obsolete messages
        activity_state = metrics.get("activity_state", "")
        if hasattr(self.arbiter, "clear_obsolete_pending"):
            self.arbiter.clear_obsolete_pending(activity_state)

        # -------------------------------------------------------------
        # 2. PRIORITY 2: Repetition Rejections & Important Form Errors
        # -------------------------------------------------------------
        # 2a. Repetition Rejected Attempt
        if rejected_attempts > self.last_rejected_attempts and rejection_reason:
            self.last_rejected_attempts = rejected_attempts
            msg = f"Repetition not counted: {rejection_reason}"
            return self.arbiter.evaluate_and_synthesize(
                event="rep_rejected",
                text=msg,
                priority=AudioPriority.CORRECTION,
                now=now
            )

        # 2b. In-flight Form Error Correction
        if form_error and form_instruction and "camera" not in form_error.lower() and "pose warning" not in form_error.lower():
            is_new_error = (form_error != self.last_form_error)
            if is_new_error:
                self.last_form_error = form_error
                return self.arbiter.evaluate_and_synthesize(
                    event="form_correction",
                    text=form_instruction,
                    priority=AudioPriority.CORRECTION,
                    now=now
                )
        else:
            # Form error cleared
            self.last_form_error = ""

        # 2c. Workout Target Completed
        if workout_completed and self.arbiter.last_spoken_text != "workout_completed":
            msg = f"Fantastic workout! You have completed all {sets_completed} target sets."
            return self.arbiter.evaluate_and_synthesize(
                event="workout_completed",
                text=msg,
                priority=AudioPriority.CORRECTION,
                now=now,
                force_audio=True
            )

        # -------------------------------------------------------------
        # 3. PRIORITY 3: Exercise Guidance (Start Posture & Readiness)
        # -------------------------------------------------------------
        if activity_state in ["START_POSITION_REQUIRED", "WAITING_FOR_START_POSITION"] and guidance_message:
            return self.arbiter.evaluate_and_synthesize(
                event="start_position_required",
                text=guidance_message,
                priority=AudioPriority.GUIDANCE,
                now=now
            )

        if activity_state in ["READY_TO_EXERCISE", "READY_TO_BEGIN"] and dispatch_prompt:
            return self.arbiter.evaluate_and_synthesize(
                event="ready_to_exercise",
                text=guidance_message or "Good, you are ready. Start your first repetition.",
                priority=AudioPriority.GUIDANCE,
                now=now
            )

        # -------------------------------------------------------------
        # 4. PRIORITY 4: Low-Priority Announcements & Praises
        # -------------------------------------------------------------
        # 4a. First Valid Repetition Praise
        if valid_reps == 1 and self.last_valid_reps <= 0:
            self.last_valid_reps = 1
            msg = "First repetition verified! Excellent control, keep this exact tempo."
            return self.arbiter.evaluate_and_synthesize(
                event="first_rep",
                text=msg,
                priority=AudioPriority.ANNOUNCEMENT,
                now=now
            )

        # 4b. Milestone Announcements (Rep 5, Rep 10)
        if valid_reps > 1 and valid_reps != self.last_valid_reps and valid_reps % 5 == 0:
            self.last_valid_reps = valid_reps
            msg = f"{valid_reps} clean repetitions completed! Great posture."
            return self.arbiter.evaluate_and_synthesize(
                event="rep_milestone",
                text=msg,
                priority=AudioPriority.ANNOUNCEMENT,
                now=now
            )

        # 4c. Idle Guidance Dispatch (Scenario C / Scenario E)
        if dispatch_prompt and guidance_message:
            return self.arbiter.evaluate_and_synthesize(
                event="idle_guidance",
                text=guidance_message,
                priority=AudioPriority.ANNOUNCEMENT,
                now=now
            )

        return None


# Global singleton instance
_FEEDBACK_SERVICE = NaturalFeedbackService()


def get_feedback_service() -> NaturalFeedbackService:
    return _FEEDBACK_SERVICE
