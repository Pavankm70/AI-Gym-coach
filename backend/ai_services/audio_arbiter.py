"""
Centralized Backend Audio Arbiter
================================
Enforces the single-speaker guarantee, 5-tier event priority hierarchy,
deduplication tracking, post-speech silence windows, and coaching intensity modes.

Priority Hierarchy:
- Priority 1: Critical safety and camera events (bypasses cooldown, interrupts lower priority)
- Priority 2: Invalid repetitions and critical technique corrections
- Priority 3: Exercise guidance and starting posture instructions
- Priority 4: Low-priority announcements (milestones, rep praise, idle reminders)
- Priority 5: Chatbot conversational audio
"""

import time
from typing import Dict, Any, Optional
from ai_services.speech_service import get_speech_service, SpeechService


class AudioPriority:
    CRITICAL_SAFETY = 1     # Camera disconnected, body out of frame, safety gate active
    CORRECTION = 2          # Rep rejected, critical form error, wrong start posture
    GUIDANCE = 3            # Movement phase cues, start position setup instructions
    ANNOUNCEMENT = 4        # Milestones (5, 10 reps), first rep praise, idle guidance
    CHATBOT = 5             # Chatbot conversational answers requested as audio


class CoachingIntensity:
    ERRORS_ONLY = "ERRORS_ONLY"   # Recommended: Only Priority 1 & 2
    STANDARD = "STANDARD"         # Priority 1, 2, 3, 4, 5
    MUTED = "MUTED"               # Zero voice output; visual HUD only


class BackendAudioArbiter:
    def __init__(
        self,
        silence_window_seconds: float = 5.0,
        speech_service: Optional[SpeechService] = None
    ):
        self.silence_window = silence_window_seconds
        self.speech_service = speech_service or get_speech_service()

        self.intensity_mode = CoachingIntensity.ERRORS_ONLY
        self.last_spoken_text = ""
        self.last_spoken_priority = 999
        self.last_speech_start_time = 0.0
        self.last_speech_end_time = 0.0
        self.is_speaking = False

        # Active tracking for deduplication across exercise phases
        self.last_active_error = ""
        self.last_active_phase = ""

    def set_coaching_intensity(self, mode: str):
        """Sets coaching intensity: ERRORS_ONLY, STANDARD, or MUTED."""
        if mode in [CoachingIntensity.ERRORS_ONLY, CoachingIntensity.STANDARD, CoachingIntensity.MUTED]:
            self.intensity_mode = mode

    def reset(self):
        """Resets the arbiter state."""
        self.last_spoken_text = ""
        self.last_spoken_priority = 999
        self.last_speech_start_time = 0.0
        self.last_speech_end_time = 0.0
        self.is_speaking = False
        self.last_active_error = ""
        self.last_active_phase = ""
        if hasattr(self.speech_service, "reset"):
            self.speech_service.reset()

    def mark_speech_started(self, text: str, priority: int, now: Optional[float] = None):
        """Called when an audio message begins playback."""
        if now is None:
            now = time.time()
        self.is_speaking = True
        self.last_spoken_text = text
        self.last_spoken_priority = priority
        self.last_speech_start_time = now

    def mark_speech_ended(self, now: Optional[float] = None):
        """Called when an audio message finishes playback, initiating silence window."""
        if now is None:
            now = time.time()
        self.is_speaking = False
        self.last_speech_end_time = now

    def is_eligible_by_mode(self, priority: int) -> bool:
        """Evaluates whether event priority is permitted under current coaching mode."""
        if self.intensity_mode == CoachingIntensity.MUTED:
            return False
        if self.intensity_mode == CoachingIntensity.ERRORS_ONLY:
            return priority in [AudioPriority.CRITICAL_SAFETY, AudioPriority.CORRECTION]
        # STANDARD allows all priorities 1 through 5
        return True

    def can_dispatch(
        self,
        text: str,
        priority: int,
        now: Optional[float] = None
    ) -> bool:
        """
        Determines whether an audio message can be dispatched under priority & cooldown rules.
        """
        if now is None:
            now = time.time()

        # 1. Coaching Intensity Mode Check
        if not self.is_eligible_by_mode(priority):
            return False

        clean_text = text.strip() if text else ""
        if not clean_text:
            return False

        # Update active speaking state based on timestamp
        if self.is_speaking and (now >= self.last_speech_end_time):
            self.is_speaking = False

        # 2. Priority 1 (Critical Safety & Camera Events)
        if priority == AudioPriority.CRITICAL_SAFETY:
            is_actively_speaking = self.is_speaking and (now < self.last_speech_end_time)

            # Don't repeat the exact same camera warning if currently actively speaking it
            if is_actively_speaking and clean_text == self.last_spoken_text:
                return False

            # If actively speaking an exercise correction (P2) or Gemini post-set review (P5),
            # do not interrupt unless genuine critical safety (e.g. camera disconnected)
            if is_actively_speaking:
                is_critical_disconnect = any(term in clean_text.lower() for term in ["disconnect", "lost connection", "critical safety"])
                if not is_critical_disconnect:
                    # Ordinary camera repositioning warning does not interrupt active exercise correction or Gemini review
                    if self.last_spoken_priority in [AudioPriority.CORRECTION, AudioPriority.CHATBOT]:
                        return False

            return True

        # 3. If currently speaking:
        # Lower priority messages cannot interrupt equal or higher priority messages.
        # Higher priority messages (e.g. P2 Form Correction) supersede lower priority (P4 Milestone).
        is_actively_speaking = self.is_speaking and (now < self.last_speech_end_time)
        if is_actively_speaking:
            if priority >= self.last_spoken_priority:
                return False
            # Higher priority message supersedes/interrupts lower priority message
            return True

        # 4. Enforce Post-Speech Silence Window (default 5.0 seconds after audio finishes)
        time_since_speech_end = now - self.last_speech_end_time
        if self.last_speech_end_time > 0 and time_since_speech_end < self.silence_window:
            return False

        # 5. Deduplication: Do not repeatedly speak the same text while active
        if clean_text == self.last_spoken_text and (now - self.last_speech_start_time < 12.0):
            return False

        return True

    def evaluate_and_synthesize(
        self,
        event: str,
        text: str,
        priority: int,
        now: Optional[float] = None,
        force_audio: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates message eligibility, applies arbitration rules, and synthesizes audio.
        Returns payload { event, text, audio, priority, can_interrupt } or None.
        """
        if now is None:
            now = time.time()

        if not force_audio and not self.can_dispatch(text, priority, now=now):
            return None

        # Synthesize audio via speech service
        audio_b64 = self.speech_service.synthesize_speech_base64(
            text=text,
            force=(priority == AudioPriority.CRITICAL_SAFETY or force_audio),
            now=now,
            is_muted=(self.intensity_mode == CoachingIntensity.MUTED)
        )

        # Update arbitration state
        can_interrupt = (self.is_speaking and priority < self.last_spoken_priority)
        self.mark_speech_started(text, priority, now=now)

        # Approximate speech duration for backend silence estimation
        approx_duration = 2.0 if priority >= AudioPriority.GUIDANCE else 1.2
        self.last_speech_end_time = now + approx_duration

        return {
            "event": event,
            "text": text,
            "audio": audio_b64,
            "priority": priority,
            "can_interrupt": can_interrupt,
            "intensity_mode": self.intensity_mode
        }

    def clear_obsolete_pending(self, activity_state: str):
        """Clears obsolete audio prompts when the user transitions between states."""
        if activity_state in ["EXERCISING", "MOVEMENT_IN_PROGRESS", "EXERCISE_IN_PROGRESS"]:
            if self.last_spoken_priority >= AudioPriority.GUIDANCE and self.last_spoken_text:
                if any(k in self.last_spoken_text.lower() for k in ["stand", "ready", "prepare", "position"]):
                    self.last_spoken_text = ""
        elif activity_state in ["RESTING", "SET_COMPLETED", "SET_COMPLETE"]:
            self.last_active_error = ""


# Global singleton arbiter instance
_AUDIO_ARBITER = BackendAudioArbiter()


def get_audio_arbiter() -> BackendAudioArbiter:
    return _AUDIO_ARBITER
