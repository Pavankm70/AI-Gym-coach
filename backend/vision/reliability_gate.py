"""
Computer-Vision & AI Safety Gatekeeper
======================================
Enforces the research requirement:
Video Frame -> MediaPipe Detection -> Camera/Pose Reliability Gate
-> If Reliable (CAMERA_READY or CAMERA_WARNING):
   Continue to exercise analysis & form error evaluation
-> If Genuinely Blocked (CAMERA_BLOCKED):
   STOP exercise evaluation, show positioning instructions without false form errors.

Provides three primary outcomes:
1. CAMERA_READY: Pose information is sufficiently reliable.
2. CAMERA_WARNING: Minor tracking or framing issue, but exercise monitoring continues.
3. CAMERA_BLOCKED: Essential pose information is unusable; exercise measurements pause.

Implements hysteresis for frame confirmation and failure filtering.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from vision.camera_validation import (
    evaluate_pose_reliability,
    GATE_READY,
    GATE_WARNING,
    GATE_BLOCKED
)


class ReliabilityGate:
    # 3 Primary Gate Outcomes
    CAMERA_READY = "CAMERA_READY"
    CAMERA_WARNING = "CAMERA_WARNING"
    CAMERA_BLOCKED = "CAMERA_BLOCKED"

    # Internal Diagnostic Readiness States
    CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
    CAMERA_CONNECTING = "CAMERA_CONNECTING"
    PERSON_NOT_DETECTED = "PERSON_NOT_DETECTED"
    CAMERA_POSITION_REQUIRED = "CAMERA_POSITION_REQUIRED"
    LANDMARKS_UNRELIABLE = "LANDMARKS_UNRELIABLE"
    EXERCISE_ANALYSIS_ACTIVE = "EXERCISE_ANALYSIS_ACTIVE"

    def __init__(self, valid_confirm_frames: int = 3, invalid_reject_frames: int = 5):
        self.prev_landmarks = None
        self.last_reliability_result = None
        self.valid_confirm_frames = valid_confirm_frames
        self.invalid_reject_frames = invalid_reject_frames
        self._consecutive_valid: int = 0
        self._consecutive_invalid: int = 0
        self._current_readiness_state: str = self.CAMERA_CONNECTING
        self._current_gate_outcome: str = self.CAMERA_BLOCKED

    def reset(self):
        """Reset hysteresis state tracking."""
        self.prev_landmarks = None
        self.last_reliability_result = None
        self._consecutive_valid = 0
        self._consecutive_invalid = 0
        self._current_readiness_state = self.CAMERA_CONNECTING
        self._current_gate_outcome = self.CAMERA_BLOCKED

    def evaluate_and_filter(
        self,
        frame: Optional[np.ndarray],
        landmarks: Optional[List[Any]],
        exercise_name: str,
        detector: Any
    ) -> Dict[str, Any]:
        """
        Executes the validation gate with 3-outcome gating, 7-state internal tracking,
        and consecutive-frame hysteresis.

        Returns a dictionary containing:
        - 'is_gated': bool (True ONLY if CAMERA_BLOCKED)
        - 'reliability': Dict of reliability assessment results including camera_readiness_state and gate_outcome
        - 'metrics': Exercise metrics
        """
        if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
            instant_state = self.CAMERA_DISCONNECTED
            gate_outcome = self.CAMERA_BLOCKED
            reliability = {
                "camera_status": "NOT_READY",
                "pose_reliability": "UNRELIABLE",
                "gate_outcome": self.CAMERA_BLOCKED,
                "person_detected": False,
                "required_landmarks_visible": False,
                "landmark_stability": "UNSTABLE",
                "camera_view": "UNSUITABLE",
                "issues": ["Camera is disconnected or frame is unreadable."],
                "suggestions": ["Ensure camera is connected and grant browser permissions."],
                "quality_metrics": {}
            }
        else:
            reliability = evaluate_pose_reliability(
                frame=frame,
                landmarks=landmarks,
                exercise_name=exercise_name,
                prev_landmarks=self.prev_landmarks
            )
            gate_outcome = reliability.get("gate_outcome", self.CAMERA_READY)

            if not reliability["person_detected"]:
                instant_state = self.PERSON_NOT_DETECTED
            elif gate_outcome == GATE_BLOCKED:
                if not reliability["required_landmarks_visible"] or any("border" in iss.lower() or "cut off" in iss.lower() or "framing" in iss.lower() for iss in reliability.get("issues", [])):
                    instant_state = self.CAMERA_POSITION_REQUIRED
                else:
                    instant_state = self.LANDMARKS_UNRELIABLE
            elif gate_outcome == GATE_WARNING:
                instant_state = self.CAMERA_WARNING
            else:
                instant_state = self.CAMERA_READY

        # Hysteresis Filter: Consecutive-frame confirmation
        if instant_state in [self.CAMERA_READY, self.CAMERA_WARNING]:
            self._consecutive_valid += 1
            self._consecutive_invalid = 0
            if self._consecutive_valid >= self.valid_confirm_frames:
                is_active = False
                if detector:
                    stage = getattr(detector, "stage", "ready")
                    reps = getattr(detector, "reps", 0)
                    if reps > 0 or stage in ["down", "up", "moving", "bottom", "curling", "pressing", "flexed", "overhead"]:
                        is_active = True
                self._current_readiness_state = self.EXERCISE_ANALYSIS_ACTIVE if is_active else self.CAMERA_READY
                self._current_gate_outcome = self.CAMERA_READY if instant_state == self.CAMERA_READY else self.CAMERA_WARNING
        else:
            self._consecutive_invalid += 1
            self._consecutive_valid = 0
            # If previously confirmed READY or ACTIVE, require invalid_reject_frames to degrade to BLOCKED
            if self._current_readiness_state in [self.CAMERA_READY, self.EXERCISE_ANALYSIS_ACTIVE, self.CAMERA_WARNING]:
                if instant_state in [self.CAMERA_DISCONNECTED, self.PERSON_NOT_DETECTED] or self._consecutive_invalid >= self.invalid_reject_frames:
                    self._current_readiness_state = instant_state
                    self._current_gate_outcome = self.CAMERA_BLOCKED
            else:
                self._current_readiness_state = instant_state
                self._current_gate_outcome = self.CAMERA_BLOCKED

        # Augment reliability dictionary
        reliability["camera_readiness_state"] = self._current_readiness_state
        reliability["gate_outcome"] = self._current_gate_outcome
        reliability["consecutive_valid_frames"] = self._consecutive_valid
        reliability["consecutive_invalid_frames"] = self._consecutive_invalid

        self.last_reliability_result = reliability

        if landmarks and len(landmarks) >= 25:
            self.prev_landmarks = landmarks

        # GATE DECISION:
        # Pause exercise evaluation ONLY when camera is genuinely blocked.
        # Hysteresis protects confirmed active/ready state from transient 1-frame spikes.
        is_blocked = False
        if instant_state in [self.CAMERA_DISCONNECTED, self.PERSON_NOT_DETECTED]:
            is_blocked = True
        elif self._current_readiness_state in [self.CAMERA_READY, self.EXERCISE_ANALYSIS_ACTIVE, self.CAMERA_WARNING]:
            is_blocked = (self._consecutive_invalid >= self.invalid_reject_frames)
        else:
            is_blocked = (reliability["camera_status"] == "NOT_READY" or reliability["pose_reliability"] == "UNRELIABLE")

        if is_blocked:
            instruction = (
                reliability["suggestions"][0]
                if reliability["suggestions"]
                else "Adjust your position within the camera view."
            )
            issue = (
                reliability["issues"][0]
                if reliability["issues"]
                else "Camera view or detected pose is unreliable."
            )

            reps = getattr(detector, "reps", 0) if detector else 0
            stage = getattr(detector, "stage", "ready") if detector else "ready"

            return {
                "is_gated": True,
                "reliability": reliability,
                "metrics": {
                    "pose_detected": reliability["person_detected"],
                    "reps": reps,
                    "stage": stage,
                    "form_error": f"Camera / Pose Warning: {issue}",
                    "form_instruction": instruction,
                    "reliability": reliability
                }
            }

        # Camera is CAMERA_READY or CAMERA_WARNING:
        # Minor camera warning does NOT stop repetition counting or normal form feedback!
        metrics = detector.process(landmarks) if detector else {}
        metrics["pose_detected"] = True
        metrics["stage"] = getattr(detector, "stage", "ready")
        metrics["reliability"] = reliability

        return {
            "is_gated": False,
            "reliability": reliability,
            "metrics": metrics
        }
