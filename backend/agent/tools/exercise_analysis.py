"""
Tool 1: Exercise Analysis Tool
==============================
Accesses current computer-vision session telemetry (joint angles, reps,
movement phase, detected form errors, camera status, pose reliability).

Strict Safety Rule:
- Does NOT analyze raw video using an LLM.
- Does NOT invent errors if the Python detector has not flagged them.
- If camera reliability is NOT_READY, warns the agent immediately.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Global active session state registry keyed by user_id
_ACTIVE_SESSIONS: Dict[int, Dict[str, Any]] = {}


def update_user_session_telemetry(user_id: int, telemetry: Dict[str, Any]):
    """Called by WebSocket frame processor to keep latest verified metrics."""
    _ACTIVE_SESSIONS[user_id] = telemetry.copy()


def get_user_session_telemetry(user_id: int) -> Optional[Dict[str, Any]]:
    return _ACTIVE_SESSIONS.get(user_id)


class ExerciseAnalysisInput(BaseModel):
    user_id: int = Field(description="The unique integer ID of the active user/athlete.")
    specific_metric: Optional[str] = Field(
        default=None,
        description="Optional metric to focus on (e.g. 'knee_angle', 'depth_status', 'back_arch_status', 'reps')."
    )


@tool(args_schema=ExerciseAnalysisInput)
def exercise_analysis_tool(user_id: int, specific_metric: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes verified computer-vision exercise session data for the current user.
    Retrieves current exercise name, repetition count, movement phase, measured
    joint angles, detected form errors, and camera/pose reliability state.
    """
    session = get_user_session_telemetry(user_id)

    if not session:
        return {
            "status": "NO_ACTIVE_SESSION",
            "message": f"No active exercise session found for User ID {user_id}. The user has not started streaming frames yet.",
            "data": None
        }

    camera_status = session.get("camera_status", "READY")
    pose_reliability = session.get("pose_reliability", "RELIABLE")
    exercise = session.get("exercise", "Unknown")
    reps = session.get("reps", 0)
    stage = session.get("stage", "ready")
    form_error = session.get("form_error")
    form_instruction = session.get("form_instruction")
    measurements = session.get("measurements", {})
    reliability_info = session.get("reliability", {})

    # SAFETY GATE CHECK:
    if camera_status == "NOT_READY" or pose_reliability == "UNRELIABLE":
        issues = reliability_info.get("issues", [])
        suggestions = reliability_info.get("suggestions", [])
        return {
            "status": "UNRELIABLE_CAMERA_VIEW",
            "exercise": exercise,
            "camera_status": camera_status,
            "pose_reliability": pose_reliability,
            "critical_notice": (
                "Camera framing or pose reliability is currently NOT READY. "
                "The computer-vision system cannot reliably evaluate exercise form. "
                "Do NOT diagnose form errors based on this unreliable frame."
            ),
            "camera_issues": issues,
            "positioning_instructions": suggestions
        }

    valid_reps = session.get("valid_reps", reps)
    rejected_attempts = session.get("rejected_attempts", 0)
    current_phase = session.get("current_phase", stage)
    rejection_reason = session.get("rejection_reason")
    activity_state = session.get("activity_state", "EXERCISE_IN_PROGRESS")
    guidance_message = session.get("guidance_message")

    # Extract relevant numerical metrics based on exercise
    analysis_summary = {
        "status": "SUCCESS",
        "exercise": exercise,
        "valid_repetitions": valid_reps,
        "reps_completed": valid_reps,
        "rejected_attempts": rejected_attempts,
        "movement_phase": current_phase,
        "activity_state": activity_state,
        "camera_status": camera_status,
        "pose_reliability": pose_reliability,
        "primary_form_error_detected": form_error,
        "rejection_reason": rejection_reason,
        "recommended_correction": form_instruction or guidance_message,
        "joint_angle_measurements": measurements
    }

    if specific_metric and specific_metric in measurements:
        analysis_summary["requested_metric"] = {
            "name": specific_metric,
            "value": measurements[specific_metric]
        }

    return analysis_summary

