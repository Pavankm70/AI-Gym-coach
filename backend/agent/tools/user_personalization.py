"""
Tool 3: User Personalization Tool
=================================
Retrieves user fitness profile, experience level, stated fitness goals,
limitations/precautions, and feedback preferences from the SQLite database.
Enables tailored, safe coaching advice without medical claims.
"""

import sys
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Ensure Main App persistence is importable
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
MAIN_APP_DIR = os.path.join(ROOT_DIR, "Main App")
if MAIN_APP_DIR not in sys.path:
    sys.path.insert(0, MAIN_APP_DIR)

from services.persistence.exercise_repository import get_user_by_id


class UserPersonalizationInput(BaseModel):
    user_id: int = Field(description="The unique integer ID of the athlete to personalize guidance for.")


@tool(args_schema=UserPersonalizationInput)
def user_personalization_tool(user_id: int) -> Dict[str, Any]:
    """
    Retrieves the authenticated user's fitness profile, including experience
    level (Beginner/Intermediate/Advanced), primary goals, feedback style,
    and physical limitations or injury precautions.
    """
    user_row = get_user_by_id(user_id)

    if not user_row:
        return {
            "status": "USER_NOT_FOUND",
            "message": f"No profile found for user_id={user_id}. Using standard beginner-friendly defaults.",
            "profile": {
                "user_id": user_id,
                "username": "Athlete",
                "fitness_level": "Beginner",
                "goals": "General Fitness",
                "preferred_feedback_style": "Encouraging",
                "experience": "Beginner (0-1 years)",
                "workout_frequency": "3 days/week",
                "limitations": "None"
            }
        }

    user_dict = dict(user_row)

    fitness_level = user_dict.get("fitness_level") or "Beginner"
    goals = user_dict.get("goals") or "General Fitness"
    feedback_style = user_dict.get("preferred_feedback_style") or "Encouraging"
    experience = user_dict.get("experience") or "Beginner (0-1 years)"
    frequency = user_dict.get("workout_frequency") or "3 days/week"
    limitations = user_dict.get("limitations") or "None"

    # Contextual tailoring suggestions for LLM
    coaching_recommendations = []
    if fitness_level.lower() == "beginner":
        coaching_recommendations.append("Use simple, encouraging cues. Avoid overly dense biomechanical jargon.")
        coaching_recommendations.append("Prioritize fundamental joint alignment and safety over maximum depth or speed.")
    elif fitness_level.lower() == "intermediate":
        coaching_recommendations.append("Provide concise technical feedback with specific numerical joint angle targets.")
    elif fitness_level.lower() == "advanced":
        coaching_recommendations.append("Provide high-level athletic cues focusing on power output, bar path, and kinetic efficiency.")

    if limitations.lower() != "none":
        coaching_recommendations.append(f"Safety Alert: User reports '{limitations}'. Avoid forcing extreme joint flexion that might irritate this condition.")

    return {
        "status": "SUCCESS",
        "profile": {
            "user_id": user_dict["id"],
            "username": user_dict["username"],
            "fitness_level": fitness_level,
            "goals": goals,
            "preferred_feedback_style": feedback_style,
            "experience": experience,
            "workout_frequency": frequency,
            "limitations": limitations
        },
        "tailoring_guidelines": coaching_recommendations
    }

