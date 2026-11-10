"""
Tool 2: Exercise Knowledge Tool (RAG)
=====================================
Retrieves evidence-based exercise science research, biomechanical analyses,
correct technique descriptions, and safety guidelines from the persistent
knowledge base.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from agent.rag.retrieval import retrieve_exercise_knowledge


class ExerciseKnowledgeInput(BaseModel):
    query: str = Field(description="The exercise science or biomechanics question (e.g. 'How deep should squats be?', 'Why is arching back bad during shoulder press?').")
    exercise: Optional[str] = Field(
        default=None,
        description="Optional exercise name to filter retrieval ('Squats', 'Push-ups', 'Biceps Curls (Dumbbell)', 'Shoulder Press', 'Lunges')."
    )


@tool(args_schema=ExerciseKnowledgeInput)
def exercise_knowledge_tool(query: str, exercise: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieves grounded exercise science literature, biomechanical standards,
    injury prevention guidance, and proper camera positioning protocols.
    Always includes source names and academic/trainer citations.
    """
    results = retrieve_exercise_knowledge(query=query, exercise_name=exercise, top_k=3)
    return results

