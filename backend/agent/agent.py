"""
Single LangChain Agent with Exactly Three Tools
===============================================
Intelligent Exercise Coaching Agent powered by LangChain and Gemini.

Architecture:
- ONE Agent, EXACTLY THREE TOOLS:
  1. Exercise Analysis Tool (verified CV telemetry)
  2. Exercise Knowledge Tool (RAG with persistent FAISS & citations)
  3. User Personalization Tool (SQLite profile & goals)
- Never multi-agent.
- Respects computer-vision reliability safety gate.
- Does not invent joint measurements or form errors.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool

from agent.tools.exercise_analysis import exercise_analysis_tool, get_user_session_telemetry
from agent.tools.exercise_knowledge import exercise_knowledge_tool
from agent.tools.user_personalization import user_personalization_tool

# Ensure environment variables are loaded
load_dotenv()

SYSTEM_PROMPT = """You are Apna AI Gym Coach, an intelligent, research-grounded exercise coach and computer-vision assistant.

You have access to EXACTLY THREE specialized tools:
1. 'exercise_analysis_tool': Use this to access the athlete's current exercise session metrics, repetitions, joint angles, movement phases, detected form errors, and camera reliability status.
2. 'exercise_knowledge_tool': Use this to retrieve evidence-based exercise science literature, biomechanical principles, common error causes, corrections, and camera framing guidelines (RAG).
3. 'user_personalization_tool': Use this to retrieve the athlete's fitness level (Beginner/Intermediate/Advanced), primary goals, feedback preferences, and injury limitations from their database profile.

CRITICAL COACHING RULES:
1. NEVER invent or guess joint angles or form errors that have not been detected by the computer-vision system.
2. If 'exercise_analysis_tool' indicates camera_status is 'NOT_READY' or pose is 'UNRELIABLE', you MUST NOT provide exercise form judgements. Instead, communicate the camera positioning instructions directly to help the athlete set up their camera.
3. Tailor explanations according to the user's fitness level and goals retrieved from 'user_personalization_tool'. If they are a beginner, keep cues intuitive, supportive, and safe. If advanced, provide biomechanical specifics.
4. When explaining exercise techniques, cite the source or standard retrieved from 'exercise_knowledge_tool'.
5. Always maintain an encouraging, professional, and safety-first coaching tone. Do not provide medical diagnoses.
"""

TOOLS: List[BaseTool] = [
    exercise_analysis_tool,
    exercise_knowledge_tool,
    user_personalization_tool
]


class GymCoachAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.tools = TOOLS
        self.agent_executor = None

        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                from langgraph.prebuilt import create_react_agent

                llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.2
                )
                self.agent_executor = create_react_agent(
                    model=llm,
                    tools=self.tools,
                    prompt=SYSTEM_PROMPT
                )
                print(f"[GymCoachAgent] LangChain ReAct agent initialized with Gemini model '{self.model_name}'.")
            except Exception as e:
                print(f"[GymCoachAgent] Notice: Could not initialize Gemini LLM agent ({e}). Using deterministic tool orchestrator.")
                self.agent_executor = None
        else:
            print("[GymCoachAgent] No active GEMINI_API_KEY found in environment. Using deterministic tool orchestrator fallback.")

    def run(
        self,
        user_message: str,
        user_id: int = 1,
        chat_history: Optional[List[Dict[str, str]]] = None,
        exercise_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the agent for a user inquiry.
        Maintains conversational context and invokes the required tools.
        """
        tools_called = []

        # 1. If Gemini Agent Executor is active, run it
        if self.agent_executor is not None and not getattr(self, "quota_exhausted", False):
            try:
                messages: List[BaseMessage] = []
                if chat_history:
                    for msg in chat_history[-6:]:  # Keep recent context
                        if msg.get("role") == "user":
                            messages.append(HumanMessage(content=msg.get("content", "")))
                        elif msg.get("role") == "assistant":
                            messages.append(AIMessage(content=msg.get("content", "")))

                # Add context prefix to user inquiry
                prompt_text = f"[Athlete User ID: {user_id}, Current Exercise: {exercise_name or 'Unspecified'}]\n{user_message}"
                messages.append(HumanMessage(content=prompt_text))

                result = self.agent_executor.invoke({"messages": messages})
                last_msg = result["messages"][-1]
                response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

                # Normalize response_text to string if it is a list of blocks/dicts
                if isinstance(response_text, list):
                    text_parts = []
                    for part in response_text:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        else:
                            text_parts.append(str(part))
                    response_text = "\n".join(text_parts)
                elif not isinstance(response_text, str):
                    response_text = str(response_text)

                # Inspect message history for tool calls made
                for m in result.get("messages", []):
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        for tc in m.tool_calls:
                            tools_called.append(tc.get("name", "unknown_tool"))

                return {
                    "response": response_text,
                    "tools_called": list(set(tools_called)),
                    "model_used": self.model_name,
                    "status": "success"
                }
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    self.quota_exhausted = True
                print(f"[GymCoachAgent] Error in LLM execution: {e}. Falling back to deterministic tool orchestrator.")

        # 2. Deterministic Tool Orchestrator (Fallback when offline or key not provided)
        return self._run_deterministic_orchestrator(user_message, user_id, exercise_name)

    def _run_deterministic_orchestrator(
        self,
        user_message: str,
        user_id: int,
        exercise_name: Optional[str]
    ) -> Dict[str, Any]:
        """
        Deterministic tool-calling pipeline that inspects the query intent,
        dispatches to the three tools, and synthesizes a grounded coaching response.
        """
        lower_q = user_message.lower()
        tools_called = []
        response_sections = []

        # A. Does query relate to user profile, goals, or experience?
        needs_profile = any(w in lower_q for w in [
            "goal", "profile", "level", "beginner", "experience", "limitation", "injury", "plan", "routine", "my"
        ])
        profile_data = None
        if needs_profile or "suggest" in lower_q or "tailor" in lower_q:
            tools_called.append("user_personalization_tool")
            res_p = user_personalization_tool.invoke({"user_id": user_id})
            profile_data = res_p.get("profile", {})
            tailoring = res_p.get("tailoring_guidelines", [])

        # B. Does query relate to recent workout session, form error, or reps?
        needs_cv_analysis = any(w in lower_q for w in [
            "rep", "mistake", "error", "form", "performance", "recent", "last", "angle",
            "knee", "arch", "hip", "sway", "deep", "depth", "posture", "camera", "frame",
            "right now", "currently", "session", "doing", "how am i"
        ])
        cv_data = None
        if needs_cv_analysis or "why" in lower_q or "how did" in lower_q or "how is" in lower_q:
            tools_called.append("exercise_analysis_tool")
            cv_data = exercise_analysis_tool.invoke({"user_id": user_id})

        # C. Does query ask for exercise knowledge, technique, or science?
        needs_rag = any(w in lower_q for w in [
            "how to", "technique", "improve", "why", "correct", "parallel", "biomechanic", "science", "rule", "posture", "camera"
        ]) or len(tools_called) == 0
        rag_data = None
        if needs_rag or not cv_data:
            tools_called.append("exercise_knowledge_tool")
            rag_data = exercise_knowledge_tool.invoke({
                "query": user_message,
                "exercise": exercise_name
            })

        # --- Synthesize Response ---
        # 1. Handle Unreliable Camera State if flagged
        if cv_data and cv_data.get("status") == "UNRELIABLE_CAMERA_VIEW":
            issues = cv_data.get("camera_issues", [])
            sugg = cv_data.get("positioning_instructions", [])
            return {
                "response": (
                    f"⚠️ **Camera Positioning Alert:**\n\n"
                    f"{cv_data.get('critical_notice')}\n\n"
                    f"**Detected Framing Issues:**\n"
                    + "\n".join([f"- {iss}" for iss in issues]) + "\n\n"
                    f"**Suggested Actions:**\n"
                    + "\n".join([f"- {s}" for s in sugg])
                ),
                "tools_called": tools_called,
                "model_used": "deterministic_safety_gate",
                "status": "success"
            }

        # 2. Performance Feedback & Joint Angles
        if cv_data and cv_data.get("status") == "SUCCESS":
            ex = cv_data.get("exercise", exercise_name or "Exercise")
            reps = cv_data.get("reps_completed", 0)
            err = cv_data.get("primary_form_error_detected")
            corr = cv_data.get("recommended_correction")
            angles = cv_data.get("joint_angle_measurements", {})

            lead = f"You completed **{reps} reps** of **{ex}**."
            if err:
                lead += f"\n\n🔍 **Detected Issue:** {err}\n💡 **Direct Cue:** {corr or 'Focus on maintaining controlled posture.'}"
            else:
                lead += "\n\n✅ **Great Form!** No major biomechanical errors were detected during your repetitions."

            if angles:
                angle_strs = [f"{k.replace('_', ' ').title()}: {v}°" for k, v in angles.items() if isinstance(v, (int, float))]
                if angle_strs:
                    lead += f"\n\n📐 **Measured Joint Kinematics:** " + ", ".join(angle_strs)

            response_sections.append(lead)

        # 3. Evidence-Based Knowledge from RAG
        if rag_data and rag_data.get("passages"):
            best_passage = rag_data["passages"][0]
            knowledge_text = (
                f"📚 **Exercise Science Context ({best_passage['section']}):**\n"
                f"{best_passage['content']}\n\n"
                f"*Source: {best_passage['source']} ({best_passage['citation']})*"
            )
            response_sections.append(knowledge_text)

        # 4. Personalized Guidance
        if profile_data:
            level = profile_data.get("fitness_level", "Beginner")
            goals = profile_data.get("goals", "General Fitness")
            personal_cue = f"🎯 **Personalized Note ({level} • Goal: {goals}):** "
            if level.lower() == "beginner":
                personal_cue += "Take your time on every repetition, focusing on smooth breathing and proper joint alignment before adding volume."
            else:
                personal_cue += "Maintain tension throughout both the concentric and eccentric phases to maximize motor unit recruitment."
            response_sections.append(personal_cue)

        final_response = "\n\n".join(response_sections)
        if not final_response.strip():
            final_response = "I am ready to coach your workout! Ask me about your current form, repetition counts, exercise biomechanics, or training tips."

        return {
            "response": final_response,
            "tools_called": list(set(tools_called)),
            "model_used": "deterministic_coach_orchestrator",
            "status": "success"
        }


# Singleton agent instance
_GLOBAL_AGENT: Optional[GymCoachAgent] = None


def get_gym_coach_agent() -> GymCoachAgent:
    global _GLOBAL_AGENT
    if _GLOBAL_AGENT is None:
        _GLOBAL_AGENT = GymCoachAgent()
    return _GLOBAL_AGENT
