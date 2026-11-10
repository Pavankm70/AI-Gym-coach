# AI Coaching Architecture: Single LangChain Agent with Exactly Three Tools
**System:** AI Gym Coach  
**Architecture Specification:** Single-Agent Tool-Calling Architecture with Retrieval-Augmented Generation (RAG)  
**LLM Engine:** Google Gemini (via `langchain-google-genai` / LangGraph ReAct)  

---

## 1. Architectural Philosophy: Why a Single Agent?

A frequent anti-pattern in modern LLM applications is constructing overly complex multi-agent graphs (e.g. separate "Analysis Agent", "Science Agent", "Personality Agent") for tasks that are inherently synchronous and query-focused. 

In an intelligent exercise coaching system:
1. **Low Latency is Paramount:** The user requires fast, cohesive responses during workout sessions. Routing queries across multiple autonomous agents introduces conversational latency and state synchronization overhead.
2. **Deterministic Tool Separation:** Clear boundary separation is achieved not by spawning independent LLM personalities, but by equipping **ONE intelligent agent** with **EXACTLY THREE specialized, well-scoped tools**:
   - Tool 1: Accesses ground-truth computer-vision kinematics.
   - Tool 2: Accesses academic sports science literature (RAG).
   - Tool 3: Accesses authenticated user preferences and limitations.
3. **Unified Reasoning:** A single agent possesses the complete conversational context, allowing it to dynamically decide whether a query requires only sports science (Tool 2), an analysis of their latest squat set (Tool 1 + Tool 2), or a personalized routine adjustment based on knee sensitivity (Tool 3 + Tool 2).

```
                      +-----------------------------+
                      |   Athlete / Chatbot UI      |
                      +--------------+--------------+
                                     |
                                     ▼
                      +-----------------------------+
                      |     FastAPI /api/chat       |
                      +--------------+--------------+
                                     |
                                     ▼
                      +-----------------------------+
                      |   Single LangChain Agent    |
                      |    (Gemini 2.5 Flash)       |
                      +--------------+--------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
         ▼                           ▼                           ▼
+------------------+       +--------------------+       +-------------------+
|     TOOL 1:      |       |      TOOL 2:       |       |      TOOL 3:      |
| Exercise Analysis|       | Exercise Knowledge |       |UserPersonalization|
|   (CV Session)   |       |       (RAG)        |       | (SQLite Profile)  |
+--------+---------+       +---------+----------+       +---------+---------+
         |                           |                            |
         ▼                           ▼                            ▼
 Live Joint Angles,        Persistent FAISS Index,       Fitness Level,
 Reps, Phase, Camera       NSCA / ACSM Biomechanics,     Goals, Injury
 Reliability Gate          Source Citations              Limitations
```

---

## 2. Detailed Responsibilities of the Three Tools

### Tool 1: Exercise Analysis Tool (`exercise_analysis_tool`)
- **Purpose:** Bridges the deterministic computer-vision pipeline to the agent.
- **Data Accessed:**
  - Active exercise name (`Squats`, `Push-ups`, `Biceps Curls`, `Shoulder Press`, `Lunges`).
  - Total reps completed and current set progress.
  - Active movement phase (`ready`, `up`, `down`, `pressing`).
  - Numerical joint angles (`knee_angle`, `elbow_angle`, `back_angle`, etc.).
  - Deterministic form issues detected by Python (e.g. *"The user's squat is not deep enough"*).
  - Pre-detector camera status (`READY` vs `NOT_READY`) and pose reliability.
- **Strict Guardrails:**
  - The tool does **NOT** accept raw video frames or ask the LLM to inspect pixels.
  - The tool does **NOT** invent form errors; if Python did not flag an issue, the tool returns `form_error: null`.
  - If camera reliability is `NOT_READY`, the tool immediately blocks form diagnosis and instructs the agent to guide camera repositioning.

### Tool 2: Exercise Knowledge Tool — RAG (`exercise_knowledge_tool`)
- **Purpose:** Provides evidence-based sports science literature, biomechanical explanations, and proper camera positioning protocols.
- **Implementation:**
  - **Knowledge Base:** Curated markdown guides authored from NSCA, ACSM, and peer-reviewed kinesiology papers.
  - **Chunking:** Semantic paragraph chunking with citation and exercise tags.
  - **Vector Store:** Persistent FAISS index on disk (`backend/agent/rag/faiss_index/`).
  - **Retrieval:** Inner-product cosine similarity search with top-k ranking and exercise filtering.
  - **Citation Guarantee:** Every retrieved passage contains the authoritative source (e.g., *Schoenfeld 2010*, *Saeterbakken 2013*, *Contreras 2012*).
- **Critical Distinction:**
  - Deterministic angle thresholds (e.g. `DOWN_THRESHOLD = 95°`) live in Python detector code.
  - RAG stores the *explanation* (e.g., why sub-parallel squatting maximizes gluteus maximus recruitment while minimizing patellar shear).

### Tool 3: User Personalization Tool (`user_personalization_tool`)
- **Purpose:** Retrieves the current authenticated athlete's fitness profile.
- **Data Accessed:**
  - `fitness_level`: Beginner, Intermediate, or Advanced.
  - `goals`: General Fitness, Strength, Hypertrophy, Mobility, Fat Loss.
  - `preferred_feedback_style`: Encouraging, Direct & Technical, Detailed Biomechanical.
  - `experience`: Training age.
  - `limitations`: Physical precautions (e.g., *"Lower back tightness"*, *"Knee meniscus sensitivity"*).
- **Coaching Tailoring:**
  - For Beginners: Generates intuitive, non-jargon cues focusing on joint safety.
  - For Advanced: Generates precise kinematic feedback focusing on bar path and kinetic chain efficiency.
  - For Stated Limitations: Emits safety alerts preventing extreme joint flexion.

---

## 3. Communication & Data Flow

1. **User asks a question** in the React chat drawer (e.g., *"Why was my last shoulder press flagged for excessive arch?"*).
2. **FastAPI `/api/chat`** forwards the prompt, user ID, and active exercise to `GymCoachAgent`.
3. **Agent Decides Tool Invocations:**
   - Calls `exercise_analysis_tool(user_id=1)` -> Retrieves verified metrics: `back_arch_status: "Excessive Arch"`, `back_angle: 135°`, `reps: 6`.
   - Calls `exercise_knowledge_tool(query="excessive lower back arch shoulder press", exercise="Shoulder Press")` -> Retrieves biomechanical explanation: lumbar hyperextension occurs when core stabilization fails or thoracic mobility is limited (*Saeterbakken & Fimland 2013*).
   - Calls `user_personalization_tool(user_id=1)` -> Discovers athlete is *Intermediate* aiming for *Hypertrophy*.
4. **Synthesis:** Gemini combines these outputs into a single, cohesive, supportive response:
   > *"You completed 6 reps of Shoulder Press. On your last repetition, your back angle dropped to 135°, triggering an Excessive Arch alert. Biomechanically, leaning back turns the vertical press into an incline press, increasing lumbar shear strain (Saeterbakken & Fimland, 2013). Focus on bracing your core and squeezing your glutes before pressing overhead."*

---

## 4. Resilience & Fallback Architecture

To ensure the AI Gym Coach remains operable under all deployment conditions:
- **Missing API Key / Offline Mode:** If `GEMINI_API_KEY` is not present, the agent automatically activates the `DeterministicGymCoachAgent` orchestrator. It directly executes the same three tools and formats a structured, grounded answer using the tool outputs.
- **RAG Semantic Embeddings Fallback:** The vector store uses FAISS with an offline deterministic character-trigram semantic embedding engine when external embedding APIs are unavailable.
- **Safety Gate Override Protection:** If camera framing is `NOT_READY`, no tool or LLM hallucination can override this state.

