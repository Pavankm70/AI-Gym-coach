import os
import sys
import base64
import time
import asyncio
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add backend and Main App directory to sys.path so modules resolve cleanly
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BACKEND_DIR)
MAIN_APP_DIR = os.path.join(ROOT_DIR, "Main App")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if MAIN_APP_DIR not in sys.path:
    sys.path.insert(0, MAIN_APP_DIR)

# Import existing detector and processing modules directly from Main App
from services.config.workout_config import EXERCISE_OPTIONS, METRICS_FIELDS
from services.vision.exercise_video_processor import VideoProcessorClass
from services.persistence.exercise_repository import (
    init_db,
    get_or_create_user,
    get_user_by_id,
    update_user_profile,
    add_exercise,
    get_users_exercises
)
from services.coaching.voice_pipeline import find_form_issue

# Import new research-based reliability gate, LangChain agent, and speech services
from vision.reliability_gate import ReliabilityGate
from vision.activity_state import ActivityStateManager
from agent.agent import get_gym_coach_agent
from agent.tools.exercise_analysis import update_user_session_telemetry
from agent.rag.retrieval import retrieve_exercise_knowledge, retrieve_exercise_knowledge_diagnostic
from ai_services.speech_service import get_speech_service
from ai_services.feedback_service import get_feedback_service
from ai_services.post_set_coach import get_post_set_coach


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database and tables
    init_db()
    # Pre-warm LangChain agent and RAG vector store
    get_gym_coach_agent()
    yield


app = FastAPI(
    title="AI Gym Coach API",
    description="Backend API, real-time WebSocket pose processor, camera reliability gate, and LangChain agent for AI Gym Coach",
    version="2.1.0",
    lifespan=lifespan
)

# Configure CORS so React (Vite dev server / production) can communicate seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str


class UpdateProfileRequest(BaseModel):
    fitness_level: Optional[str] = "Beginner"
    goals: Optional[str] = "General Fitness"
    preferred_feedback_style: Optional[str] = "Encouraging"
    experience: Optional[str] = "Beginner (0-1 years)"
    workout_frequency: Optional[str] = "3 days/week"
    limitations: Optional[str] = "None"
    coaching_intensity: Optional[str] = "ERRORS_ONLY"


class SaveWorkoutRequest(BaseModel):
    user_id: int
    exercise_name: str
    reps: int
    sets: int
    time_taken: float


class ChatRequest(BaseModel):
    message: str
    user_id: int = 1
    exercise: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    request_voice: Optional[bool] = False
    coaching_intensity: Optional[str] = "ERRORS_ONLY"


class SpeechRequest(BaseModel):
    text: str


class PostSetCoachingRequest(BaseModel):
    user_id: int = 1
    exercise_name: str
    set_number: int = 1
    session_id: Optional[str] = None
    set_data: Dict[str, Any]
    request_voice: Optional[bool] = True


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "app": "AI Gym Coach API",
        "reliability_gate": "active",
        "agent": "active"
    }


@app.get("/api/exercises")
def get_exercises():
    """Return available exercises and their metric configurations."""
    return {
        "exercises": EXERCISE_OPTIONS,
        "metrics_fields": METRICS_FIELDS
    }


@app.post("/api/user/login")
def login(req: LoginRequest):
    """Authenticate or register user using SQLite persistence."""
    cleaned = req.username.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    user = get_or_create_user(cleaned)
    user_dict = dict(user)
    user_dict["user_id"] = user_dict["id"]
    return user_dict


@app.get("/api/user/profile/{user_id}")
def get_profile(user_id: int):
    """Retrieve user fitness profile."""
    row = get_user_by_id(user_id)
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(row)


@app.post("/api/user/profile/{user_id}")
def update_profile(user_id: int, req: UpdateProfileRequest):
    """Update user fitness profile and preferences."""
    updated = update_user_profile(user_id, req.model_dump(exclude_unset=True))
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return dict(updated)


@app.get("/api/history/{user_id}")
def get_history(user_id: int):
    """Get workout history for a user."""
    rows = get_users_exercises(user_id)
    history = [
        {
            "id": row["id"],
            "exercise": row["exercise_name"],
            "reps": row["reps"],
            "sets": row["sets"],
            "time": row["time"],
            "date": row["created_at"]
        }
        for row in rows
    ]
    return {"history": history}


@app.post("/api/workout/save")
def save_workout(req: SaveWorkoutRequest):
    """Save completed workout sets to the database."""
    add_exercise(
        req.user_id,
        req.exercise_name,
        req.reps,
        req.sets,
        int(req.time_taken)
    )
    return {"status": "success", "saved": True}


@app.post("/api/chat")
def chat_with_coach(req: ChatRequest):
    """
    Conversational AI Coaching endpoint powered by a single LangChain agent
    with three specialized tools: Exercise Analysis, RAG Knowledge, and Personalization.
    """
    cleaned_msg = req.message.strip()
    if not cleaned_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    agent = get_gym_coach_agent()
    result = agent.run(
        user_message=cleaned_msg,
        user_id=req.user_id,
        chat_history=req.chat_history,
        exercise_name=req.exercise
    )
    if req.request_voice and result.get("response"):
        from ai_services.audio_arbiter import get_audio_arbiter, AudioPriority
        arbiter = get_audio_arbiter()
        first_sentence = result["response"].split(".")[0].strip() + "."
        voice_res = arbiter.evaluate_and_synthesize(
            event="chatbot_speech",
            text=first_sentence,
            priority=AudioPriority.CHATBOT
        )
        if voice_res:
            result["spoken_audio"] = voice_res.get("audio")
            result["spoken_text"] = voice_res.get("text")

    return result


@app.post("/api/coaching/post-set")
def get_post_set_coaching(req: PostSetCoachingRequest):
    """
    Generates intelligent, PDF RAG-grounded Gemini coaching for a completed set.
    Uses actual session measurements and authentic PDF research literature.
    """
    coach = get_post_set_coach()
    result = coach.generate_post_set_coaching(
        user_id=req.user_id,
        exercise_name=req.exercise_name,
        set_number=req.set_number,
        set_data=req.set_data,
        session_id=req.session_id,
        request_voice=req.request_voice if req.request_voice is not None else True
    )
    return result


@app.post("/api/speech/synthesize")
def synthesize_speech(req: SpeechRequest):
    """Synthesizes text into base64 speech audio using gTTS."""
    service = get_speech_service()
    audio = service.synthesize_speech_base64(req.text, force=True)
    return {"audio": audio}


@app.get("/api/rag/diagnostic")
def get_rag_diagnostic(query: str, exercise: Optional[str] = None):
    """Retrieval diagnostic endpoint exposing chunk IDs, similarity scores, and filters."""
    return retrieve_exercise_knowledge_diagnostic(query=query, exercise_name=exercise)


def map_form_instruction(issue: Optional[str]) -> Optional[str]:
    """Provide concise, supportive text instruction for detected issues."""
    if not issue:
        return None
    if "squat is not deep enough" in issue:
        return "Squat lower until your thighs are parallel to the ground."
    if "leaning too far forward" in issue:
        return "Keep your chest high and maintain an upright torso."
    if "body is not straight" in issue:
        return "Tighten your abs to align shoulders, hips and ankles."
    if "hips are sagging" in issue:
        return "Lift your hips to maintain a flat, neutral plank position."
    if "hips are too high" in issue:
        return "Lower your hips to avoid piking up."
    if "swinging their torso" in issue:
        return "Hold your body steady and curl using only your arms."
    if "elbow is drifting" in issue:
        return "Keep elbows glued close to your sides."
    if "Excessive Arch" in issue or "arching their lower back excessively" in issue:
        return "Tuck your pelvis and brace your core to prevent lower back arch."
    if "Slight back arch" in issue:
        return "Brace your core muscles firmly during the press."
    if "losing balance" in issue:
        return "Widen your stance to hip-width for better balance."
    if "No pose detected" in issue or "step into" in issue:
        return "Please position yourself fully within the camera view."
    return issue


@app.websocket("/ws/pose-stream")
async def pose_stream(websocket: WebSocket):
    """
    Real-time WebSocket endpoint for camera frame processing, pose reliability
    validation, and exercise form analysis.

    Enforces the Safety Gate and Bounded Latest-Frame Processing:
    - Bounded single-frame queue drops stale frames immediately
    - Thread-pool inference prevents event-loop blocking
    - Selective JPEG transmission: raw landmarks JSON in canvas mode
    """
    await websocket.accept()

    processor = VideoProcessorClass()
    reliability_gate = ReliabilityGate()
    feedback_service = get_feedback_service()
    activity_mgr = ActivityStateManager()

    current_exercise = "Squats"
    processor.set_exercise(current_exercise)

    target_sets = 3
    reps_per_set = 10
    current_user_id = 1

    # Bounded Latest-Frame Queue (maxsize=1)
    frame_queue = asyncio.Queue(maxsize=1)
    frames_received = 0
    frames_processed = 0
    frames_dropped = 0
    stop_event = asyncio.Event()

    def sync_process_frame_payload(frame_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nonlocal current_exercise, target_sets, reps_per_set, current_user_id, frames_received, frames_processed, frames_dropped

        t_backend_recv = frame_data.get("t_backend_recv", time.time() * 1000.0)
        t_capture = float(frame_data.get("t_capture", 0.0))
        frame_id = frame_data.get("frame_id", 0)
        session_id = frame_data.get("session_id", "session_default")
        view_mode = frame_data.get("view_mode", "canvas")

        image_data = frame_data.get("image", "")
        if not image_data:
            return None

        if "user_id" in frame_data:
            current_user_id = int(frame_data["user_id"])

        # Strip base64 header if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        raw_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(raw_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return None

        if "exercise" in frame_data and frame_data["exercise"] != current_exercise:
            current_exercise = frame_data["exercise"]
            processor.set_exercise(current_exercise)
            activity_mgr.reset()

        t_mp_start = time.time() * 1000.0
        annotated_img, raw_metrics = processor.process_frame_bgr(frame)
        t_mp_done = time.time() * 1000.0

        detector = processor._detectors.get(current_exercise)
        raw_landmarks = getattr(processor, "_current_landmarks", None)
        pose_detected = raw_metrics.get("pose_detected", False)

        # Evaluate frame with research-based Reliability Gate
        gate_result = reliability_gate.evaluate_and_filter(
            frame=frame,
            landmarks=raw_landmarks,
            exercise_name=current_exercise,
            detector=detector
        )

        reliability = gate_result["reliability"]
        camera_status = reliability["camera_status"]
        pose_reliability = reliability["pose_reliability"]

        valid_reps = raw_metrics.get("valid_reps", raw_metrics.get("reps", 0))
        rejected_attempts = raw_metrics.get("rejected_attempts", 0)
        current_phase = raw_metrics.get("current_phase", raw_metrics.get("stage", "ready"))
        rejection_reason = raw_metrics.get("rejection_reason")
        last_rep_valid = raw_metrics.get("last_rep_valid")

        if valid_reps is not None and reps_per_set > 0 and target_sets > 0:
            sets_completed = valid_reps // reps_per_set
            current_set_reps = valid_reps % reps_per_set
            workout_completed = sets_completed >= target_sets
            set_completed = (current_set_reps == 0 and valid_reps > 0 and not workout_completed)
        else:
            sets_completed = 0
            current_set_reps = 0
            workout_completed = False
            set_completed = False

        activity_info = activity_mgr.update(
            camera_connected=True,
            pose_detected=pose_detected,
            camera_status=camera_status,
            pose_reliability=pose_reliability,
            exercise_name=current_exercise,
            current_phase=current_phase,
            valid_reps=valid_reps,
            rejected_attempts=rejected_attempts,
            workout_completed=workout_completed,
            landmarks=raw_landmarks,
            set_completed=set_completed
        )

        metrics = raw_metrics.copy()
        metrics["reps"] = valid_reps
        metrics["valid_reps"] = valid_reps
        metrics["rejected_attempts"] = rejected_attempts
        metrics["current_phase"] = current_phase
        metrics["rejection_reason"] = rejection_reason
        metrics["last_rep_valid"] = last_rep_valid
        metrics["current_set_reps"] = current_set_reps
        metrics["sets_completed"] = sets_completed
        metrics["set_completed"] = set_completed
        metrics["target_sets"] = target_sets
        metrics["reps_per_set"] = reps_per_set
        metrics["workout_completed"] = workout_completed
        metrics["exercise"] = current_exercise
        metrics["camera_status"] = camera_status
        metrics["pose_reliability"] = pose_reliability
        metrics["reliability"] = reliability
        metrics["activity_state"] = activity_info["activity_state"]
        metrics["guidance_message"] = activity_info["guidance_message"]
        metrics["dispatch_prompt"] = activity_info["dispatch_prompt"]
        metrics["is_idle"] = activity_info["is_idle"]
        metrics["coaching_intensity"] = frame_data.get("coaching_intensity", "ERRORS_ONLY")

        gate_outcome = reliability.get("gate_outcome", "CAMERA_READY")
        is_blocked = (gate_outcome == "CAMERA_BLOCKED" or camera_status in ["NOT_READY", "BLOCKED"])

        if is_blocked:
            issues = reliability.get("issues", [])
            sugg = reliability.get("suggestions", [])
            form_error = f"Camera / Pose Warning: {issues[0]}" if issues else "Camera framing or pose is unreliable."
            form_instruction = sugg[0] if sugg else (activity_info["guidance_message"] or "Please position yourself within camera view.")
        elif not pose_detected:
            form_error = "Person not detected."
            form_instruction = activity_info["guidance_message"] or "Please stand in front of the camera."
        else:
            # Camera is READY or WARNING: evaluate actual exercise form errors!
            raw_issue = find_form_issue(current_exercise, metrics)
            form_error = raw_issue
            form_instruction = map_form_instruction(raw_issue)

        metrics["form_error"] = form_error
        metrics["form_instruction"] = form_instruction

        t_detector_done = time.time() * 1000.0

        update_user_session_telemetry(
            user_id=current_user_id,
            telemetry={
                "user_id": current_user_id,
                "exercise": current_exercise,
                "reps": valid_reps,
                "valid_reps": valid_reps,
                "rejected_attempts": rejected_attempts,
                "stage": metrics.get("stage", "ready"),
                "current_phase": current_phase,
                "rejection_reason": rejection_reason,
                "activity_state": activity_info["activity_state"],
                "guidance_message": activity_info["guidance_message"],
                "camera_status": camera_status,
                "pose_reliability": pose_reliability,
                "form_error": form_error,
                "form_instruction": form_instruction,
                "measurements": {
                    k: v for k, v in metrics.items()
                    if k not in ["diagnostic", "reliability", "form_error", "form_instruction", "exercise", "activity_state", "guidance_message"]
                },
                "reliability": reliability
            }
        )

        voice_cue = feedback_service.evaluate_feedback(
            exercise=current_exercise,
            metrics=metrics,
            camera_status=camera_status,
            pose_reliability=pose_reliability
        )
        if voice_cue:
            metrics["spoken_feedback"] = voice_cue

        # Extract normalized 33 landmarks
        landmarks_data = []
        if raw_landmarks:
            for lm in raw_landmarks:
                landmarks_data.append({
                    "x": round(float(lm.x), 4),
                    "y": round(float(lm.y), 4),
                    "z": round(float(lm.z), 4),
                    "visibility": round(float(lm.visibility), 3)
                })

        # Only encode JPEG image if client explicitly requested 'processed' server view
        response_image = None
        if view_mode == "processed":
            _, buffer = cv2.imencode(".jpg", annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            b64_output = base64.b64encode(buffer).decode("utf-8")
            response_image = f"data:image/jpeg;base64,{b64_output}"

        t_backend_send = time.time() * 1000.0

        timing_telemetry = {
            "t_capture": t_capture,
            "t_backend_recv": round(t_backend_recv, 1),
            "inference_ms": round(t_mp_done - t_mp_start, 1),
            "detector_ms": round(t_detector_done - t_mp_done, 1),
            "server_ms": round(t_backend_send - t_backend_recv, 1),
            "t_backend_send": round(t_backend_send, 1),
            "frames_received": frames_received,
            "frames_processed": frames_processed,
            "frames_dropped": frames_dropped
        }

        return {
            "type": "pose_update",
            "session_id": session_id,
            "frame_id": frame_id,
            "image": response_image,
            "landmarks": landmarks_data,
            "metrics": metrics,
            "timing": timing_telemetry
        }

    async def worker_loop():
        while not stop_event.is_set():
            try:
                frame_item = await frame_queue.get()
                nonlocal frames_processed
                frames_processed += 1
                payload = await asyncio.to_thread(sync_process_frame_payload, frame_item)
                if payload:
                    await websocket.send_json(payload)
                frame_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except Exception:
                    pass

    worker_task = asyncio.create_task(worker_loop())

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "frame")

            if msg_type == "config":
                if "exercise" in data:
                    current_exercise = data["exercise"]
                    processor.set_exercise(current_exercise)
                    activity_mgr.reset()
                if "target_sets" in data:
                    target_sets = int(data["target_sets"])
                if "reps_per_set" in data:
                    reps_per_set = int(data["reps_per_set"])
                if "user_id" in data:
                    current_user_id = int(data["user_id"])
                await websocket.send_json({
                    "type": "config_ack",
                    "exercise": current_exercise,
                    "target_sets": target_sets,
                    "reps_per_set": reps_per_set
                })
                continue

            if msg_type == "reset":
                ex_to_reset = data.get("exercise", current_exercise)
                processor.reset_exercise(ex_to_reset)
                activity_mgr.reset()
                await websocket.send_json({
                    "type": "reset_ack",
                    "exercise": ex_to_reset
                })
                continue

            if msg_type == "frame":
                frames_received += 1
                data["t_backend_recv"] = time.time() * 1000.0

                # Bounded Latest-Frame Processing:
                # If a frame is already waiting in the queue, drop it and take newest!
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                        frames_dropped += 1
                    except (asyncio.QueueEmpty, ValueError):
                        pass

                await frame_queue.put(data)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        stop_event.set()
        if worker_task:
            worker_task.cancel()


# Serve compiled React frontend if built
FRONTEND_DIST = os.path.join(ROOT_DIR, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static_frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
