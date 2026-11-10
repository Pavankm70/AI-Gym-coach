import React, { useState, useEffect, useRef, useCallback } from 'react';
import Camera from './components/Camera';
import RepetitionCounter from './components/RepetitionCounter';
import JointAngles from './components/JointAngles';
import FeedbackPanel from './components/FeedbackPanel';
import ExerciseDashboard from './components/ExerciseDashboard';
import DiagnosticPanel from './components/DiagnosticPanel';
import Chatbot from './components/Chatbot';
import VoiceFeedbackControls from './components/VoiceFeedbackControls';
import UserProfileModal from './components/UserProfileModal';
import audioArbiter, { AudioPriority } from './services/AudioArbiter';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL || `${window.location.protocol}//${window.location.hostname}:8000`;
const WS_BASE = import.meta.env.VITE_WS_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.hostname}:8000`;

export default function App() {
  // Exercise & Plan State
  const [exercises, setExercises] = useState([
    'Squats',
    'Push-ups',
    'Biceps Curls (Dumbbell)',
    'Shoulder Press',
    'Lunges'
  ]);
  const [selectedExercise, setSelectedExercise] = useState('Squats');
  const [targetSets, setTargetSets] = useState(3);
  const [repsPerSet, setRepsPerSet] = useState(10);
  const [workoutActive, setWorkoutActive] = useState(false);
  const [workoutStartTime, setWorkoutStartTime] = useState(null);

  // User & History State
  const [user, setUser] = useState(() => {
    const saved = localStorage.getItem('gym_coach_user');
    return saved ? JSON.parse(saved) : { user_id: 1, id: 1, username: 'athlete_1', fitness_level: 'Beginner' };
  });
  const [history, setHistory] = useState([]);
  const [isProfileModalOpen, setIsProfileModalOpen] = useState(false);

  // Voice Coaching Intensity & Arbiter State
  const [coachingIntensity, setCoachingIntensity] = useState(() => {
    return localStorage.getItem('gym_coach_coaching_mode') || 'ERRORS_ONLY';
  });
  const coachingIntensityRef = useRef(coachingIntensity);
  useEffect(() => {
    coachingIntensityRef.current = coachingIntensity;
  }, [coachingIntensity]);

  const handleIntensityChange = (mode) => {
    setCoachingIntensity(mode);
    localStorage.setItem('gym_coach_coaching_mode', mode);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          exercise: selectedExerciseRef.current,
          target_sets: targetSetsRef.current,
          reps_per_set: repsPerSetRef.current,
          user_id: userRef.current?.user_id || userRef.current?.id || 1,
          coaching_intensity: mode
        })
      );
    }
  };

  // Post-Set Gemini Coaching State
  const [postSetCoaching, setPostSetCoaching] = useState(null);
  const [isReviewLoading, setIsReviewLoading] = useState(false);
  const lastCoachedSetRef = useRef(0);
  const isFetchingPostSetRef = useRef(false);

  // Video & WebSocket Stream State
  const [isStreaming, setIsStreaming] = useState(false);
  const [processedImage, setProcessedImage] = useState(null);
  const [landmarks, setLandmarks] = useState([]);
  const [viewMode, setViewMode] = useState('canvas'); // 'canvas' (zero-lag overlay) | 'processed' | 'raw'
  const [cameraError, setCameraError] = useState(null);
  const [fps, setFps] = useState(0);
  const [latencyInfo, setLatencyInfo] = useState({
    roundTripMs: 0,
    inferenceMs: 0,
    detectorMs: 0,
    serverMs: 0,
    networkMs: 0
  });

  // Diagnostic & Telemetry State
  const [cameraStatus, setCameraStatus] = useState('idle'); // 'idle' | 'requesting' | 'connected' | 'error'
  const [wsStatus, setWsStatus] = useState('disconnected'); // 'disconnected' | 'connecting' | 'connected' | 'error'
  const [framesSent, setFramesSent] = useState(0);
  const [framesReceived, setFramesReceived] = useState(0);
  const [diagnostic, setDiagnostic] = useState({
    model_loaded: true,
    poses_detected_count: 0,
    landmarks_count: 0,
    key_landmarks_visibility: {},
    detector_called: false
  });

  // Real-time Exercise Metrics & Safety Gate Telemetry
  const [metrics, setMetrics] = useState({
    pose_detected: false,
    reps: 0,
    current_set_reps: 0,
    sets_completed: 0,
    stage: 'ready',
    knee_angle: 0,
    back_angle: 0,
    depth_status: 'N/A',
    elbow_angle: 0,
    left_elbow_angle: 0,
    right_elbow_angle: 0,
    selected_side: 'both',
    body_alignment: 'N/A',
    hip_status: 'N/A',
    shoulder_status: 'N/A',
    swing_status: 'N/A',
    extension_status: 'N/A',
    back_arch_status: 'N/A',
    front_knee_angle: 0,
    torso_angle: 0,
    balance_status: 'N/A',
    workout_completed: false,
    form_error: null,
    form_instruction: null,
    camera_status: 'NOT_READY',
    pose_reliability: 'UNRELIABLE',
    reliability: null,
    spoken_feedback: null
  });

  // Refs for stable, loop-safe closure execution
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const wsRef = useRef(null);
  const isStreamingRef = useRef(false);
  const isAwaitingFrameRef = useRef(false);
  const lastFrameSentTimeRef = useRef(performance.now());
  const framesSentCountRef = useRef(0);
  const framesReceivedCountRef = useRef(0);
  const animFrameIdRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastFpsCalcTimeRef = useRef(performance.now());

  const selectedExerciseRef = useRef(selectedExercise);
  const targetSetsRef = useRef(targetSets);
  const repsPerSetRef = useRef(repsPerSet);
  const userRef = useRef(user);
  const viewModeRef = useRef(viewMode);
  const latencyHistoryRef = useRef([]);
  const sessionIdRef = useRef('sess_' + Math.random().toString(36).substring(2, 9));
  const frameIdRef = useRef(0);

  // Keep refs synchronized with state
  useEffect(() => {
    selectedExerciseRef.current = selectedExercise;
  }, [selectedExercise]);
  useEffect(() => {
    targetSetsRef.current = targetSets;
  }, [targetSets]);
  useEffect(() => {
    repsPerSetRef.current = repsPerSet;
  }, [repsPerSet]);
  useEffect(() => {
    userRef.current = user;
  }, [user]);
  useEffect(() => {
    viewModeRef.current = viewMode;
  }, [viewMode]);

  // Automatically trigger PDF RAG-grounded Gemini coaching when a set completes
  useEffect(() => {
    const completedSets = metrics.sets_completed || 0;
    const actState = metrics.activity_state || '';
    const isRestingPhase = [
      'SET_COMPLETED',
      'EXERCISE_COMPLETED',
      'RESTING',
      'READY_TO_BEGIN',
      'READY_TO_EXERCISE'
    ].includes(actState);

    if (
      completedSets > 0 &&
      completedSets > lastCoachedSetRef.current &&
      isRestingPhase &&
      !isFetchingPostSetRef.current
    ) {
      lastCoachedSetRef.current = completedSets;
      isFetchingPostSetRef.current = true;
      setIsReviewLoading(true);

      fetch(`${API_BASE}/api/coaching/post-set`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: user?.user_id || user?.id || 1,
          exercise_name: selectedExerciseRef.current,
          set_number: completedSets,
          session_id: sessionIdRef.current,
          set_data: {
            attempts: (metrics.valid_reps || 0) + (metrics.rejected_attempts || 0),
            valid_reps: metrics.valid_reps || 0,
            rejected_attempts: metrics.rejected_attempts || 0,
            main_error: metrics.form_error,
            form_errors: metrics.form_error ? [metrics.form_error] : [],
            joint_measurements: {
              knee_angle: metrics.knee_angle,
              elbow_angle: metrics.elbow_angle,
              back_angle: metrics.back_angle,
              torso_angle: metrics.torso_angle
            },
            movement_quality: {
              depth_status: metrics.depth_status,
              body_alignment: metrics.body_alignment,
              hip_status: metrics.hip_status,
              balance_status: metrics.balance_status
            },
            camera_status: metrics.camera_status,
            pose_reliability: metrics.pose_reliability
          },
          request_voice: true
        })
      })
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data) {
            setPostSetCoaching(data);
            if (data.coaching_text) {
              audioArbiter.requestSpeech({
                text: data.spoken_text || data.coaching_text.split('.')[0] + '.',
                audio: data.spoken_audio || null,
                priority: AudioPriority.CHATBOT,
                source: 'post_set_coach'
              });
            }
          }
        })
        .catch((err) => console.warn('Post-set coaching trigger error:', err))
        .finally(() => {
          isFetchingPostSetRef.current = false;
          setIsReviewLoading(false);
        });
    }
  }, [metrics.sets_completed, metrics.activity_state, user]);

  // Fetch available exercises from backend on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/exercises`)
      .then((res) => res.json())
      .then((data) => {
        if (data && data.exercises && data.exercises.length > 0) {
          setExercises(data.exercises);
        }
      })
      .catch((err) => {
        console.warn('Backend API not yet reachable at mount, using defaults:', err.message);
      });
  }, []);

  // Fetch workout history for current user
  const fetchHistory = useCallback(() => {
    const currentUserId = user?.user_id || user?.id;
    if (!currentUserId) return;
    fetch(`${API_BASE}/api/history/${currentUserId}`)
      .then((res) => res.json())
      .then((data) => {
        if (data && data.history) {
          setHistory(data.history);
        }
      })
      .catch((err) => console.error('Failed to fetch history:', err));
  }, [user]);

  useEffect(() => {
    if (user && (user.user_id || user.id)) {
      fetchHistory();
    }
  }, [user, fetchHistory]);

  // Handle User Login
  const handleLogin = async (username) => {
    try {
      const res = await fetch(`${API_BASE}/api/user/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username })
      });
      const data = await res.json();
      if (res.ok) {
        const normalizedUser = {
          ...data,
          user_id: data.user_id || data.id
        };
        setUser(normalizedUser);
        localStorage.setItem('gym_coach_user', JSON.stringify(normalizedUser));
      } else {
        alert(data.detail || 'Login failed');
      }
    } catch (err) {
      console.error('Login error:', err);
      const fallbackUser = { user_id: 1, id: 1, username };
      setUser(fallbackUser);
      localStorage.setItem('gym_coach_user', JSON.stringify(fallbackUser));
    }
  };

  // Switch Exercise
  const handleSelectExercise = (exercise) => {
    setSelectedExercise(exercise);
    selectedExerciseRef.current = exercise;
    lastCoachedSetRef.current = 0;
    setPostSetCoaching(null);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'config',
          exercise,
          target_sets: targetSetsRef.current,
          reps_per_set: repsPerSetRef.current,
          user_id: userRef.current?.user_id || userRef.current?.id || 1,
          coaching_intensity: coachingIntensityRef.current
        })
      );
    }
  };

  // Reset Repetition State
  const handleResetReps = () => {
    lastCoachedSetRef.current = 0;
    setPostSetCoaching(null);
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          type: 'reset',
          exercise: selectedExerciseRef.current
        })
      );
    }
    setMetrics((prev) => ({
      ...prev,
      reps: 0,
      current_set_reps: 0,
      sets_completed: 0,
      stage: 'ready',
      workout_completed: false
    }));
  };

  // Workout Session Controls
  const handleStartWorkout = () => {
    setWorkoutActive(true);
    setWorkoutStartTime(Date.now());
    handleResetReps();
    if (!isStreamingRef.current) {
      startCamera();
    }
  };

  const handleEndWorkout = async () => {
    const elapsedSec = workoutStartTime ? Math.round((Date.now() - workoutStartTime) / 1000) : 0;
    setWorkoutActive(false);

    const currentUserId = user?.user_id || user?.id;
    if (currentUserId && metrics.reps > 0) {
      try {
        await fetch(`${API_BASE}/api/workout/save`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: currentUserId,
            exercise_name: selectedExerciseRef.current,
            reps: metrics.reps,
            sets: metrics.sets_completed || 1,
            time_taken: elapsedSec
          })
        });
        fetchHistory();
      } catch (err) {
        console.error('Failed to save workout session:', err);
      }
    }
  };

  // Robust Continuous Frame Capture Loop (25 FPS Paced)
  const captureLoop = useCallback(() => {
    if (!isStreamingRef.current) return;

    const video = videoRef.current;
    const ws = wsRef.current;

    if (
      video &&
      video.readyState >= 2 &&
      video.videoWidth > 0 &&
      video.videoHeight > 0 &&
      ws &&
      ws.readyState === WebSocket.OPEN
    ) {
      const now = performance.now();
      const intervalMs = 40; // 25 FPS pacing

      // Send next frame if interval elapsed
      if (now - lastFrameSentTimeRef.current >= intervalMs) {
        if (!canvasRef.current) {
          canvasRef.current = document.createElement('canvas');
        }

        const canvas = canvasRef.current;
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;

        // Preserve webcam native aspect ratio (prevents squishing 16:9 into 4:3)
        const targetW = 640;
        const targetH = Math.round(640 * (vh / vw)) || 480;
        if (canvas.width !== targetW || canvas.height !== targetH) {
          canvas.width = targetW;
          canvas.height = targetH;
        }

        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        ctx.drawImage(video, 0, 0, targetW, targetH);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.65);

        lastFrameSentTimeRef.current = now;
        framesSentCountRef.current += 1;
        setFramesSent(framesSentCountRef.current);

        ws.send(
          JSON.stringify({
            type: 'frame',
            session_id: sessionIdRef.current,
            frame_id: frameIdRef.current++,
            image: dataUrl,
            exercise: selectedExerciseRef.current,
            target_sets: targetSetsRef.current,
            reps_per_set: repsPerSetRef.current,
            user_id: userRef.current?.user_id || userRef.current?.id || 1,
            view_mode: viewModeRef.current,
            coaching_intensity: coachingIntensityRef.current,
            t_capture: now
          })
        );
      }
    }

    if (isStreamingRef.current) {
      animFrameIdRef.current = requestAnimationFrame(captureLoop);
    }
  }, []);

  // Start Camera
  const startCamera = async () => {
    setCameraError(null);
    setCameraStatus('requesting');
    setWsStatus('connecting');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        },
        audio: false
      });

      mediaStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => {});
      }
      setCameraStatus('connected');

      // Initialize WebSocket connection to FastAPI
      const wsUrl = `${WS_BASE}/ws/pose-stream`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setWsStatus('connected');
        isStreamingRef.current = true;
        setIsStreaming(true);
        latencyHistoryRef.current = [];

        ws.send(
          JSON.stringify({
            type: 'config',
            exercise: selectedExerciseRef.current,
            target_sets: targetSetsRef.current,
            reps_per_set: repsPerSetRef.current,
            user_id: userRef.current?.user_id || userRef.current?.id || 1,
            view_mode: viewModeRef.current,
            coaching_intensity: coachingIntensityRef.current
          })
        );

        if (animFrameIdRef.current) {
          cancelAnimationFrame(animFrameIdRef.current);
        }
        animFrameIdRef.current = requestAnimationFrame(captureLoop);
      };

      ws.onmessage = (event) => {
        framesReceivedCountRef.current += 1;
        setFramesReceived(framesReceivedCountRef.current);

        try {
          const data = JSON.parse(event.data);
          if (data.type === 'pose_update') {
            if (data.image) {
              setProcessedImage(data.image);
            }
            if (data.landmarks) {
              setLandmarks(data.landmarks);
            }
            if (data.metrics) {
              setMetrics((prev) => ({
                ...prev,
                ...data.metrics
              }));
              if (data.metrics.diagnostic) {
                setDiagnostic(data.metrics.diagnostic);
              }
            }

            // Real-time Pipeline Latency Telemetry with Rolling P95 and Median
            if (data.timing) {
              const now = performance.now();
              const tCap = data.timing.t_capture || now;
              const rtt = Math.max(1, Math.round(now - tCap));
              const srv = data.timing.server_ms || 0;
              const inf = data.timing.inference_ms || 0;
              const det = data.timing.detector_ms || 0;
              const net = Math.max(0, Math.round(rtt - srv));

              latencyHistoryRef.current.push(rtt);
              if (latencyHistoryRef.current.length > 60) {
                latencyHistoryRef.current.shift();
              }
              const sorted = [...latencyHistoryRef.current].sort((a, b) => a - b);
              const medianMs = sorted[Math.floor(sorted.length * 0.5)] || rtt;
              const p95Ms = sorted[Math.floor(sorted.length * 0.95)] || rtt;

              setLatencyInfo({
                roundTripMs: rtt,
                medianMs,
                p95Ms,
                landmarkAgeMs: rtt,
                inferenceMs: inf,
                detectorMs: det,
                serverMs: srv,
                networkMs: net,
                framesProcessed: data.timing.frames_processed,
                framesDropped: data.timing.frames_dropped
              });
            }

            // Calculate FPS
            frameCountRef.current += 1;
            const now = performance.now();
            const elapsed = now - lastFpsCalcTimeRef.current;
            if (elapsed >= 1000) {
              setFps(Math.round((frameCountRef.current * 1000) / elapsed));
              frameCountRef.current = 0;
              lastFpsCalcTimeRef.current = now;
            }
          }
        } catch (err) {
          console.error('Error parsing WS message:', err);
        }
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        setWsStatus('error');
        setCameraError('WebSocket connection error. Check that backend server is running on port 8000.');
      };

      ws.onclose = () => {
        setWsStatus('disconnected');
        isStreamingRef.current = false;
        setIsStreaming(false);
        isAwaitingFrameRef.current = false;
      };
    } catch (err) {
      console.error('Camera access error:', err);
      setCameraStatus('error');
      setCameraError(
        err.name === 'NotAllowedError'
          ? 'Camera permission denied. Please allow camera access in browser permissions.'
          : `Camera error: ${err.message}`
      );
      isStreamingRef.current = false;
      setIsStreaming(false);
    }
  };

  // Stop Camera
  const stopCamera = () => {
    isStreamingRef.current = false;
    setIsStreaming(false);
    isAwaitingFrameRef.current = false;

    if (animFrameIdRef.current) {
      cancelAnimationFrame(animFrameIdRef.current);
      animFrameIdRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setCameraStatus('idle');
    setWsStatus('disconnected');
    setProcessedImage(null);
    setLandmarks([]);
    setLatencyInfo({ roundTripMs: 0, inferenceMs: 0, detectorMs: 0, serverMs: 0, networkMs: 0 });
    setFps(0);
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-logo">🏋️‍♂️</div>
          <div className="brand-text">
            <h1 className="brand-title">AI GYM COACH</h1>
            <p className="brand-subtitle">Research-Based Exercise Monitoring & Agentic AI Coaching</p>
          </div>
        </div>

        <div className="header-actions">
          {/* Voice Feedback Audio & Arbiter Priority Controls */}
          <VoiceFeedbackControls
            spokenFeedback={metrics.spoken_feedback}
            coachingIntensity={coachingIntensity}
            onIntensityChange={handleIntensityChange}
          />

          <div className="status-badge-chip">
            <span className={`status-indicator ${isStreaming ? 'online' : 'offline'}`} />
            {isStreaming ? `${fps} FPS • Pipeline Active` : 'Ready'}
          </div>

          {/* User Profile Button */}
          {user && (
            <button
              type="button"
              className="user-badge-btn"
              onClick={() => setIsProfileModalOpen(true)}
              title="Click to view and edit fitness profile"
            >
              <span>Athlete:</span> <strong>{user.username}</strong>
              <span className="profile-edit-icon">⚙️</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Dashboard Layout */}
      <main className="dashboard-layout">
        {/* Left Column: Camera Feed & Real-time Visuals */}
        <section className="dashboard-col col-primary">
          <Camera
            isStreaming={isStreaming}
            onStart={startCamera}
            onStop={stopCamera}
            processedImage={processedImage}
            landmarks={landmarks}
            poseDetected={metrics.pose_detected}
            cameraError={cameraError}
            currentExercise={selectedExercise}
            viewMode={viewMode}
            setViewMode={setViewMode}
            videoRef={videoRef}
            fps={fps}
            reliability={metrics.reliability}
            latencyInfo={latencyInfo}
          />

          {/* Feedback & Form Coaching Panel with Activity Guidance & Safety Gate Notice */}
          <FeedbackPanel
            formError={metrics.form_error}
            formInstruction={metrics.form_instruction}
            poseDetected={metrics.pose_detected}
            isStreaming={isStreaming}
            exercise={selectedExercise}
            reliability={metrics.reliability}
            cameraStatus={metrics.camera_status}
            activityState={metrics.activity_state}
            guidanceMessage={metrics.guidance_message}
            rejectionReason={metrics.rejection_reason}
            lastRepValid={metrics.last_rep_valid}
          />

          {/* Diagnostic & Telemetry HUD Panel */}
          <DiagnosticPanel
            cameraStatus={cameraStatus}
            wsStatus={wsStatus}
            framesSent={framesSent}
            framesReceived={framesReceived}
            fps={fps}
            diagnostic={diagnostic}
            metrics={metrics}
            exercise={selectedExercise}
            latencyInfo={latencyInfo}
          />
        </section>

        {/* Right Column: Metrics, Joint Angles & Workout Management */}
        <section className="dashboard-col col-secondary">
          {/* Repetition & Validated Movement Phase Counter */}
          <RepetitionCounter
            reps={metrics.reps || 0}
            validReps={metrics.valid_reps !== undefined ? metrics.valid_reps : (metrics.reps || 0)}
            rejectedAttempts={metrics.rejected_attempts || 0}
            currentPhase={metrics.current_phase || metrics.stage || 'READY'}
            rejectionReason={metrics.rejection_reason}
            lastRepValid={metrics.last_rep_valid}
            currentSetReps={metrics.current_set_reps || 0}
            repsPerSet={repsPerSet}
            setsCompleted={metrics.sets_completed || 0}
            targetSets={targetSets}
            stage={metrics.stage || 'ready'}
            workoutCompleted={metrics.workout_completed}
            metrics={metrics}
            exercise={selectedExercise}
            onReset={handleResetReps}
          />

          {/* Dynamic Joint Angle Gauges */}
          <JointAngles
            exercise={selectedExercise}
            metrics={metrics}
          />

          {/* Exercise Plan, Setup & Workout History */}
          <ExerciseDashboard
            exercises={exercises}
            selectedExercise={selectedExercise}
            onSelectExercise={handleSelectExercise}
            targetSets={targetSets}
            setTargetSets={setTargetSets}
            repsPerSet={repsPerSet}
            setRepsPerSet={setRepsPerSet}
            workoutActive={workoutActive}
            onStartWorkout={handleStartWorkout}
            onEndWorkout={handleEndWorkout}
            user={user}
            onLogin={handleLogin}
            history={history}
            onRefreshHistory={fetchHistory}
            isStreaming={isStreaming}
            postSetCoaching={postSetCoaching}
            isReviewLoading={isReviewLoading}
          />
        </section>
      </main>

      {/* AI Conversational Chatbot Section (Single Agent with 3 Tools) */}
      <section className="chatbot-section">
        <Chatbot
          userId={user?.user_id || user?.id || 1}
          currentExercise={selectedExercise}
          apiBase={API_BASE}
        />
      </section>

      {/* User Profile Modal */}
      <UserProfileModal
        isOpen={isProfileModalOpen}
        onClose={() => setIsProfileModalOpen(false)}
        user={user}
        onUpdateUser={(updated) => {
          setUser(updated);
          localStorage.setItem('gym_coach_user', JSON.stringify(updated));
          if (updated && updated.coaching_intensity) {
            handleIntensityChange(updated.coaching_intensity);
          }
        }}
        apiBase={API_BASE}
      />

      {/* Footer */}
      <footer className="app-footer">
        <p>AI Gym Coach • Real-Time Human Pose Analysis, Camera Reliability Gate & Agentic Coaching</p>
      </footer>
    </div>
  );
}
