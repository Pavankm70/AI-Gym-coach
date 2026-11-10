import React, { useRef, useEffect, useState, useCallback } from 'react';
import CameraReliabilityBadge from './CameraReliabilityBadge';
import { getDisplayedVideoRect, mapLandmarkToCanvas } from '../utils/coordinateMapping';
import { PoseOneEuroFilter } from '../utils/oneEuroFilter';

// MediaPipe Full-Body Pose Connection Pairs
const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Shoulders and arms
  [11, 23], [12, 24], [23, 24],                     // Torso
  [23, 25], [24, 26], [25, 27], [26, 28],           // Hips to knees to ankles
  [27, 29], [28, 30], [29, 31], [30, 32]            // Feet and heels
];

// Diagnostic landmark labels
const DIAGNOSTIC_LABELS = {
  0: 'Nose',
  11: 'L.Shoulder',
  12: 'R.Shoulder',
  13: 'L.Elbow',
  14: 'R.Elbow',
  15: 'L.Wrist',
  16: 'R.Wrist',
  23: 'L.Hip',
  24: 'R.Hip',
  25: 'L.Knee',
  26: 'R.Knee',
  27: 'L.Ankle',
  28: 'R.Ankle'
};

export default function Camera({
  isStreaming,
  onStart,
  onStop,
  processedImage,
  landmarks = [],
  poseDetected,
  cameraError,
  currentExercise,
  viewMode,
  setViewMode,
  videoRef,
  fps,
  reliability,
  latencyInfo,
  lastLandmarkTimestamp = 0
}) {
  const overlayCanvasRef = useRef(null);
  const containerRef = useRef(null);
  const animFrameRef = useRef(null);
  const filterRef = useRef(new PoseOneEuroFilter(1.2, 0.008, 1.0));
  const latestLandmarksRef = useRef([]);
  const lastUpdateTimestampRef = useRef(0);

  const [showDebugMarkers, setShowDebugMarkers] = useState(false);

  // Sync incoming landmarks to stable ref and run 1€ filter
  useEffect(() => {
    if (landmarks && landmarks.length > 0 && poseDetected) {
      const now = performance.now();
      const filtered = filterRef.current.filterLandmarks(landmarks, now);
      latestLandmarksRef.current = filtered;
      lastUpdateTimestampRef.current = now;
    } else {
      latestLandmarksRef.current = [];
      filterRef.current.reset();
    }
  }, [landmarks, poseDetected]);

  // High-performance 60 FPS requestAnimationFrame render loop
  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');

    const render = () => {
      if (!isStreaming || viewMode !== 'canvas') {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        animFrameRef.current = requestAnimationFrame(render);
        return;
      }

      const container = containerRef.current || canvas.parentElement;
      const cWidth = container ? container.clientWidth : 640;
      const cHeight = container ? container.clientHeight : 480;
      const dpr = window.devicePixelRatio || 1;

      // Handle HiDPI / Retina backing store
      const targetCanvasW = Math.round(cWidth * dpr);
      const targetCanvasH = Math.round(cHeight * dpr);
      if (canvas.width !== targetCanvasW || canvas.height !== targetCanvasH) {
        canvas.width = targetCanvasW;
        canvas.height = targetCanvasH;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, cWidth, cHeight);

      // Verify landmark freshness (drop if older than 250 ms to prevent ghosting)
      const now = performance.now();
      const ageMs = now - lastUpdateTimestampRef.current;
      const isFresh = ageMs < 250 && latestLandmarksRef.current.length > 0;

      // Draw subtle alignment crosshair if in diagnostic mode
      if (showDebugMarkers) {
        ctx.strokeStyle = 'rgba(0, 242, 254, 0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(cWidth / 2, 0);
        ctx.lineTo(cWidth / 2, cHeight);
        ctx.moveTo(0, cHeight / 2);
        ctx.lineTo(cWidth, cHeight / 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      if (isFresh && poseDetected) {
        const vRect = getDisplayedVideoRect(video, cWidth, cHeight, 'cover');
        const pts = latestLandmarksRef.current.map(lm => mapLandmarkToCanvas(lm, vRect, false));

        // 1. Draw Skeleton Connection Lines
        ctx.lineWidth = 3.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = '#00f2fe'; // Neon cyan

        for (const [startIdx, endIdx] of POSE_CONNECTIONS) {
          const p1 = pts[startIdx];
          const p2 = pts[endIdx];

          if (p1 && p2 && p1.visible && p2.visible) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        }

        // 2. Draw Joint Circles
        for (let i = 0; i < pts.length; i++) {
          const p = pts[i];
          if (p && p.visible) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4.5, 0, 2 * Math.PI);
            ctx.fillStyle = '#4facfe'; // Electric blue
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = '#ffffff';
            ctx.stroke();

            // 3. Draw Diagnostic Markers / Labels
            if (showDebugMarkers && DIAGNOSTIC_LABELS[i]) {
              const label = DIAGNOSTIC_LABELS[i];
              ctx.font = '600 11px Inter, system-ui, sans-serif';
              const textWidth = ctx.measureText(label).width;

              // Text background bubble
              ctx.fillStyle = 'rgba(10, 15, 29, 0.85)';
              ctx.fillRect(p.x + 8, p.y - 12, textWidth + 8, 16);
              ctx.strokeStyle = '#00f2fe';
              ctx.lineWidth = 1;
              ctx.strokeRect(p.x + 8, p.y - 12, textWidth + 8, 16);

              // Text string
              ctx.fillStyle = '#ffffff';
              ctx.fillText(label, p.x + 12, p.y);
            }
          }
        }
      }

      ctx.restore();
      animFrameRef.current = requestAnimationFrame(render);
    };

    animFrameRef.current = requestAnimationFrame(render);

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isStreaming, viewMode, poseDetected, showDebugMarkers]);

  return (
    <div className="camera-card">
      <div className="camera-header">
        <div className="camera-title-group">
          <span className="live-pill">
            <span className={`pulse-dot ${isStreaming ? 'active' : ''}`} />
            {isStreaming ? 'LIVE STREAM' : 'CAMERA STANDBY'}
          </span>
          <span className="exercise-badge">{currentExercise}</span>
        </div>

        {isStreaming && (
          <div className="camera-header-controls">
            <button
              type="button"
              className={`toggle-btn ${showDebugMarkers ? 'active' : ''}`}
              onClick={() => setShowDebugMarkers(prev => !prev)}
              title="Toggle joint names and alignment crosshair"
            >
              🎯 {showDebugMarkers ? 'Markers: ON' : 'Markers: OFF'}
            </button>

            <div className="view-mode-toggle">
              <button
                type="button"
                className={`toggle-btn ${viewMode === 'canvas' ? 'active' : ''}`}
                onClick={() => setViewMode('canvas')}
                title="Native 60 FPS video with zero-latency transparent canvas overlay"
              >
                Zero-Lag Overlay
              </button>
              <button
                type="button"
                className={`toggle-btn ${viewMode === 'processed' ? 'active' : ''}`}
                onClick={() => setViewMode('processed')}
                title="Backend OpenCV annotated video feed"
              >
                Server Render
              </button>
              <button
                type="button"
                className={`toggle-btn ${viewMode === 'raw' ? 'active' : ''}`}
                onClick={() => setViewMode('raw')}
                title="Direct clean webcam video feed without skeleton"
              >
                Raw Feed
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="video-viewport" ref={containerRef}>
        {/* Active Native Video element: plays directly from webcam with 0ms visual latency */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`raw-video ${isStreaming ? (viewMode === 'processed' ? 'hidden' : 'visible') : 'hidden'}`}
        />

        {/* Real-time Transparent Canvas Skeleton Overlay directly on top of native video */}
        {isStreaming && viewMode === 'canvas' && (
          <canvas
            ref={overlayCanvasRef}
            className="skeleton-overlay-canvas"
          />
        )}

        {/* Backend OpenCV processed JPEG frame */}
        {isStreaming && viewMode === 'processed' && processedImage && (
          <img
            src={processedImage}
            alt="Real-time MediaPipe Annotated Pose"
            className="processed-frame"
          />
        )}

        {/* Placeholder when camera is not running */}
        {!isStreaming && (
          <div className="viewport-placeholder">
            <div className="placeholder-icon">🏋️‍♂️</div>
            <h3>Camera is Disconnected</h3>
            <p>Click <strong>Start Camera</strong> to activate real-time MediaPipe pose detection, camera validation, and exercise monitoring.</p>
            {cameraError && (
              <div className="camera-error-banner">
                <strong>Camera Access Notice:</strong> {cameraError}
              </div>
            )}
          </div>
        )}

        {/* Status badges HUD */}
        {isStreaming && (
          <div className="viewport-overlays">
            <div className={`status-pill ${poseDetected ? 'detected' : 'missing'}`}>
              <span className="dot" />
              {poseDetected ? 'Pose Detected' : 'No Body Detected'}
            </div>
            {fps > 0 && <div className="fps-pill">{fps} FPS</div>}
            {latencyInfo && latencyInfo.roundTripMs > 0 && (
              <div className="latency-pill" title="End-to-End Latency">
                ⚡ {latencyInfo.roundTripMs} ms
              </div>
            )}
          </div>
        )}
      </div>

      {/* Camera-View Validation & Pose Reliability Assessment HUD */}
      {isStreaming && (
        <CameraReliabilityBadge reliability={reliability} />
      )}

      {/* Camera Controls */}
      <div className="camera-controls">
        {!isStreaming ? (
          <button
            type="button"
            className="btn btn-primary btn-large"
            onClick={onStart}
          >
            <span className="btn-icon">▶</span> Start Camera
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-danger btn-large"
            onClick={onStop}
          >
            <span className="btn-icon">⏹</span> Stop Camera
          </button>
        )}
      </div>
    </div>
  );
}
