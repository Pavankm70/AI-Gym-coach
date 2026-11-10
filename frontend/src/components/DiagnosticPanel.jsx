import React, { useState } from 'react';

export default function DiagnosticPanel({
  cameraStatus,
  wsStatus,
  framesSent,
  framesReceived,
  fps,
  diagnostic = {},
  metrics = {},
  exercise,
  latencyInfo = {}
}) {
  const [isOpen, setIsOpen] = useState(true);

  // Status computation
  const getOverallState = () => {
    if (cameraStatus !== 'connected') {
      return {
        label: 'Camera Standby',
        desc: 'Webcam is not streaming frames. Click Start Camera.',
        color: 'danger'
      };
    }
    if (wsStatus !== 'connected') {
      return {
        label: 'WebSocket Disconnected',
        desc: 'Not connected to Python backend (port 8000).',
        color: 'danger'
      };
    }
    if (!diagnostic.model_loaded) {
      return {
        label: 'MediaPipe Initializing',
        desc: 'pose_landmarker_full.task is initializing.',
        color: 'warning'
      };
    }
    if (!metrics.pose_detected) {
      return {
        label: 'No Body Detected',
        desc: metrics.guidance_message || 'MediaPipe found 0 bodies. Please stand in front of the camera.',
        color: 'warning'
      };
    }

    if (metrics.camera_status === 'NOT_READY') {
      return {
        label: 'Camera Safety Gate Paused',
        desc: metrics.form_error || 'Camera view unsuitable for reliable exercise analysis.',
        color: 'warning'
      };
    }

    return {
      label: 'Pipeline Active & Validating',
      desc: `Tracking active. ${exercise} phase: ${metrics.current_phase || metrics.stage || 'READY'}.`,
      color: 'success'
    };
  };

  const stateInfo = getOverallState();
  const vis = diagnostic.key_landmarks_visibility || {};

  return (
    <div className="diagnostic-card">
      <div className="diagnostic-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="diagnostic-header-left">
          <span className="diag-icon">⚡</span>
          <span className="diag-title">Pipeline Telemetry & Latency HUD</span>
          <span className={`diag-badge badge-${stateInfo.color}`}>
            {stateInfo.label}
          </span>
        </div>
        <button type="button" className="diag-toggle-btn">
          {isOpen ? '▲ Collapse' : '▼ Expand'}
        </button>
      </div>

      {isOpen && (
        <div className="diagnostic-body">
          <div className="diag-status-summary">
            <strong>System Status:</strong> {stateInfo.desc}
          </div>

          {/* Research-Grade Latency Breakdown Section */}
          <div className="latency-breakdown-row">
            <div className="latency-box">
              <span className="latency-num">{latencyInfo.medianMs ?? latencyInfo.roundTripMs ?? 0} ms</span>
              <span className="latency-sub">Median Latency</span>
            </div>
            <div className="latency-box">
              <span className="latency-num">{latencyInfo.p95Ms ?? latencyInfo.roundTripMs ?? 0} ms</span>
              <span className="latency-sub">P95 Latency</span>
            </div>
            <div className="latency-box">
              <span className="latency-num">{latencyInfo.landmarkAgeMs ?? latencyInfo.roundTripMs ?? 0} ms</span>
              <span className="latency-sub">Landmark Age</span>
            </div>
            <div className="latency-box">
              <span className="latency-num">{latencyInfo.inferenceMs ?? 0} ms</span>
              <span className="latency-sub">MediaPipe Task</span>
            </div>
            <div className="latency-box">
              <span className="latency-num">{latencyInfo.detectorMs ?? 0} ms</span>
              <span className="latency-sub">Kinematics</span>
            </div>
            <div className="latency-box">
              <span className="latency-num">{latencyInfo.networkMs ?? 0} ms</span>
              <span className="latency-sub">Network / IO</span>
            </div>
          </div>

          <div className="diag-grid">
            <div className="diag-item">
              <span className="diag-label">1. Activity State</span>
              <span className="diag-val val-info">
                {metrics.activity_state || 'WAITING_FOR_USER'}
              </span>
            </div>

            <div className="diag-item">
              <span className="diag-label">2. Movement Phase</span>
              <span className="diag-val val-good">
                {(metrics.current_phase || metrics.stage || 'READY').toUpperCase()}
              </span>
            </div>

            <div className="diag-item">
              <span className="diag-label">3. Valid vs Rejected Reps</span>
              <span className="diag-val val-info">
                {metrics.valid_reps ?? metrics.reps ?? 0} Valid / {metrics.rejected_attempts ?? 0} Rejected
              </span>
            </div>

            <div className="diag-item">
              <span className="diag-label">4. Capture & Pipeline FPS</span>
              <span className="diag-val val-good">
                {fps} FPS ({framesSent} sent / {latencyInfo.framesProcessed ?? framesReceived} proc / {latencyInfo.framesDropped ?? 0} drop)
              </span>
            </div>

            <div className="diag-item">
              <span className="diag-label">5. MediaPipe Task Model</span>
              <span className={`diag-val ${diagnostic.model_loaded ? 'val-good' : 'val-warn'}`}>
                {diagnostic.model_loaded ? 'Loaded (pose_landmarker_full)' : 'Initializing...'}
              </span>
            </div>

            <div className="diag-item">
              <span className="diag-label">6. Poses Detected</span>
              <span className={`diag-val ${diagnostic.poses_detected_count > 0 ? 'val-good' : 'val-warn'}`}>
                {diagnostic.poses_detected_count ?? (metrics.pose_detected ? 1 : 0)}
              </span>
            </div>

            <div className="diag-item diag-span-2">
              <span className="diag-label">7. Key Joint Tracking Visibility (Threshold: 0.50)</span>
              <div className="diag-vis-pills">
                <span className={`vis-pill ${vis.left_hip >= 0.5 ? 'pill-good' : 'pill-low'}`}>
                  L-Hip: {vis.left_hip ?? '0.00'}
                </span>
                <span className={`vis-pill ${vis.left_knee >= 0.5 ? 'pill-good' : 'pill-low'}`}>
                  L-Knee: {vis.left_knee ?? '0.00'}
                </span>
                <span className={`vis-pill ${vis.left_ankle >= 0.5 ? 'pill-good' : 'pill-low'}`}>
                  L-Ankle: {vis.left_ankle ?? '0.00'}
                </span>
                <span className={`vis-pill ${vis.right_hip >= 0.5 ? 'pill-good' : 'pill-low'}`}>
                  R-Hip: {vis.right_hip ?? '0.00'}
                </span>
                <span className={`vis-pill ${vis.right_knee >= 0.5 ? 'pill-good' : 'pill-low'}`}>
                  R-Knee: {vis.right_knee ?? '0.00'}
                </span>
                <span className={`vis-pill ${vis.right_ankle >= 0.5 ? 'pill-good' : 'pill-low'}`}>
                  R-Ankle: {vis.right_ankle ?? '0.00'}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
