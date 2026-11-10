import React, { useState } from 'react';

export default function RepetitionCounter({
  reps = 0,
  validReps = 0,
  rejectedAttempts = 0,
  currentPhase = 'READY',
  rejectionReason = null,
  lastRepValid = null,
  currentSetReps = 0,
  repsPerSet = 10,
  setsCompleted = 0,
  targetSets = 3,
  stage = 'ready',
  workoutCompleted = false,
  metrics = {},
  exercise = 'Squats',
  onReset
}) {
  const [showDebug, setShowDebug] = useState(false);

  const displayReps = validReps !== undefined ? validReps : reps;
  const currentRepProgress = repsPerSet > 0 ? Math.min(100, Math.round((currentSetReps / repsPerSet) * 100)) : 0;
  const overallProgress = targetSets > 0 ? Math.min(100, Math.round((setsCompleted / targetSets) * 100)) : 0;

  // Format active exercise-specific phase
  const activePhase = (currentPhase || stage || 'READY').toUpperCase();

  const getPhaseColorClass = (phaseStr) => {
    switch (phaseStr) {
      case 'OVERHEAD':
      case 'BOTTOM':
      case 'FLEXED':
      case 'COMPLETE':
        return 'phase-peak';
      case 'PRESSING':
      case 'CURLING':
      case 'PUSHING':
      case 'RISING':
        return 'phase-concentric';
      case 'LOWERING':
      case 'RETURNING':
        return 'phase-eccentric';
      case 'RACK':
      case 'START':
      case 'READY':
        return 'phase-ready';
      case 'RESTING':
        return 'phase-resting';
      default:
        return 'phase-default';
    }
  };

  return (
    <div className="rep-counter-card">
      <div className="rep-counter-header">
        <h3 className="card-title">Repetition & Form Validation Monitor</h3>
        <div className="header-actions-group">
          <button
            type="button"
            className={`btn-debug-toggle ${showDebug ? 'active' : ''}`}
            onClick={() => setShowDebug(!showDebug)}
            title="Toggle Kinematic Debug & Angle HUD"
          >
            {showDebug ? 'Hide HUD' : 'Kinematic HUD'}
          </button>
          <button
            type="button"
            className="btn-text-action"
            onClick={onReset}
            title="Reset repetition counters"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="rep-primary-display">
        {/* Valid Repetitions */}
        <div className="rep-big-number">
          <span className="count-number">{displayReps}</span>
          <span className="count-label">VALID REPS</span>
        </div>

        {/* Rejected Repetitions Counter */}
        <div className="rep-rejected-box" title={rejectionReason || 'No rejected repetitions'}>
          <span className="rejected-count">{rejectedAttempts}</span>
          <span className="rejected-label">REJECTED ATTEMPTS</span>
        </div>

        {/* Exercise-Specific Movement Phase */}
        <div className="stage-indicator">
          <span className="stage-caption">Exercise Phase</span>
          <span className={`stage-badge ${getPhaseColorClass(activePhase)}`}>
            {activePhase}
          </span>
        </div>
      </div>

      {/* Rejection Notification Banner */}
      {lastRepValid === false && rejectionReason && (
        <div className="rep-rejected-banner">
          ⚠️ <strong>Repetition Not Counted:</strong> {rejectionReason}
        </div>
      )}

      {workoutCompleted && (
        <div className="workout-completed-banner">
          🎉 <strong>Target Workout Completed!</strong> Fantastic effort!
        </div>
      )}

      <div className="set-metrics-grid">
        <div className="set-metric-box">
          <div className="metric-header">
            <span className="metric-title">Current Set</span>
            <span className="metric-value">
              {currentSetReps} <span className="metric-max">/ {repsPerSet}</span>
            </span>
          </div>
          <div className="progress-bar-bg">
            <div
              className="progress-bar-fill accent-fill"
              style={{ width: `${currentRepProgress}%` }}
            />
          </div>
        </div>

        <div className="set-metric-box">
          <div className="metric-header">
            <span className="metric-title">Sets Completed</span>
            <span className="metric-value">
              {setsCompleted} <span className="metric-max">/ {targetSets}</span>
            </span>
          </div>
          <div className="progress-bar-bg">
            <div
              className="progress-bar-fill success-fill"
              style={{ width: `${overallProgress}%` }}
            />
          </div>
        </div>
      </div>

      {/* Movement-Phase and Joint Telemetry Debugging Panel */}
      {showDebug && (
        <div className="debug-telemetry-hud">
          <div className="debug-hud-title">🔬 Real-time Kinematic Telemetry ({exercise})</div>
          <div className="debug-grid">
            <div className="debug-item">
              <span className="debug-k">Movement Phase:</span>
              <span className="debug-v">{activePhase}</span>
            </div>
            <div className="debug-item">
              <span className="debug-k">Valid Reps:</span>
              <span className="debug-v">{displayReps}</span>
            </div>
            <div className="debug-item">
              <span className="debug-k">Rejected Attempts:</span>
              <span className="debug-v">{rejectedAttempts}</span>
            </div>
            {metrics.knee_angle !== undefined && (
              <div className="debug-item">
                <span className="debug-k">Knee Angle:</span>
                <span className="debug-v">{metrics.knee_angle}°</span>
              </div>
            )}
            {metrics.elbow_angle !== undefined && (
              <div className="debug-item">
                <span className="debug-k">Elbow Angle:</span>
                <span className="debug-v">{metrics.elbow_angle}°</span>
              </div>
            )}
            {metrics.back_angle !== undefined && (
              <div className="debug-item">
                <span className="debug-k">Torso / Back Angle:</span>
                <span className="debug-v">{metrics.back_angle}°</span>
              </div>
            )}
            {metrics.depth_status && (
              <div className="debug-item">
                <span className="debug-k">Depth Status:</span>
                <span className="debug-v">{metrics.depth_status}</span>
              </div>
            )}
            {metrics.extension_status && (
              <div className="debug-item">
                <span className="debug-k">Extension State:</span>
                <span className="debug-v">{metrics.extension_status}</span>
              </div>
            )}
            {metrics.balance_status && (
              <div className="debug-item">
                <span className="debug-k">Balance:</span>
                <span className="debug-v">{metrics.balance_status}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
