import React from 'react';

export default function FeedbackPanel({
  formError,
  formInstruction,
  poseDetected,
  isStreaming,
  exercise,
  reliability,
  cameraStatus,
  activityState,
  guidanceMessage,
  rejectionReason,
  lastRepValid
}) {
  const isCameraBlocked = (
    cameraStatus === 'NOT_READY'
    || cameraStatus === 'BLOCKED'
    || (reliability && (reliability.camera_status === 'NOT_READY' || reliability.gate_outcome === 'CAMERA_BLOCKED'))
  );

  const isCameraWarning = (
    cameraStatus === 'WARNING'
    || (reliability && (reliability.camera_status === 'WARNING' || reliability.gate_outcome === 'CAMERA_WARNING'))
  );

  const cameraIssues = reliability?.issues || [];
  const cameraSuggestions = reliability?.suggestions || [];

  return (
    <div className="feedback-card">
      <div className="feedback-header">
        <div className="header-left">
          <span className="feedback-icon">🤖</span>
          <h3 className="card-title">AI Gym Coach Real-Time Coaching</h3>
        </div>

        <div className="header-status-group">
          {/* Subtle Non-Intrusive Camera Status Pill */}
          {isStreaming && (
            <span
              className={`camera-indicator-pill ${
                isCameraBlocked ? 'pill-blocked' : isCameraWarning ? 'pill-warning' : 'pill-ready'
              }`}
              title={cameraSuggestions[0] || (isCameraBlocked ? 'Camera Blocked' : isCameraWarning ? 'Camera Warning' : 'Camera Ready')}
            >
              <span className="indicator-dot" />
              {isCameraBlocked ? 'Camera Blocked' : isCameraWarning ? 'Camera Warning' : 'Camera Ready'}
            </span>
          )}

          <span className={`feedback-type-badge ${isCameraBlocked ? 'badge-gated' : 'badge-active'}`}>
            {isCameraBlocked
              ? 'Safety Gate: Paused'
              : activityState
              ? activityState.replace(/_/g, ' ')
              : 'Active Analysis'}
          </span>
        </div>
      </div>

      <div className="feedback-body">
        {!isStreaming ? (
          <div className="feedback-idle">
            <p>
              Camera is currently on standby. Start the camera to receive real-time corrective feedback on your {exercise}{' '}
              technique.
            </p>
          </div>
        ) : isCameraBlocked ? (
          /* Genuinely Blocked Camera View */
          <div className="feedback-alert alert-warning">
            <div className="alert-badge">Camera Status</div>
            <div className="alert-title">Camera Blocked</div>
            <div className="alert-message">
              Camera view is blocked. Please move back and make sure your required body joints are visible.
            </div>
            <div className="alert-safety-notice">
              🛡️ <em>Exercise form evaluation is paused until proper camera framing is restored.</em>
            </div>
          </div>
        ) : !poseDetected ? (
          /* Person Not Detected */
          <div className="feedback-alert alert-warning">
            <div className="alert-badge">Positioning Notice</div>
            <div className="alert-title">Person Not Detected</div>
            <div className="alert-message">
              {guidanceMessage || 'Person not detected. Please stand in front of the camera.'}
            </div>
          </div>
        ) : activityState === 'START_POSITION_REQUIRED' || activityState === 'WAITING_FOR_START_POSITION' ? (
          /* Prompt for Starting Position */
          <div className="feedback-alert alert-info">
            <div className="alert-badge">Get Into Starting Position</div>
            <div className="alert-title">Starting Position Needed</div>
            <div className="alert-message">
              {guidanceMessage || `Assume the proper starting position for ${exercise}.`}
            </div>
          </div>
        ) : activityState === 'READY_TO_EXERCISE' || activityState === 'READY_TO_BEGIN' ? (
          /* Ready to Begin */
          <div className="feedback-alert alert-success">
            <div className="alert-badge">Ready to Exercise</div>
            <div className="alert-title">Starting Posture Confirmed</div>
            <div className="alert-message">
              {guidanceMessage || 'Good, you are ready. Start your first repetition.'}
            </div>
          </div>
        ) : activityState === 'SET_COMPLETE' || activityState === 'SET_COMPLETED' ? (
          /* Set Complete */
          <div className="feedback-alert alert-info">
            <div className="alert-badge">Set Completed</div>
            <div className="alert-title">Set Complete!</div>
            <div className="alert-message">
              {guidanceMessage || 'Great work! Take a short rest before your next set.'}
            </div>
          </div>
        ) : activityState === 'RESTING' ? (
          /* Resting Between Sets */
          <div className="feedback-alert alert-info">
            <div className="alert-badge">Resting</div>
            <div className="alert-title">Rest Interval</div>
            <div className="alert-message">
              {guidanceMessage || 'Catch your breath and begin your next repetition whenever you are ready.'}
            </div>
          </div>
        ) : lastRepValid === false && rejectionReason ? (
          /* Rejected Repetition Attempt with Specific Cause */
          <div className="feedback-alert alert-danger">
            <div className="alert-badge">Repetition Rejected</div>
            <div className="alert-title">{rejectionReason}</div>
            <div className="alert-instruction">
              <strong>Requirement:</strong> Ensure full range of motion without form breakdown for the rep to count.
            </div>
          </div>
        ) : formError && !formError.toLowerCase().includes('camera') ? (
          /* Actual Exercise Form Correction */
          <div className="feedback-alert alert-danger">
            <div className="alert-badge">Form Correction</div>
            <div className="alert-title">{formError}</div>
            {formInstruction && (
              <div className="alert-instruction">
                <strong>Corrective Cue:</strong> {formInstruction}
              </div>
            )}
          </div>
        ) : (
          /* Clean Form In Progress */
          <div className="feedback-alert alert-success">
            <div className="alert-badge">Exercise in Progress</div>
            <div className="alert-title">Form looks good!</div>
            <div className="alert-message">
              Maintain your cadence and control through the full range of motion.
            </div>
          </div>
        )}
      </div>

      <div className="exercise-tips-footer">
        <span className="tips-label">Posture Rule:</span>
        {exercise === 'Squats' && (
          <span className="tips-text">Drop your hips until knees bend below 100° for valid depth; maintain back angle above 75°.</span>
        )}
        {exercise === 'Push-ups' && (
          <span className="tips-text">Lower chest until elbows bend below 90°; keep body straight and prevent hips from sagging or piking.</span>
        )}
        {exercise === 'Biceps Curls (Dumbbell)' && (
          <span className="tips-text">Curl until elbow angle is below 50°; keep elbows pinned to sides and prevent torso swinging.</span>
        )}
        {exercise === 'Shoulder Press' && (
          <span className="tips-text">Press overhead past 155° with hands reaching overhead; brace core to prevent lower back arch.</span>
        )}
        {exercise === 'Lunges' && (
          <span className="tips-text">Bend front knee below 100°; maintain an upright torso and hip-width lateral stance for balance.</span>
        )}
      </div>
    </div>
  );
}
