import React from 'react';

export default function CameraReliabilityBadge({ reliability }) {
  if (!reliability) {
    return (
      <div className="reliability-badge-card standby">
        <span className="rel-icon">📷</span>
        <span className="rel-label">Camera Validation: Standby</span>
      </div>
    );
  }

  const cameraStatus = reliability.camera_status || 'NOT_READY';
  const poseReliability = reliability.pose_reliability || 'UNRELIABLE';
  const issues = reliability.issues || [];
  const suggestions = reliability.suggestions || [];
  const viewSuitability = reliability.camera_view || 'SUITABLE';
  const stability = reliability.landmark_stability || 'GOOD';

  const isReady = cameraStatus === 'READY' && poseReliability !== 'UNRELIABLE';
  const isDegraded = poseReliability === 'DEGRADED';

  let statusClass = 'not-ready';
  let statusText = 'Camera View Not Ready';

  if (isReady && !isDegraded) {
    statusClass = 'ready';
    statusText = 'Camera & Pose Reliable';
  } else if (isDegraded) {
    statusClass = 'degraded';
    statusText = 'Suboptimal Camera Angle';
  }

  return (
    <div className={`reliability-badge-card ${statusClass}`}>
      <div className="reliability-top-row">
        <div className="reliability-status-group">
          <span className={`status-indicator-dot ${statusClass}`} />
          <span className="reliability-main-title">{statusText}</span>
        </div>
        <div className="reliability-sub-tags">
          <span className={`badge-tag view-${viewSuitability.toLowerCase()}`}>
            View: {viewSuitability}
          </span>
          <span className={`badge-tag stab-${stability.toLowerCase()}`}>
            Stability: {stability}
          </span>
        </div>
      </div>

      {/* When issues exist, show repositioning suggestions clearly */}
      {!isReady && issues.length > 0 && (
        <div className="reliability-alert-box">
          <div className="rel-issue-header">
            ⚠️ <strong>Positioning Required Before Exercise Evaluation:</strong>
          </div>
          <ul className="rel-issues-list">
            {issues.map((iss, i) => (
              <li key={i} className="issue-item">{iss}</li>
            ))}
          </ul>
          {suggestions.length > 0 && (
            <div className="rel-suggestions-box">
              <span className="sugg-title">👉 Action:</span> {suggestions.join(' ')}
            </div>
          )}
        </div>
      )}

      {isReady && isDegraded && issues.length > 0 && (
        <div className="reliability-suboptimal-note">
          ℹ️ {issues[0]} {suggestions[0]}
        </div>
      )}
    </div>
  );
}

