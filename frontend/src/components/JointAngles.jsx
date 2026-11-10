import React from 'react';

export default function JointAngles({ exercise, metrics = {} }) {
  const getStatusBadge = (label, status) => {
    let className = 'badge-neutral';
    const s = (status || '').toUpperCase();

    if (
      s === 'GOOD DEPTH' ||
      s === 'STRAIGHT' ||
      s === 'LEVEL' ||
      s === 'STABLE' ||
      s === 'NO SWING' ||
      s === 'FULL EXTENSION' ||
      s === 'NEUTRAL' ||
      s === 'BALANCED'
    ) {
      className = 'badge-success';
    } else if (
      s === 'TOO HIGH' ||
      s === 'POOR FORM' ||
      s === 'SAGGING' ||
      s === 'PIKED UP' ||
      s === 'ELBOW DRIFTING' ||
      s === 'SWINGING' ||
      s === 'EXCESSIVE ARCH' ||
      s === 'OFF BALANCE'
    ) {
      className = 'badge-danger';
    } else if (s === 'SLIGHT BEND' || s === 'SLIGHT ARCH' || s === 'NEARLY EXTENDED') {
      className = 'badge-warning';
    }

    return <span className={`status-tag ${className}`}>{status || 'N/A'}</span>;
  };

  const renderExerciseMetrics = () => {
    switch (exercise) {
      case 'Squats':
        return (
          <>
            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Knee Angle</span>
                <span className="angle-target">Target: &lt; 100° (down), &ge; 160° (up)</span>
              </div>
              <div className="angle-value-row">
                <span className="angle-number">{metrics.knee_angle ?? 0}°</span>
                {getStatusBadge('Depth', metrics.depth_status)}
              </div>
            </div>

            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Back Angle</span>
                <span className="angle-target">Threshold: &ge; 130°</span>
              </div>
              <div className="angle-value-row">
                <span className="angle-number">{metrics.back_angle ?? 0}°</span>
                <span className="angle-detail">Torso Posture</span>
              </div>
            </div>
          </>
        );

      case 'Push-ups':
        return (
          <>
            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Elbow Angle</span>
                <span className="angle-target">Target: &lt; 90° (down), &gt; 160° (up)</span>
              </div>
              <div className="angle-value-row">
                <span className="angle-number">{metrics.elbow_angle ?? 0}°</span>
              </div>
            </div>

            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Body Alignment</span>
                <span className="angle-target">Spine & Legs</span>
              </div>
              <div className="angle-value-row">
                {getStatusBadge('Body', metrics.body_alignment)}
              </div>
            </div>

            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Hip Position</span>
                <span className="angle-target">Tolerance: 0.08</span>
              </div>
              <div className="angle-value-row">
                {getStatusBadge('Hip', metrics.hip_status)}
              </div>
            </div>
          </>
        );

      case 'Biceps Curls (Dumbbell)':
        return (
          <>
            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Elbow Angle</span>
                <span className="angle-target">Target: &lt; 50° (up), &gt; 160° (down)</span>
              </div>
              <div className="angle-value-row">
                <span className="angle-number">{metrics.elbow_angle ?? 0}°</span>
              </div>
            </div>

            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Shoulder Stability</span>
                <span className="angle-target">Drift: &le; 0.06</span>
              </div>
              <div className="angle-value-row">
                {getStatusBadge('Shoulder', metrics.shoulder_status)}
              </div>
            </div>

            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Torso Swing</span>
                <span className="angle-target">Threshold: &le; 15°</span>
              </div>
              <div className="angle-value-row">
                {getStatusBadge('Swing', metrics.swing_status)}
              </div>
            </div>
          </>
        );

      case 'Shoulder Press':
        return (
          <>
            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Elbow Angle</span>
                <span className="angle-target">Target: &gt; 160° (up), &lt; 90° (down)</span>
              </div>
              <div className="angle-value-row">
                <span className="angle-number">{metrics.elbow_angle ?? 0}°</span>
                {getStatusBadge('Extension', metrics.extension_status)}
              </div>
            </div>

            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Back Arch</span>
                <span className="angle-target">Lumbar spine alignment</span>
              </div>
              <div className="angle-value-row">
                {getStatusBadge('Arch', metrics.back_arch_status)}
              </div>
            </div>
          </>
        );

      case 'Lunges':
        return (
          <>
            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Front Knee Angle</span>
                <span className="angle-target">Target: &lt; 100° (down), &gt; 160° (up)</span>
              </div>
              <div className="angle-value-row">
                <span className="angle-number">{metrics.front_knee_angle ?? 0}°</span>
              </div>
            </div>

            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Torso Angle</span>
                <span className="angle-target">Vertical posture</span>
              </div>
              <div className="angle-value-row">
                <span className="angle-number">{metrics.torso_angle ?? 0}°</span>
              </div>
            </div>

            <div className="angle-card">
              <div className="angle-meta">
                <span className="angle-name">Lateral Balance</span>
                <span className="angle-target">Offset: &le; 0.10</span>
              </div>
              <div className="angle-value-row">
                {getStatusBadge('Balance', metrics.balance_status)}
              </div>
            </div>
          </>
        );

      default:
        return <p className="text-muted">Select an exercise to view angles.</p>;
    }
  };

  return (
    <div className="joint-angles-card">
      <div className="card-header">
        <h3 className="card-title">Live Joint Angles & Biometrics</h3>
        <span className="exercise-tag">{exercise}</span>
      </div>

      <div className="angles-grid">
        {renderExerciseMetrics()}
      </div>
    </div>
  );
}
