import React, { useState, useEffect } from 'react';
import audioArbiter, { AudioPriority } from '../services/AudioArbiter';

export default function ExerciseDashboard({
  exercises = [],
  selectedExercise,
  onSelectExercise,
  targetSets,
  setTargetSets,
  repsPerSet,
  setRepsPerSet,
  workoutActive,
  onStartWorkout,
  onEndWorkout,
  user,
  onLogin,
  history = [],
  onRefreshHistory,
  isStreaming,
  postSetCoaching = null,
  isReviewLoading = false
}) {
  const [usernameInput, setUsernameInput] = useState('');
  const [activeTab, setActiveTab] = useState('plan'); // 'plan' | 'review' | 'history'

  // Auto-switch to review tab when fresh post-set coaching is loading or arrives
  useEffect(() => {
    if (isReviewLoading || postSetCoaching) {
      setActiveTab('review');
    }
  }, [isReviewLoading, postSetCoaching]);

  const handleLoginSubmit = (e) => {
    e.preventDefault();
    if (usernameInput.trim()) {
      onLogin(usernameInput.trim());
    }
  };

  const handleReplayVoice = () => {
    if (postSetCoaching && postSetCoaching.coaching_text) {
      audioArbiter.requestSpeech({
        text: postSetCoaching.spoken_text || postSetCoaching.coaching_text.split('.')[0] + '.',
        audio: postSetCoaching.spoken_audio || null,
        priority: AudioPriority.CHATBOT,
        source: 'post_set_coach'
      });
    }
  };

  return (
    <div className="card dashboard-card">
      <div className="dashboard-header">
        <div>
          <h3 className="card-title">Training Management</h3>
          <span className="card-subtitle">Set Targets & Biomechanical History</span>
        </div>

        {!user && (
          <form onSubmit={handleLoginSubmit} className="login-form">
            <input
              type="text"
              placeholder="Enter username"
              value={usernameInput}
              onChange={(e) => setUsernameInput(e.target.value)}
              className="input-text"
            />
            <button type="submit" className="btn btn-secondary btn-sm">
              Set User
            </button>
          </form>
        )}

        <div className="dashboard-tab-pills">
          <button
            type="button"
            className={`tab-pill ${activeTab === 'plan' ? 'active' : ''}`}
            onClick={() => setActiveTab('plan')}
          >
            Workout Plan
          </button>
          <button
            type="button"
            className={`tab-pill ${activeTab === 'review' ? 'active' : ''}`}
            onClick={() => setActiveTab('review')}
          >
            AI Review {postSetCoaching && <span className="tab-badge-dot" title="Fresh Post-Set Coaching Available" />}
          </button>
          <button
            type="button"
            className={`tab-pill ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('history');
              if (user && onRefreshHistory) onRefreshHistory();
            }}
          >
            History ({history.length})
          </button>
        </div>
      </div>

      {activeTab === 'plan' && (
        <div className="plan-section">
          <div className="form-group">
            <label htmlFor="exercise-select" className="form-label">
              Choose Exercise
            </label>
            <select
              id="exercise-select"
              value={selectedExercise}
              onChange={(e) => onSelectExercise(e.target.value)}
              disabled={workoutActive}
              className="select-input"
            >
              {exercises.map((ex) => (
                <option key={ex} value={ex}>
                  {ex}
                </option>
              ))}
            </select>
          </div>

          <div className="grid-2-col">
            <div className="form-group">
              <label htmlFor="target-sets" className="form-label">
                Target Sets
              </label>
              <input
                id="target-sets"
                type="number"
                min="1"
                max="50"
                value={targetSets}
                onChange={(e) => setTargetSets(Math.max(1, parseInt(e.target.value) || 1))}
                disabled={workoutActive}
                className="input-text"
              />
            </div>

            <div className="form-group">
              <label htmlFor="reps-per-set" className="form-label">
                Reps per Set
              </label>
              <input
                id="reps-per-set"
                type="number"
                min="1"
                max="100"
                value={repsPerSet}
                onChange={(e) => setRepsPerSet(Math.max(1, parseInt(e.target.value) || 1))}
                disabled={workoutActive}
                className="input-text"
              />
            </div>
          </div>

          <div className="workout-control-actions">
            {!workoutActive ? (
              <button
                type="button"
                className="btn btn-primary btn-block btn-lg"
                onClick={onStartWorkout}
                disabled={!isStreaming}
              >
                {!isStreaming ? 'Start Camera Stream to Begin' : '🚀 Start Active Workout'}
              </button>
            ) : (
              <button
                type="button"
                className="btn btn-danger btn-block btn-lg"
                onClick={onEndWorkout}
              >
                ⏹ Complete & Save Workout
              </button>
            )}
          </div>
        </div>
      )}

      {activeTab === 'review' && (
        <div className="review-section">
          {isReviewLoading ? (
            <div className="empty-coaching-box review-loading">
              <span className="empty-icon">🧠</span>
              <h4>Gemini is reviewing your set...</h4>
              <p className="text-muted">
                Synthesizing kinematics, repetition validity, and retrieving grounded research from exercise science PDF literature.
              </p>
            </div>
          ) : !postSetCoaching ? (
            <div className="empty-coaching-box">
              <span className="empty-icon">🧠</span>
              <h4>No Post-Set Coaching Recorded Yet</h4>
              <p className="text-muted">
                Complete a set of {selectedExercise} ({repsPerSet} repetitions) and your AI Gym Coach will generate
                a research-grounded performance review cited from sports science PDF literature.
              </p>
            </div>
          ) : (
            <div className="post-set-coaching-card">
              <div className="coaching-card-header">
                <div className="coaching-title-group">
                  <span className="coaching-badge-set">Set {postSetCoaching.set_number} Review</span>
                  <span className="coaching-exercise-name">{postSetCoaching.exercise}</span>
                </div>
                <div className="coaching-meta-badges">
                  {postSetCoaching.rag_grounded ? (
                    <span className="rag-grounded-badge" title="Grounded in peer-reviewed exercise science PDF literature">
                      📄 PDF RAG Grounded
                    </span>
                  ) : (
                    <span className="general-coaching-badge">General AI Coaching</span>
                  )}
                  <button
                    type="button"
                    className="btn-replay-voice"
                    onClick={handleReplayVoice}
                    title="Play voice coaching through Audio Arbiter"
                  >
                    🔊 Speak
                  </button>
                </div>
              </div>

              {/* Set Performance Statistics Matrix */}
              <div className="coaching-stats-grid">
                <div className="stat-pill">
                  <span className="stat-label">Total Attempts</span>
                  <span className="stat-value">{postSetCoaching.attempts}</span>
                </div>
                <div className="stat-pill success">
                  <span className="stat-label">Valid Reps</span>
                  <span className="stat-value">{postSetCoaching.valid_reps}</span>
                </div>
                <div className="stat-pill warning">
                  <span className="stat-label">Rejected Reps</span>
                  <span className="stat-value">{postSetCoaching.rejected_reps}</span>
                </div>
                <div className="stat-pill neutral">
                  <span className="stat-label">Main Focus</span>
                  <span className="stat-value error-text">{postSetCoaching.main_error || 'Clean Form'}</span>
                </div>
              </div>

              {/* Gemini Coaching Narrative */}
              <div className="coaching-narrative-box">
                <div className="narrative-header">
                  <span className="coach-avatar">🏋️‍♂️</span>
                  <strong>AI Coach Feedback (Gemini + PDF RAG)</strong>
                </div>
                <p className="coaching-narrative-text">{postSetCoaching.coaching_text}</p>
              </div>

              {/* PDF Citations Box */}
              {postSetCoaching.citations && postSetCoaching.citations.length > 0 && (
                <div className="citations-container">
                  <h5 className="citations-title">
                    <span>📚 Grounded Peer-Reviewed PDF Sources:</span>
                  </h5>
                  {postSetCoaching.citations.map((c, i) => (
                    <div key={i} className="citation-item-card">
                      <div className="citation-header-row">
                        <span className="citation-pdf-name">📄 {c.pdf_name}</span>
                        <span className="citation-page-badge">Page {c.page_number}</span>
                      </div>
                      <div className="citation-academic-ref">{c.citation}</div>
                      <p className="citation-excerpt">"{c.content}"</p>
                    </div>
                  ))}
                </div>
              )}

              {/* Next Set & Rest Guidance */}
              <div className="coaching-rest-guidance-box" style={{ marginTop: '1rem', padding: '0.75rem 1rem', background: 'rgba(56, 189, 248, 0.08)', borderRadius: '8px', border: '1px solid rgba(56, 189, 248, 0.2)', display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.9rem' }}>
                <span className="rest-icon" style={{ fontSize: '1.25rem' }}>⏱️</span>
                <span><strong>Rest & Next Set Guidance:</strong> Rest 60–90 seconds to recover ATP-CP energy stores, then resume when ready for Set {postSetCoaching.set_number + 1}.</span>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'history' && (
        <div className="history-section">
          {history.length === 0 ? (
            <p className="text-muted">No workout history recorded yet for this user.</p>
          ) : (
            <div className="table-responsive">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Exercise</th>
                    <th>Reps</th>
                    <th>Sets</th>
                    <th>Time (s)</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((row) => (
                    <tr key={row.id}>
                      <td className="exercise-cell">{row.exercise}</td>
                      <td>{row.reps}</td>
                      <td>{row.sets}</td>
                      <td>{row.time}s</td>
                      <td className="date-cell">
                        {new Date(row.date).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
