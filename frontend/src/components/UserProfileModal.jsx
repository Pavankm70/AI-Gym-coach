import React, { useState, useEffect } from 'react';

export default function UserProfileModal({ isOpen, onClose, user, onUpdateUser, apiBase }) {
  const [profile, setProfile] = useState({
    fitness_level: 'Beginner',
    goals: 'General Fitness',
    preferred_feedback_style: 'Encouraging',
    experience: 'Beginner (0-1 years)',
    workout_frequency: '3 days/week',
    limitations: 'None',
    coaching_intensity: 'ERRORS_ONLY'
  });
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    if (user && user.user_id) {
      // Fetch current profile from backend
      fetch(`${apiBase}/api/user/profile/${user.user_id}`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data) {
            setProfile({
              fitness_level: data.fitness_level || 'Beginner',
              goals: data.goals || 'General Fitness',
              preferred_feedback_style: data.preferred_feedback_style || 'Encouraging',
              experience: data.experience || 'Beginner (0-1 years)',
              workout_frequency: data.workout_frequency || '3 days/week',
              limitations: data.limitations || 'None',
              coaching_intensity: data.coaching_intensity || 'ERRORS_ONLY'
            });
          }
        })
        .catch((err) => console.warn('Could not fetch user profile:', err));
    }
  }, [user, apiBase, isOpen]);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!user || !user.user_id) return;
    setIsSaving(true);
    setSaveSuccess(false);

    try {
      const res = await fetch(`${apiBase}/api/user/profile/${user.user_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(profile)
      });
      if (res.ok) {
        const updated = await res.json();
        setSaveSuccess(true);
        if (onUpdateUser) {
          onUpdateUser({ ...user, ...updated });
        }
        setTimeout(() => {
          setSaveSuccess(false);
          onClose();
        }, 1200);
      }
    } catch (err) {
      console.error('Failed to update profile:', err);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="modal-backdrop">
      <div className="modal-dialog">
        <div className="modal-header">
          <div className="modal-title-group">
            <span className="modal-icon">⚙️</span>
            <div>
              <h3>Athlete Profile & AI Coaching Preferences</h3>
              <p>Personalize how your AI Coach tailors instructions and feedback</p>
            </div>
          </div>
          <button type="button" className="modal-close-btn" onClick={onClose}>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-form">
          <div className="form-group">
            <label className="form-label">Athlete Username</label>
            <input
              type="text"
              className="form-input"
              value={user?.username || 'Athlete'}
              disabled
            />
          </div>

          <div className="form-row">
            <div className="form-group flex-1">
              <label className="form-label">Fitness Level</label>
              <select
                className="form-select"
                value={profile.fitness_level}
                onChange={(e) => setProfile({ ...profile, fitness_level: e.target.value })}
              >
                <option value="Beginner">Beginner</option>
                <option value="Intermediate">Intermediate</option>
                <option value="Advanced">Advanced</option>
              </select>
            </div>

            <div className="form-group flex-1">
              <label className="form-label">Feedback Style</label>
              <select
                className="form-select"
                value={profile.preferred_feedback_style}
                onChange={(e) => setProfile({ ...profile, preferred_feedback_style: e.target.value })}
              >
                <option value="Encouraging">Encouraging & Supportive</option>
                <option value="Direct">Direct & Technical</option>
                <option value="Detailed">Detailed Biomechanical</option>
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group flex-1">
              <label className="form-label">Primary Fitness Goal</label>
              <select
                className="form-select"
                value={profile.goals}
                onChange={(e) => setProfile({ ...profile, goals: e.target.value })}
              >
                <option value="General Fitness">General Fitness & Health</option>
                <option value="Strength & Hypertrophy">Strength & Muscle Hypertrophy</option>
                <option value="Form & Movement Mastery">Form & Movement Mastery</option>
                <option value="Joint Mobility & Safety">Joint Mobility & Injury Prevention</option>
                <option value="Fat Loss & Conditioning">Fat Loss & Conditioning</option>
              </select>
            </div>

            <div className="form-group flex-1">
              <label className="form-label">Workout Frequency</label>
              <select
                className="form-select"
                value={profile.workout_frequency}
                onChange={(e) => setProfile({ ...profile, workout_frequency: e.target.value })}
              >
                <option value="2-3 days/week">2-3 days / week</option>
                <option value="3-4 days/week">3-4 days / week</option>
                <option value="4-5 days/week">4-5 days / week</option>
                <option value="6+ days/week">6+ days / week</option>
              </select>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              Physical Limitations or Injury Precautions
              <span className="label-hint"> (e.g. 'Knee meniscus sensitivity', 'Shoulder impingement', or 'None')</span>
            </label>
            <input
              type="text"
              className="form-input"
              value={profile.limitations}
              onChange={(e) => setProfile({ ...profile, limitations: e.target.value })}
              placeholder="e.g. None, Lower back tightness, Knee pain when squatting below 90°"
            />
          </div>

          <div className="form-group">
            <label className="form-label">
              Voice Coaching Intensity
              <span className="label-hint"> (Controls single-speaker voice priority & frequency)</span>
            </label>
            <select
              className="form-select"
              value={profile.coaching_intensity}
              onChange={(e) => setProfile({ ...profile, coaching_intensity: e.target.value })}
            >
              <option value="ERRORS_ONLY">Errors Only (Recommended) — Safety & form corrections only</option>
              <option value="STANDARD">Standard — Corrections, phase guidance, milestones & chatbot voice</option>
              <option value="MUTED">Muted — Silent; visual HUD, angles & rep counters remain 100% active</option>
            </select>
          </div>

          {saveSuccess && (
            <div className="save-success-banner">
              ✅ Profile updated! AI Coach will personalize responses accordingly.
            </div>
          )}

          <div className="modal-actions">
            <button type="button" className="btn btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={isSaving}>
              {isSaving ? 'Saving...' : 'Save Profile'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

