import React, { useState, useEffect } from 'react';
import audioArbiter, { AudioPriority, CoachingMode, AudioState } from '../services/AudioArbiter';

export default function VoiceFeedbackControls({
  spokenFeedback,
  coachingIntensity,
  onIntensityChange
}) {
  const [arbiterSnapshot, setArbiterSnapshot] = useState(() => audioArbiter.getSnapshot());

  // Subscribe to AudioArbiter state updates
  useEffect(() => {
    const unsubscribe = audioArbiter.subscribe((snapshot) => {
      setArbiterSnapshot(snapshot);
    });
    return () => {
      unsubscribe();
    };
  }, []);

  // Sync external prop into arbiter if passed from App
  useEffect(() => {
    if (coachingIntensity && coachingIntensity !== arbiterSnapshot.mode) {
      audioArbiter.setCoachingMode(coachingIntensity);
    }
  }, [coachingIntensity, arbiterSnapshot.mode]);

  // Route incoming spoken feedback from WebSocket metrics through arbiter
  useEffect(() => {
    if (spokenFeedback && spokenFeedback.text) {
      const priority = spokenFeedback.priority || AudioPriority.CORRECTION;
      audioArbiter.requestSpeech({
        text: spokenFeedback.text,
        audio: spokenFeedback.audio || null,
        priority: priority,
        source: 'websocket'
      });
    }
  }, [spokenFeedback]);

  const handleModeSelect = (newMode) => {
    audioArbiter.setCoachingMode(newMode);
    if (onIntensityChange) {
      onIntensityChange(newMode);
    }
  };

  const { state, mode, isSpeaking, isCooldown, cooldownRemaining } = arbiterSnapshot;

  return (
    <div className="voice-controls-chip">
      {/* 3-Option Coaching Intensity Selector */}
      <div className="intensity-selector-group" role="group" aria-label="Coaching Voice Intensity">
        <button
          type="button"
          className={`intensity-btn ${mode === CoachingMode.ERRORS_ONLY ? 'active' : ''}`}
          onClick={() => handleModeSelect(CoachingMode.ERRORS_ONLY)}
          title="Recommended: Speaks only critical safety alerts and exercise corrections"
        >
          Errors Only (Rec)
        </button>
        <button
          type="button"
          className={`intensity-btn ${mode === CoachingMode.STANDARD ? 'active' : ''}`}
          onClick={() => handleModeSelect(CoachingMode.STANDARD)}
          title="Standard: Speaks safety, corrections, posture guidance, milestones & chatbot"
        >
          Standard
        </button>
        <button
          type="button"
          className={`intensity-btn ${mode === CoachingMode.MUTED ? 'active muted' : ''}`}
          onClick={() => handleModeSelect(CoachingMode.MUTED)}
          title="Mute: Completely silent; HUD and text remain 100% active"
        >
          🔇 Muted
        </button>
      </div>

      {/* Real-time State Indicator */}
      <div className={`arbiter-status-pill status-${state.toLowerCase()}`}>
        {isSpeaking && (
          <span className="speaking-wave" title="Coach is speaking aloud">
            <span className="bar b1" />
            <span className="bar b2" />
            <span className="bar b3" />
          </span>
        )}
        <span className="arbiter-state-label">
          {mode === CoachingMode.MUTED
            ? 'Voice Muted'
            : isSpeaking
            ? 'Coach Speaking'
            : isCooldown
            ? `Cooldown (${cooldownRemaining}s)`
            : 'Coach Ready'}
        </span>
      </div>
    </div>
  );
}
