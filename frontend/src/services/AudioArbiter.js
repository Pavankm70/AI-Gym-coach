/**
 * AI Gym Coach — Centralized Frontend Audio Arbiter
 * ==================================================
 * Single-speaker audio arbiter enforcing:
 * 1. Mutual Exclusion: Only ONE voice plays at any moment.
 * 2. 5-Tier Priority Hierarchy: Priority 1 (Safety) can interrupt lower-priority audio.
 * 3. Priority Replacement: At most ONE pending message (higher priority replaces lower).
 * 4. Audio Busy Lock: States IDLE, PLAYING, COOLDOWN, MUTED.
 * 5. Post-Speech Silence: Enforces 5.0s quiet gap after ordinary messages.
 * 6. Coaching Intensity Modes: ERRORS_ONLY (recommended), STANDARD, MUTED.
 * 7. Unified Playback: Manages HTML5 <audio> and Web Speech API without simultaneous overlap.
 */

export const AudioPriority = {
  CRITICAL_SAFETY: 1,  // Camera disconnected, body cut off, safety gate active
  CORRECTION: 2,       // Rep rejected, critical form error, wrong start posture
  GUIDANCE: 3,         // Movement phase instructions, start posture setup
  ANNOUNCEMENT: 4,     // Milestones (5, 10 reps), first rep praise, idle reminders
  CHATBOT: 5           // Chatbot conversational voice responses
};

export const AudioState = {
  IDLE: 'IDLE',
  PLAYING: 'PLAYING',
  COOLDOWN: 'COOLDOWN',
  MUTED: 'MUTED'
};

export const CoachingMode = {
  ERRORS_ONLY: 'ERRORS_ONLY', // Recommended: Speaks only Priority 1 & 2
  STANDARD: 'STANDARD',       // Speaks Priority 1 through 5
  MUTED: 'MUTED'              // Zero voice output; visual HUD only
};

class CentralizedAudioArbiter {
  constructor() {
    this.state = AudioState.IDLE;
    this.mode = CoachingMode.ERRORS_ONLY;
    this.silenceWindowMs = 5000; // 5 seconds post-speech silence

    this.currentMessage = null; // { text, audio, priority, source }
    this.pendingMessage = null; // At most ONE pending message
    this.lastSpokenText = '';
    this.lastSpokenEndTimestamp = 0;

    this.audioElement = null;
    this.cooldownTimerId = null;
    this.subscribers = new Set();

    // Initialize HTML5 Audio element
    if (typeof window !== 'undefined') {
      this.audioElement = new Audio();
      this.audioElement.preload = 'auto';

      this.audioElement.onended = () => this._handleSpeechEnded();
      this.audioElement.onerror = (e) => this._handleSpeechError('HTML5 Audio playback error', e);
      this.audioElement.onpause = () => {
        if (this.state === AudioState.PLAYING && !this.audioElement.ended) {
          // Handled if paused externally
        }
      };

      // Load persisted intensity mode if available
      try {
        const saved = localStorage.getItem('gym_coach_coaching_mode');
        if (saved && Object.values(CoachingMode).includes(saved)) {
          this.mode = saved;
          if (saved === CoachingMode.MUTED) {
            this.state = AudioState.MUTED;
          }
        }
      } catch (e) {
        // localStorage unavailable
      }
    }
  }

  // Subscribe UI components to state changes
  subscribe(callback) {
    this.subscribers.add(callback);
    callback(this.getSnapshot());
    return () => this.subscribers.delete(callback);
  }

  _notify() {
    const snapshot = this.getSnapshot();
    for (const sub of this.subscribers) {
      try {
        sub(snapshot);
      } catch (e) {
        console.error('Arbiter subscriber notification error:', e);
      }
    }
  }

  getSnapshot() {
    const now = (typeof performance !== 'undefined') ? performance.now() : Date.now();
    let cooldownRemaining = 0;
    if (this.state === AudioState.COOLDOWN) {
      const elapsed = now - this.lastSpokenEndTimestamp;
      cooldownRemaining = Math.max(0, Math.ceil((this.silenceWindowMs - elapsed) / 1000));
    }

    return {
      state: this.state,
      mode: this.mode,
      currentText: this.currentMessage ? this.currentMessage.text : '',
      currentPriority: this.currentMessage ? this.currentMessage.priority : null,
      isSpeaking: this.state === AudioState.PLAYING,
      isCooldown: this.state === AudioState.COOLDOWN,
      isMuted: this.mode === CoachingMode.MUTED,
      cooldownRemaining
    };
  }

  setCoachingMode(newMode) {
    if (!Object.values(CoachingMode).includes(newMode)) return;

    this.mode = newMode;
    try {
      localStorage.setItem('gym_coach_coaching_mode', newMode);
    } catch (e) {}

    if (newMode === CoachingMode.MUTED) {
      this.stopAll();
      this.state = AudioState.MUTED;
    } else if (this.state === AudioState.MUTED) {
      this.state = AudioState.IDLE;
    }
    this._notify();
  }

  isPriorityPermitted(priority) {
    if (this.mode === CoachingMode.MUTED) return false;
    if (this.mode === CoachingMode.ERRORS_ONLY) {
      return priority === AudioPriority.CRITICAL_SAFETY || priority === AudioPriority.CORRECTION;
    }
    // STANDARD allows all
    return true;
  }

  /**
   * Main entry point for all audio playback requests.
   *
   * @param {object} item
   * @param {string} item.text - Spoken sentence text
   * @param {string} [item.audio] - Base64 data:audio/mp3 URI
   * @param {number} item.priority - AudioPriority enum (1 to 5)
   * @param {string} [item.source] - 'websocket' | 'chatbot' | 'local'
   */
  requestSpeech({ text, audio = null, priority = AudioPriority.CORRECTION, source = 'system' }) {
    if (!text || !text.trim()) return false;
    const cleanText = text.trim();

    // 1. Mode Permission Check
    if (!this.isPriorityPermitted(priority)) {
      return false;
    }

    const now = (typeof performance !== 'undefined') ? performance.now() : Date.now();

    // 2. Prevent repeating identical message while still active
    if (cleanText === this.lastSpokenText && (now - this.lastSpokenEndTimestamp < 8000)) {
      return false;
    }

    // 3. Priority 1 (Critical Safety) can interrupt lower-priority playing audio immediately,
    // UNLESS the active audio is an exercise form correction (P2) or Gemini post-set review (P5),
    // in which case only genuine critical disconnects/emergencies can interrupt.
    if (priority === AudioPriority.CRITICAL_SAFETY) {
      if (this.state === AudioState.PLAYING) {
        const isCriticalDisconnect = /disconnect|lost connection|critical safety/i.test(cleanText);
        const isProtectedAudio = (
          this.currentMessage?.priority === AudioPriority.CORRECTION ||
          this.currentMessage?.priority === AudioPriority.CHATBOT ||
          this.currentMessage?.source === 'post_set_coach'
        );

        if (isProtectedAudio && !isCriticalDisconnect) {
          // Do not interrupt active form correction or Gemini review for ordinary camera repositioning
          if (!this.pendingMessage || priority < this.pendingMessage.priority) {
            this.pendingMessage = { text: cleanText, audio, priority, source, requestedAt: now };
          }
          return false;
        }

        if (this.currentMessage?.priority > AudioPriority.CRITICAL_SAFETY) {
          this._stopCurrentAudio();
        }
      }
      this._playMessage({ text: cleanText, audio, priority, source });
      return true;
    }

    // 4. If audio is actively playing:
    if (this.state === AudioState.PLAYING) {
      // Form corrections (Priority 2) supersede/interrupt lower priority announcements (Priority 4/5)
      if (priority < (this.currentMessage?.priority || 999)) {
        this._stopCurrentAudio();
        this._playMessage({ text: cleanText, audio, priority, source });
        return true;
      }

      // Otherwise evaluate pending slot: At most ONE pending message
      // A higher priority message replaces a lower priority pending message
      if (!this.pendingMessage || priority < this.pendingMessage.priority) {
        this.pendingMessage = { text: cleanText, audio, priority, source, requestedAt: now };
      }
      return false;
    }

    // 5. If in COOLDOWN (post-speech silence window)
    if (this.state === AudioState.COOLDOWN) {
      const elapsed = now - this.lastSpokenEndTimestamp;
      if (elapsed < this.silenceWindowMs) {
        // Queue if higher priority than pending
        if (!this.pendingMessage || priority < this.pendingMessage.priority) {
          this.pendingMessage = { text: cleanText, audio, priority, source, requestedAt: now };
        }
        return false;
      }
      // Cooldown has elapsed
      this.state = AudioState.IDLE;
    }

    // 6. State is IDLE -> Play message immediately
    this._playMessage({ text: cleanText, audio, priority, source });
    return true;
  }

  _playMessage(item) {
    this.currentMessage = item;
    this.lastSpokenText = item.text;
    this.state = AudioState.PLAYING;
    this._notify();

    // Option A: Play base64 MP3 audio from backend
    if (item.audio && this.audioElement) {
      try {
        // Cancel browser Web Speech if active
        if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
          window.speechSynthesis.cancel();
        }

        this.audioElement.src = item.audio;
        const playPromise = this.audioElement.play();
        if (playPromise !== undefined) {
          playPromise.catch((err) => {
            console.warn('[AudioArbiter] HTML5 play failed, falling back to Web Speech API:', err);
            this._fallbackWebSpeech(item.text);
          });
        }
        return;
      } catch (err) {
        console.warn('[AudioArbiter] Audio element error:', err);
        this._fallbackWebSpeech(item.text);
        return;
      }
    }

    // Option B: Native Browser SpeechSynthesis API
    this._fallbackWebSpeech(item.text);
  }

  _fallbackWebSpeech(text) {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      this._handleSpeechError('SpeechSynthesis not supported');
      return;
    }

    try {
      if (this.audioElement) {
        try {
          this.audioElement.pause();
          this.audioElement.currentTime = 0;
        } catch (e) {}
      }
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.05;
      utterance.pitch = 1.0;

      utterance.onend = () => this._handleSpeechEnded();
      utterance.onerror = (e) => this._handleSpeechError('SpeechSynthesis utterance error', e);

      window.speechSynthesis.speak(utterance);
    } catch (e) {
      this._handleSpeechError('SpeechSynthesis exception', e);
    }
  }

  /**
   * Clears obsolete pending messages when activity state transitions
   * e.g. prevents old idle prompts from playing after the user has begun exercising.
   */
  clearObsoletePending(newActivityState) {
    if (!newActivityState) return;
    const state = String(newActivityState).toUpperCase();

    // If user has started exercising, discard pending idle prompts or setup guidance
    if (['EXERCISING', 'MOVEMENT_IN_PROGRESS', 'EXERCISE_IN_PROGRESS', 'EXERCISE_STARTED'].includes(state)) {
      if (this.pendingMessage && this.pendingMessage.priority >= AudioPriority.GUIDANCE) {
        this.pendingMessage = null;
      }
      if (this.state === AudioState.PLAYING && this.currentMessage?.priority >= AudioPriority.ANNOUNCEMENT) {
        this._stopCurrentAudio();
        this.state = AudioState.IDLE;
        this.currentMessage = null;
        this._notify();
      }
    }

    // If user entered rest or completed set, discard pending in-flight movement corrections
    if (['RESTING', 'SET_COMPLETE', 'SET_COMPLETED', 'EXERCISE_COMPLETED', 'WORKOUT_COMPLETE'].includes(state)) {
      if (this.pendingMessage && this.pendingMessage.priority === AudioPriority.CORRECTION) {
        this.pendingMessage = null;
      }
    }
  }

  _handleSpeechEnded() {
    const now = (typeof performance !== 'undefined') ? performance.now() : Date.now();
    this.lastSpokenEndTimestamp = now;
    this.currentMessage = null;

    if (this.mode === CoachingMode.MUTED) {
      this.state = AudioState.MUTED;
      this.pendingMessage = null;
      this._notify();
      return;
    }

    // Enter post-speech silence window (COOLDOWN)
    this.state = AudioState.COOLDOWN;
    this._notify();

    if (this.cooldownTimerId) {
      clearTimeout(this.cooldownTimerId);
    }

    this.cooldownTimerId = setTimeout(() => {
      this.cooldownTimerId = null;
      if (this.state === AudioState.COOLDOWN) {
        this.state = AudioState.IDLE;
        this._notify();

        // Process pending message if one arrived during speech/cooldown
        if (this.pendingMessage) {
          const next = this.pendingMessage;
          this.pendingMessage = null;
          // Check if pending message has not expired (> 6s old)
          const currNow = (typeof performance !== 'undefined') ? performance.now() : Date.now();
          if (currNow - next.requestedAt < 6000) {
            this.requestSpeech(next);
          }
        }
      }
    }, this.silenceWindowMs);
  }

  _handleSpeechError(reason, err = null) {
    console.warn(`[AudioArbiter] Resetting lock due to error: ${reason}`, err);
    this.currentMessage = null;
    this.pendingMessage = null;
    this.state = (this.mode === CoachingMode.MUTED) ? AudioState.MUTED : AudioState.IDLE;
    this._notify();
  }

  _stopCurrentAudio() {
    if (this.audioElement) {
      try {
        this.audioElement.pause();
        this.audioElement.currentTime = 0;
      } catch (e) {}
    }
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch (e) {}
    }
  }

  stopAll() {
    this._stopCurrentAudio();
    if (this.cooldownTimerId) {
      clearTimeout(this.cooldownTimerId);
      this.cooldownTimerId = null;
    }
    this.currentMessage = null;
    this.pendingMessage = null;
    this.state = (this.mode === CoachingMode.MUTED) ? AudioState.MUTED : AudioState.IDLE;
    this._notify();
  }
}

// Global Singleton Export
export const audioArbiter = new CentralizedAudioArbiter();
export default audioArbiter;
