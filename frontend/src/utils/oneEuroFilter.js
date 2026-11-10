/**
 * One Euro Filter for Human Pose Landmarks
 * ========================================
 * Reference: Casiez, G., Roussel, N., & Vogel, F. (2012).
 * "1 € filter: a simple speed-based low-pass filter for noisy input in interactive systems."
 * ACM CHI 2012.
 *
 * Smooths high-frequency camera landmark jitter when stationary (f_min cutoff)
 * while dynamically increasing cutoff frequency (via beta) during fast movement
 * to eliminate lag and phase delay.
 */

class LowPassFilter {
  constructor(alpha = 1.0, initVal = 0.0) {
    self = this;
    this.alpha = alpha;
    this.s = initVal;
    this.initialized = false;
  }

  filter(value, alpha) {
    this.alpha = alpha;
    if (!this.initialized) {
      this.s = value;
      this.initialized = true;
      return value;
    }
    this.s = alpha * value + (1.0 - alpha) * this.s;
    return this.s;
  }

  last() {
    return this.s;
  }

  reset() {
    this.initialized = false;
    this.s = 0.0;
  }
}

class OneEuroFilter1D {
  constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
    this.minCutoff = minCutoff; // Minimum cutoff frequency (Hz)
    this.beta = beta;           // Speed coefficient
    this.dCutoff = dCutoff;     // Derivative cutoff frequency (Hz)
    this.xFilter = new LowPassFilter();
    this.dxFilter = new LowPassFilter();
    this.lastTime = null;
  }

  _alpha(cutoff, dt) {
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(value, timestamp) {
    if (this.lastTime === null || timestamp === undefined) {
      this.lastTime = timestamp || Date.now();
      return this.xFilter.filter(value, 1.0);
    }

    const dt = Math.max(0.001, (timestamp - this.lastTime) / 1000.0);
    this.lastTime = timestamp;

    // Estimate derivative
    const prevX = this.xFilter.last();
    const dx = (value - prevX) / dt;
    const edx = this.dxFilter.filter(dx, this._alpha(this.dCutoff, dt));

    // Adaptive cutoff frequency based on movement speed
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(value, this._alpha(cutoff, dt));
  }

  reset() {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.lastTime = null;
  }
}

/**
 * Filter manager for 33 full-body MediaPipe pose landmarks.
 */
export class PoseOneEuroFilter {
  constructor(minCutoff = 1.2, beta = 0.008, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.filters = [];
    for (let i = 0; i < 33; i++) {
      this.filters.push({
        x: new OneEuroFilter1D(minCutoff, beta, dCutoff),
        y: new OneEuroFilter1D(minCutoff, beta, dCutoff),
        z: new OneEuroFilter1D(minCutoff, beta, dCutoff)
      });
    }
  }

  filterLandmarks(landmarks, timestamp) {
    if (!landmarks || landmarks.length === 0) return [];

    const now = timestamp || performance.now();
    return landmarks.map((lm, idx) => {
      if (idx >= this.filters.length || !lm) return lm;

      const f = this.filters[idx];
      return {
        x: f.x.filter(lm.x, now),
        y: f.y.filter(lm.y, now),
        z: lm.z !== undefined ? f.z.filter(lm.z, now) : 0,
        visibility: lm.visibility ?? 1.0
      };
    });
  }

  reset() {
    for (const f of this.filters) {
      f.x.reset();
      f.y.reset();
      f.z.reset();
    }
  }
}

