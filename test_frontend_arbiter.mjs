/**
 * AI Gym Coach — Frontend AudioArbiter Unit Tests (Tests A through K)
 * =================================================================
 * Runs under Node.js to verify that CentralizedAudioArbiter satisfies:
 * - Test A: Simultaneous ordinary events -> mutual exclusion, only one plays
 * - Test B: Form correction arrives during announcement -> supersedes/interrupts
 * - Test C: Critical safety warning interrupts immediately
 * - Test D: Consecutive frame error deduplication -> prevents repeated speech
 * - Test E: Post-speech silence window (5.0s) strictly respected
 * - Test F: Coaching intensity ERRORS_ONLY -> blocks milestones, permits corrections
 * - Test G: Coaching intensity STANDARD -> permits all priorities
 * - Test H: Coaching intensity MUTED -> silences all speech
 * - Test I: Spoken text matches requested text
 * - Test J: Playback error resets busy lock cleanly to IDLE
 * - Test K: Component unsubscribe cleans up without duplicate listeners
 */

import { audioArbiter, AudioPriority, AudioState, CoachingMode } from './frontend/src/services/AudioArbiter.js';

function assert(condition, message) {
  if (!condition) {
    console.error(`❌ ASSERTION FAILED: ${message}`);
    process.exit(1);
  }
}

console.log('===============================================================');
console.log('RUNNING FRONTEND AUDIO ARBITER TEST SUITE (TESTS A THROUGH K)');
console.log('===============================================================');

// Mock browser globals for Node environment
globalThis.window = globalThis;
globalThis.speechSynthesis = {
  cancel: () => {},
  speak: () => {}
};
globalThis.SpeechSynthesisUtterance = function(text) {
  this.text = text;
};

// Mock audio element for Node environment
if (!audioArbiter.audioElement) {
  audioArbiter.audioElement = {
    play: () => Promise.resolve(),
    pause: () => {},
    src: '',
    currentTime: 0
  };
}

// Test A: Simultaneous ordinary events
console.log('\n--- [Test A] Simultaneous Events Mutual Exclusion ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.STANDARD);
const p1 = audioArbiter.requestSpeech({ text: 'Squat deeper to reach parallel.', priority: AudioPriority.CORRECTION });
assert(p1 === true, 'First event should play');
assert(audioArbiter.state === AudioState.PLAYING, 'State should be PLAYING');

// Second event of equal priority while playing
const p2 = audioArbiter.requestSpeech({ text: 'Keep your torso upright.', priority: AudioPriority.CORRECTION });
assert(p2 === false, 'Second event should not play simultaneously');
assert(audioArbiter.pendingMessage !== null, 'Second event queued in pending slot');
console.log('  Test A PASSED: Only one event plays, no audio overlap.');

// Test B: Form correction supersedes milestone
console.log('\n--- [Test B] Priority Supersession (Form Correction > Milestone) ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.STANDARD);
const m1 = audioArbiter.requestSpeech({ text: 'Milestone: 5 reps complete!', priority: AudioPriority.ANNOUNCEMENT });
assert(m1 === true, 'Milestone begins playing');
assert(audioArbiter.currentMessage.priority === AudioPriority.ANNOUNCEMENT);

// Form correction (Priority 2) arrives while announcement (Priority 4) is playing
const c1 = audioArbiter.requestSpeech({ text: 'Knees collapsing inward! Push knees out.', priority: AudioPriority.CORRECTION });
assert(c1 === true, 'Priority 2 must supersede/interrupt Priority 4');
assert(audioArbiter.currentMessage.priority === AudioPriority.CORRECTION);
assert(audioArbiter.currentMessage.text === 'Knees collapsing inward! Push knees out.');
console.log('  Test B PASSED: Form correction interrupts playing milestone.');

// Test C: Critical camera warning interrupts immediately
console.log('\n--- [Test C] Safety Warning Immediate Interrupt ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.STANDARD);
audioArbiter.requestSpeech({ text: 'Lower slowly for 3 seconds.', priority: AudioPriority.GUIDANCE });
assert(audioArbiter.state === AudioState.PLAYING);

const safe = audioArbiter.requestSpeech({
  text: 'Camera position required. Ensure feet and head are visible.',
  priority: AudioPriority.CRITICAL_SAFETY
});
assert(safe === true, 'Critical safety alert must interrupt playing guidance');
assert(audioArbiter.currentMessage.priority === AudioPriority.CRITICAL_SAFETY);
console.log('  Test C PASSED: Critical safety alert interrupts playing audio.');

// Test D: Deduplication prevents repeated speech while error persists
console.log('\n--- [Test D] Consecutive Frame Error Deduplication ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.STANDARD);
audioArbiter.requestSpeech({ text: 'Push your knees outward.', priority: AudioPriority.CORRECTION });
audioArbiter._handleSpeechEnded(); // Audio ends, enters cooldown

// Same error arriving right after
const dup = audioArbiter.requestSpeech({ text: 'Push your knees outward.', priority: AudioPriority.CORRECTION });
assert(dup === false, 'Duplicate message must be blocked by deduplication');
console.log('  Test D PASSED: Deduplication suppresses repeated speech.');

// Test E: Post-speech silence window (5.0s)
console.log('\n--- [Test E] Post-Speech Silence Window Enforced ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.STANDARD);
audioArbiter.requestSpeech({ text: 'Repetition valid. Good depth.', priority: AudioPriority.GUIDANCE });
audioArbiter._handleSpeechEnded(); // enters COOLDOWN
assert(audioArbiter.state === AudioState.COOLDOWN, 'State should be COOLDOWN after speech ends');

// Ordinary guidance during cooldown
const early = audioArbiter.requestSpeech({ text: 'Keep your core engaged.', priority: AudioPriority.GUIDANCE });
assert(early === false, 'Ordinary speech must be blocked during cooldown');
console.log('  Test E PASSED: Post-speech silence window strictly enforced.');

// Test F: Coaching intensity ERRORS_ONLY mode
console.log('\n--- [Test F] ERRORS_ONLY Coaching Intensity ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.ERRORS_ONLY);
assert(audioArbiter.isPriorityPermitted(AudioPriority.CRITICAL_SAFETY) === true, 'P1 allowed');
assert(audioArbiter.isPriorityPermitted(AudioPriority.CORRECTION) === true, 'P2 allowed');
assert(audioArbiter.isPriorityPermitted(AudioPriority.GUIDANCE) === false, 'P3 blocked');
assert(audioArbiter.isPriorityPermitted(AudioPriority.ANNOUNCEMENT) === false, 'P4 blocked');
assert(audioArbiter.isPriorityPermitted(AudioPriority.CHATBOT) === false, 'P5 blocked');
console.log('  Test F PASSED: ERRORS_ONLY blocks guidance, milestones and chatbot.');

// Test G: Coaching intensity STANDARD mode
console.log('\n--- [Test G] STANDARD Coaching Intensity ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.STANDARD);
assert(audioArbiter.isPriorityPermitted(AudioPriority.CRITICAL_SAFETY) === true);
assert(audioArbiter.isPriorityPermitted(AudioPriority.CORRECTION) === true);
assert(audioArbiter.isPriorityPermitted(AudioPriority.GUIDANCE) === true);
assert(audioArbiter.isPriorityPermitted(AudioPriority.ANNOUNCEMENT) === true);
assert(audioArbiter.isPriorityPermitted(AudioPriority.CHATBOT) === true);
console.log('  Test G PASSED: STANDARD mode permits all priority levels.');

// Test H: Coaching intensity MUTED mode
console.log('\n--- [Test H] MUTED Coaching Intensity ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.MUTED);
assert(audioArbiter.isPriorityPermitted(AudioPriority.CRITICAL_SAFETY) === false);
assert(audioArbiter.isPriorityPermitted(AudioPriority.CORRECTION) === false);
assert(audioArbiter.isPriorityPermitted(AudioPriority.GUIDANCE) === false);
assert(audioArbiter.isPriorityPermitted(AudioPriority.ANNOUNCEMENT) === false);
assert(audioArbiter.isPriorityPermitted(AudioPriority.CHATBOT) === false);
const mutedReq = audioArbiter.requestSpeech({ text: 'Test text', priority: AudioPriority.CORRECTION });
assert(mutedReq === false, 'All requests blocked when MUTED');
console.log('  Test H PASSED: MUTED mode completely silences all voice output.');

// Test I: Spoken text matches requested text
console.log('\n--- [Test I] Spoken Text Matches Requested Text ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.STANDARD);
const sampleText = 'Maintain a straight vertical spine throughout the movement.';
audioArbiter.requestSpeech({ text: sampleText, priority: AudioPriority.CORRECTION });
assert(audioArbiter.currentMessage.text === sampleText, 'Current message text matches requested text');
assert(audioArbiter.lastSpokenText === sampleText, 'Last spoken text matches requested text');
console.log('  Test I PASSED: Spoken text matches displayed text.');

// Test J: Playback error resets busy lock cleanly to IDLE
console.log('\n--- [Test J] Playback Error Lock Recovery ---');
audioArbiter.stopAll();
audioArbiter.setCoachingMode(CoachingMode.STANDARD);
audioArbiter.requestSpeech({ text: 'Some text', priority: AudioPriority.CORRECTION });
assert(audioArbiter.state === AudioState.PLAYING);
// Trigger error handler
audioArbiter._handleSpeechError('Simulated network decode failure');
assert(audioArbiter.state === AudioState.IDLE, 'State must reset to IDLE after error');
assert(audioArbiter.currentMessage === null, 'Current message cleared');
assert(audioArbiter.pendingMessage === null, 'Pending message cleared');
console.log('  Test J PASSED: Busy lock resets cleanly to IDLE on error.');

// Test K: React unsubscribe teardown without lingering listeners
console.log('\n--- [Test K] Component Subscription & Unsubscribe Cleanup ---');
let notifyCount = 0;
const testListener = (snap) => { notifyCount++; };
const unsub = audioArbiter.subscribe(testListener);
assert(audioArbiter.subscribers.has(testListener), 'Subscriber added');
const countBefore = notifyCount;

// Trigger notification
audioArbiter.setCoachingMode(CoachingMode.STANDARD);
assert(notifyCount > countBefore, 'Listener received update');

// Unsubscribe
unsub();
assert(!audioArbiter.subscribers.has(testListener), 'Subscriber cleanly removed');
const countAfter = notifyCount;
audioArbiter.setCoachingMode(CoachingMode.ERRORS_ONLY);
assert(notifyCount === countAfter, 'Unsubscribed listener receives zero further updates');
console.log('  Test K PASSED: Clean teardown, no lingering listeners or memory leaks.');

console.log('\n===============================================================');
console.log('ALL FRONTEND AUDIO ARBITER TESTS A THROUGH K PASSED (100%)');
console.log('===============================================================');
