# AI GYM COACH — Single-Speaker Audio Arbiter & Voice Priority System Report

## 1. Executive Summary & Problem Resolution

In multi-component fitness applications integrating real-time computer vision, biomechanical heuristics, and AI conversational models, uncoordinated voice feedback produces severe cognitive overload. Previously, multiple components triggered speech concurrently:
- Camera framing warnings collided with rep counter announcements.
- Idle guidance collided with form correction instructions.
- Chatbot conversational speech clashed with real-time exercise feedback.
- Repetitive cues triggered on consecutive video frames without deduplication.

To resolve these conflicts, a **Single-Speaker Audio Arbiter and Intelligent Priority System** has been engineered and deployed across the Python backend (`backend/ai_services/audio_arbiter.py`) and React frontend (`frontend/src/services/AudioArbiter.js`).

### Key Invariants Enforced:
1. **Single-Speaker Invariant**: At any millisecond $t$, strictly $\le 1$ voice utterance can be active.
2. **5-Tier Priority Hierarchy**: Critical safety and camera framing events (Priority 1) can immediately interrupt playing audio. Form corrections (Priority 2) supersede milestones and guidance.
3. **Queue Bounding**: At most **one** pending message is retained in memory. Higher-priority incoming events overwrite lower-priority pending events, preventing queued speech storms.
4. **Post-Speech Silence Window**: A strict 5.0-second silence gap is enforced after speech termination before any ordinary instruction can be spoken.
5. **Deduplication Engine**: Identical cues cannot repeat while an error remains active (8.0s to 12.0s suppression window).
6. **Coaching Intensity Modes**: Athletes can choose between `ERRORS_ONLY` (default, recommended), `STANDARD`, and `MUTED`.

---

## 2. System Architecture

```mermaid
flowchart TD
    subgraph EventSources ["Voice Event Sources"]
        E1["Priority 1: Camera Disconnect / Out of Frame"]
        E2["Priority 2: Rep Rejected / Form Error"]
        E3["Priority 3: Posture Setup / Movement Guidance"]
        E4["Priority 4: Milestones / Praise / Idle Tips"]
        E5["Priority 5: Conversational Chatbot Speech"]
    end

    subgraph CentralizedArbiter ["Centralized Audio Arbiter"]
        ModeCheck{"Mode Permitted?\n(ERRORS_ONLY / STANDARD / MUTED)"}
        DedupCheck{"Deduplication Check\n(Is identical cue within window?)"}
        BusyCheck{"Is Coach Currently Speaking?"}
        InterruptEval{"Priority < Current Priority?\n(P1 Safety or P2 Correction)"}
        SilenceCheck{"Post-Speech Cooldown Elapsed?\n(>= 5.0s silence)"}
        AudioExec["Audio Playback Engine\n(HTML5 Audio / Web Speech API)"]
        PendingSlot["Pending Slot (Max 1)\nHigher Priority Overwrites"]
    end

    subgraph AudioOutput ["Unified Audio Output"]
        Speaker(("Single Speaker Output"))
    end

    E1 --> ModeCheck
    E2 --> ModeCheck
    E3 --> ModeCheck
    E4 --> ModeCheck
    E5 --> ModeCheck

    ModeCheck -- Permitted --> DedupCheck
    ModeCheck -- Blocked --> Discard["Discard Speech (Text HUD Active)"]
    DedupCheck -- Fresh --> BusyCheck
    DedupCheck -- Duplicate --> Discard

    BusyCheck -- "Yes (Busy)" --> InterruptEval
    InterruptEval -- "Yes (Higher Priority)" --> AudioExec
    InterruptEval -- "No (Equal/Lower)" --> PendingSlot

    BusyCheck -- "No (Idle/Cooldown)" --> SilenceCheck
    SilenceCheck -- "Elapsed" --> AudioExec
    SilenceCheck -- "In Cooldown" --> PendingSlot

    AudioExec --> Speaker
```

---

## 3. 5-Tier Event Priority System Matrix

| Priority Level | Category | Source Components | Example Utterance | Preemption / Interrupt Capability |
| :--- | :--- | :--- | :--- | :--- |
| **Priority 1** | **Critical Safety & Camera** | `ReliabilityGate`, `CameraValidation` | *"Camera position required. Ensure feet and head are visible."* | **Immediate Preemption**: Interrupts any currently playing audio of lower priority, bypasses post-speech cooldown. |
| **Priority 2** | **Form Corrections & Rep Rejections** | Exercise Detectors (`Squat`, `PushUp`, `ShoulderPress`, etc.) | *"Knees collapsing inward! Push knees outward over toes."* | **Supersedes Announcements**: Preempts or replaces Priority 3, 4, 5. Cannot interrupt Priority 1. |
| **Priority 3** | **Exercise Guidance & Setup** | `ActivityStateManager`, Starting Posture Gates | *"Stand upright with feet shoulder-width apart to begin."* | **Non-preempting**: Must wait for idle state and respect 5.0s silence window. Supersedes Priority 4 & 5 pending. |
| **Priority 4** | **Announcements & Praise** | Repetition Counters, Milestone Engine, Idle Monitor | *"Great work! Five repetitions completed."* | **Low-priority**: Dispatched only in `STANDARD` mode. Blocked during workout in `ERRORS_ONLY` mode. |
| **Priority 5** | **Conversational Chatbot** | LangChain Gemini ReAct Agent (`/api/chat`) | *"Squats target your quadriceps, hamstrings, and gluteus maximus."* | **Conversational**: Dispatched only on user request when athlete is idle or in `STANDARD` mode. First clean sentence spoken. |

---

## 4. Playback State Machine

The Audio Arbiter transitions through four deterministic states:

```mermaid
stateDiagram-v2
    [*] --> IDLE
    IDLE --> PLAYING : requestSpeech() accepted
    PLAYING --> PLAYING : Priority 1 / 2 interrupts lower priority
    PLAYING --> COOLDOWN : onended / playback finished
    PLAYING --> IDLE : onerror / decode failure (lock release)
    COOLDOWN --> IDLE : 5.0s silence window expires
    COOLDOWN --> PLAYING : Priority 1 (Safety) bypasses cooldown
    COOLDOWN --> PLAYING : Pending message dispatches upon timer expiry
    IDLE --> MUTED : user selects MUTED mode
    PLAYING --> MUTED : user selects MUTED mode (stopAll)
    COOLDOWN --> MUTED : user selects MUTED mode
    MUTED --> IDLE : user selects ERRORS_ONLY or STANDARD
```

### State Definitions:
1. **`IDLE`**: Ready to receive and speak eligible coaching messages immediately.
2. **`PLAYING`**: HTML5 `<audio>` or Web Speech API utterance is actively producing sound. Mutex busy lock is locked.
3. **`COOLDOWN`**: Audio has completed playback. A 5.0-second timer ensures real silence so the athlete can focus on physical movement.
4. **`MUTED`**: Audio synthesis and browser speech are completely suppressed. All visual HUD cards, badges, angle overlays, and counters remain 100% operational.

---

## 5. Coaching Intensity Modes & Athlete Control

Athletes configure coaching voice density directly via the header controls chip or their athlete profile modal:

```
+---------------------------------------------------------------------------------+
|  [ Errors Only (Rec) ]   [ Standard ]   [ 🔇 Muted ]      [ Coach Ready / Speaking ]  |
+---------------------------------------------------------------------------------+
```

### 1. `ERRORS_ONLY` (Default & Recommended)
- Speaks **only Priority 1 (Safety)** and **Priority 2 (Technique Corrections & Rep Rejections)**.
- Keeps the coach quiet during normal valid repetitions and setup phases.
- Eliminates voice spam during intense sets.

### 2. `STANDARD`
- Speaks Priorities 1 through 5.
- Includes posture guidance, milestone praise (5, 10 reps), idle coaching tips, and chatbot voice answers.

### 3. `MUTED`
- Completely silences all voice output across all components.
- Visual feedback, angle overlays, rep counters, and text recommendations continue updating in real time.

### State Persistence:
- Synchronized with `localStorage.getItem('gym_coach_coaching_mode')`.
- Persisted in the SQLite `users` table via `coaching_intensity TEXT DEFAULT 'ERRORS_ONLY'`.
- Streamed over WebSocket in both `config` messages and per-frame telemetry.

---

## 6. Dual-Engine Speech Playback & Robust Lock Release

To ensure reliable playback across mobile, desktop, and background environments:
1. **Primary Playback (HTML5 `<audio>`)**: Decodes high-clarity base64 MP3 generated server-side.
2. **Fallback Playback (Web Speech API `SpeechSynthesis`)**: Automatically takes over if HTML5 network or codec decode fails.
3. **Guaranteed Lock Release**:
   - `onended`: Transitions cleanly into `COOLDOWN`.
   - `onerror`: Catches file decoding or network failure and resets lock immediately to `IDLE` (preventing permanent lock freeze).
   - `stopAll()`: Stops audio, cancels browser utterances, and resets state upon React component unmount.

---

## 7. Automated Test Suite (Tests A through K)

The Arbiter architecture has been verified across both Python backend and React frontend via an 11-point test suite:

| Test ID | Verification Target | Observed Behavior | Status |
| :--- | :--- | :--- | :---: |
| **Test A** | Simultaneous ordinary events | First event plays; second event queued without overlap | **PASSED (100%)** |
| **Test B** | Form correction arrives while milestone plays | Priority 2 immediately interrupts and supersedes Priority 4 | **PASSED (100%)** |
| **Test C** | Critical camera warning during guidance | Priority 1 bypasses cooldown and interrupts lower-priority audio | **PASSED (100%)** |
| **Test D** | Consecutive identical form errors | First error spoken; subsequent identical errors suppressed | **PASSED (100%)** |
| **Test E** | Post-speech silence window | Audio at 2.0s and 4.9s rejected; audio at 5.1s accepted | **PASSED (100%)** |
| **Test F** | `ERRORS_ONLY` coaching mode | Announcements, guidance, and chatbot blocked; errors & safety allowed | **PASSED (100%)** |
| **Test G** | `STANDARD` coaching mode | All priorities 1 through 5 permitted with priority arbitration | **PASSED (100%)** |
| **Test H** | `MUTED` coaching mode | All voice suppressed; visual HUD and metrics 100% active | **PASSED (100%)** |
| **Test I** | Dynamic Gemini speech-text sync | Spoken sentence matches the first sentence of displayed chat text | **PASSED (100%)** |
| **Test J** | Audio playback failure recovery | Busy lock resets cleanly to `IDLE` without permanent freeze | **PASSED (100%)** |
| **Test K** | React unmount / subscriber teardown | Subscribers deregister cleanly; zero memory leaks or dangling callbacks | **PASSED (100%)** |

---

## 8. Conclusion

With the deployment of the **Centralized Audio Arbiter**, the AI Gym Coach delivers a single-speaker fitness coaching experience. Voice feedback is synchronized, safety alerts take instant preemption, and repetitive voice spam is eliminated.

