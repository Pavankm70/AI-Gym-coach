"""
Camera-View Validation and Pose-Reliability Assessment Module
============================================================
Research Contribution for Intelligent Exercise Monitoring.

Implements pre-detector validation checks to ensure video frame quality,
exercise-specific landmark visibility, framing boundaries, temporal stability,
and viewpoint suitability before any exercise biomechanics or form errors
are evaluated.

Provides three distinct outcomes:
1. CAMERA_READY: Pose information is sufficiently reliable for all measurements.
2. CAMERA_WARNING: Minor framing or tracking issue, but exercise monitoring continues.
3. CAMERA_BLOCKED: Essential pose information is unusable; exercise evaluation pauses.

Protects athletes from erroneous feedback while ensuring camera reliability
operates quietly in the background without nuisance interruptions during exercise.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np


class CameraValidationConfig:
    """Configurable thresholds for exercise-specific validation."""
    MIN_BRIGHTNESS = 30.0        # Grayscale mean [0..255]
    MAX_BRIGHTNESS = 245.0
    MIN_BLUR_VAR = 25.0          # Laplacian variance
    MIN_IMAGE_WIDTH = 320
    MIN_IMAGE_HEIGHT = 240

    # Normalized margins near image borders [0..1]
    BORDER_MARGIN_Y_TOP = 0.01
    BORDER_MARGIN_Y_BOTTOM = 0.985
    BORDER_MARGIN_X_LEFT = 0.015
    BORDER_MARGIN_X_RIGHT = 0.985

    # Landmark visibility threshold
    MIN_LANDMARK_VISIBILITY = 0.40

    # Jitter / instability threshold (normalized displacement per frame)
    MAX_STABLE_JITTER = 0.32


# Outcome Constants
GATE_READY = "CAMERA_READY"
GATE_WARNING = "CAMERA_WARNING"
GATE_BLOCKED = "CAMERA_BLOCKED"


# Exercise-specific landmark dependency mapping
# MediaPipe Landmark Indices:
# 11: Left Shoulder, 12: Right Shoulder
# 13: Left Elbow,    14: Right Elbow
# 15: Left Wrist,    16: Right Wrist
# 23: Left Hip,      24: Right Hip
# 25: Left Knee,     26: Right Knee
# 27: Left Ankle,    28: Right Ankle
EXERCISE_REQUIRED_LANDMARKS = {
    "Squats": {
        "primary": [23, 24, 25, 26, 27, 28], # Hips, knees, ankles
        "secondary": [11, 12],                # Shoulders (for torso angle)
        "names": {
            23: "Left Hip", 24: "Right Hip",
            25: "Left Knee", 26: "Right Knee",
            27: "Left Ankle", 28: "Right Ankle",
            11: "Left Shoulder", 12: "Right Shoulder"
        },
        "full_body": True,
        "overhead_clearance": False,
    },
    "Push-ups": {
        "primary": [11, 12, 13, 14, 15, 16, 23, 24], # Shoulders, elbows, wrists, hips
        "secondary": [27, 28],                        # Ankles
        "names": {
            11: "Left Shoulder", 12: "Right Shoulder",
            13: "Left Elbow", 14: "Right Elbow",
            15: "Left Wrist", 16: "Right Wrist",
            23: "Left Hip", 24: "Right Hip",
            27: "Left Ankle", 28: "Right Ankle"
        },
        "full_body": False,
        "overhead_clearance": False,
    },
    "Biceps Curls (Dumbbell)": {
        "primary": [11, 12, 13, 14, 15, 16], # Shoulders, elbows, wrists
        "secondary": [23, 24],                # Hips (for torso swing)
        "names": {
            11: "Left Shoulder", 12: "Right Shoulder",
            13: "Left Elbow", 14: "Right Elbow",
            15: "Left Wrist", 16: "Right Wrist",
            23: "Left Hip", 24: "Right Hip"
        },
        "full_body": False,
        "overhead_clearance": False,
    },
    "Shoulder Press": {
        "primary": [11, 12, 13, 14, 15, 16], # Shoulders, elbows, wrists
        "secondary": [23, 24, 25, 26],        # Hips, knees (for back arch)
        "names": {
            11: "Left Shoulder", 12: "Right Shoulder",
            13: "Left Elbow", 14: "Right Elbow",
            15: "Left Wrist", 16: "Right Wrist",
            23: "Left Hip", 24: "Right Hip"
        },
        "full_body": False,
        "overhead_clearance": True, # Wrists must not get cut off at top of frame
    },
    "Lunges": {
        "primary": [23, 24, 25, 26, 27, 28], # Hips, knees, ankles
        "secondary": [11, 12],                # Shoulders
        "names": {
            23: "Left Hip", 24: "Right Hip",
            25: "Left Knee", 26: "Right Knee",
            27: "Left Ankle", 28: "Right Ankle",
            11: "Left Shoulder", 12: "Right Shoulder"
        },
        "full_body": True,
        "overhead_clearance": False,
    },
}


def check_camera_frame_quality(frame: Optional[np.ndarray]) -> Tuple[bool, bool, List[str], List[str], Dict[str, Any]]:
    """
    Check 1: Camera Frame Quality
    Returns (is_ok, is_fatal, issues, suggestions, metrics).
    """
    issues = []
    suggestions = []
    metrics = {
        "width": 0,
        "height": 0,
        "brightness": 0.0,
        "blur_score": 0.0
    }

    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False, True, ["Video frame is missing or unreadable."], ["Ensure your camera is connected and permissions are granted."], metrics

    h, w = frame.shape[:2]
    metrics["width"] = w
    metrics["height"] = h

    if w < CameraValidationConfig.MIN_IMAGE_WIDTH or h < CameraValidationConfig.MIN_IMAGE_HEIGHT:
        issues.append(f"Video resolution {w}x{h} is below minimum requirement of {CameraValidationConfig.MIN_IMAGE_WIDTH}x{CameraValidationConfig.MIN_IMAGE_HEIGHT}.")
        suggestions.append("Switch to a higher camera resolution (at least 640x480 recommended).")

    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    brightness = float(np.mean(gray))
    metrics["brightness"] = round(brightness, 1)

    is_fatal = False
    if brightness < CameraValidationConfig.MIN_BRIGHTNESS:
        issues.append("Camera environment is too dark (underexposed).")
        suggestions.append("Turn on additional room lighting or face towards a light source.")
        is_fatal = True
    elif brightness > CameraValidationConfig.MAX_BRIGHTNESS:
        issues.append("Camera environment is heavily overexposed / washed out.")
        suggestions.append("Reduce harsh direct backlight or glare behind you.")

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["blur_score"] = round(blur_var, 1)

    if blur_var < CameraValidationConfig.MIN_BLUR_VAR:
        issues.append("Camera feed is severely blurred or out of focus.")
        suggestions.append("Clean camera lens or adjust camera focus.")
        if blur_var < 10.0:
            is_fatal = True

    is_ok = len(issues) == 0
    return is_ok, is_fatal, issues, suggestions, metrics


def check_person_detected(landmarks: Optional[List[Any]]) -> Tuple[bool, List[str], List[str]]:
    """
    Check 2: Person Detection
    Verifies that human body landmarks are detected in the scene.
    """
    if landmarks is None or len(landmarks) == 0:
        return (
            False,
            ["No person detected in the camera frame."],
            ["Step into the camera frame and face the lens to begin."]
        )
    return True, [], []


def check_required_landmarks_visible(
    landmarks: List[Any],
    exercise_name: str
) -> Tuple[bool, bool, List[str], List[str]]:
    """
    Check 3: Required Landmarks Visibility (Configured Separately per Exercise)
    Does NOT require all 33 landmarks.
    Returns (is_ready, is_fatal, issues, suggestions).
    """
    config = EXERCISE_REQUIRED_LANDMARKS.get(exercise_name, EXERCISE_REQUIRED_LANDMARKS["Squats"])
    primary_indices = config["primary"]
    names = config["names"]

    # Squats & Lunges: Need at least one leg clearly visible
    if exercise_name in ["Squats", "Lunges"]:
        left_leg_ok = (
            len(landmarks) > 27
            and getattr(landmarks[23], "visibility", 0.0) >= 0.35
            and getattr(landmarks[25], "visibility", 0.0) >= 0.35
            and getattr(landmarks[27], "visibility", 0.0) >= 0.30
        )
        right_leg_ok = (
            len(landmarks) > 28
            and getattr(landmarks[24], "visibility", 0.0) >= 0.35
            and getattr(landmarks[26], "visibility", 0.0) >= 0.35
            and getattr(landmarks[28], "visibility", 0.0) >= 0.30
        )
        if not (left_leg_ok or right_leg_ok):
            return (
                False,
                True,
                ["Lower body joints (hips, knees, ankles) are occluded or not clearly visible."],
                ["Ensure your legs and feet are unobstructed by furniture or loose clothing."]
            )
        return True, False, [], []

    # Shoulder Press: Need at least one arm (shoulder, elbow, wrist) clearly visible
    if exercise_name == "Shoulder Press":
        left_arm_ok = (
            len(landmarks) > 15
            and getattr(landmarks[11], "visibility", 0.0) >= 0.35
            and getattr(landmarks[13], "visibility", 0.0) >= 0.35
            and getattr(landmarks[15], "visibility", 0.0) >= 0.35
        )
        right_arm_ok = (
            len(landmarks) > 16
            and getattr(landmarks[12], "visibility", 0.0) >= 0.35
            and getattr(landmarks[14], "visibility", 0.0) >= 0.35
            and getattr(landmarks[16], "visibility", 0.0) >= 0.35
        )
        if not (left_arm_ok or right_arm_ok):
            return (
                False,
                True,
                ["Arm joints (shoulders, elbows, wrists) are not clearly visible."],
                ["Position the camera so your upper body and arms are clearly in view."]
            )
        return True, False, [], []

    # Biceps Curls: Need at least one arm clearly visible
    if exercise_name == "Biceps Curls (Dumbbell)":
        left_arm_ok = (
            len(landmarks) > 15
            and getattr(landmarks[11], "visibility", 0.0) >= 0.35
            and getattr(landmarks[13], "visibility", 0.0) >= 0.35
            and getattr(landmarks[15], "visibility", 0.0) >= 0.35
        )
        right_arm_ok = (
            len(landmarks) > 16
            and getattr(landmarks[12], "visibility", 0.0) >= 0.35
            and getattr(landmarks[14], "visibility", 0.0) >= 0.35
            and getattr(landmarks[16], "visibility", 0.0) >= 0.35
        )
        if not (left_arm_ok or right_arm_ok):
            return (
                False,
                True,
                ["Arm joints (shoulder, elbow, wrist) are not clearly visible."],
                ["Ensure your arms and dumbbells are clearly in view of the camera."]
            )
        return True, False, [], []

    # Push-ups: Need at least one arm and torso/hips visible
    if exercise_name == "Push-ups":
        left_side_ok = (
            len(landmarks) > 23
            and getattr(landmarks[11], "visibility", 0.0) >= 0.35
            and getattr(landmarks[13], "visibility", 0.0) >= 0.35
            and getattr(landmarks[15], "visibility", 0.0) >= 0.35
            and getattr(landmarks[23], "visibility", 0.0) >= 0.35
        )
        right_side_ok = (
            len(landmarks) > 24
            and getattr(landmarks[12], "visibility", 0.0) >= 0.35
            and getattr(landmarks[14], "visibility", 0.0) >= 0.35
            and getattr(landmarks[16], "visibility", 0.0) >= 0.35
            and getattr(landmarks[24], "visibility", 0.0) >= 0.35
        )
        if not (left_side_ok or right_side_ok):
            return (
                False,
                True,
                ["Upper body joints for push-ups are not clearly visible."],
                ["Position the camera at floor level from a side angle to capture your push-up plank."]
            )
        return True, False, [], []

    # Generic fallback: check if more than half of primary landmarks have good visibility
    missing = [
        names.get(idx, f"Joint {idx}") for idx in primary_indices
        if idx >= len(landmarks) or getattr(landmarks[idx], "visibility", 0.0) < CameraValidationConfig.MIN_LANDMARK_VISIBILITY
    ]
    if len(missing) > len(primary_indices) // 2:
        return (
            False,
            True,
            [f"Required landmarks not visible: {', '.join(missing[:4])}."],
            ["Adjust your position so required body parts are clearly visible."]
        )

    return True, False, [], []


def check_full_body_visibility(
    landmarks: List[Any],
    exercise_name: str
) -> Tuple[bool, bool, List[str], List[str]]:
    """
    Check 4: Framing Boundaries & Tolerance
    Distinguishes minor movements within the frame from genuinely blocked views.
    Returns (is_ok, is_fatal, issues, suggestions).
    """
    config = EXERCISE_REQUIRED_LANDMARKS.get(exercise_name, EXERCISE_REQUIRED_LANDMARKS["Squats"])
    issues = []
    suggestions = []
    is_fatal = False

    # Check ankle bounds for lower-body exercises
    if config["full_body"]:
        left_ankle = landmarks[27] if len(landmarks) > 27 else None
        right_ankle = landmarks[28] if len(landmarks) > 28 else None
        left_knee = landmarks[25] if len(landmarks) > 25 else None
        right_knee = landmarks[26] if len(landmarks) > 26 else None

        if left_ankle and right_ankle:
            # Both ankles cut off at bottom border
            if left_ankle.y > CameraValidationConfig.BORDER_MARGIN_Y_BOTTOM and right_ankle.y > CameraValidationConfig.BORDER_MARGIN_Y_BOTTOM:
                issues.append("Ankle and feet landmarks are cut off at the bottom of the camera frame.")
                suggestions.append("Move the camera backwards or tilt slightly downward to capture your entire body from head to toe.")
                # Fatal if knees are also cut off or if ankles are severely off-screen
                if (left_knee and left_knee.y > 0.95) or (right_knee and right_knee.y > 0.95) or (left_ankle.y >= 0.99 and right_ankle.y >= 0.99):
                    is_fatal = True

    # Check overhead clearance for shoulder press
    if config["overhead_clearance"]:
        left_wrist = landmarks[15] if len(landmarks) > 15 else None
        right_wrist = landmarks[16] if len(landmarks) > 16 else None
        if left_wrist and right_wrist:
            if left_wrist.y < CameraValidationConfig.BORDER_MARGIN_Y_TOP or right_wrist.y < CameraValidationConfig.BORDER_MARGIN_Y_TOP:
                issues.append("Hands are cut off at the top of the camera frame when reaching overhead.")
                suggestions.append("Tilt your camera up or step backwards so your overhead reach fits completely in frame.")
                if left_wrist.y <= 0.0 or right_wrist.y <= 0.0:
                    is_fatal = True

    # Check severe lateral cropping (only if torso center is completely off-screen)
    left_shoulder = landmarks[11] if len(landmarks) > 11 else None
    right_shoulder = landmarks[12] if len(landmarks) > 12 else None
    if left_shoulder and right_shoulder:
        mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2.0
        # Tolerant range: [0.03 .. 0.97]
        if mid_shoulder_x < 0.03:
            issues.append("You are too close to the left edge of the camera view.")
            suggestions.append("Shift slightly towards the center of the camera frame.")
            is_fatal = (max(left_shoulder.x, right_shoulder.x) < 0.01)
        elif mid_shoulder_x > 0.97:
            issues.append("You are too close to the right edge of the camera view.")
            suggestions.append("Shift slightly towards the center of the camera frame.")
            is_fatal = (min(left_shoulder.x, right_shoulder.x) > 0.99)

    is_ok = len(issues) == 0
    return is_ok, is_fatal, issues, suggestions


def check_landmark_stability(
    current_landmarks: List[Any],
    prev_landmarks: Optional[List[Any]]
) -> Tuple[bool, str, List[str], List[str]]:
    """
    Check 5: Landmark Temporal Stability
    Measures frame-to-frame displacement of torso anchors to filter tracking teleportation.
    """
    if prev_landmarks is None or len(prev_landmarks) < 25 or len(current_landmarks) < 25:
        return True, "GOOD", [], []

    dx_sh = current_landmarks[11].x - current_landmarks[12].x
    dy_sh = current_landmarks[11].y - current_landmarks[12].y
    shoulder_dist = math.sqrt(dx_sh**2 + dy_sh**2)
    if shoulder_dist < 0.05:
        shoulder_dist = 0.20

    anchors = [11, 12, 23, 24]
    displacements = []
    for idx in anchors:
        if idx < len(current_landmarks) and idx < len(prev_landmarks):
            p1 = current_landmarks[idx]
            p2 = prev_landmarks[idx]
            dist = math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
            displacements.append(dist / shoulder_dist)

    max_disp = max(displacements) if displacements else 0.0

    if max_disp > CameraValidationConfig.MAX_STABLE_JITTER:
        return (
            False,
            "UNSTABLE",
            ["High landmark tracking jitter / rapid position jump detected."],
            ["Hold still for a moment to allow the pose tracker to stabilize."]
        )

    return True, "GOOD", [], []


def check_camera_view_suitability(
    landmarks: List[Any],
    exercise_name: str
) -> Tuple[bool, str, List[str], List[str]]:
    """
    Check 6: Camera View Suitability
    Provides non-blocking viewpoint guidance. Suboptimal views yield CAMERA_WARNING,
    never CAMERA_BLOCKED, unless required joints are totally unmeasurable.
    """
    issues = []
    suggestions = []

    if len(landmarks) < 25:
        return True, "SUITABLE", [], []

    ls, rs = landmarks[11], landmarks[12]
    shoulder_width = abs(ls.x - rs.x)
    mid_shoulder_y = (ls.y + rs.y) / 2.0
    mid_hip_y = (landmarks[23].y + landmarks[24].y) / 2.0
    torso_height = abs(mid_hip_y - mid_shoulder_y)
    aspect_ratio = (shoulder_width / torso_height) if torso_height > 0.01 else 1.0

    # Shoulder Press: Extreme side profile warning only
    if exercise_name == "Shoulder Press":
        if aspect_ratio < 0.16 and (ls.visibility < 0.35 or rs.visibility < 0.35):
            issues.append("Side profile angle detected. One arm may be occluded.")
            suggestions.append("Face the camera directly for optimal overhead shoulder press monitoring.")

    status = "SUITABLE" if len(issues) == 0 else "SUBOPTIMAL"
    return len(issues) == 0, status, issues, suggestions


def evaluate_pose_reliability(
    frame: Optional[np.ndarray],
    landmarks: Optional[List[Any]],
    exercise_name: str,
    prev_landmarks: Optional[List[Any]] = None
) -> Dict[str, Any]:
    """
    Check 7: Complete 3-Outcome Pose Reliability Evaluation
    Classifies into:
    1. CAMERA_READY (status='READY', reliability='RELIABLE')
    2. CAMERA_WARNING (status='WARNING', reliability='DEGRADED')
    3. CAMERA_BLOCKED (status='NOT_READY', reliability='UNRELIABLE')
    """
    all_issues = []
    all_suggestions = []

    # 1. Frame Quality
    frame_ok, frame_fatal, q_issues, q_sugg, q_metrics = check_camera_frame_quality(frame)
    all_issues.extend(q_issues)
    all_suggestions.extend(q_sugg)

    if frame_fatal or not frame_ok:
        return {
            "camera_status": "NOT_READY",
            "pose_reliability": "UNRELIABLE",
            "gate_outcome": GATE_BLOCKED,
            "person_detected": False,
            "required_landmarks_visible": False,
            "landmark_stability": "UNSTABLE",
            "camera_view": "UNSUITABLE",
            "issues": all_issues,
            "suggestions": all_suggestions,
            "quality_metrics": q_metrics
        }

    # 2. Person Detected
    person_ok, p_issues, p_sugg = check_person_detected(landmarks)
    all_issues.extend(p_issues)
    all_suggestions.extend(p_sugg)

    if not person_ok or landmarks is None:
        return {
            "camera_status": "NOT_READY",
            "pose_reliability": "UNRELIABLE",
            "gate_outcome": GATE_BLOCKED,
            "person_detected": False,
            "required_landmarks_visible": False,
            "landmark_stability": "UNSTABLE",
            "camera_view": "UNSUITABLE",
            "issues": all_issues,
            "suggestions": all_suggestions,
            "quality_metrics": q_metrics
        }

    # 3. Required Landmarks Visible (Exercise-specific)
    req_ok, req_fatal, r_issues, r_sugg = check_required_landmarks_visible(landmarks, exercise_name)
    all_issues.extend(r_issues)
    all_suggestions.extend(r_sugg)

    # 4. Framing Boundaries
    frame_bounds_ok, bounds_fatal, b_issues, b_sugg = check_full_body_visibility(landmarks, exercise_name)
    all_issues.extend(b_issues)
    all_suggestions.extend(b_sugg)

    # 5. Temporal Stability
    stab_ok, stab_status, s_issues, s_sugg = check_landmark_stability(landmarks, prev_landmarks)
    all_issues.extend(s_issues)
    all_suggestions.extend(s_sugg)

    # 6. Viewpoint Suitability
    view_ok, view_status, v_issues, v_sugg = check_camera_view_suitability(landmarks, exercise_name)
    all_issues.extend(v_issues)
    all_suggestions.extend(v_sugg)

    # Deduplicate issues and suggestions
    unique_issues = list(dict.fromkeys(all_issues))
    unique_suggestions = list(dict.fromkeys(all_suggestions))

    # Determine 3-Gate Outcome
    is_fatal_block = (
        req_fatal
        or bounds_fatal
        or (not stab_ok)
    )

    if is_fatal_block:
        gate_outcome = GATE_BLOCKED
        status = "NOT_READY"
        reliability = "UNRELIABLE"
    elif (not frame_bounds_ok) or (not req_ok) or (not view_ok) or (len(unique_issues) > 0):
        gate_outcome = GATE_WARNING
        status = "WARNING"
        reliability = "DEGRADED"
    else:
        gate_outcome = GATE_READY
        status = "READY"
        reliability = "RELIABLE"

    return {
        "camera_status": status,
        "pose_reliability": reliability,
        "gate_outcome": gate_outcome,
        "person_detected": person_ok,
        "required_landmarks_visible": req_ok,
        "landmark_stability": stab_status,
        "camera_view": view_status,
        "issues": unique_issues,
        "suggestions": unique_suggestions,
        "quality_metrics": q_metrics
    }
