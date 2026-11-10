import os
import cv2
try:
    import av
except ImportError:
    av = None
import numpy as np
import mediapipe as mp
import threading
try:
    from streamlit_webrtc import VideoProcessorBase
except ImportError:
    class VideoProcessorBase:
        pass
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from detectors.squat import SquatDetector
from detectors.pushup import PushUpDetector
from detectors.biceps_curl import BicepsCurlDetector
from detectors.shoulder_press import ShoulderPressDetector
from detectors.lunges import LungesDetector
from services.config.workout_config import POSE_CONNECTIONS


class VideoProcessorClass(VideoProcessorBase):
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_metrics = None
        self._exercise_type = "Squats"

        model_path = os.path.join(os.getcwd(), "ml_models", "pose_landmarker_full.task")
        if not os.path.exists(model_path):
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "ml_models",
                "pose_landmarker_full.task"
            )
        base_option = python.BaseOptions(model_asset_path=model_path)

        options = vision.PoseLandmarkerOptions(
            base_options=base_option,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)

        self._detectors = {
            "Squats": SquatDetector(),
            "Push-ups": PushUpDetector(),
            "Biceps Curls (Dumbbell)": BicepsCurlDetector(),
            "Shoulder Press": ShoulderPressDetector(),
            "Lunges": LungesDetector(),
        }

        self._frame_timestamps_ms = 0
    
    def set_latest_metrics(self, metrics):
        with self._lock:
            self._latest_metrics = metrics.copy()

    def get_latest_metrics(self):
        with self._lock:
            return None if self._latest_metrics is None else self._latest_metrics.copy()
        
    def set_exercise(self, exercise_type):
        with self._lock:
            self._exercise_type = exercise_type

    def get_exercise(self):
        with self._lock:
            return self._exercise_type
        
    def _draw_skeleton(self, img, landmarks):
        h, w = img.shape[:2]

        for start_idx, end_idx in POSE_CONNECTIONS:
            p1 = landmarks[start_idx]
            p2 = landmarks[end_idx]

            if p1.visibility > 0.7 and p2.visibility > 0.7:
                cv2.line(
                    img,
                    (int(p1.x * w), int(p1.y * h)),
                    (int(p2.x * w), int(p2.y * h)),
                    (0, 255, 0),
                    8
                )
        
        for lm in landmarks:
            if lm.visibility > 0.7:
                cv2.circle(
                    img, 
                    (int(lm.x * w), int(lm.y * h)),
                    8,
                    (255, 0, 0),
                    -1
                )
            
    def _draw_no_pose_warnings(self, img):
        cv2.putText(
            img,
            "NO POSE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            "PLEASE FACE THE CAMERA",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_overlays(self, img, metrics, ex_type):
        if ex_type == "Squats":
            self._draw_squats_overlays(img, metrics)
        elif ex_type == "Push-ups":
            self._draw_pushup_overlays(img, metrics)
        elif ex_type == "Biceps Curls (Dumbbell)":
            self._draw_curl_overlays(img, metrics)
        elif ex_type == "Shoulder Press":
            self._draw_press_overlays(img, metrics)
        elif ex_type == "Lunges":
            self._draw_lunge_overlays(img, metrics)


    def _draw_squats_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"DEPTH: {metrics['depth_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    
    def _draw_pushup_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"BODY: {metrics['body_alignment']} | HIP: {metrics['hip_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_curl_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"SWING: {metrics['swing_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_press_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"EXT: {metrics['extension_status']} | BACK: {metrics['back_arch_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def _draw_lunge_overlays(self, img, metrics):
        h, _ = img.shape[:2]

        cv2.putText(
            img,
            f"BALANCE: {metrics['balance_status']}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    def reset_exercise(self, ex_type=None):
        with self._lock:
            if ex_type:
                det = self._detectors.get(ex_type)
                if det:
                    det.reset()
            else:
                for det in self._detectors.values():
                    det.reset()
            self._latest_metrics = None

    def process_frame_bgr(self, bgr_image):
        image = np.asarray(
            cv2.flip(bgr_image, 1),
            dtype=np.uint8
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

        self._frame_timestamps_ms += 30
        result = self._landmarker.detect_for_video(mp_image, self._frame_timestamps_ms)

        diagnostic = {
            "model_loaded": True,
            "poses_detected_count": len(result.pose_landmarks) if result.pose_landmarks else 0,
            "landmarks_count": 0,
            "key_landmarks_visibility": {},
            "detector_called": False,
        }

        metrics = {}
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            self._current_landmarks = landmarks
            diagnostic["landmarks_count"] = len(landmarks)
            diagnostic["key_landmarks_visibility"] = {
                "left_hip": round(float(landmarks[23].visibility), 2) if len(landmarks) > 23 else 0.0,
                "left_knee": round(float(landmarks[25].visibility), 2) if len(landmarks) > 25 else 0.0,
                "left_ankle": round(float(landmarks[27].visibility), 2) if len(landmarks) > 27 else 0.0,
                "right_hip": round(float(landmarks[24].visibility), 2) if len(landmarks) > 24 else 0.0,
                "right_knee": round(float(landmarks[26].visibility), 2) if len(landmarks) > 26 else 0.0,
                "right_ankle": round(float(landmarks[28].visibility), 2) if len(landmarks) > 28 else 0.0,
            }

            self._draw_skeleton(image, landmarks)

            ex_type = self.get_exercise()

            detector = self._detectors.get(ex_type)

            if detector:
                metrics = detector.process(landmarks)

                metrics["pose_detected"] = True
                metrics["stage"] = detector.stage or "ready"
                diagnostic["detector_called"] = True

                self._draw_overlays(image, metrics, ex_type)

                self.set_latest_metrics(metrics)
        else:
            self._current_landmarks = None
            self._draw_no_pose_warnings(image)

            with self._lock:
                if self._latest_metrics is not None:
                    self._latest_metrics["pose_detected"] = False
                    metrics = self._latest_metrics.copy()
                else:
                    self._latest_metrics = {"pose_detected": False}
                    metrics = {"pose_detected": False}

        metrics["diagnostic"] = diagnostic
        return image, metrics

    def recv(self, frame):
        if av is None:
            raise RuntimeError("PyAV is required for streamlit-webrtc recv()")
        bgr_frame = frame.to_ndarray(format="bgr24")
        image, _ = self.process_frame_bgr(bgr_frame)
        return av.VideoFrame.from_ndarray(image, format="bgr24")
    