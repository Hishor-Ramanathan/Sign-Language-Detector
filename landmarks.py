"""MediaPipe Holistic tracking shared by extract_dataset.py, train.py and detect.py.

Uses the MediaPipe Tasks API (HolisticLandmarker). The legacy `mp.solutions.holistic` API
used by Sign-Language-Detection-using-MediaPipe is gone from mediapipe releases after 0.10.21.
Drawing colours are copied from that project.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import drawing_utils

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
             "holistic_landmarker/float16/latest/holistic_landmarker.task")
MODEL_PATH = Path(__file__).with_name("holistic_landmarker.task")

SEQUENCE_LENGTH = 30                                 # frames per sample the model sees
KEYPOINTS_PER_FRAME = 33 * 4 + 21 * 3 * 2 + 478 * 3  # pose + both hands + face (incl. iris) = 1692

_spec = drawing_utils.DrawingSpec
_FACE_CONTOURS = vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
_HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def _model_file():
    if not MODEL_PATH.exists():
        print(f"Downloading {MODEL_PATH.name} (~14 MB, first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)


class HolisticTracker:
    """Runs HolisticLandmarker over consecutive BGR frames of one video or webcam stream.

    Use one tracker per video: it tracks between frames, so reusing it across videos
    would carry the previous clip's landmarks into the next one.
    """

    def __init__(self):
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_model_file()),
            running_mode=vision.RunningMode.VIDEO)
        self._model = vision.HolisticLandmarker.create_from_options(options)
        self._last_ms = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._model.close()

    def process(self, frame_bgr, timestamp_ms):
        # VIDEO mode rejects timestamps that don't increase; some MP4s and webcams repeat them.
        self._last_ms = max(int(timestamp_ms), self._last_ms + 1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return self._model.detect_for_video(image, self._last_ms)


def draw_face_and_hands(image, results):
    """Draw face contours and both hands onto image in place (pose is tracked but not drawn)."""
    drawing_utils.draw_landmarks(image, results.face_landmarks, _FACE_CONTOURS,
                                 _spec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                 _spec(color=(80, 255, 121), thickness=1, circle_radius=1))
    drawing_utils.draw_landmarks(image, results.left_hand_landmarks, _HAND_CONNECTIONS,
                                 _spec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                 _spec(color=(121, 44, 250), thickness=2, circle_radius=2))
    drawing_utils.draw_landmarks(image, results.right_hand_landmarks, _HAND_CONNECTIONS,
                                 _spec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                 _spec(color=(245, 66, 230), thickness=2, circle_radius=2))


def has_hands(results):
    return bool(results.left_hand_landmarks or results.right_hand_landmarks)


def extract_keypoints(results):
    """Flatten one frame's landmarks to a 1692-long vector; missing body parts become zeros."""
    def flat(landmarks, count, with_visibility=False):
        if not landmarks:
            return np.zeros(count * (4 if with_visibility else 3))
        return np.array([[p.x, p.y, p.z, p.visibility] if with_visibility else [p.x, p.y, p.z]
                         for p in landmarks]).flatten()

    return np.concatenate([
        flat(results.pose_landmarks, 33, with_visibility=True),
        flat(results.left_hand_landmarks, 21),
        flat(results.right_hand_landmarks, 21),
        flat(results.face_landmarks, 478),
    ])
