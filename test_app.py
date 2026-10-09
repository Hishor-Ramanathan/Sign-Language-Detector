"""Smoke checks for the app's file handling and the detector's smoothing.  Run: python test_app.py"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app import ClipRecorder, clean_sign_name, forget_extracted, trim_clip
from detect import STABLE_PREDICTIONS, SignDetector, draw_output_field, draw_probability_bars
from landmarks import SEQUENCE_LENGTH


def frame_count(path):
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def check_sign_names():
    assert clean_sign_name("  Guten Tag ") == "guten_tag"
    assert clean_sign_name("Tschüss!") == "tschüss"
    assert clean_sign_name("   ") == ""


def check_record_trim_forget():
    recorder = ClipRecorder()
    recorder.start("hallo")
    for i in range(40):
        recorder.add(np.full((120, 160, 3), i * 5, np.uint8))
    path, frames, _ = recorder.stop()
    assert path.parent == Path("videos/hallo") and frames == 40 and frame_count(path) == 40
    assert not recorder.is_recording
    empty = ClipRecorder()
    empty.start("hallo")
    assert empty.stop() is None  # Stop pressed before any frame arrived: nothing saved

    trim_clip(path, 5, 24)
    assert frame_count(path) == 20

    npy = Path("dataset/hallo") / f"{path.stem}.npy"
    npy.parent.mkdir(parents=True)
    npy.touch()
    forget_extracted(path)
    assert not npy.exists()


def check_detector_reports_each_sign_once():
    no_landmarks = SimpleNamespace(pose_landmarks=[], left_hand_landmarks=[], right_hand_landmarks=[], face_landmarks=[])
    always_hallo = lambda batch, training: np.array([[0.9, 0.1]])
    detector = SignDetector(always_hallo, ["hallo", "danke"], threshold=0.5)
    reports = [detector.step(no_landmarks) for _ in range(SEQUENCE_LENGTH + 50)]
    first = SEQUENCE_LENGTH - 1 + STABLE_PREDICTIONS - 1  # window full, then 10 agreeing predictions
    assert reports[first] == ("hallo", 0.9), reports[first - 1:first + 2]
    assert all(r is None for i, r in enumerate(reports) if i != first)


def check_overlay_draws():
    for count in (2, 6):
        labels = [f"sign_{i}" for i in range(count)]
        frame = np.full((480, 640, 3), 200, np.uint8)
        draw_probability_bars(frame, np.linspace(0, 1, count), labels, 0.5, labels[-1])
        draw_output_field(frame, (labels[-1], 1.0), labels)
        assert (frame[0, 0] < 200).all() and (frame[0, 0] > 0).all()  # banner is see-through, not opaque
    draw_output_field(frame, None, [])  # nothing detected yet


if __name__ == "__main__":
    os.chdir(tempfile.mkdtemp())  # the app works relative to the current folder
    check_sign_names()
    check_record_trim_forget()
    check_detector_reports_each_sign_once()
    check_overlay_draws()
    print("all checks passed")
