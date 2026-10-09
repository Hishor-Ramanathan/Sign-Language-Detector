"""Smoke checks for the app's file handling and the detector's smoothing.  Run: python test_app.py"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from app import (ClipRecorder, TrainingProgress, clean_sign_name, forget_extracted, number_clips, reference_picture,
                 save_photo, trim_clip)
from detect import STABLE_PREDICTIONS, SignDetector, draw_output_field, draw_probability_bars
from extract_dataset import choose_split
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
    assert path == Path("videos/hallo/hallo_001.mp4") and frames == 40 and frame_count(path) == 40
    assert not recorder.is_recording
    empty = ClipRecorder()
    empty.start("hallo")
    assert empty.stop() is None  # Stop pressed before any frame arrived: nothing saved

    trim_clip(path, 5, 24)
    assert frame_count(path) == 20

    npy = Path("dataset/val/hallo") / f"{path.stem}.npy"
    npy.parent.mkdir(parents=True)
    npy.touch()
    forget_extracted(path)
    assert not npy.exists()


def check_number_clips():
    folder, samples = Path("videos/danke"), Path("dataset/test/danke")
    folder.mkdir(parents=True)
    samples.mkdir(parents=True)
    for name in ("b", "a", "danke_005"):
        (folder / f"{name}.mp4").write_text(name)
    (samples / "b.npy").write_text("b")
    assert len(number_clips("danke")) == 3
    assert [(f.name, f.read_text()) for f in sorted(folder.iterdir())] == [
        ("danke_001.mp4", "a"), ("danke_002.mp4", "b"), ("danke_003.mp4", "danke_005")]
    assert [(f.name, f.read_text()) for f in samples.iterdir()] == [("danke_002.npy", "b")]
    assert number_clips("danke") == []  # already numbered: nothing moves


def check_split_is_70_15_15(root=Path("split_check")):
    for i in range(10):
        sample = root / choose_split(root, "hallo") / "hallo" / f"{i}.npy"
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.touch()
    counts = {split: len(list((root / split / "hallo").glob("*.npy"))) for split in ("train", "val", "test")}
    assert counts == {"train": 7, "val": 2, "test": 1}, counts


def check_save_photo():
    frame = np.zeros((120, 160, 3), np.uint8)
    hand = SimpleNamespace(left_hand_landmarks=[object()], right_hand_landmarks=[])
    no_hand = SimpleNamespace(left_hand_landmarks=[], right_hand_landmarks=[])
    assert save_photo("a", frame, hand) == Path("images/a/a_001.jpg")
    assert save_photo("a", frame, hand) == Path("images/a/a_002.jpg")
    assert save_photo("a", frame, no_hand) is None
    assert save_photo("ä", frame, hand) == Path("images/ä/ä_001.jpg")  # non-ASCII folder still works
    assert cv2.imdecode(np.fromfile("images/a/a_001.jpg", np.uint8), cv2.IMREAD_COLOR).shape == frame.shape


def check_reference_picture():
    assert reference_picture("a") is None  # no references/ folder yet
    Path("references").mkdir()
    Path("references/a.png").touch()
    assert reference_picture("a") == Path("references/a.png") and reference_picture("b") is None


def check_training_progress_reads_train_py_output():
    progress = TrainingProgress()
    output = ["14 train / 4 val / 2 test samples, labels: ['danke', 'hallo']",
              "epoch 1/500  accuracy 4%  loss 0.935  |  val accuracy 83%  val loss 0.651",
              "epoch 2/500  accuracy 87%  loss 0.646  |  val accuracy 83%  val loss 0.445",
              "best epoch 2 (val accuracy 83%)",
              "test accuracy 75% (3/4 correct)", "  danke: 1/2", "  hallo: 2/2", "saved model.keras and labels.txt"]
    assert [progress.read(line) for line in output] == [False, True, True, True, True, True, True, False]
    assert progress.epochs == [1, 2] and progress.accuracy == [0.04, 0.87] and progress.val_loss == [0.651, 0.445]
    assert progress.best_epoch == 2
    assert progress.test_score == "Test accuracy 75% (3/4 correct)   danke 1/2   hallo 2/2"


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
    check_number_clips()
    check_split_is_70_15_15()
    check_save_photo()
    check_reference_picture()
    check_training_progress_reads_train_py_output()
    check_detector_reports_each_sign_once()
    check_overlay_draws()
    print("all checks passed")
