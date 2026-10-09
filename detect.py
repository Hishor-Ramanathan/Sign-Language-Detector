"""Detect signs on the webcam or an MP4, drawing face/hand landmarks and an output field.

    python detect.py                      # webcam
    python detect.py --video clip.mp4     # a video file
    python detect.py --preview            # landmarks only, no trained model needed

Output field (banner at the top): the sign detected right now with its confidence, and the
last 5 detected signs. Each newly detected sign is also printed and appended to detections.csv.
Press q to quit.
"""
import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from landmarks import SEQUENCE_LENGTH, HolisticTracker, draw_face_and_hands, extract_keypoints

MODEL_FILE = Path("model.keras")
LABELS_FILE = Path("labels.txt")
STABLE_PREDICTIONS = 10  # a sign must win this many predictions in a row before it's reported
FONT = cv2.FONT_HERSHEY_DUPLEX
BANNER_HEIGHT = 84
PANEL_WIDTH = 260
# Colours are BGR, as OpenCV expects.
WHITE = (255, 255, 255)
MUTED = (170, 170, 170)
DARK = (20, 20, 20)
TRACK = (70, 70, 70)
GREEN = (94, 197, 34)    # the detected sign
AMBER = (11, 158, 245)   # above the threshold, but not yet stable long enough to be reported
GREY = (130, 130, 130)   # below the threshold


class DetectionLog:
    """Prints each newly detected sign and appends it to a CSV file."""

    def __init__(self, path, source):
        is_new = not path.exists()
        self._file = path.open("a", newline="", encoding="utf-8")
        self._csv = csv.writer(self._file)
        self._source = source
        if is_new:
            self._csv.writerow(["wall_time", "source", "video_time_s", "sign", "confidence"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._file.close()

    def record(self, sign, confidence, video_time_s):
        now = datetime.now().isoformat(timespec="seconds")
        print(f"{now}  {sign}  {confidence:.0%}")
        video_time = "" if video_time_s is None else f"{video_time_s:.2f}"
        self._csv.writerow([now, self._source, video_time, sign, f"{confidence:.3f}"])
        self._file.flush()  # keep the CSV usable if the window is killed


class SignDetector:
    """Turns a stream of per-frame landmarks into smoothed sign detections and draws the overlay."""

    def __init__(self, model, labels, threshold):
        self.labels = labels
        self._model = model
        self._threshold = threshold
        self.reset()

    def reset(self):
        """Forget all frames, e.g. after the stream was paused, so old frames don't leak into a prediction."""
        self._window = deque(maxlen=SEQUENCE_LENGTH)      # last 30 frames of keypoints = one model input
        self._recent = deque(maxlen=STABLE_PREDICTIONS)   # last predicted label indexes, for smoothing
        self._history = deque(maxlen=5)                   # signs shown in the top banner
        self._probabilities = None
        self._detected = None

    def step(self, results):
        """Feed one frame's landmarks. Returns (sign, confidence) when a *new* sign is detected, else None."""
        self._window.append(extract_keypoints(results))
        self._detected = None
        if len(self._window) < SEQUENCE_LENGTH:
            return None
        # Calling the model directly is faster than predict() for a single sample.
        self._probabilities = np.asarray(self._model(np.array([self._window]), training=False))[0]
        best = int(np.argmax(self._probabilities))
        self._recent.append(best)
        if self._recent.count(best) < STABLE_PREDICTIONS or self._probabilities[best] <= self._threshold:
            return None
        self._detected = (self.labels[best], float(self._probabilities[best]))
        if self._history and self._history[-1] == self._detected[0]:
            return None
        self._history.append(self._detected[0])
        return self._detected

    def draw(self, image):
        if self._probabilities is not None:
            detected_sign = self._detected[0] if self._detected else None
            draw_probability_bars(image, self._probabilities, self.labels, self._threshold, detected_sign)
        draw_output_field(image, self._detected, self._history)


def load_detector(threshold=0.5):
    """SignDetector for model.keras + labels.txt, or None if nothing has been trained yet."""
    if not MODEL_FILE.exists():
        return None
    from tensorflow.keras.models import load_model  # slow import (5-10 s), so only pay it once a model exists
    labels = LABELS_FILE.read_text(encoding="utf-8").split()
    return SignDetector(load_model(MODEL_FILE), labels, threshold)


def darken(image, left, top, right, bottom):
    """See-through dark panel: the camera stays visible behind the text."""
    image[top:bottom, left:right] //= 3


def put_text(image, text, origin, scale, color, thickness=1):
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_output_field(image, detected, history):
    """Top banner: the detected sign with a confidence badge and meter, and the recent signs."""
    width = image.shape[1]
    darken(image, 0, 0, width, BANNER_HEIGHT)
    if detected:
        sign, confidence = detected
        put_text(image, sign.upper(), (12, 44), 1.1, WHITE, 2)
        (sign_width, _), _ = cv2.getTextSize(sign.upper(), FONT, 1.1, 2)
        badge = f"{confidence:.0%}"
        (badge_width, badge_height), _ = cv2.getTextSize(badge, FONT, 0.7, 2)
        left = 24 + sign_width
        cv2.rectangle(image, (left, 18), (left + badge_width + 16, 48), GREEN, -1)
        put_text(image, badge, (left + 8, 33 + badge_height // 2), 0.7, DARK, 2)
        cv2.rectangle(image, (0, BANNER_HEIGHT - 4), (int(width * confidence), BANNER_HEIGHT), GREEN, -1)
    else:
        put_text(image, "No sign yet", (12, 44), 1.1, GREY, 2)
    put_text(image, "Recent: " + (", ".join(history) or "-"), (12, 70), 0.55, MUTED)


def draw_probability_bars(image, probabilities, labels, threshold, detected_sign):
    """One row per sign: name, confidence %, and a bar with a white tick at the threshold.

    Rows keep the labels.txt order so they don't jump around; the leading sign gets a white label.
    """
    left, top, row_height = 8, BANNER_HEIGHT + 8, 30
    track_left, track_right = left + 8, left + PANEL_WIDTH - 8
    track_width = track_right - track_left
    darken(image, left, top, left + PANEL_WIDTH, top + 8 + row_height * len(labels))
    leader = int(np.argmax(probabilities))
    for i, (label, probability) in enumerate(zip(labels, probabilities)):
        y = top + 8 + i * row_height
        if label == detected_sign:
            bar_color = GREEN
        elif probability > threshold:
            bar_color = AMBER
        else:
            bar_color = GREY
        text_color = WHITE if i == leader else MUTED
        percent = f"{probability:.0%}"
        (percent_width, _), _ = cv2.getTextSize(percent, FONT, 0.5, 1)
        put_text(image, label, (track_left, y + 12), 0.5, text_color)
        put_text(image, percent, (track_right - percent_width, y + 12), 0.5, text_color)
        cv2.rectangle(image, (track_left, y + 16), (track_right, y + 22), TRACK, -1)
        cv2.rectangle(image, (track_left, y + 16), (track_left + int(probability * track_width), y + 22), bar_color, -1)
        tick = track_left + int(threshold * track_width)
        cv2.line(image, (tick, y + 14), (tick, y + 24), WHITE, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, help="MP4 to run on instead of the webcam")
    parser.add_argument("--threshold", type=float, default=0.5, help="min confidence to report a sign")
    parser.add_argument("--log", type=Path, default=Path("detections.csv"))
    parser.add_argument("--preview", action="store_true", help="only draw landmarks; no model needed")
    args = parser.parse_args()

    detector = None
    if not args.preview:
        detector = load_detector(args.threshold)
        if detector is None:
            raise SystemExit(f"No {MODEL_FILE} yet - run train.py first (see README.md), or use --preview")
    capture = cv2.VideoCapture(str(args.video) if args.video else 0)

    with HolisticTracker() as tracker, DetectionLog(args.log, str(args.video or "webcam")) as log:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            video_time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 if args.video else None
            timestamp_ms = video_time_s * 1000 if args.video else time.monotonic() * 1000
            results = tracker.process(frame, timestamp_ms)
            draw_face_and_hands(frame, results)
            if detector is not None:
                new_sign = detector.step(results)
                if new_sign:
                    log.record(*new_sign, video_time_s)
                detector.draw(frame)
            cv2.imshow("Sign Language Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
