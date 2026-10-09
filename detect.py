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
from tensorflow.keras.models import load_model

from landmarks import SEQUENCE_LENGTH, HolisticTracker, draw_face_and_hands, extract_keypoints

STABLE_PREDICTIONS = 10  # a sign must win this many predictions in a row before it's reported
BAR_COLORS = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (200, 60, 200), (60, 200, 200)]
WHITE = (255, 255, 255)


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
        self._file.close()

    def record(self, sign, confidence, video_time_s):
        now = datetime.now().isoformat(timespec="seconds")
        print(f"{now}  {sign}  {confidence:.0%}")
        video_time = "" if video_time_s is None else f"{video_time_s:.2f}"
        self._csv.writerow([now, self._source, video_time, sign, f"{confidence:.3f}"])
        self._file.flush()  # keep the CSV usable if the window is killed


def draw_output_field(image, detected, history):
    width = image.shape[1]
    cv2.rectangle(image, (0, 0), (width, 40), (245, 117, 16), -1)
    cv2.putText(image, " ".join(history), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, 40), (width, 80), (40, 40, 40), -1)
    text = f"Detected: {detected[0]} ({detected[1]:.0%})" if detected else "Detected: -"
    cv2.putText(image, text, (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)


def draw_probability_bars(image, probabilities, labels):
    for i, (label, probability) in enumerate(zip(labels, probabilities)):
        top = 90 + i * 40
        cv2.rectangle(image, (0, top), (int(probability * 100), top + 30), BAR_COLORS[i % len(BAR_COLORS)], -1)
        # Dark outline keeps the label readable when the bar is short and the background is light.
        cv2.putText(image, label, (0, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, label, (0, top + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, help="MP4 to run on instead of the webcam")
    parser.add_argument("--threshold", type=float, default=0.5, help="min confidence to report a sign")
    parser.add_argument("--log", type=Path, default=Path("detections.csv"))
    parser.add_argument("--preview", action="store_true", help="only draw landmarks; no model needed")
    args = parser.parse_args()

    model, labels = None, []
    if not args.preview:
        if not Path("model.keras").exists():
            raise SystemExit("No model.keras yet - run train.py first (see README.md), or use --preview")
        model = load_model("model.keras")
        labels = Path("labels.txt").read_text(encoding="utf-8").split()
    capture = cv2.VideoCapture(str(args.video) if args.video else 0)

    window = deque(maxlen=SEQUENCE_LENGTH)      # last 30 frames of keypoints = one model input
    recent = deque(maxlen=STABLE_PREDICTIONS)   # last predicted label indexes, for smoothing
    history = deque(maxlen=5)                   # signs shown in the top banner

    with HolisticTracker() as tracker, DetectionLog(args.log, str(args.video or "webcam")) as log:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            video_time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 if args.video else None
            timestamp_ms = video_time_s * 1000 if args.video else time.monotonic() * 1000
            results = tracker.process(frame, timestamp_ms)
            draw_face_and_hands(frame, results)
            window.append(extract_keypoints(results))

            detected = None
            if model is not None and len(window) == SEQUENCE_LENGTH:
                # Calling the model directly is faster than predict() for a single sample.
                probabilities = model(np.array([window]), training=False).numpy()[0]
                best = int(np.argmax(probabilities))
                recent.append(best)
                if recent.count(best) == STABLE_PREDICTIONS and probabilities[best] > args.threshold:
                    detected = (labels[best], float(probabilities[best]))
                    if not history or history[-1] != detected[0]:
                        history.append(detected[0])
                        log.record(*detected, video_time_s)
                draw_probability_bars(frame, probabilities, labels)

            draw_output_field(frame, detected, history)
            cv2.imshow("Sign Language Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
