"""Sign Language Detector app: record, review and detect in one window.

    python app.py

Tabs (one module each in tabs/):
  Record   - webcam with face/hand landmarks. Pick or type a sign, then Record / Stop.
             Clips are saved to videos/<sign>/<sign>_<hf user>_001.mp4, _002, ..., the layout extract_dataset.py
             reads, and uploaded to the team's Hugging Face repo (see clip_sync.py).
  Alphabet - webcam with landmarks and, next to it, references/<letter>.jpg showing the hand shape.
             Pick a letter, show its hand shape, press Snap (or Space).
             Photos are saved to images/<letter>/<letter>_001.jpg, _002, ...; a frame without a hand isn't saved.
             Letters that move (Z, J, Ä, ...): Record / Stop saves a clip to videos/<letter>/, as in the Record tab.
  Clips    - every clip and photo grouped by sign: play it, scrub through it, check hand tracking, trim it, delete it.
             Only the clip's owner can trim or delete it. Sync fetches the team's clips.
  Detect   - live detection with the output field, same as detect.py.
Training runs on the command line: python extract_dataset.py, then python train.py.
"""
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2

from clip_sync import ClipSync
from detect import DetectionLog
from landmarks import HolisticTracker
from tabs.alphabet_tab import AlphabetTab
from tabs.clips_tab import ClipsTab
from tabs.detect_tab import DetectTab
from tabs.record_tab import RecordTab


class SignLanguageApp:
    """One webcam and one tracker shared by the camera tabs; the Clips tab doesn't use the camera."""

    def __init__(self, root):
        self._root = root
        self._camera = cv2.VideoCapture(0)
        self._tracker = HolisticTracker()
        self._log = DetectionLog(Path("detections.csv"), "app webcam")
        sync = ClipSync()  # asks Hugging Face who is logged in, once

        self._notebook = ttk.Notebook(root)
        self._notebook.pack(fill="both", expand=True)
        self._notebook.enable_traversal()  # Ctrl+Tab / Ctrl+Shift+Tab switch tabs
        self._detect = DetectTab(self._notebook, self._log)
        self._record = RecordTab(self._notebook, sync)
        self._alphabet = AlphabetTab(self._notebook, sync)
        self._clips = ClipsTab(self._notebook, sync)
        self._notebook.add(self._record, text="Record")
        self._notebook.add(self._alphabet, text="Alphabet")
        self._notebook.add(self._clips, text="Clips")
        self._notebook.add(self._detect, text="Detect")
        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        root.bind("<space>", self._on_space)
        root.protocol("WM_DELETE_WINDOW", self._close)
        self._tick()

    def _current_tab(self):
        return self._notebook.nametowidget(self._notebook.select())

    def _on_space(self, event):
        # Not while typing (Combobox is an Entry) or on a focused button (Space already presses it).
        if self._current_tab() is self._alphabet and not isinstance(event.widget, (ttk.Entry, ttk.Button)):
            self._alphabet.snap()

    def _on_tab_changed(self, _event):
        tab = self._current_tab()
        if tab is not self._record:
            self._record.stop_recording()
        if tab is not self._alphabet:
            self._alphabet.stop_recording()
        if tab is not self._clips:
            self._clips.pause()
        if tab is self._clips:
            self._clips.refresh()
        if tab is self._detect:
            self._detect.activate()

    def _tick(self):
        tab = self._current_tab()
        if tab is not self._clips:  # the only tab without the camera
            ok, frame = self._camera.read()  # blocks until the next frame, which paces this loop
            if ok:
                tab.show_camera_frame(frame, self._tracker.process(frame, time.monotonic() * 1000))
        self._root.after(10, self._tick)

    def _close(self):
        self._record.stop_recording()
        self._alphabet.stop_recording()
        self._clips.pause()
        self._camera.release()
        self._tracker.close()
        self._log.close()
        self._root.destroy()


def main():
    root = tk.Tk()
    root.title("Sign Language Detector")
    SignLanguageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
