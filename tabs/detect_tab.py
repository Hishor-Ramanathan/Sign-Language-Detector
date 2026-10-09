"""Detect tab: live detection with the output field, same as detect.py."""
from tkinter import ttk

from detect import load_detector
from landmarks import draw_face_and_hands
from tabs.common import show_frame


class DetectTab(ttk.Frame):
    def __init__(self, parent, log):
        super().__init__(parent, padding=8)
        self._log = log
        self._detector = None
        self._needs_load = True  # loaded on first visit; restart the app after training a new model
        self._status = ttk.Label(self)
        self._status.pack(fill="x")
        self._video = ttk.Label(self)
        self._video.pack(pady=6)

    def activate(self):
        if self._needs_load:
            self._status.configure(text="Loading model...")
            self.update_idletasks()
            self._detector = load_detector()
            self._needs_load = False
        if self._detector is None:
            self._status.configure(text="No trained model yet: record clips in the Record tab, then run "
                                        "'python extract_dataset.py' and 'python train.py' and restart the app.")
        else:
            self._detector.reset()  # frames from before the tab switch would mix into the next prediction
            self._status.configure(text="")  # the overlay already lists every sign

    def show_camera_frame(self, frame, results):
        draw_face_and_hands(frame, results)
        if self._detector is not None:
            new_sign = self._detector.step(results)
            if new_sign:
                self._log.record(*new_sign, None)
            self._detector.draw(frame)
        show_frame(self._video, frame)
