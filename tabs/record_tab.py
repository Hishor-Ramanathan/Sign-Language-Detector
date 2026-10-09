"""Record tab: webcam with landmarks; Record / Stop saves a clip of the sign and uploads it."""
import time
from tkinter import ttk

import cv2

from clip_sync import VIDEOS_DIR, new_clip_path
from landmarks import draw_face_and_hands
from tabs.common import TARGET_SAMPLES, clean_sign_name, in_background, list_clips, list_signs, show_frame, write_mp4

RED = (0, 0, 255)


class ClipRecorder:
    """Collects raw webcam frames between start() and stop(), then saves them as one MP4.

    Frames are buffered so the file can be written at the frame rate actually reached
    (tracking slows the loop below the camera's nominal 30 fps); otherwise playback runs too fast.
    """
    MAX_SECONDS = 20  # ponytail: frames live in memory (~20 MB/s at 640x480); stream to disk if clips get long

    def __init__(self):
        self._frames = None
        self._sign = None
        self._started = 0.0

    @property
    def is_recording(self):
        return self._frames is not None

    @property
    def seconds(self):
        return time.monotonic() - self._started

    def start(self, sign):
        self._frames, self._sign, self._started = [], sign, time.monotonic()

    def add(self, frame):
        self._frames.append(frame.copy())  # copy: the caller draws landmarks on its frame afterwards

    def stop(self, owner=None):
        """Save the clip under owner's name (see new_clip_path). Returns (path, frame count, seconds),
        or None if no frame was captured."""
        frames, seconds = self._frames, self.seconds
        self._frames = None
        if not frames:
            return None
        path = new_clip_path(VIDEOS_DIR / self._sign, owner)
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = min(max(len(frames) / seconds, 1.0), 60.0)  # OpenCV silently writes nothing at absurd rates
        write_mp4(path, frames, fps)
        return path, len(frames), seconds


class RecordTab(ttk.Frame):
    def __init__(self, parent, sync):
        super().__init__(parent, padding=8)
        self._sync = sync
        self._recorder = ClipRecorder()

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Label(controls, text="Sign:").pack(side="left")
        self._sign = ttk.Combobox(controls, values=list_signs(), width=24)
        self._sign.pack(side="left", padx=4)
        self._sign.bind("<<ComboboxSelected>>", lambda _: self._update_count())
        self._sign.bind("<KeyRelease>", lambda _: self._update_count())
        self._record_button = ttk.Button(controls, text="● Record", command=self.toggle_recording)
        self._record_button.pack(side="left", padx=4)
        self._count = ttk.Label(controls)
        self._count.pack(side="left", padx=8)

        self._video = ttk.Label(self)
        self._video.pack(pady=6)
        self._status = ttk.Label(self, text="Type a new sign or pick one, then press Record.")
        self._status.pack(fill="x")

    def show_camera_frame(self, frame, results):
        if self._recorder.is_recording:
            self._recorder.add(frame)  # raw frame, before landmarks are drawn on it
            if self._recorder.seconds > ClipRecorder.MAX_SECONDS:
                self.toggle_recording()
        draw_face_and_hands(frame, results)
        if self._recorder.is_recording:
            cv2.circle(frame, (20, 20), 8, RED, -1)
            cv2.putText(frame, f"REC {self._recorder.seconds:.1f}s", (35, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        show_frame(self._video, frame)

    def toggle_recording(self):
        if self._recorder.is_recording:
            saved = self._recorder.stop(self._sync.owner)
            self._record_button.configure(text="● Record")
            if saved:
                path, frames, seconds = saved
                in_background(self._sync.push, path)
                upload = "uploading" if self._sync.owner else "not uploaded: not logged in to Hugging Face"
                self._status.configure(text=f"Saved {path}  ({frames} frames, {seconds:.1f} s), {upload}")
            self._sign.configure(values=list_signs())
            self._update_count()
            return
        sign = clean_sign_name(self._sign.get())
        if not sign:
            self._status.configure(text="Type a sign name first.")
            return
        self._sign.set(sign)
        self._recorder.start(sign)
        self._record_button.configure(text="■ Stop")
        self._status.configure(text=f"Recording '{sign}'... press Stop when the sign is done.")

    def stop_recording(self):
        if self._recorder.is_recording:
            self.toggle_recording()

    def _update_count(self):
        sign = clean_sign_name(self._sign.get())
        self._count.configure(text=f"{len(list_clips(sign))}/{TARGET_SAMPLES} clips" if sign else "")
