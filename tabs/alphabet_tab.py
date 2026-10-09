"""Alphabet tab: photos (or, for letters that move, clips) of fingerspelled letters, next to a reference picture."""
import string
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from landmarks import draw_face_and_hands, has_hands
from tabs.common import IMAGES_DIR, TARGET_SAMPLES, clean_sign_name, list_clips, list_photos, show_frame
from tabs.record_tab import ClipRecorder, draw_recording_badge, save_and_upload

REFERENCES_DIR = Path("references")  # <letter>.jpg showing the hand shape; local only, often stock pictures
REFERENCE_SIZE = (220, 220)


def numbered_path(folder, number, suffix):
    """videos/hallo, 1, '.mp4' -> videos/hallo/hallo_001.mp4"""
    return folder / f"{folder.name}_{number:03d}{suffix}"  # 3 digits: name order stays number order up to 999


def next_numbered_path(folder, suffix):
    number = len(list(folder.glob(f"*{suffix}"))) + 1
    while numbered_path(folder, number, suffix).exists():  # a gap left by a file copied in by hand
        number += 1
    return numbered_path(folder, number, suffix)


def save_photo(letter, frame, results):
    """Save the raw webcam frame as images/<letter>/<letter>_001.jpg, _002, ...
    Returns the path, or None if no hand is visible: a letter photo without a hand teaches nothing."""
    if not has_hands(results):
        return None
    path = next_numbered_path(IMAGES_DIR / letter, ".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".jpg", frame)[1].tobytes())  # cv2.imwrite can't write to non-ASCII paths (ä)
    return path


def reference_picture(letter):
    """references/<letter>.jpg (or .png, ...) showing how to sign the letter, or None."""
    return next(REFERENCES_DIR.glob(f"{letter}.*"), None) if letter else None


class AlphabetTab(ttk.Frame):
    def __init__(self, parent, sync):
        super().__init__(parent, padding=8)
        self._sync = sync
        self._recorder = ClipRecorder()
        self._frame = self._results = None  # latest raw webcam frame and its tracking, for snap()

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Label(controls, text="Letter:").pack(side="left")
        self._letter = ttk.Combobox(controls, values=list(string.ascii_uppercase), width=6)
        self._letter.set("A")
        self._letter.pack(side="left", padx=4)
        self._letter.bind("<<ComboboxSelected>>", lambda _: self._letter_chosen())
        self._letter.bind("<Return>", lambda _: self._letter_chosen())
        self._letter.bind("<KeyRelease>", lambda _: self._letter_changed())
        ttk.Button(controls, text="📷 Snap", command=self.snap).pack(side="left", padx=4)
        self._record_button = ttk.Button(controls, text="● Record", command=self.toggle_recording)
        self._record_button.pack(side="left", padx=4)
        self._count = ttk.Label(controls)
        self._count.pack(side="left", padx=8)

        view = ttk.Frame(self)
        view.pack(pady=6)
        self._video = ttk.Label(view)
        self._video.pack(side="left")
        self._reference = ttk.Label(view, justify="center")
        self._reference.pack(side="left", padx=(12, 0))
        self._status = ttk.Label(self, text="Show the letter's hand shape, then press Snap or Space. "
                                            "For a letter that moves (Z, J, Ä, ...) press Record instead.")
        self._status.pack(fill="x")
        self._letter_changed()

    def show_camera_frame(self, frame, results):
        self._frame, self._results = frame.copy(), results  # copy: the photo is saved without landmarks
        if self._recorder.is_recording:
            self._recorder.add(frame)
            if self._recorder.seconds > ClipRecorder.MAX_SECONDS:
                self.toggle_recording()
        draw_face_and_hands(frame, results)
        if self._recorder.is_recording:
            draw_recording_badge(frame, self._recorder.seconds)
        show_frame(self._video, frame)

    def snap(self):
        letter = clean_sign_name(self._letter.get())
        if not letter:
            self._status.configure(text="Type a letter first.")
            return
        if self._frame is None:
            return
        path = save_photo(letter, self._frame, self._results)
        self._status.configure(text=f"Saved {path}" if path else "No hand found - not saved.")
        self._update_count()

    def toggle_recording(self):
        """Record a clip of the letter into videos/<letter>/, like the Record tab does for a sign."""
        if self._recorder.is_recording:
            message = save_and_upload(self._recorder, self._sync)
            self._record_button.configure(text="● Record")
            if message:
                self._status.configure(text=message)
            self._update_count()
            return
        letter = clean_sign_name(self._letter.get())
        if not letter:
            self._status.configure(text="Type a letter first.")
            return
        self._recorder.start(letter)
        self._record_button.configure(text="■ Stop")
        self._status.configure(text=f"Recording '{letter}'... press Stop when the letter is done.")

    def stop_recording(self):
        if self._recorder.is_recording:
            self.toggle_recording()

    def _letter_chosen(self):
        self._letter_changed()
        self.focus_set()  # out of the letter box, so Space snaps instead of typing a space

    def _letter_changed(self):
        self._update_count()
        self._show_reference()

    def _show_reference(self):
        path = reference_picture(clean_sign_name(self._letter.get()))
        if path is None:
            self._reference.configure(image="", text="No reference picture.\nAdd one as references/<letter>.jpg")
            self._reference.image = None
            return
        picture = Image.open(path)
        picture.thumbnail(REFERENCE_SIZE)
        image = ImageTk.PhotoImage(picture)
        self._reference.configure(image=image, text="")
        self._reference.image = image  # Tk keeps no reference of its own (see show_frame)

    def _update_count(self):
        letter = clean_sign_name(self._letter.get())
        if not letter:
            self._count.configure(text="")
            return
        self._count.configure(text=f"{len(list_photos(letter))}/{TARGET_SAMPLES} photos · "
                                   f"{len(list_clips(letter))}/{TARGET_SAMPLES} clips")
