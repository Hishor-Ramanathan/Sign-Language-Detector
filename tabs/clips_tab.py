"""Clips tab: every clip and photo grouped by sign - play, scrub, check hand tracking, trim, delete, sync."""
import shutil
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2

from clip_sync import DATASET_DIR, NOT_REACHABLE, owner_of
from extract_dataset import forget_extracted, read_photo
from landmarks import HolisticTracker, draw_face_and_hands, has_hands
from tabs.common import (TARGET_SAMPLES, fit, in_background, list_clips, list_photos, list_signs, show_frame,
                         write_mp4)


def clip_seconds(path):
    capture = cv2.VideoCapture(str(path))
    frames, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return frames / fps if fps else 0.0


def read_frames(path):
    """(frames at display size, frame rate). A photo is a clip of one frame, so everything else works on it too;
    trimming re-reads the original file."""
    if path.suffix == ".jpg":
        photo = read_photo(path)
        return ([] if photo is None else [fit(photo)]), 30.0
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(fit(frame))
    capture.release()
    return frames, fps


def trim_clip(path, first, last):
    """Overwrite the clip with only frames first..last (inclusive), keeping its resolution and frame rate."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    kept = []
    for index in range(last + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if index >= first:
            kept.append(frame)
    capture.release()
    # Write elsewhere first: a half-written file in videos/ would be picked up as a clip.
    temporary = Path(tempfile.gettempdir()) / f"trim_{path.name}"
    write_mp4(temporary, kept, fps)
    shutil.move(temporary, path)


class ClipsTab(ttk.Frame):
    def __init__(self, parent, sync):
        super().__init__(parent, padding=8)
        self._sync = sync
        self._frames, self._results = [], []
        self._path, self._fps = None, 30.0
        self._index = self._first = self._last = 0
        self._playing, self._after_id = False, None
        self._generation = 0  # bumped per loaded clip so a stale analysis thread stops

        library = ttk.Frame(self)
        library.pack(side="left", fill="y")
        self._tree = ttk.Treeview(library, columns=("length",), selectmode="browse", height=24)
        self._tree.heading("#0", text="Clip")
        self._tree.heading("length", text="Length")
        self._tree.column("#0", width=280)
        self._tree.column("length", width=60, anchor="e")
        self._tree.pack(fill="y", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        ttk.Button(library, text="⟳ Sync with team", command=self.sync).pack(fill="x", pady=(4, 0))

        player = ttk.Frame(self, padding=(12, 0, 0, 0))
        player.pack(side="left", fill="both", expand=True)
        self._video = ttk.Label(player, text="Select a clip on the left.")
        self._video.pack()
        self._slider = ttk.Scale(player, from_=0, to=0, command=self._on_scrub)
        self._slider.pack(fill="x", pady=4)
        buttons = ttk.Frame(player)
        buttons.pack(fill="x")
        self._play_button = ttk.Button(buttons, text="▶ Play", command=self._toggle_play)
        self._play_button.pack(side="left")
        ttk.Button(buttons, text="Set start", command=self._set_first).pack(side="left", padx=(12, 2))
        ttk.Button(buttons, text="Set end", command=self._set_last).pack(side="left", padx=2)
        ttk.Button(buttons, text="Save trim", command=self._save_trim).pack(side="left", padx=2)
        ttk.Button(buttons, text="Delete clip", command=self._delete).pack(side="right")
        self._overlay = tk.BooleanVar(value=True)
        ttk.Checkbutton(buttons, text="Show landmarks", variable=self._overlay,
                        command=lambda: self._show(self._index)).pack(side="right", padx=12)
        self._info = ttk.Label(player)
        self._info.pack(fill="x", pady=4)

    def sync(self):
        """Fetch the team's clips and upload yours (see clip_sync.py), then list them."""
        self._unload("Syncing with Hugging Face...")  # the loaded clip may be dropped by the sync
        self.update_idletasks()
        try:
            message = self._sync.pull()  # ponytail: blocks the window while downloading; thread it if that gets long
        except NOT_REACHABLE as error:
            message = f"Sync stopped: {error}"
        self._video.configure(text=message)
        self.refresh()

    def refresh(self):
        self._tree.delete(*self._tree.get_children())
        for sign in list_signs():
            clips, photos = list_clips(sign), list_photos(sign)
            counts = [f"{len(items)}/{TARGET_SAMPLES} {kind}" for items, kind in ((clips, "clips"), (photos, "photos"))
                      if items]
            node = self._tree.insert("", "end", text=f"{sign}   ({', '.join(counts)})", open=True)
            for clip in clips:
                self._tree.insert(node, "end", iid=str(clip), text=clip.name, values=(f"{clip_seconds(clip):.1f} s",))
            for photo in photos:
                self._tree.insert(node, "end", iid=str(photo), text=photo.name, values=("photo",))

    def _on_select(self, _event):
        selected = self._tree.selection()
        if selected and selected[0].endswith((".mp4", ".jpg")):  # sign rows have generated ids
            self._load(Path(selected[0]))

    def _load(self, path):
        self.pause()
        frames, self._fps = read_frames(path)
        if not frames:
            self._info.configure(text=f"Could not read {path.name}.")
            return
        self._path, self._frames = path, frames
        self._results = [None] * len(frames)
        self._index, self._first, self._last = 0, 0, len(frames) - 1
        self._slider.configure(to=self._last)
        self._generation += 1
        threading.Thread(target=self._analyse, args=(self._generation, frames, self._results, self._fps),
                         daemon=True).start()
        self._show(0)
        self._redraw_while_analysing(self._generation)

    def _redraw_while_analysing(self, generation):
        """Refresh the paused frame while analysis runs, so overlay and hand count appear without scrubbing."""
        if generation != self._generation or not self._frames:
            return
        if not self._playing:
            self._show(self._index)
        if any(result is None for result in self._results):
            self.after(250, self._redraw_while_analysing, generation)

    def _analyse(self, generation, frames, results, fps):
        """Track every frame in the background so the overlay and hand count fill in while you watch."""
        with HolisticTracker() as tracker:
            for index, frame in enumerate(frames):
                if generation != self._generation:
                    return
                results[index] = tracker.process(frame, index * 1000 / fps)

    def _show(self, index):
        if not self._frames:
            return
        frame = self._frames[index].copy()
        if self._overlay.get() and self._results[index] is not None:
            draw_face_and_hands(frame, self._results[index])
        show_frame(self._video, frame)
        self._slider.set(index)
        analysed = [r for r in self._results if r is not None]
        hands = sum(has_hands(r) for r in analysed)
        tracking = f"hands found in {hands}/{len(analysed)} frames" + (
            "" if len(analysed) == len(self._frames) else " (analysing...)")
        self._info.configure(text=f"{self._path.name}   frame {index + 1}/{len(self._frames)}   "
                                  f"start {self._first + 1}  end {self._last + 1}   {tracking}")

    def _on_scrub(self, value):
        index = int(float(value))
        if index != self._index and self._frames:
            self._index = index
            self._show(index)

    def _toggle_play(self):
        if self._playing:
            self.pause()
            return
        if not self._frames:
            return
        self._playing = True
        self._play_button.configure(text="⏸ Pause")
        if not self._first <= self._index < self._last:
            self._index = self._first
        self._play_step()

    def _play_step(self):
        self._show(self._index)
        # Loop inside start..end so the trimmed result can be watched before saving.
        self._index = self._index + 1 if self._index < self._last else self._first
        self._after_id = self.after(int(1000 / self._fps), self._play_step)

    def pause(self):
        self._playing = False
        self._play_button.configure(text="▶ Play")
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _set_first(self):
        self._first = min(self._index, self._last)
        self._show(self._index)

    def _set_last(self):
        self._last = max(self._index, self._first)
        self._show(self._index)

    def _save_trim(self):
        if not self._frames or not self._can_edit():
            return
        if (self._first, self._last) == (0, len(self._frames) - 1):
            self._info.configure(text="Set a start and/or end first, then Save trim.")
            return
        if not messagebox.askyesno("Save trim", f"Overwrite {self._path.name} with frames "
                                                f"{self._first + 1}-{self._last + 1}? This can't be undone."):
            return
        self.pause()
        path = self._path
        trim_clip(path, self._first, self._last)
        forget_extracted(DATASET_DIR, path)
        in_background(self._sync.push, path)
        self.refresh()
        self._tree.selection_set(str(path))  # reloads the trimmed clip via <<TreeviewSelect>>

    def _delete(self):
        if not self._path or not self._can_edit():
            return
        if not messagebox.askyesno("Delete clip", f"Delete {self._path.name}? This can't be undone."):
            return
        self._path.unlink()
        forget_extracted(DATASET_DIR, self._path)
        in_background(self._sync.remove, self._path)
        self._unload("Clip deleted. Select another clip.")
        self.refresh()

    def _can_edit(self):
        """Only the owner changes a clip: anyone else's change would be undone by the next sync."""
        if self._sync.can_edit(self._path):
            return True
        self._info.configure(text=f"Only {owner_of(self._path)} can trim or delete {self._path.name}.")
        return False

    def _unload(self, message):
        self.pause()
        self._generation += 1  # stop its analysis
        self._frames, self._path = [], None
        self._video.configure(image="", text=message)
        self._info.configure(text="")
