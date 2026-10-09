"""Sign Language Detector app: record, review, train and detect in one window.

    python app.py

Tabs:
  Record  - webcam with face/hand landmarks. Pick or type a sign, then Record / Stop.
            Clips are saved to videos/<sign>/<sign>_001.mp4, _002, ..., the layout extract_dataset.py reads.
  Alphabet - webcam with landmarks. Pick a letter, show its hand shape, press Snap (or Space).
            Photos are saved to images/<letter>/<letter>_001.jpg, _002, ...; a frame without a hand isn't saved.
  Clips  - every clip grouped by sign: play it, scrub through it, check hand tracking, trim it, delete it.
            Opening the tab renumbers each sign's clips to <sign>_001.mp4, ... (covers clips copied in by hand).
  Training - "Build dataset + Train" runs extract_dataset.py and then train.py. Live charts of accuracy and
            loss per epoch (train vs val), the best epoch, then the test score per sign.
  Detect  - live detection with the output field, same as detect.py.
"""
import os
import queue
import re
import shutil
import string
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from detect import DetectionLog, load_detector
from extract_dataset import extracted_path
from landmarks import HolisticTracker, draw_face_and_hands, has_hands

VIDEOS_DIR = Path("videos")
DATASET_DIR = Path("dataset")
IMAGES_DIR = Path("images")
TARGET_SAMPLES = 30         # clips per sign (photos per letter) the README recommends
DISPLAY_SIZE = (560, 420)   # largest size a frame is shown at; keeps the window on a laptop screen
RED = (0, 0, 255)


# --- clip files -------------------------------------------------------------------------------

def clean_sign_name(text):
    """'Guten Tag ' -> 'guten_tag'. The folder name becomes the label, so keep it filesystem-safe."""
    return re.sub(r"\W+", "_", text.strip().lower()).strip("_")


def list_signs():
    return sorted(d.name for d in VIDEOS_DIR.iterdir() if d.is_dir()) if VIDEOS_DIR.exists() else []


def list_clips(sign):
    return sorted((VIDEOS_DIR / sign).glob("*.mp4"))


def numbered_path(folder, number, suffix):
    """videos/hallo, 1, '.mp4' -> videos/hallo/hallo_001.mp4"""
    return folder / f"{folder.name}_{number:03d}{suffix}"  # 3 digits: name order stays number order up to 999


def next_numbered_path(folder, suffix):
    number = len(list(folder.glob(f"*{suffix}"))) + 1
    while numbered_path(folder, number, suffix).exists():  # a gap left by a file copied in by hand
        number += 1
    return numbered_path(folder, number, suffix)


def number_clips(sign):
    """Rename the sign's clips to <sign>_001.mp4, _002, ... in name order; each clip's .npy follows it.
    Returns the paths that were renamed.

    Everything first moves to a temporary name, so no rename can land on a clip that hasn't moved yet
    (hallo_002 -> hallo_001 while the old hallo_001 is still there)."""
    moves = [(clip, numbered_path(VIDEOS_DIR / sign, number, ".mp4"))
             for number, clip in enumerate(list_clips(sign), start=1)]
    moves = [(clip, target) for clip, target in moves if clip != target]
    staged = []
    for clip, target in moves:
        sample = extracted_path(DATASET_DIR, sign, clip.stem)
        if sample:
            sample = sample.rename(sample.with_suffix(".renaming"))
        staged.append((clip.rename(clip.with_suffix(".renaming")), sample, target))
    for clip, sample, target in staged:
        clip.rename(target)
        if sample:
            sample.rename(sample.with_name(f"{target.stem}.npy"))
    return [clip for clip, _ in moves]


def clip_seconds(path):
    capture = cv2.VideoCapture(str(path))
    frames, fps = capture.get(cv2.CAP_PROP_FRAME_COUNT), capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return frames / fps if fps else 0.0


def write_mp4(path, frames, fps):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


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


def save_photo(letter, frame, results):
    """Save the raw webcam frame as images/<letter>/<letter>_001.jpg, _002, ...
    Returns the path, or None if no hand is visible: a letter photo without a hand teaches nothing."""
    if not has_hands(results):
        return None
    path = next_numbered_path(IMAGES_DIR / letter, ".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".jpg", frame)[1].tobytes())  # cv2.imwrite can't write to non-ASCII paths (ä)
    return path


def forget_extracted(path):
    """Delete the clip's .npy so extract_dataset.py rebuilds it from the changed clip."""
    sample = extracted_path(DATASET_DIR, path.parent.name, path.stem)
    if sample:
        sample.unlink()


# --- display helpers ----------------------------------------------------------------------------

def fit(frame):
    height, width = frame.shape[:2]
    scale = min(DISPLAY_SIZE[0] / width, DISPLAY_SIZE[1] / height, 1)
    return frame if scale == 1 else cv2.resize(frame, (int(width * scale), int(height * scale)))


def show_frame(label, frame_bgr):
    image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(fit(frame_bgr), cv2.COLOR_BGR2RGB)))
    label.configure(image=image)
    label.image = image  # Tk keeps no reference of its own; without this the image is garbage-collected


# --- recording and training ---------------------------------------------------------------------

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

    def stop(self):
        """Save the clip. Returns (path, frame count, seconds), or None if no frame was captured."""
        frames, seconds = self._frames, self.seconds
        self._frames = None
        if not frames:
            return None
        path = next_numbered_path(VIDEOS_DIR / self._sign, ".mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = min(max(len(frames) / seconds, 1.0), 60.0)  # OpenCV silently writes nothing at absurd rates
        write_mp4(path, frames, fps)
        return path, len(frames), seconds


class TrainingJob:
    """Runs extract_dataset.py then train.py in child processes. Output lines arrive on `lines`;
    None marks the end, after which `succeeded` is set."""
    SCRIPTS = ("extract_dataset.py", "train.py")
    _NATIVE_LOG = re.compile(r"^[WIE]\d{4} |^WARNING: All log messages")  # MediaPipe/TF C++ log noise

    def __init__(self):
        self.lines = queue.Queue()
        self.succeeded = None
        self._thread = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        self.succeeded = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.succeeded = all(self._run_script(script) for script in self.SCRIPTS)
        self.lines.put(None)

    def _run_script(self, script):
        self.lines.put(f"$ python {script}")
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).with_name(script))],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        for line in process.stdout:
            if not self._NATIVE_LOG.match(line):
                self.lines.put(line.rstrip())
        return process.wait() == 0


class TrainingProgress:
    """The numbers in train.py's output lines, collected for the Training tab's charts."""
    _EPOCH = re.compile(r"epoch (\d+)/\d+  accuracy (\d+)%  loss ([\d.]+)  \|  val accuracy (\d+)%  val loss ([\d.]+)")
    _BEST = re.compile(r"best epoch (\d+)")
    _PER_SIGN = re.compile(r"  (\S+): (\d+/\d+)$")

    def __init__(self):
        self.epochs, self.accuracy, self.loss, self.val_accuracy, self.val_loss = [], [], [], [], []
        self.best_epoch = None
        self.test_score = ""  # "Test accuracy 88% (7/8 correct)   danke 4/4   hallo 3/4"

    def read(self, line):
        """Take in one output line. Returns True if it was a score line."""
        if match := self._EPOCH.match(line):
            epoch, accuracy, loss, val_accuracy, val_loss = match.groups()
            self.epochs.append(int(epoch))
            self.accuracy.append(int(accuracy) / 100)
            self.loss.append(float(loss))
            self.val_accuracy.append(int(val_accuracy) / 100)
            self.val_loss.append(float(val_loss))
        elif match := self._BEST.match(line):
            self.best_epoch = int(match[1])
        elif line.startswith("test accuracy"):
            self.test_score = line.replace("test", "Test", 1)
        elif self.test_score and (match := self._PER_SIGN.match(line)):
            self.test_score += f"   {match[1]} {match[2]}"
        else:
            return False
        return True


# --- tabs ---------------------------------------------------------------------------------------

class RecordTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=8)
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
            saved = self._recorder.stop()
            self._record_button.configure(text="● Record")
            if saved:
                path, frames, seconds = saved
                self._status.configure(text=f"Saved {path}  ({frames} frames, {seconds:.1f} s)")
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


class TrainingTab(ttk.Frame):
    """Runs the training job and shows its progress: live train-vs-val charts, then the test score."""
    TRAIN_COLOR, VAL_COLOR = "#1f77b4", "#ff7f0e"  # matplotlib's default blue and orange

    def __init__(self, parent, on_trained):
        super().__init__(parent, padding=8)
        self._on_trained = on_trained
        self._job = TrainingJob()
        self._progress = TrainingProgress()

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        self._train_button = ttk.Button(controls, text="Build dataset + Train", command=self._start)
        self._train_button.pack(side="left")
        self._status = ttk.Label(controls, text="Press 'Build dataset + Train' once each sign has its clips.")
        self._status.pack(side="left", padx=12)

        # Figure, not pyplot: pyplot keeps global state and can open windows of its own.
        figure = Figure(figsize=(9.6, 3.4), dpi=100, layout="constrained")
        self._accuracy_chart, self._loss_chart = figure.subplots(1, 2)
        self._canvas = FigureCanvasTkAgg(figure, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, pady=6)
        self._draw_charts()

        self._test_score = ttk.Label(self, font=("Segoe UI", 11, "bold"))
        self._test_score.pack(fill="x")
        self._log = tk.Text(self, height=6, state="disabled")
        self._log.pack(fill="both", pady=(6, 0))

    def _start(self):
        if self._job.running:
            return
        self._progress = TrainingProgress()
        self._draw_charts()
        self._test_score.configure(text="")
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")
        self._train_button.configure(state="disabled")
        self._status.configure(text="Building dataset... (new clips are tracked first, this can take a while)")
        self._job.start()

    def poll(self):
        """Move the job's new output lines into the log and charts. Called from the app's main loop."""
        scores_changed = False
        while True:
            try:
                line = self._job.lines.get_nowait()
            except queue.Empty:
                break
            if line is None:
                self._finish()
                continue
            if self._progress.read(line) and line.startswith("epoch "):
                self._status.configure(text=f"Training... {line}")
            scores_changed |= line.startswith(("epoch ", "best epoch"))
            self._log.configure(state="normal")
            self._log.insert("end", line + "\n")
            self._log.see("end")
            self._log.configure(state="disabled")
        if scores_changed:
            self._draw_charts()

    def _finish(self):
        self._train_button.configure(state="normal")
        if not self._job.succeeded:
            self._status.configure(text="Training failed - see the log below.")
            return
        self._status.configure(text="Training done - open the Detect tab.")
        self._test_score.configure(text=self._progress.test_score)
        self._on_trained()

    def _draw_charts(self):
        progress = self._progress
        for chart, title, train, val in ((self._accuracy_chart, "Accuracy", progress.accuracy, progress.val_accuracy),
                                         (self._loss_chart, "Loss", progress.loss, progress.val_loss)):
            chart.clear()
            chart.set_title(title)
            chart.set_xlabel("epoch")
            chart.grid(alpha=0.3)
            chart.plot(progress.epochs, train, label="train", color=self.TRAIN_COLOR)
            chart.plot(progress.epochs, val, label="val", color=self.VAL_COLOR)
            if progress.best_epoch:
                chart.axvline(progress.best_epoch, color="gray", linestyle="--",
                              label=f"best epoch {progress.best_epoch}")
            chart.set_xlim(1, max(len(progress.epochs), 10))
            chart.legend(loc="lower right" if chart is self._accuracy_chart else "upper right")
        self._accuracy_chart.set_ylim(0, 1.02)
        self._accuracy_chart.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        self._canvas.draw_idle()  # redraws once when Tk is idle, however many epochs arrived


class AlphabetTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=8)
        self._frame = self._results = None  # latest raw webcam frame and its tracking, for snap()

        controls = ttk.Frame(self)
        controls.pack(fill="x")
        ttk.Label(controls, text="Letter:").pack(side="left")
        self._letter = ttk.Combobox(controls, values=list(string.ascii_uppercase), width=6)
        self._letter.set("A")
        self._letter.pack(side="left", padx=4)
        self._letter.bind("<<ComboboxSelected>>", lambda _: self._letter_chosen())
        self._letter.bind("<Return>", lambda _: self._letter_chosen())
        self._letter.bind("<KeyRelease>", lambda _: self._update_count())
        ttk.Button(controls, text="📷 Snap", command=self.snap).pack(side="left", padx=4)
        self._count = ttk.Label(controls)
        self._count.pack(side="left", padx=8)

        self._video = ttk.Label(self)
        self._video.pack(pady=6)
        self._status = ttk.Label(self, text="Show the letter's hand shape, then press Snap or Space.")
        self._status.pack(fill="x")
        self._update_count()

    def show_camera_frame(self, frame, results):
        self._frame, self._results = frame.copy(), results  # copy: the photo is saved without landmarks
        draw_face_and_hands(frame, results)
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

    def _letter_chosen(self):
        self._update_count()
        self.focus_set()  # out of the letter box, so Space snaps instead of typing a space

    def _update_count(self):
        letter = clean_sign_name(self._letter.get())
        photos = len(list((IMAGES_DIR / letter).glob("*.jpg"))) if letter else 0
        self._count.configure(text=f"{photos}/{TARGET_SAMPLES} photos" if letter else "")


class ClipsTab(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=8)
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
        ttk.Button(library, text="Refresh", command=self.refresh).pack(fill="x", pady=(4, 0))

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

    def refresh(self):
        renamed = [old for sign in list_signs() for old in number_clips(sign)]
        if self._path in renamed:  # its name now belongs to another clip; Trim/Delete must not hit that one
            self._unload("Clips were renumbered. Select a clip on the left.")
        self._tree.delete(*self._tree.get_children())
        for sign in list_signs():
            clips = list_clips(sign)
            node = self._tree.insert("", "end", text=f"{sign}   ({len(clips)}/{TARGET_SAMPLES})", open=True)
            for clip in clips:
                self._tree.insert(node, "end", iid=str(clip), text=clip.name, values=(f"{clip_seconds(clip):.1f} s",))

    def _on_select(self, _event):
        selected = self._tree.selection()
        if selected and selected[0].endswith(".mp4"):  # sign rows have generated ids
            self._load(Path(selected[0]))

    def _load(self, path):
        self.pause()
        capture = cv2.VideoCapture(str(path))
        self._fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(fit(frame))  # display size only; trimming re-reads the original file
        capture.release()
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
        if not self._frames:
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
        forget_extracted(path)
        self.refresh()
        self._tree.selection_set(str(path))  # reloads the trimmed clip via <<TreeviewSelect>>

    def _delete(self):
        if not self._path or not messagebox.askyesno("Delete clip", f"Delete {self._path.name}? This can't be undone."):
            return
        self._path.unlink()
        forget_extracted(self._path)
        self._unload("Clip deleted. Select another clip.")
        self.refresh()

    def _unload(self, message):
        self.pause()
        self._generation += 1  # stop its analysis
        self._frames, self._path = [], None
        self._video.configure(image="", text=message)
        self._info.configure(text="")


class DetectTab(ttk.Frame):
    def __init__(self, parent, log):
        super().__init__(parent, padding=8)
        self._log = log
        self._detector = None
        self._needs_load = True
        self._status = ttk.Label(self)
        self._status.pack(fill="x")
        self._video = ttk.Label(self)
        self._video.pack(pady=6)

    def model_changed(self):
        self._needs_load = True

    def activate(self):
        if self._needs_load:
            self._status.configure(text="Loading model...")
            self.update_idletasks()
            self._detector = load_detector()
            self._needs_load = False
        if self._detector is None:
            self._status.configure(text="No trained model yet: record clips in the Record tab, then press "
                                        "'Build dataset + Train' in the Training tab.")
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


# --- app ----------------------------------------------------------------------------------------

class SignLanguageApp:
    """One webcam and one tracker shared by the camera tabs; the Clips and Training tabs don't use the camera."""

    def __init__(self, root):
        self._root = root
        self._camera = cv2.VideoCapture(0)
        self._tracker = HolisticTracker()
        self._log = DetectionLog(Path("detections.csv"), "app webcam")

        self._notebook = ttk.Notebook(root)
        self._notebook.pack(fill="both", expand=True)
        self._notebook.enable_traversal()  # Ctrl+Tab / Ctrl+Shift+Tab switch tabs
        self._detect = DetectTab(self._notebook, self._log)
        self._record = RecordTab(self._notebook)
        self._alphabet = AlphabetTab(self._notebook)
        self._clips = ClipsTab(self._notebook)
        self._training = TrainingTab(self._notebook, on_trained=self._detect.model_changed)
        self._notebook.add(self._record, text="Record")
        self._notebook.add(self._alphabet, text="Alphabet")
        self._notebook.add(self._clips, text="Clips")
        self._notebook.add(self._training, text="Training")
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
        if tab is not self._clips:
            self._clips.pause()
        if tab is self._clips:
            self._clips.refresh()
        if tab is self._detect:
            self._detect.activate()

    def _tick(self):
        tab = self._current_tab()
        if tab not in (self._clips, self._training):  # these two don't need the camera
            ok, frame = self._camera.read()  # blocks until the next frame, which paces this loop
            if ok:
                tab.show_camera_frame(frame, self._tracker.process(frame, time.monotonic() * 1000))
        self._training.poll()
        self._root.after(10, self._tick)

    def _close(self):
        self._record.stop_recording()
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
