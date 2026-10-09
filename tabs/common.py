"""What several tabs share: where clips and photos live, sign names, writing MP4s, showing frames, background work."""
import re
import threading
from pathlib import Path

import cv2
from PIL import Image, ImageTk

from clip_sync import VIDEOS_DIR

IMAGES_DIR = Path("images")  # images/<letter>/<letter>_001.jpg, photos from the Alphabet tab
TARGET_SAMPLES = 30         # clips per sign (photos per letter) the README recommends
DISPLAY_SIZE = (560, 420)   # largest size a frame is shown at; keeps the window on a laptop screen


def clean_sign_name(text):
    """'Guten Tag ' -> 'guten_tag'. The folder name becomes the label, so keep it filesystem-safe."""
    return re.sub(r"\W+", "_", text.strip().lower()).strip("_")


def list_signs():
    """Every sign with clips or photos; a letter may have only photos."""
    folders = [d for root in (VIDEOS_DIR, IMAGES_DIR) if root.exists() for d in root.iterdir() if d.is_dir()]
    # any(): skips images/train/ etc. from the Hub download, which hold letter folders, not photos
    return sorted({folder.name for folder in folders if folder.parent == VIDEOS_DIR or any(folder.glob("*.jpg"))})


def list_clips(sign):
    return sorted((VIDEOS_DIR / sign).glob("*.mp4"))


def list_photos(sign):
    return sorted((IMAGES_DIR / sign).glob("*.jpg"))


def write_mp4(path, frames, fps):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def fit(frame):
    height, width = frame.shape[:2]
    scale = min(DISPLAY_SIZE[0] / width, DISPLAY_SIZE[1] / height, 1)
    return frame if scale == 1 else cv2.resize(frame, (int(width * scale), int(height * scale)))


def in_background(target, *args):
    """Run target(*args) on a daemon thread so a network call (an upload) doesn't freeze the window."""
    threading.Thread(target=target, args=args, daemon=True).start()


def show_frame(label, frame_bgr):
    image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(fit(frame_bgr), cv2.COLOR_BGR2RGB)))
    label.configure(image=image)
    label.image = image  # Tk keeps no reference of its own; without this the image is garbage-collected
