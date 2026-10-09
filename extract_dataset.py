"""Turn labelled MP4s and photos into training samples, split into train / val / test.

    videos/<label>/<any name>.mp4  ->  dataset/<split>/<label>/<same name>.npy         (shape 30 x 1692)
    images/<label>/<any name>.jpg  ->  dataset/<split>/<label>/<same name>_photo.npy

The folder name is the label. Every frame is tracked (the tracker needs consecutive frames),
then 30 frames spread evenly across the clip are kept, so clip length doesn't matter.
A photo is a letter held still: its one frame repeated 30 times, so the same model learns from photos and clips.

Each new sample goes to the split of its label that is furthest below its share of SPLITS.
Samples never move once placed, so a test clip can't leak into training on a later run.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from landmarks import SEQUENCE_LENGTH, HolisticTracker, extract_keypoints, has_hands

# train: the model learns from it. val: scored every epoch, picks when to stop. test: scored once at the end.
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def sample_stem(source):
    """The .npy name for a clip or photo: a_001.jpg -> a_001_photo, so it can't clash with clip a_001.mp4."""
    return f"{source.stem}_photo" if source.suffix == ".jpg" else source.stem


def extracted_path(root, label, stem):
    """The clip's .npy in whichever split it was put, or None if it isn't extracted yet."""
    for split in SPLITS:
        path = root / split / label / f"{stem}.npy"
        if path.exists():
            return path
    return None


def forget_extracted(root, clip):
    """Delete the clip's (or photo's) .npy so the next run rebuilds it from the changed clip."""
    sample = extracted_path(root, clip.parent.name, sample_stem(clip))
    if sample:
        sample.unlink()


def choose_split(root, label):
    """The split of this label furthest below its share once one more sample is added."""
    counts = {split: len(list((root / split / label).glob("*.npy"))) for split in SPLITS}
    total = sum(counts.values()) + 1
    return max(SPLITS, key=lambda split: SPLITS[split] * total - counts[split])


def video_to_sample(video_path):
    """Return (sample of shape (30, 1692), frames with a hand visible, total frames)."""
    capture = cv2.VideoCapture(str(video_path))
    keypoints, hand_frames = [], 0
    with HolisticTracker() as tracker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            results = tracker.process(frame, capture.get(cv2.CAP_PROP_POS_MSEC))
            keypoints.append(extract_keypoints(results))
            hand_frames += has_hands(results)
    capture.release()
    if not keypoints:
        return None, 0, 0
    # Short clips (< 30 frames) repeat frames here rather than failing.
    picks = np.linspace(0, len(keypoints) - 1, SEQUENCE_LENGTH).round().astype(int)
    return np.array(keypoints)[picks], hand_frames, len(keypoints)


def read_photo(path):
    """The photo as a BGR frame, or None. cv2.imread can't read non-ASCII paths (ä)."""
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def photo_to_sample(photo_path):
    """Return (sample of shape (30, 1692), 1 if a hand is visible else 0, 1 frame) - the same as video_to_sample."""
    frame = read_photo(photo_path)
    if frame is None:
        return None, 0, 0
    # ponytail: one tracker (model load) per photo, since photos aren't consecutive frames; batch if it gets slow
    with HolisticTracker() as tracker:
        results = tracker.process(frame, 0)
    return np.tile(extract_keypoints(results), (SEQUENCE_LENGTH, 1)), int(has_hands(results)), 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--images", type=Path, default=Path("images"),
                        help="the Alphabet tab's photos, images/<letter>/*.jpg")
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--force", action="store_true", help="re-extract clips and photos that already have a .npy")
    args = parser.parse_args()

    sources = sorted(args.videos.glob("*/*.mp4")) + sorted(args.images.glob("*/*.jpg"))
    if not sources:
        raise SystemExit(f"No MP4s under {args.videos}/<label>/ or photos under {args.images}/<label>/ - see README.md")

    for source in sources:
        label, stem = source.parent.name, sample_stem(source)
        existing = extracted_path(args.out, label, stem)
        if existing and not args.force:
            continue
        to_sample = photo_to_sample if source.suffix == ".jpg" else video_to_sample
        sample, hand_frames, total = to_sample(source)
        if sample is None:
            print(f"SKIP  {source}: could not read any frames")
            continue
        out_file = existing or args.out / choose_split(args.out, label) / label / f"{stem}.npy"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_file, sample)
        warning = "  <-- WARNING: hands rarely visible, check this clip" if hand_frames < total / 2 else ""
        print(f"saved {out_file}  ({total} frames, hands in {hand_frames}){warning}")


if __name__ == "__main__":
    main()
