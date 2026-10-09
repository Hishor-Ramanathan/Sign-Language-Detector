"""Turn labelled MP4s into training samples, split into train / val / test.

    videos/<label>/<any name>.mp4  ->  dataset/<split>/<label>/<same name>.npy   (shape 30 x 1692)

The folder name is the label. Every frame is tracked (the tracker needs consecutive frames),
then 30 frames spread evenly across the clip are kept, so clip length doesn't matter.

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


def extracted_path(root, label, stem):
    """The clip's .npy in whichever split it was put, or None if it isn't extracted yet."""
    for split in SPLITS:
        path = root / split / label / f"{stem}.npy"
        if path.exists():
            return path
    return None


def forget_extracted(root, clip):
    """Delete the clip's .npy so the next run rebuilds it from the changed clip."""
    sample = extracted_path(root, clip.parent.name, clip.stem)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=Path("videos"))
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--force", action="store_true", help="re-extract videos that already have a .npy")
    args = parser.parse_args()

    videos = sorted(args.videos.glob("*/*.mp4"))
    if not videos:
        raise SystemExit(f"No MP4s found under {args.videos}/<label>/ - see README.md")

    for video in videos:
        label = video.parent.name
        existing = extracted_path(args.out, label, video.stem)
        if existing and not args.force:
            continue
        sample, hand_frames, total = video_to_sample(video)
        if sample is None:
            print(f"SKIP  {video}: could not read any frames")
            continue
        out_file = existing or args.out / choose_split(args.out, label) / label / f"{video.stem}.npy"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_file, sample)
        warning = "  <-- WARNING: hands rarely visible, check this clip" if hand_frames < total / 2 else ""
        print(f"saved {out_file}  ({total} frames, hands in {hand_frames}){warning}")


if __name__ == "__main__":
    main()
