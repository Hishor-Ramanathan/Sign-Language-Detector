"""Smoke checks for the app's file handling, clip sync and the detector's smoothing.  Run: python test_app.py"""
import hashlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from clip_sync import ClipSync, new_clip_path, owner_of
from detect import STABLE_PREDICTIONS, SignDetector, draw_output_field, draw_probability_bars
from extract_dataset import choose_split, forget_extracted
from import_alphabet import SPLITS, target_path
from landmarks import SEQUENCE_LENGTH
from tabs.alphabet_tab import reference_picture, save_photo
from tabs.clips_tab import trim_clip
from tabs.common import clean_sign_name
from tabs.record_tab import ClipRecorder


def frame_count(path):
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def check_sign_names():
    assert clean_sign_name("  Guten Tag ") == "guten_tag"
    assert clean_sign_name("Tschüss!") == "tschüss"
    assert clean_sign_name("   ") == ""


def check_record_trim_forget():
    recorder = ClipRecorder()
    recorder.start("hallo")
    for i in range(40):
        recorder.add(np.full((120, 160, 3), i * 5, np.uint8))
    path, frames, _ = recorder.stop()
    assert path == Path("videos/hallo/hallo_001.mp4") and frames == 40 and frame_count(path) == 40
    assert not recorder.is_recording
    empty = ClipRecorder()
    empty.start("hallo")
    assert empty.stop() is None  # Stop pressed before any frame arrived: nothing saved

    trim_clip(path, 5, 24)
    assert frame_count(path) == 20

    npy = Path("dataset/val/hallo") / f"{path.stem}.npy"
    npy.parent.mkdir(parents=True)
    npy.touch()
    forget_extracted(Path("dataset"), path)
    assert not npy.exists()


def check_clip_names():
    assert owner_of(Path("videos/guten_tag/guten_tag_hr_007.mp4")) == "hr"
    assert owner_of(Path("videos/guten_tag/guten_tag_007.mp4")) is None  # recorded while logged out
    assert owner_of(Path("videos/guten_tag/IMG_1234.mp4")) is None      # copied in from a phone
    folder = Path("names_check/hallo")
    folder.mkdir(parents=True)
    assert new_clip_path(folder, "jr") == folder / "hallo_jr_001.mp4"
    for name in ("hallo_jr_001", "hallo_jr_004", "hallo_hr_009", "hallo_002"):
        (folder / f"{name}.mp4").touch()
    assert new_clip_path(folder, "jr") == folder / "hallo_jr_005.mp4"  # after the highest: no reuse
    assert new_clip_path(folder, None) == folder / "hallo_003.mp4"


class FakeHub:
    """The few HfApi calls ClipSync makes, on a dict of path -> bytes."""

    def __init__(self, user, files):
        self.user, self.files, self.commits = user, dict(files), 0

    def whoami(self):
        return {"name": self.user}

    def list_repo_tree(self, repo_id, recursive, repo_type):
        return [SimpleNamespace(path=path, lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))
                for path, data in self.files.items()]

    def create_commit(self, repo_id, operations, commit_message, repo_type):
        self.commits += 1
        self.files.update({op.path_in_repo: Path(op.path_or_fileobj).read_bytes() for op in operations})

    def hf_hub_download(self, repo_id, filename, repo_type, local_dir):
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[filename])


def check_clip_sync():
    os.chdir(tempfile.mkdtemp())
    me, bob = "hr", "jr"  # OWNER_TAGS["Hishor-R"] and a teammate
    hub = FakeHub("Hishor-R", {f"videos/hallo/hallo_{me}_001.mp4": b"mine, already up",
                         f"videos/hallo/hallo_{bob}_001.mp4": b"bob's new clip",
                         f"videos/hallo/hallo_{bob}_002.mp4": b"bob trimmed this",
                         "videos/hallo/uploaded_by_hand.mp4": b"no tag: left alone"})
    local = {f"videos/hallo/hallo_{me}_001.mp4": b"mine, already up",
             "videos/hallo/IMG_1234.mp4": b"from my phone",
             f"videos/hallo/hallo_{bob}_002.mp4": b"bob before trimming",
             f"videos/hallo/hallo_{bob}_009.mp4": b"bob deleted this"}
    for path, data in local.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
    for stem in ("IMG_1234", f"hallo_{bob}_009"):
        Path("dataset/train/hallo").mkdir(parents=True, exist_ok=True)
        Path(f"dataset/train/hallo/{stem}.npy").touch()

    sync = ClipSync(hub)
    sync.pull()

    assert hub.files == {f"videos/hallo/hallo_{me}_001.mp4": b"mine, already up",
                         f"videos/hallo/hallo_{me}_002.mp4": b"from my phone",  # adopted, then uploaded
                         f"videos/hallo/hallo_{bob}_001.mp4": b"bob's new clip",
                         f"videos/hallo/hallo_{bob}_002.mp4": b"bob trimmed this",
                         "videos/hallo/uploaded_by_hand.mp4": b"no tag: left alone"}, hub.files.keys()
    assert hub.commits == 1  # only the adopted clip; the unchanged one isn't uploaded again
    assert {p.as_posix(): p.read_bytes() for p in Path("videos").glob("*/*.mp4")} == {
        path: data for path, data in hub.files.items() if "uploaded_by_hand" not in path}
    assert sorted(p.name for p in Path("dataset").rglob("*.npy")) == [f"hallo_{me}_002.npy"]  # moved / dropped
    assert sync.can_edit(Path(f"videos/hallo/hallo_{me}_002.mp4"))
    assert not sync.can_edit(Path(f"videos/hallo/hallo_{bob}_001.mp4"))


def check_alphabet_import():
    source = Path("external/schauerstoff-dgs-manual-alphabet")
    assert target_path(source / "letters_7/Sch/Sch_23.png") == Path("images/train/sch/sch_p07_023.jpg")
    assert target_path(source / "letters_11/O/0_5.png") == Path("images/test/o/o_p11_005.jpg")  # label from folder
    assert target_path(source / "letters_12/A/A_50.png") == Path("images/validation/a/a_p12_050.jpg")
    people = [person for split in SPLITS.values() for person in split]
    assert sorted(people) == list(range(1, 13))  # all 12 people, each in exactly one split


def check_split_is_70_15_15(root=Path("split_check")):
    for i in range(10):
        sample = root / choose_split(root, "hallo") / "hallo" / f"{i}.npy"
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.touch()
    counts = {split: len(list((root / split / "hallo").glob("*.npy"))) for split in ("train", "val", "test")}
    assert counts == {"train": 7, "val": 2, "test": 1}, counts


def check_save_photo():
    frame = np.zeros((120, 160, 3), np.uint8)
    hand = SimpleNamespace(left_hand_landmarks=[object()], right_hand_landmarks=[])
    no_hand = SimpleNamespace(left_hand_landmarks=[], right_hand_landmarks=[])
    assert save_photo("a", frame, hand) == Path("images/a/a_001.jpg")
    assert save_photo("a", frame, hand) == Path("images/a/a_002.jpg")
    assert save_photo("a", frame, no_hand) is None
    assert save_photo("ä", frame, hand) == Path("images/ä/ä_001.jpg")  # non-ASCII folder still works
    assert cv2.imdecode(np.fromfile("images/a/a_001.jpg", np.uint8), cv2.IMREAD_COLOR).shape == frame.shape


def check_reference_picture():
    assert reference_picture("a") is None  # no references/ folder yet
    Path("references").mkdir()
    Path("references/a.png").touch()
    assert reference_picture("a") == Path("references/a.png") and reference_picture("b") is None


def check_detector_reports_each_sign_once():
    no_landmarks = SimpleNamespace(pose_landmarks=[], left_hand_landmarks=[], right_hand_landmarks=[], face_landmarks=[])
    always_hallo = lambda batch, training: np.array([[0.9, 0.1]])
    detector = SignDetector(always_hallo, ["hallo", "danke"], threshold=0.5)
    reports = [detector.step(no_landmarks) for _ in range(SEQUENCE_LENGTH + 50)]
    first = SEQUENCE_LENGTH - 1 + STABLE_PREDICTIONS - 1  # window full, then 10 agreeing predictions
    assert reports[first] == ("hallo", 0.9), reports[first - 1:first + 2]
    assert all(r is None for i, r in enumerate(reports) if i != first)


def check_overlay_draws():
    for count in (2, 6):
        labels = [f"sign_{i}" for i in range(count)]
        frame = np.full((480, 640, 3), 200, np.uint8)
        draw_probability_bars(frame, np.linspace(0, 1, count), labels, 0.5, labels[-1])
        draw_output_field(frame, (labels[-1], 1.0), labels)
        assert (frame[0, 0] < 200).all() and (frame[0, 0] > 0).all()  # banner is see-through, not opaque
    draw_output_field(frame, None, [])  # nothing detected yet


if __name__ == "__main__":
    os.chdir(tempfile.mkdtemp())  # the app works relative to the current folder
    check_sign_names()
    check_record_trim_forget()
    check_clip_names()
    check_alphabet_import()
    check_split_is_70_15_15()
    check_save_photo()
    check_reference_picture()
    check_detector_reports_each_sign_once()
    check_overlay_draws()
    check_clip_sync()  # last: it moves to a folder of its own
    print("all checks passed")
