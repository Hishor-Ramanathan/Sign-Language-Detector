"""Share videos/ with the team through a private Hugging Face dataset repo.

    python clip_sync.py       (or the Sync button in the app's Clips tab)

Every clip has one owner, whose tag (OWNER_TAGS) is in its name: videos/hallo/hallo_hr_001.mp4.
Only the owner uploads, trims or deletes it, so two people never write the same file.

pull() brings this folder and the repo in step:
  1. clips without an owner (recorded while logged out, copied in from a phone) become yours
  2. your clips that are missing or different on the Hub are uploaded, in one commit
  3. other people's clips that were deleted or trimmed on the Hub are dropped here, with their .npy
  4. clips on the Hub that aren't here are downloaded
Logged out or offline, nothing is synced and everything else works as before; the next pull catches up.
Log in once with:  .venv\\Scripts\\hf auth login   (a token with write access to the repo)
"""
import glob
import hashlib
import re
from pathlib import Path

import httpx2
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import EntryNotFoundError, LocalTokenNotFoundError, OfflineModeIsEnabled

from extract_dataset import extracted_path, forget_extracted

REPO_ID = "Rakobra/sign-hands-german"
VIDEOS_DIR = Path("videos")
DATASET_DIR = Path("dataset")
# Every way the Hub can't be reached: no token, no network, server or permission error.
NOT_REACHABLE = (LocalTokenNotFoundError, OfflineModeIsEnabled, httpx2.HTTPError)
# Hugging Face user -> the tag in their clip names: 2-4 lowercase letters. Add yourself before your first sync.
OWNER_TAGS = {"Hishor-R": "hr"}


def owner_of(clip):
    """videos/hallo/hallo_hr_001.mp4 -> 'hr'; hallo_001.mp4 or IMG_1234.mp4 -> None."""
    match = re.fullmatch(rf"{re.escape(clip.parent.name)}_([a-z]{{2,4}})_\d{{3}}", clip.stem)
    return match[1] if match else None


def new_clip_path(folder, owner):
    """The next name for a clip recorded into folder: hallo_<tag>_007.mp4, or hallo_007.mp4 without an owner.
    Numbers continue after the highest, so a deleted clip's name isn't reused."""
    prefix = f"{folder.name}_{owner}_" if owner else f"{folder.name}_"
    taken = [int(clip.stem[-3:]) for clip in folder.glob(f"{glob.escape(prefix)}[0-9][0-9][0-9].mp4")]
    return folder / f"{prefix}{max(taken, default=0) + 1:03d}.mp4"


def sha256(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class ClipSync:
    """Uploads, deletes and downloads clips in the team's dataset repo.
    `owner` is your tag from OWNER_TAGS, or None when logged out, offline or not in OWNER_TAGS."""

    def __init__(self, api=None, repo_id=REPO_ID):
        self._api = api or HfApi()
        self._repo = repo_id
        try:
            self.user = self._api.whoami()["name"]
        except NOT_REACHABLE:
            self.user = None
        self.owner = OWNER_TAGS.get(self.user)

    def can_edit(self, clip):
        """Only the owner trims or deletes a clip; a clip nobody owns yet belongs to whoever has it."""
        return owner_of(clip) in (None, self.owner)

    def push(self, clip):
        """Upload a clip you recorded or trimmed. Returns True if it reached the Hub; if not, pull() retries it."""
        if not self.owner or owner_of(clip) != self.owner:
            return False
        try:
            self._api.upload_file(path_or_fileobj=clip, path_in_repo=clip.as_posix(), repo_id=self._repo,
                                  repo_type="dataset", commit_message=f"Add {clip.name}")
        except NOT_REACHABLE:
            return False
        return True

    def remove(self, clip):
        """Delete your clip on the Hub too (call after deleting it here)."""
        if not self.owner or owner_of(clip) != self.owner:
            return
        try:
            self._api.delete_file(clip.as_posix(), self._repo, repo_type="dataset",
                                  commit_message=f"Delete {clip.name}")
        except EntryNotFoundError:
            pass  # never got uploaded
        except NOT_REACHABLE:
            pass  # ponytail: offline delete isn't remembered, the next pull downloads the clip again

    def pull(self):
        """Returns a one-line summary; prints each clip it renames or drops."""
        if not self.user:
            return "Not synced: log in with '.venv\\Scripts\\hf auth login', or no internet. Using local clips."
        if not self.owner:
            return f"Not synced: add '{self.user}' and your tag to OWNER_TAGS in clip_sync.py."
        # Untagged files on the Hub (uploaded by hand) are left alone: downloaded, they'd be adopted by whoever syncs.
        remote = {entry.path: entry.lfs.sha256
                  for entry in self._api.list_repo_tree(self._repo, recursive=True, repo_type="dataset")
                  if entry.path.startswith(f"{VIDEOS_DIR}/") and entry.path.endswith(".mp4")
                  and owner_of(Path(entry.path))}
        for clip in self._local_clips():
            if owner_of(clip) is None:
                self._adopt(clip)

        changed = [clip for clip in self._local_clips()
                   if owner_of(clip) == self.owner and remote.get(clip.as_posix()) != sha256(clip)]
        if changed:
            self._api.create_commit(
                self._repo, [CommitOperationAdd(clip.as_posix(), clip) for clip in changed],
                commit_message=f"Add {len(changed)} clips from {self.owner}", repo_type="dataset")

        for clip in self._local_clips():
            if owner_of(clip) != self.owner and remote.get(clip.as_posix()) != sha256(clip):
                clip.unlink()
                forget_extracted(DATASET_DIR, clip)
                print(f"dropped {clip} (deleted or changed on the Hub)")

        missing = [path for path in remote if not Path(path).exists()]
        for path in missing:
            self._api.hf_hub_download(self._repo, path, repo_type="dataset", local_dir=".")
        return f"Synced: {len(changed)} clips uploaded, {len(missing)} downloaded."

    @staticmethod
    def _local_clips():
        return sorted(VIDEOS_DIR.glob("*/*.mp4"))

    def _adopt(self, clip):
        """hallo/IMG_1234.mp4 -> hallo/hallo_<tag>_008.mp4, taking its .npy along."""
        target = new_clip_path(clip.parent, self.owner)
        sample = extracted_path(DATASET_DIR, clip.parent.name, clip.stem)
        clip.rename(target)
        if sample:
            sample.rename(sample.with_name(f"{target.stem}.npy"))
        print(f"renamed {clip} -> {target.name}")


if __name__ == "__main__":
    try:
        print(ClipSync().pull())
    except NOT_REACHABLE as error:  # lost the connection halfway; the next pull carries on
        print(f"Sync stopped: {error}")
