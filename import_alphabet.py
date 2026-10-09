"""One-off import: the DGS manual alphabet PNGs -> images/<split>/<letter>/ as JPGs, in this project and on the Hub.

    python import_alphabet.py

Source: external/schauerstoff-dgs-manual-alphabet/letters_<person>/<Letter>/<Letter>_<n>.png
        12 people, 50 photos each of A-Z and Sch; CC BY-SA 4.0, see images/SOURCE.md.
        external/ was deleted once imported, so the source is read from SOURCE_REVISION, a commit that still has it.
Output: images/<split>/<letter>/<letter>_p<person>_<n>.jpg, e.g. images/train/sch/sch_p07_023.jpg

Split by person, so nobody is in two splits: with 50 near-identical photos per person, a random split would
let the model recognise the person instead of the hand shape, and the test score would be too good.
The photos are written into this project (images/ is in .gitignore) and uploaded from there. Re-running is safe:
converted photos are kept and the upload skips what is already on the Hub.
Teammates get them without converting:
    .venv\\Scripts\\hf download Rakobra/sign-hands-german --repo-type dataset --include "images/*" --local-dir .
"""
import re
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from PIL import Image

from clip_sync import REPO_ID

SOURCE = "external/schauerstoff-dgs-manual-alphabet"
SOURCE_REVISION = "3cd0dfbc84acf1577ed63cb9f730bae09c70db41"
SPLITS = {"train": range(1, 9), "validation": (9, 12), "test": (10, 11)}  # people; README.txt says who is who
JPEG_QUALITY = 95  # the app's photos are JPGs too


def split_of(person):
    return next(split for split, people in SPLITS.items() if person in people)


def target_path(source):
    """.../letters_7/Sch/Sch_23.png -> images/train/sch/sch_p07_023.jpg"""
    person = int(re.fullmatch(r"letters_(\d+)", source.parent.parent.name)[1])
    letter = source.parent.name.lower()
    number = int(source.stem.rsplit("_", 1)[1])
    return Path("images", split_of(person), letter, f"{letter}_p{person:02d}_{number:03d}.jpg")


def main():
    print("downloading the PNGs (about 2 GB, into the Hugging Face cache)...")
    snapshot = Path(snapshot_download(REPO_ID, repo_type="dataset", revision=SOURCE_REVISION,
                                      allow_patterns=f"{SOURCE}/letters_*/*/*.png"))
    sources = sorted((snapshot / SOURCE).glob("letters_*/*/*.png"))
    print(f"converting {len(sources)} photos into images/<split>/...")
    for source in sources:
        target = target_path(source)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.open(source).convert("RGB").save(target, quality=JPEG_QUALITY)

    print("uploading (split into several commits; re-run to resume if it stops)...")
    # Only the split folders: images/<letter>/ next to them holds the app's own photos, which aren't shared.
    HfApi().upload_folder(repo_id=REPO_ID, repo_type="dataset", folder_path=".",
                          allow_patterns=[f"images/{split}/*/*.jpg" for split in SPLITS],
                          commit_message="Add the DGS manual alphabet as images/<split>/<letter>/, split by person")
    print("done")


if __name__ == "__main__":
    main()
