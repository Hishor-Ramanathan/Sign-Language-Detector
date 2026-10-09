# Sign Language Detector

Detects signs from a webcam or video. MediaPipe tracks the face, hands and body, an LSTM classifies
each 30-frame window, and the result is shown in an **output field** on screen, printed to the
console and logged to `detections.csv`.

Starting with German Sign Language (DGS), designed to extend to other sign languages.

Goals:

- Understand the ML pipeline end to end: data → features → training → evaluation.
- Learn what MediaPipe offers for hand/pose tracking.
- End with a clean, usable product.

Based on [Sign-Language-Detection-using-MediaPipe](../Sign-Language-Detection-using-MediaPipe), moved to
the current MediaPipe **Tasks API** (`HolisticLandmarker`). The old `mp.solutions.holistic` API that
project uses was retired in 2023 and is not in mediapipe releases after 0.10.21.

```
videos/<sign>/*.mp4
images/<letter>/*.jpg  --extract_dataset.py-->  dataset/{train,val,test}/<sign>/*.npy  --train.py-->  model.keras + labels.txt  --detect.py-->  output field
```

## Status

| Sign | Clips in `videos/` | Converted to `dataset/` | Needed |
|---|---|---|---|
| `guten_tag` | 1 (`guten_tag_001.mp4`: 75 frames, hands found in all 75) | 1 × `(30, 1692)` in `train/` | 30+ |
| `none` (other movements, idle hands) | 0 | 0 | 30+ |

Training needs at least 30 clips per sign and at least 2 signs. Next steps:

1. Add more clips to `videos/guten_tag/` and `videos/none/`.
2. `python extract_dataset.py` (already-converted clips are skipped).
3. `python train.py`
4. `python detect.py`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The MediaPipe model (`holistic_landmarker.task`, ~14 MB) downloads itself on first run. Tracking,
training and detection all run on your machine. The only other network access is the clip sync with
the team's Hugging Face repo (see [Sharing clips with the team](#sharing-clips-with-the-team)); logged
out, nothing is uploaded and everything works offline.

## The app: record, review, train, detect

```bash
python app.py
```

One window with four tabs (Ctrl+Tab switches between them). Each tab is one module in `tabs/`:

| Tab | What you do there |
|---|---|
| **Record** | Webcam with face/hand landmarks. Type a new sign (or pick one), press **● Record**, sign, press **■ Stop**. The clip is saved to `videos/<sign>/<sign>_<your tag>_001.mp4`, `_002`, … and uploaded to the team repo in the background; the counter shows how many clips the sign has out of the 30 target. Clips are saved without the landmark drawing. |
| **Alphabet** | Photos of fingerspelled letters. Pick a letter (A–Z, or type one like `Ä`), show its hand shape, press **📷 Snap** or **Space**. Next to the camera the tab shows `references/<letter>.jpg` (or `.png`) as a guide to the hand shape, if there is one; `references/` is in `.gitignore` because such pictures are usually someone else's. The photo is saved to `images/<letter>/<letter>_001.jpg`, `_002`, … without the landmark drawing; the counter shows photos and clips out of the 30 target. A frame with no hand found isn't saved. Letters that move (J, Z, Ä, Ö, Ü, SCH in DGS) are better recorded as clips: **● Record** / **■ Stop** saves one to `videos/<letter>/` and uploads it, just like the Record tab. What's inside `images/` is in `.gitignore`, like `videos/`. `extract_dataset.py` turns each photo into a training sample (the one frame repeated 30 times), so one model learns letters from photos and clips. |
| **Clips** | Every clip and photo, grouped by sign, with its length (`photo` for a photo). Select one to watch it (a photo shows as a one-frame clip): **Play** loops it, the slider scrubs, **Show landmarks** overlays the tracking, and the status line says in how many frames hands were found. To cut a clip, move to the first good frame and press **Set start**, then to the last and press **Set end**; Play now loops just that part. **Save trim** overwrites the clip with it. **Delete clip** removes it. Both ask first and can't be undone; the clip's `.npy` is dropped so the next training re-extracts it, and the change goes to the team repo. Only the clip's owner (the tag in the file name) can trim or delete it. **⟳ Sync with team** fetches the team's clips and uploads yours (see below). |
| **Detect** | Live detection with the output field, same as `detect.py`. Detections go to the console and `detections.csv`. |

Sign names are cleaned into folder names: `Guten Tag` becomes `guten_tag`. A recording stops by itself after 20 s.

The sections below explain the same steps for the command-line scripts, and how to use MP4s from elsewhere (e.g. a phone).

## 1. Record your MP4s

Make **one short clip per repetition of a sign**. Put the clips in a folder named after the sign.
**The folder name is the label**, so the file names don't matter; the next sync renames them to `<sign>_<your tag>_001.mp4` (e.g. `guten_tag_hr_001.mp4`), … anyway. What's inside `videos/` is in `.gitignore`
(only the empty folder is committed): clips are shared through Hugging Face, never through git.

```
videos/
├── hallo/
│   ├── hallo_001.mp4
│   ├── hallo_002.mp4
│   └── ...
├── danke/
│   ├── danke_001.mp4
│   └── ...
└── guten_tag/
    └── ...
```

Recording tips:

| Topic | Advice |
|---|---|
| Length | 1–3 s, one sign per clip. Start just before the sign, stop right after it. |
| Amount | At least **30 clips per sign** (the original project used 30). More is better. |
| Labels | At least 2 signs. Use lowercase folder names with no spaces (`thank_you`, not `Thank You`). |
| Framing | Face and both hands fully in view, upper body visible, good light, plain background. |
| Variety | Change distance, lighting, clothes, and people across clips so the model doesn't learn the background. |
| Phone videos | Any frame rate or resolution works. Each clip is resampled to 30 frames. |

Clip length doesn't need to match. 30 evenly spaced frames are taken from each clip.

## Sharing clips with the team

Clips live in the private Hugging Face dataset repo
[`Rakobra/sign-hands-german`](https://huggingface.co/datasets/Rakobra/sign-hands-german), next to the code's
`videos/` layout. One-time setup per person:

1. A Hugging Face account that is a member of the `Rakobra` organisation (write access).
2. A token with **Write** permission from <https://huggingface.co/settings/tokens>, then in your own terminal
   (the token isn't shown and is stored in your user folder, not in the project):
   ```bash
   .venv\Scripts\hf auth login
   .venv\Scripts\hf auth whoami      # check: prints your user name
   ```
3. Your user name and a tag of 2–4 lowercase letters (your initials) in `OWNER_TAGS` in `clip_sync.py`,
   e.g. `{"Hishor-R": "hr"}`. Commit that, so everyone's app knows the tag.

Every clip has one **owner**, whose tag is in the file name: `hallo_hr_001.mp4`. Only the owner uploads,
trims or deletes a clip, so two people never write the same file.

| When | What happens |
|---|---|
| You stop a recording | the clip is uploaded in the background |
| You trim or delete your clip | the change is uploaded |
| **⟳ Sync with team** in the Clips tab, or `python clip_sync.py` | 1. clips without an owner (recorded while logged out, copied in from a phone) get your tag<br>2. your clips that are missing or different on the Hub are uploaded<br>3. other people's clips that were deleted or trimmed on the Hub are dropped here, with their `.npy`<br>4. clips on the Hub that aren't here yet are downloaded |

Logged out or offline, clips are saved locally and the next sync uploads them. A clip you delete while
offline comes back with the next sync; delete it again once online.

Each computer splits the clips into train / val / test itself, so test scores from two computers aren't
directly comparable. The clips show your faces: keep the repo private, and only record people who agreed.

### The alphabet dataset on the Hub

The same repo holds the DGS manual alphabet as photos, ready for an image classifier:
`images/{train,validation,test}/<letter>/<letter>_p<person>_<photo>.jpg`, 27 letters (`a` … `z`, `sch`),
12 people × 50 photos each. Hugging Face reads the split from the first folder and the label from the second.
To get them:

```bash
.venv\Scripts\hf download Rakobra/sign-hands-german --repo-type dataset --include "images/*" --local-dir .
```

That puts them in this project under `images/train/`, `images/validation/`, `images/test/` (in `.gitignore`), next
to the Alphabet tab's own photos in `images/<letter>/`, which aren't shared. With the `datasets` library you can
also load them straight from the Hub: `load_dataset("Rakobra/sign-hands-german", "alphabet")`.

It's split **by person** (train 1–8, validation 9 and 12, test 10 and 11), so the test score shows how well a
model reads the hand shape of someone it has never seen. `import_alphabet.py` made it once from the original
PNGs (Kaggle, by schauerstoff, **CC BY-SA 4.0**: credit the author and share anything built from it under the
same license; see `images/SOURCE.md` on the Hub). The PNGs were then removed and are only in the repo's history. `extract_dataset.py` doesn't read these yet, only the Alphabet tab's own `images/<letter>/`.

## 2. Convert the MP4s into a labelled dataset

```bash
python extract_dataset.py                 # reads videos/ and images/<letter>/, writes dataset/
python extract_dataset.py --force         # re-extract everything
```

What happens for each `videos/<sign>/<clip>.mp4`:

1. Every frame goes through MediaPipe `HolisticLandmarker`.
2. Each frame becomes one vector of **1692 numbers**:
   pose 33×(x,y,z,visibility) + left hand 21×(x,y,z) + right hand 21×(x,y,z) + face 478×(x,y,z).
   A body part that isn't found is all zeros.
3. 30 frames spread evenly across the clip are kept and saved as `dataset/<split>/<sign>/<clip>.npy`
   with shape `(30, 1692)`.

Each sign is split 70 / 15 / 15 (`SPLITS` in `extract_dataset.py`):

```
dataset/
├── train/<sign>/*.npy   the model learns from these
├── val/<sign>/*.npy     scored after every epoch; early stopping keeps the best epoch
└── test/<sign>/*.npy    scored once at the end, on clips the model never saw
```

A new sample goes to whichever split of its sign is furthest below its share. Samples never move once
placed, so a test clip can't end up in training on a later run. With 10 clips a sign gets 7 / 2 / 1.

Clips that already have a `.npy` are skipped, so you can add new videos and run it again.
A line ending in `WARNING: hands rarely visible` means the hands were found in fewer than half the
frames. Re-record that clip or delete it.

## 3. Train

```bash
python train.py                  # writes model.keras and labels.txt
python train.py --epochs 300
```

It uses the same LSTM as the original project. It learns from `train/`, and training stops early
once the loss on `val/` stops improving. Each epoch prints one score line:

```
epoch 12/500  accuracy 83%  loss 0.412  |  val accuracy 75%  val loss 0.550
```

At the end it prints the best epoch (its weights are the ones saved) and the **test accuracy**, per sign:

```
best epoch 40 (val accuracy 90%)
test accuracy 88% (7/8 correct)
  danke: 4/4
  hallo: 3/4
```

`accuracy` is on clips the model learns from, so it climbs towards 100% anyway. The test accuracy is
the honest number. Training stops with a message until `val/` and `test/` each have a sample.

## 4. Detect

```bash
python detect.py                          # webcam
python detect.py --video some_clip.mp4    # a video file
python detect.py --threshold 0.8          # report only confident detections
python detect.py --preview                # landmarks only - check framing before recording, no model needed
```

Press `q` to quit. The window shows:

- the face and hand landmarks (pose is tracked for the model but not drawn)
- **Output field**: an orange bar with the last 5 detected signs, then `Detected: <sign> (<confidence>)`
  for the sign recognised right now
- a probability bar for each sign

A sign only counts as detected after it wins 10 predictions in a row with confidence above
`--threshold`. Each new detection is also:

- printed: `2026-10-09T10:41:07  hello  97%`
- appended to `detections.csv`: `wall_time, source, video_time_s, sign, confidence`

## Files

| File | Purpose |
|---|---|
| `landmarks.py` | MediaPipe tracking, face/hand drawing, keypoint extraction (shared) |
| `extract_dataset.py` | MP4s and photos → labelled `.npy` dataset |
| `train.py` | dataset → `model.keras` + `labels.txt` |
| `detect.py` | webcam/MP4 → landmarks + output field + console + CSV |
| `app.py` | one window with the Record / Alphabet / Clips / Detect tabs |
| `tabs/` | one module per tab, plus `common.py` for what they share |
| `clip_sync.py` | shares `videos/` with the team's Hugging Face repo |
| `import_alphabet.py` | one-off: the original alphabet PNGs (Hub history) → `images/<split>/<letter>/` on the Hub |
| `test_app.py` | smoke checks: `python test_app.py` |
