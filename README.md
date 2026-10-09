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
videos/<sign>/*.mp4  --extract_dataset.py-->  dataset/<sign>/*.npy  --train.py-->  model.keras + labels.txt  --detect.py-->  output field
```

## Status

| Sign | Clips in `videos/` | Converted to `dataset/` | Needed |
|---|---|---|---|
| `guten_tag` | 1 (`110276_GUTEN_TAG.mp4`: 75 frames, hands found in all 75) | 1 × `(30, 1692)` | 30+ |
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

The MediaPipe model (`holistic_landmarker.task`, ~14 MB) downloads itself on first run. That
download is the only network access. Tracking, training and detection all run on your machine,
and no video or landmarks are uploaded. Once the model file exists, everything works offline.

## 1. Record your MP4s

Make **one short clip per repetition of a sign**. Put the clips in a folder named after the sign.
**The folder name is the label**, so the file names don't matter. `videos/` is in `.gitignore`, so
the clips stay on your machine and are never committed.

```
videos/
├── hello/
│   ├── hello_01.mp4
│   ├── hello_02.mp4
│   └── ...
├── thanks/
│   ├── thanks_01.mp4
│   └── ...
└── iloveyou/
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

## 2. Convert the MP4s into a labelled dataset

```bash
python extract_dataset.py                 # reads videos/, writes dataset/
python extract_dataset.py --force         # re-extract everything
```

What happens for each `videos/<sign>/<clip>.mp4`:

1. Every frame goes through MediaPipe `HolisticLandmarker`.
2. Each frame becomes one vector of **1692 numbers**:
   pose 33×(x,y,z,visibility) + left hand 21×(x,y,z) + right hand 21×(x,y,z) + face 478×(x,y,z).
   A body part that isn't found is all zeros.
3. 30 frames spread evenly across the clip are kept and saved as `dataset/<sign>/<clip>.npy`
   with shape `(30, 1692)`.

Clips that already have a `.npy` are skipped, so you can add new videos and run it again.
A line ending in `WARNING: hands rarely visible` means the hands were found in fewer than half the
frames. Re-record that clip or delete it.

## 3. Train

```bash
python train.py                  # writes model.keras and labels.txt
python train.py --epochs 300
```

It uses the same LSTM as the original project. 10% of the clips are held out for validation, and
training stops early once validation loss stops improving.

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
| `extract_dataset.py` | MP4s → labelled `.npy` dataset |
| `train.py` | dataset → `model.keras` + `labels.txt` |
| `detect.py` | webcam/MP4 → landmarks + output field + console + CSV |
