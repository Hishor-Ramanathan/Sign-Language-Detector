"""Train the LSTM sign classifier on dataset/<label>/*.npy  ->  model.keras + labels.txt.

Architecture is the one from Sign-Language-Detection-using-MediaPipe's notebook.
"""
import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from landmarks import KEYPOINTS_PER_FRAME, SEQUENCE_LENGTH


def load_dataset(root):
    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    samples, targets = [], []
    for index, label in enumerate(labels):
        for file in sorted((root / label).glob("*.npy")):
            samples.append(np.load(file))
            targets.append(index)
    return np.array(samples), to_categorical(targets, len(labels)), labels


def build_model(num_labels):
    model = Sequential([
        Input((SEQUENCE_LENGTH, KEYPOINTS_PER_FRAME)),
        LSTM(64, return_sequences=True, activation="relu"),
        LSTM(128, return_sequences=True, activation="relu"),
        LSTM(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(num_labels, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    x, y, labels = load_dataset(args.data)
    if len(labels) < 2:
        raise SystemExit(f"Need at least 2 label folders in {args.data}/, found {labels}")
    print(f"{len(x)} samples, labels: {labels}")

    # validation_split takes the *last* 10%, so shuffle first or it would hold out one whole label.
    order = np.random.default_rng(0).permutation(len(x))
    model = build_model(len(labels))
    model.fit(x[order], y[order], epochs=args.epochs, validation_split=0.1,
              callbacks=[EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)])

    model.save("model.keras")
    Path("labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    print("saved model.keras and labels.txt")


if __name__ == "__main__":
    main()
