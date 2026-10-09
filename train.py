"""Train the LSTM sign classifier on dataset/{train,val,test}/<label>/*.npy  ->  model.keras + labels.txt.

Learns from train/, scores val/ after every epoch (early stopping keeps the best epoch),
then scores test/ once: clips the model never saw, so that number is the honest one.
Architecture is the one from Sign-Language-Detection-using-MediaPipe's notebook.
"""
import argparse
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from landmarks import KEYPOINTS_PER_FRAME, SEQUENCE_LENGTH


def load_split(folder, labels):
    samples, targets = [], []
    for index, label in enumerate(labels):
        for file in sorted((folder / label).glob("*.npy")):
            samples.append(np.load(file))
            targets.append(index)
    return np.array(samples), to_categorical(targets, len(labels))


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


def print_test_score(model, x_test, y_test, labels):
    predicted = model.predict(x_test, verbose=0).argmax(axis=1)
    truth = y_test.argmax(axis=1)
    hits = predicted == truth
    print(f"test accuracy {hits.mean():.0%} ({hits.sum()}/{len(hits)} correct)")
    for index, label in enumerate(labels):
        mine = truth == index
        if mine.any():
            print(f"  {label}: {hits[mine].sum()}/{mine.sum()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    train_dir = args.data / "train"
    labels = sorted(d.name for d in train_dir.iterdir() if d.is_dir()) if train_dir.is_dir() else []
    if len(labels) < 2:
        raise SystemExit(f"Need at least 2 label folders in {train_dir}/, found {labels}")
    x_train, y_train = load_split(train_dir, labels)
    x_val, y_val = load_split(args.data / "val", labels)
    x_test, y_test = load_split(args.data / "test", labels)
    for split, x in (("val", x_val), ("test", x_test)):
        if not len(x):
            raise SystemExit(f"No samples in {args.data / split}/ yet: record more clips per sign "
                             f"(about every 7th clip lands in {split}/)")
    print(f"{len(x_train)} train / {len(x_val)} val / {len(x_test)} test samples, labels: {labels}")

    print_epoch_score = LambdaCallback(on_epoch_end=lambda epoch, logs: print(
        f"epoch {epoch + 1}/{args.epochs}  accuracy {logs['categorical_accuracy']:.0%}  loss {logs['loss']:.3f}"
        f"  |  val accuracy {logs['val_categorical_accuracy']:.0%}  val loss {logs['val_loss']:.3f}"))
    early_stopping = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)
    model = build_model(len(labels))
    history = model.fit(x_train, y_train, epochs=args.epochs, validation_data=(x_val, y_val),
                        callbacks=[print_epoch_score, early_stopping], verbose=0)

    best = early_stopping.best_epoch  # restore_best_weights put this epoch's weights back
    print(f"best epoch {best + 1} (val accuracy {history.history['val_categorical_accuracy'][best]:.0%})")
    print_test_score(model, x_test, y_test, labels)

    model.save("model.keras")
    Path("labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    print("saved model.keras and labels.txt")


if __name__ == "__main__":
    main()
