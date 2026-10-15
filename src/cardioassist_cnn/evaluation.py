"""Metrics for a labeled evaluation split.

These functions compare predictions with ground-truth ejection fraction.
They do not decide which videos belong in the test set. Call them only with
labels from a split that was not used to train the model or to choose
preprocessing statistics.
"""

from __future__ import annotations

import math

import numpy as np


def _paired_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if truth.size == 0:
        raise ValueError("Evaluation needs at least one labeled video.")
    if truth.shape != predicted.shape:
        raise ValueError(
            f"Ground truth has {truth.size} values and predictions have {predicted.size}."
        )
    if np.isnan(truth).any() or np.isnan(predicted).any():
        raise ValueError("Ground truth and predictions must not contain NaN.")
    return truth, predicted


def mean_absolute_error(y_true, y_pred) -> float:
    """Average of the absolute differences, in the same units as the labels."""

    truth, predicted = _paired_arrays(y_true, y_pred)
    return float(np.mean(np.abs(predicted - truth)))


def root_mean_squared_error(y_true, y_pred) -> float:
    """Square root of the average squared difference."""

    truth, predicted = _paired_arrays(y_true, y_pred)
    return float(math.sqrt(np.mean((predicted - truth) ** 2)))


def pearson_correlation(y_true, y_pred) -> float:
    """Pearson correlation between labels and predictions."""

    truth, predicted = _paired_arrays(y_true, y_pred)
    if truth.size < 2:
        raise ValueError("Correlation needs at least two labeled videos.")
    if np.std(truth) == 0 or np.std(predicted) == 0:
        raise ValueError("Correlation is undefined when one side does not vary.")
    return float(np.corrcoef(truth, predicted)[0, 1])


def coefficient_of_determination(y_true, y_pred) -> float:
    """One minus the residual sum of squares over the total sum of squares.

    This is the R² reported by the EchoNet-Dynamic paper. It is not the
    Pearson correlation.
    """

    truth, predicted = _paired_arrays(y_true, y_pred)
    total = float(np.sum((truth - np.mean(truth)) ** 2))
    if total == 0:
        raise ValueError("R² is undefined when the labels do not vary.")
    residual = float(np.sum((truth - predicted) ** 2))
    return 1.0 - residual / total


def labeled_videos(data_dir, split: str) -> list[tuple[str, float]]:
    """Return ``(filename, ejection fraction)`` for one official split.

    The list is the one ``echonet.datasets.Echo`` keeps after it drops videos
    with fewer than two tracings. The training split is rejected here because
    its videos were used to compute the normalization constants.
    """

    import echonet

    from pathlib import Path

    name = split.lower()
    if name in {"train", "all"}:
        raise ValueError(
            f"Split {split!r} includes videos used to compute normalization. "
            "Score 'val' or 'test'."
        )
    root = Path(data_dir)
    dataset = echonet.datasets.Echo(root=str(root), split=name, target_type="EF")
    ef_index = dataset.header.index("EF")
    return [
        (filename, float(outcome[ef_index]))
        for filename, outcome in zip(dataset.fnames, dataset.outcome)
    ]


def score_split(
    data_dir,
    checkpoint_path,
    normalization_path,
    split: str = "test",
    clip_start: int = 0,
    device: str = "auto",
    predictions_path=None,
) -> dict:
    """Predict one frame-0 clip per video and compare it with the file label.

    This is not the paper's beat-averaged protocol. Each prediction uses
    ``clip_start`` and the saved training-split normalization.
    """

    import csv
    import logging
    from pathlib import Path

    from cardioassist_cnn.inference import EFPredictor

    logging.getLogger("cardioassist_cnn").setLevel(logging.WARNING)
    rows = labeled_videos(data_dir, split)
    if not rows:
        raise ValueError(f"No videos were kept for split {split!r}.")
    predictor = EFPredictor(
        checkpoint_path=checkpoint_path,
        normalization_path=normalization_path,
        device=device,
        clip_start=clip_start,
    )
    root = Path(data_dir)
    destination = None if predictions_path is None else Path(predictions_path)
    handle = None
    writer = None
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        handle = destination.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            handle,
            fieldnames=("FileName", "label_ef", "predicted_ef", "status", "device", "forward_seconds"),
        )
        writer.writeheader()
    labels: list[float] = []
    predictions: list[float] = []
    failures = 0
    try:
        for index, (filename, label) in enumerate(rows, start=1):
            result = predictor.predict(root / "Videos" / filename)
            predicted = result.ejection_fraction_percent
            if result.status == "ok" and predicted is not None:
                labels.append(label)
                predictions.append(predicted)
            else:
                failures += 1
            if writer is not None and handle is not None:
                writer.writerow(
                    {
                        "FileName": filename,
                        "label_ef": label,
                        "predicted_ef": "" if predicted is None else predicted,
                        "status": result.status,
                        "device": result.device,
                        "forward_seconds": "" if result.forward_seconds is None else result.forward_seconds,
                    }
                )
                handle.flush()
            if index == 1 or index % 50 == 0 or index == len(rows):
                print(
                    f"{index}/{len(rows)} status={result.status} device={result.device}",
                    flush=True,
                )
    finally:
        if handle is not None:
            handle.close()
    summary = {
        "split": split.lower(),
        "clip_start": clip_start,
        "videos": len(rows),
        "scored": len(labels),
        "failures": failures,
        "device": predictor.loaded.device.type,
        "mean_absolute_error": mean_absolute_error(labels, predictions),
        "root_mean_squared_error": root_mean_squared_error(labels, predictions),
        "pearson_correlation": pearson_correlation(labels, predictions),
        "r2": coefficient_of_determination(labels, predictions),
        "predictions_path": "" if destination is None else str(destination),
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    """Score one split. The default split is test, with the clip starting at frame 0."""

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Compare one frame-0 clip per video with the FileList ejection-fraction "
            "label. This is not the paper's beat-averaged test."
        )
    )
    parser.add_argument("--data-dir", default="data/EchoNet-Dynamic")
    parser.add_argument("--checkpoint", default="models/r2plus1d_18_32_2_pretrained.pt")
    parser.add_argument("--normalization", default="models/normalization.json")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--clip-start", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", default="output/test_frame0_predictions.csv")
    args = parser.parse_args(argv)
    summary = score_split(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        normalization_path=args.normalization,
        split=args.split,
        clip_start=args.clip_start,
        device=args.device,
        predictions_path=args.output,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
