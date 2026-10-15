"""Score the test split by averaging one clip around each detected beat.

This follows ``scripts/beat_by_beat_analysis.R`` and the systole detector in
``echonet/utils/segmentation.py``. It is the procedure behind the paper's
reported test-set figures. A video with no matched beat is left out of the
average, which is what that script's inner join does.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from cardioassist_cnn.evaluation import (
    coefficient_of_determination,
    labeled_videos,
    mean_absolute_error,
    pearson_correlation,
    root_mean_squared_error,
)
from cardioassist_cnn.inference import EFPredictor
from cardioassist_cnn.preprocessing import (
    load_video_rgb,
    normalize_video,
    resize_to_model_input,
    sample_clip,
)
from cardioassist_cnn.segmentation import (
    LoadedSegmentation,
    beat_clip_starts,
    load_segmentation_model,
    systole_frames,
    ventricle_sizes,
)

LOGGER = logging.getLogger(__name__)


def _predict_starts(
    predictor: EFPredictor,
    segmentation: LoadedSegmentation,
    normalized: np.ndarray,
    starts: list[int],
) -> tuple[np.ndarray, LoadedSegmentation]:
    """Run one normalized video at several clip starts."""

    clips = []
    for start in starts:
        clip = sample_clip(
            normalized,
            frames=predictor.loaded.frames,
            period=predictor.loaded.period,
            clip_start=start,
        )
        clips.append(torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0))
    batch = torch.cat(clips, dim=0)
    module = predictor.loaded.module
    device = predictor.loaded.device
    module.eval()
    with torch.inference_mode():
        try:
            output = module(batch.to(device))
            if device.type == "mps":
                torch.mps.synchronize()
        except RuntimeError:
            if device.type == "cpu":
                raise
            LOGGER.info("Beat forward pass failed on %s. Retrying on CPU.", device.type)
            module.to("cpu")
            segmentation.module.to("cpu")
            predictor.loaded = type(predictor.loaded)(
                module=module,
                device=torch.device("cpu"),
                model_id=predictor.loaded.model_id,
                frames=predictor.loaded.frames,
                period=predictor.loaded.period,
                checkpoint_path=predictor.loaded.checkpoint_path,
            )
            segmentation = _move_segmentation(segmentation, torch.device("cpu"))
            output = module(batch.to("cpu"))
    return output.detach().view(-1).cpu().numpy(), segmentation


def _move_segmentation(loaded: LoadedSegmentation, device: torch.device) -> LoadedSegmentation:
    loaded.module.to(device)
    return LoadedSegmentation(
        module=loaded.module,
        device=device,
        model_id=loaded.model_id,
        checkpoint_path=loaded.checkpoint_path,
    )


def score_beats(
    data_dir,
    ef_checkpoint,
    segmentation_checkpoint,
    normalization_path,
    split: str = "test",
    device: str = "auto",
    predictions_path=None,
    segmentation_batch_size: int = 16,
) -> dict:
    """Average beat-centered clips and compare them with file labels."""

    import csv

    logging.getLogger("cardioassist_cnn").setLevel(logging.WARNING)
    rows = labeled_videos(data_dir, split)
    if not rows:
        raise ValueError(f"No videos were kept for split {split!r}.")
    predictor = EFPredictor(
        checkpoint_path=ef_checkpoint,
        normalization_path=normalization_path,
        device=device,
        clip_start=0,
    )
    segmentation = load_segmentation_model(
        segmentation_checkpoint,
        predictor.loaded.device,
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
            fieldnames=(
                "FileName",
                "label_ef",
                "predicted_ef",
                "n_beats",
                "status",
                "device",
            ),
        )
        writer.writeheader()

    labels: list[float] = []
    predictions: list[float] = []
    failures = 0
    no_beats = 0
    try:
        for index, (filename, label) in enumerate(rows, start=1):
            status = "ok"
            predicted_mean = None
            n_beats = 0
            try:
                raw = load_video_rgb(root / "Videos" / filename)
                resized = resize_to_model_input(raw)
                normalized = normalize_video(resized, predictor.mean, predictor.std)
                frame_tensor = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(normalized, (1, 0, 2, 3)))
                )
                try:
                    sizes = ventricle_sizes(
                        segmentation,
                        frame_tensor,
                        batch_size=segmentation_batch_size,
                    )
                except RuntimeError:
                    if segmentation.device.type == "cpu":
                        raise
                    segmentation = _move_segmentation(segmentation, torch.device("cpu"))
                    predictor.loaded.module.to("cpu")
                    predictor.loaded = type(predictor.loaded)(
                        module=predictor.loaded.module,
                        device=torch.device("cpu"),
                        model_id=predictor.loaded.model_id,
                        frames=predictor.loaded.frames,
                        period=predictor.loaded.period,
                        checkpoint_path=predictor.loaded.checkpoint_path,
                    )
                    sizes = ventricle_sizes(
                        segmentation,
                        frame_tensor,
                        batch_size=segmentation_batch_size,
                    )
                starts = beat_clip_starts(
                    systole_frames(sizes),
                    frame_count=resized.shape[1],
                    frames=predictor.loaded.frames,
                    period=predictor.loaded.period,
                )
                n_beats = len(starts)
                if not starts:
                    status = "no_beat"
                    no_beats += 1
                else:
                    values, segmentation = _predict_starts(
                        predictor,
                        segmentation,
                        normalized,
                        starts,
                    )
                    predicted_mean = float(np.mean(values))
                    labels.append(label)
                    predictions.append(predicted_mean)
            except (OSError, RuntimeError, ValueError) as exc:
                status = "failed"
                failures += 1
                LOGGER.warning("Beat scoring failed for %s: %s", filename, exc)
            if writer is not None and handle is not None:
                writer.writerow(
                    {
                        "FileName": filename,
                        "label_ef": label,
                        "predicted_ef": "" if predicted_mean is None else predicted_mean,
                        "n_beats": n_beats,
                        "status": status,
                        "device": predictor.loaded.device.type,
                    }
                )
                handle.flush()
            if index == 1 or index % 25 == 0 or index == len(rows):
                print(
                    f"{index}/{len(rows)} status={status} device={predictor.loaded.device.type}",
                    flush=True,
                )
    finally:
        if handle is not None:
            handle.close()
    if not labels:
        raise ValueError("No beat-averaged predictions were produced.")
    return {
        "split": split.lower(),
        "protocol": "beat_average_frame_minus_32",
        "videos": len(rows),
        "scored": len(labels),
        "no_beats": no_beats,
        "failures": failures,
        "device": predictor.loaded.device.type,
        "mean_absolute_error": mean_absolute_error(labels, predictions),
        "root_mean_squared_error": root_mean_squared_error(labels, predictions),
        "pearson_correlation": pearson_correlation(labels, predictions),
        "r2": coefficient_of_determination(labels, predictions),
        "predictions_path": "" if destination is None else str(destination),
    }


def main(argv: list[str] | None = None) -> int:
    """Score beat-averaged ejection fraction. The default split is test."""

    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Average one ejection-fraction clip around each segmented beat "
            "and compare it with the FileList label."
        )
    )
    parser.add_argument("--data-dir", default="data/EchoNet-Dynamic")
    parser.add_argument("--checkpoint", default="models/r2plus1d_18_32_2_pretrained.pt")
    parser.add_argument(
        "--segmentation-checkpoint",
        default="models/deeplabv3_resnet50_random.pt",
    )
    parser.add_argument("--normalization", default="models/normalization.json")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", default="output/test_beat_predictions.csv")
    args = parser.parse_args(argv)
    summary = score_beats(
        data_dir=args.data_dir,
        ef_checkpoint=args.checkpoint,
        segmentation_checkpoint=args.segmentation_checkpoint,
        normalization_path=args.normalization,
        split=args.split,
        device=args.device,
        predictions_path=args.output,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
