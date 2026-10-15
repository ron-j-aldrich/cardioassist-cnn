"""Predict synthetic acquisition severity from a degraded echocardiography video.

This network is not the ejection-fraction model. Its target is
``severity_label`` from ``degradation.py``: 0 is the unchanged video and 1
is the strongest pose. A view passes when the prediction is at or below
``ACCEPT_SEVERITY``. That cutoff is a chosen operating point, not a value
fit to ejection-fraction error.

Training uses the EchoNet-Dynamic ``TRAIN`` videos. The epoch is kept or
discarded using ``VAL`` only. ``TEST`` is not read here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cardioassist_cnn.degradation import degrade_video
from cardioassist_cnn.model import select_device
from cardioassist_cnn.preprocessing import (
    load_video_rgb,
    normalize_video,
    read_normalization,
    resize_to_model_input,
)

QUALITY_MODEL_ID = "cardioassist-severity-cnn-v1"
QUALITY_ROLE = "synthetic_acquisition_severity"
QUALITY_FRAMES = 8
ACCEPT_SEVERITY = 0.25


class SeverityNet(nn.Module):
    """Map eight normalized frames to one severity in (0, 1)."""

    def __init__(self, frames: int = QUALITY_FRAMES) -> None:
        super().__init__()
        self.frames = frames
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * frames, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] != self.frames:
            raise ValueError(
                f"Expected shape (batch, 3, {self.frames}, height, width), got {tuple(video.shape)}."
            )
        batch, channels, frames, height, width = video.shape
        flat = video.transpose(1, 2).reshape(batch, channels * frames, height, width)
        features = self.encoder(flat).flatten(1)
        return torch.sigmoid(self.head(features).squeeze(-1))


@dataclass(frozen=True)
class LoadedQuality:
    module: SeverityNet
    device: torch.device
    model_id: str
    frames: int
    threshold: float
    checkpoint_path: Path


def quality_frame_indices(frame_count: int, frames: int = QUALITY_FRAMES) -> tuple[int, ...]:
    """Evenly spaced frame numbers, including the first and the last."""

    if frame_count < 1 or frames < 1:
        raise ValueError("Frame sampling needs at least one frame.")
    if frame_count == 1 or frames == 1:
        return tuple(0 for _ in range(frames))
    positions = np.linspace(0, frame_count - 1, frames)
    return tuple(int(round(position)) for position in positions)


def passes_severity(score: float, threshold: float = ACCEPT_SEVERITY) -> bool:
    """True when predicted severity is low enough to accept the view."""

    if not np.isfinite(score) or not np.isfinite(threshold):
        raise ValueError("Severity and threshold must be finite.")
    if threshold < 0 or threshold > 1:
        raise ValueError("Severity threshold must be between 0 and 1.")
    return float(score) <= float(threshold)


def quality_clip_from_video(
    video: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    frames: int = QUALITY_FRAMES,
) -> np.ndarray:
    """Eight-frame clip, float32 ``(3, frames, 112, 112)``, using training-split normalization."""

    resized = resize_to_model_input(video)
    normalized = normalize_video(resized, mean, std)
    indices = quality_frame_indices(resized.shape[1], frames)
    return np.ascontiguousarray(normalized[:, indices, :, :], dtype=np.float32)


def example_from_decoded(
    video: np.ndarray,
    pose: Sequence[float],
    mean: np.ndarray,
    std: np.ndarray,
    frames: int = QUALITY_FRAMES,
) -> tuple[np.ndarray, float]:
    """Degrade one decoded video and return ``(clip, severity_label)``.

    The clip uses the same 0–255 training-split normalization as the
    ejection-fraction model.
    """

    degraded, settings = degrade_video(video, pose)
    return quality_clip_from_video(degraded, mean, std, frames), settings.severity_label


def sample_pose(generator: np.random.Generator) -> tuple[float, float, float]:
    values = generator.uniform(-1.0, 1.0, size=3)
    return float(values[0]), float(values[1]), float(values[2])


def validation_pose(filename: str) -> tuple[float, float, float]:
    """One stable pose per file, so a validation score can be repeated."""

    digest = hashlib.sha256(filename.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32)
    return sample_pose(np.random.default_rng(seed))


def fit_examples(
    model: SeverityNet,
    clips: Sequence[np.ndarray],
    labels: Sequence[float],
    steps: int,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Train on an in-memory batch. Returns the mean squared error at each step."""

    if steps < 1 or len(clips) == 0 or len(clips) != len(labels):
        raise ValueError("Training needs at least one step and one labeled clip.")
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    inputs = torch.from_numpy(np.stack(clips)).to(device)
    targets = torch.tensor(labels, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def predict_severity(loaded: LoadedQuality, clip: np.ndarray) -> float:
    """Severity prediction for one clip of shape ``(3, frames, 112, 112)``."""

    tensor = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)).unsqueeze(0)
    tensor = tensor.to(loaded.device)
    loaded.module.eval()
    with torch.no_grad():
        score = loaded.module(tensor)
    return float(score.detach().cpu())


def save_quality_checkpoint(
    path: str | Path,
    model: SeverityNet,
    threshold: float = ACCEPT_SEVERITY,
    history: list[dict] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "role": QUALITY_ROLE,
            "model_id": QUALITY_MODEL_ID,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "frames": model.frames,
            "threshold": threshold,
            "history": [] if history is None else history,
        },
        destination,
    )


def load_quality_model(path: str | Path, device: str | None = "auto") -> LoadedQuality:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("role") != QUALITY_ROLE:
        raise ValueError(f"Checkpoint role must be {QUALITY_ROLE!r}.")
    frames = int(checkpoint["frames"])
    model = SeverityNet(frames=frames)
    model.load_state_dict(checkpoint["state_dict"])
    selected = select_device(device)
    model.to(selected)
    model.eval()
    return LoadedQuality(
        module=model,
        device=selected,
        model_id=str(checkpoint.get("model_id", QUALITY_MODEL_ID)),
        frames=frames,
        threshold=float(checkpoint.get("threshold", ACCEPT_SEVERITY)),
        checkpoint_path=checkpoint_path,
    )


def split_filenames(data_dir: str | Path, split: str) -> list[str]:
    """Filenames kept by the official loader for one split."""

    import echonet

    name = split.lower()
    if name not in {"train", "val"}:
        raise ValueError("Split must be 'train' or 'val'. TEST is reserved for a later report.")
    dataset = echonet.datasets.Echo(root=str(data_dir), split=name, target_type="EF")
    return list(dataset.fnames)


def _video_path(data_dir: Path, filename: str) -> Path:
    return data_dir / "Videos" / filename


def _run_epoch(
    model: SeverityNet,
    data_dir: Path,
    filenames: list[str],
    poses: Sequence[tuple[float, float, float]],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[float], list[float]]:
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    seen = 0
    predictions: list[float] = []
    labels: list[float] = []
    clips: list[np.ndarray] = []
    batch_labels: list[float] = []

    def flush() -> None:
        nonlocal total_loss, seen
        if not clips:
            return
        inputs = torch.from_numpy(np.stack(clips)).to(device)
        targets = torch.tensor(batch_labels, dtype=torch.float32, device=device)
        if train:
            if optimizer is None:
                raise ValueError("Training requires an optimizer.")
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss = nn.functional.mse_loss(output, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(inputs)
                loss = nn.functional.mse_loss(output, targets)
        count = len(batch_labels)
        total_loss += float(loss.detach().cpu()) * count
        seen += count
        predictions.extend(float(value) for value in output.detach().cpu())
        labels.extend(batch_labels)
        clips.clear()
        batch_labels.clear()

    for index, (filename, pose) in enumerate(zip(filenames, poses), start=1):
        video = load_video_rgb(_video_path(data_dir, filename))
        clip, label = example_from_decoded(video, pose, mean, std, frames=model.frames)
        clips.append(clip)
        batch_labels.append(label)
        if len(clips) == batch_size:
            flush()
        if index % 200 == 0:
            phase = "train" if train else "val"
            print(f"{phase} {index}/{len(filenames)}", flush=True)
    flush()
    if seen == 0:
        raise ValueError("The epoch had no videos.")
    return total_loss / seen, predictions, labels


def train_quality_model(
    data_dir: str | Path,
    normalization_path: str | Path,
    output_path: str | Path,
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    device: str = "auto",
    train_limit: int | None = None,
    val_limit: int | None = None,
) -> dict:
    """Fit on ``TRAIN`` and keep the epoch with the lowest ``VAL`` mean absolute error."""

    from cardioassist_cnn.evaluation import mean_absolute_error

    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    root = Path(data_dir)
    mean, std, _ = read_normalization(normalization_path)
    train_names = split_filenames(root, "train")
    val_names = split_filenames(root, "val")
    if train_limit is not None:
        train_names = train_names[:train_limit]
    if val_limit is not None:
        val_names = val_names[:val_limit]
    if not train_names or not val_names:
        raise ValueError("Training and validation both need at least one video.")
    selected = select_device(device)
    model = SeverityNet().to(selected)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = np.random.default_rng(seed)
    best_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        order = generator.permutation(len(train_names))
        epoch_names = [train_names[int(index)] for index in order]
        train_poses = [sample_pose(generator) for _ in epoch_names]
        train_loss, _, _ = _run_epoch(
            model, root, epoch_names, train_poses, mean, std, selected, batch_size, train=True, optimizer=optimizer
        )
        val_poses = [validation_pose(name) for name in val_names]
        _, val_predictions, val_labels = _run_epoch(
            model, root, val_names, val_poses, mean, std, selected, batch_size, train=False
        )
        val_mae = mean_absolute_error(val_labels, val_predictions)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae, "val_videos": len(val_names)})
        print(
            f"epoch {epoch} train_mse {train_loss:.6f} val_mae {val_mae:.6f} videos {len(val_names)}",
            flush=True,
        )
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    save_quality_checkpoint(output_path, model, threshold=ACCEPT_SEVERITY, history=history)
    return {
        "model_id": QUALITY_MODEL_ID,
        "threshold": ACCEPT_SEVERITY,
        "pass_rule": "predicted_severity <= threshold",
        "train_videos": len(train_names),
        "best_val_mae": best_mae,
        "history": history,
        "checkpoint": str(output_path),
        "device": selected.type,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import logging

    parser = argparse.ArgumentParser(
        description=(
            "Train the severity model on TRAIN videos and select the epoch with VAL. "
            "TEST is not used. The target is the synthetic pose severity, not ejection fraction."
        )
    )
    parser.add_argument("--data-dir", default="data/EchoNet-Dynamic")
    parser.add_argument("--normalization", default="models/normalization.json")
    parser.add_argument("--output", default="models/severity_cnn.pt")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    args = parser.parse_args(argv)
    logging.getLogger("cardioassist_cnn").setLevel(logging.WARNING)
    summary = train_quality_model(
        data_dir=args.data_dir,
        normalization_path=args.normalization,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
