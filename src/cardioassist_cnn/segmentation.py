"""Load the official EchoNet-Dynamic left-ventricle segmentation checkpoint.

The architecture is torchvision's DeepLabV3 ResNet-50 with one output
channel, which is the construction in ``echonet/utils/segmentation.py``.
Per-frame ventricle size and the systole frames follow that file. The beat
window then follows ``scripts/beat_by_beat_analysis.R``: the ejection-fraction
clip starts 32 frames before each systole frame, and a negative start is
clamped to 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import torchvision

SEGMENTATION_MODEL_ID = "echonet-dynamic-deeplabv3_resnet50-v1.0.0"
EXPECTED_SEGMENTATION_BYTES = 317_401_511


@dataclass(frozen=True)
class LoadedSegmentation:
    """Segmentation model kept in memory for repeated videos."""

    module: torch.nn.Module
    device: torch.device
    model_id: str
    checkpoint_path: Path


def load_segmentation_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> LoadedSegmentation:
    """Build DeepLabV3 ResNet-50 and load the official segmentation checkpoint.

    ``weights=None`` and ``weights_backbone=None`` skip the torchvision
    downloads. The EchoNet checkpoint replaces every parameter. The saved
    ``frames`` and ``period`` belong to segmentation training. Frame-wise
    size measurement does not use them.
    """

    from cardioassist_cnn.model import select_device

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    size = path.stat().st_size
    if size != EXPECTED_SEGMENTATION_BYTES:
        raise ValueError(
            f"Checkpoint size is {size} bytes. The v1.0.0 release asset is "
            f"{EXPECTED_SEGMENTATION_BYTES} bytes: {path}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "Expected the EchoNet training dictionary with a 'state_dict' entry."
        )
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=False,
    )
    last = model.classifier[-1]
    model.classifier[-1] = torch.nn.Conv2d(
        last.in_channels,
        1,
        kernel_size=last.kernel_size,
    )
    state = {}
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("module."):
            state[key[len("module.") :]] = value
        else:
            state[key] = value
    weight = state.get("classifier.4.weight")
    if weight is None or tuple(weight.shape) != (1, last.in_channels, 1, 1):
        raise ValueError(
            "The checkpoint output layer is not the one-channel segmentation head."
        )
    model.load_state_dict(state, strict=True)
    selected = device if device is not None else select_device("auto")
    model.to(selected)
    model.eval()
    return LoadedSegmentation(
        module=model,
        device=selected,
        model_id=SEGMENTATION_MODEL_ID,
        checkpoint_path=path,
    )


def ventricle_sizes(
    loaded: LoadedSegmentation,
    frames: torch.Tensor,
    batch_size: int = 16,
) -> np.ndarray:
    """Return the number of segmented pixels in each frame.

    ``frames`` is float32 ``(frames, 3, height, width)``, already normalized.
    A pixel counts when the single logit is greater than 0, which is the
    rule in ``echonet/utils/segmentation.py``.
    """

    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError(
            f"Expected frames shaped (frames, 3, height, width), got {tuple(frames.shape)}."
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    counts = []
    loaded.module.eval()
    with torch.inference_mode():
        for start in range(0, frames.shape[0], batch_size):
            batch = frames[start : start + batch_size].to(loaded.device)
            logits = loaded.module(batch)["out"]
            if loaded.device.type == "mps":
                torch.mps.synchronize()
            counts.append((logits[:, 0] > 0).flatten(1).sum(dim=1).detach().cpu().numpy())
    if not counts:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(counts)


def systole_frames(size: np.ndarray) -> np.ndarray:
    """Return frame indexes of detected systole.

    The trim points, distance, and prominence are the values in
    ``echonet/utils/segmentation.py``. Peaks are found on the negated size,
    so a small ventricle is a peak.
    """

    values = np.asarray(size, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return np.zeros((0,), dtype=np.int64)
    ordered = sorted(values.tolist())
    trim_min = ordered[round(len(ordered) ** 0.05)]
    trim_max = ordered[round(len(ordered) ** 0.95)]
    trim_range = trim_max - trim_min
    peaks, _properties = scipy.signal.find_peaks(
        -values,
        distance=20,
        prominence=(0.50 * trim_range),
    )
    return np.asarray(peaks, dtype=np.int64)


def beat_clip_starts(
    systole: np.ndarray,
    frame_count: int,
    frames: int = 32,
    period: int = 2,
) -> list[int]:
    """Map each systole frame to the ejection-fraction clip that covers it.

    ``scripts/beat_by_beat_analysis.R`` subtracts 32 from the systole frame
    and clamps a negative result to 0. A start that the official all-clips
    sampler cannot produce is omitted, because that script inner-joins the
    two tables.
    """

    if frames < 1 or period < 1:
        raise ValueError("frames and period must be positive.")
    needed = frames * period
    available = frame_count if frame_count >= needed else needed
    last_start = available - (frames - 1) * period - 1
    starts = []
    for frame in np.asarray(systole).tolist():
        start = int(frame) - frames
        if start < 0:
            start = 0
        if start <= last_start:
            starts.append(start)
    return starts
