"""Turn one echocardiography video into the tensor R(2+1)D-18 expects.

The steps follow ``echonet.utils.loadvideo`` and ``echonet.datasets.Echo``.
Dataset videos are already 112 by 112. A video of another size is resized
with cubic interpolation because that is how the official dataset was
created, before the model ever sees it.
"""

from __future__ import annotations

import csv
import json
import logging
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

import echonet

LOGGER = logging.getLogger(__name__)

SPATIAL_SIZE = 112
TRAINING_NORMALIZATION_ROLE = "echonet_dynamic_training_split"


def load_video_rgb(path: str | Path) -> np.ndarray:
    """Read a video as uint8 RGB with shape ``(channels=3, frames, height, width)``.

    OpenCV decodes BGR. The official loader converts each frame to RGB, and
    this function uses that loader.
    """

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    video = echonet.utils.loadvideo(str(video_path))
    if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] == 0:
        raise ValueError(
            f"Video decoded to shape {tuple(video.shape)}. "
            "Expected (channels=3, frames, height, width) with at least one frame."
        )
    if video.shape[2] == 0 or video.shape[3] == 0:
        raise ValueError(f"Video has an empty frame size: {tuple(video.shape)}")
    return video


def resize_to_model_input(video: np.ndarray) -> np.ndarray:
    """Resize frames to 112 by 112 when the file is not already that size.

    ``video`` is uint8 ``(3, frames, height, width)``. Cubic interpolation
    matches the dataset description on https://echonet.github.io/dynamic/.
    """

    channels, frame_count, height, width = video.shape
    if height == SPATIAL_SIZE and width == SPATIAL_SIZE:
        LOGGER.debug("Frames are already %dx%d; no resize applied.", SPATIAL_SIZE, SPATIAL_SIZE)
        return video

    LOGGER.info(
        "Resizing %d frames from %dx%d to %dx%d with cubic interpolation.",
        frame_count,
        width,
        height,
        SPATIAL_SIZE,
        SPATIAL_SIZE,
    )
    resized = np.empty((frame_count, SPATIAL_SIZE, SPATIAL_SIZE, channels), dtype=np.uint8)
    frames_last = np.transpose(video, (1, 2, 3, 0))
    for index, frame in enumerate(frames_last):
        resized[index] = cv2.resize(
            frame,
            (SPATIAL_SIZE, SPATIAL_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )
    return np.transpose(resized, (3, 0, 1, 2))


def normalize_video(video: np.ndarray, mean: np.ndarray | float, std: np.ndarray | float) -> np.ndarray:
    """Subtract the training mean and divide by the training standard deviation.

    Pixels stay on the 0–255 scale used by ``loadvideo``. They are not divided
    by 255. ``mean`` and ``std`` must come from the EchoNet-Dynamic training
    split. Validation or test statistics would leak evaluation data into the input.
    """

    values = video.astype(np.float32)
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    if np.any(std_array == 0):
        raise ValueError("Normalization standard deviation contains 0.")
    if mean_array.ndim == 0:
        values -= float(mean_array)
    elif tuple(mean_array.shape) == (3,):
        values -= mean_array.reshape(3, 1, 1, 1)
    else:
        raise ValueError(f"Mean must be a scalar or shape (3,), got {tuple(mean_array.shape)}.")
    if std_array.ndim == 0:
        values /= float(std_array)
    elif tuple(std_array.shape) == (3,):
        values /= std_array.reshape(3, 1, 1, 1)
    else:
        raise ValueError(f"Std must be a scalar or shape (3,), got {tuple(std_array.shape)}.")
    return values


def sample_clip(video: np.ndarray, frames: int, period: int, clip_start: int) -> np.ndarray:
    """Take ``frames`` frames, keeping every ``period``-th frame.

    With the official EF settings, ``frames`` is 32 and ``period`` is 2, so
    the clip spans 64 original frames. Index 0 is the first decoded frame.
    A short video is padded with zeros after normalization, which the official
    dataset class treats as the mean color.
    """

    if frames < 1 or period < 1:
        raise ValueError("frames and period must be positive.")
    channels, frame_count, height, width = video.shape
    needed = frames * period
    if frame_count < needed:
        pad_count = needed - frame_count
        LOGGER.info("Padding %d frames so a %d-frame clip can be sampled.", pad_count, frames)
        pad = np.zeros((channels, pad_count, height, width), dtype=video.dtype)
        video = np.concatenate((video, pad), axis=1)
        frame_count = needed
    last_start = frame_count - (frames - 1) * period - 1
    if clip_start < 0 or clip_start > last_start:
        raise ValueError(
            f"clip_start={clip_start} is outside 0..{last_start} for {frame_count} frames."
        )
    indices = clip_start + period * np.arange(frames)
    LOGGER.debug(
        "Clip indices start=%d period=%d count=%d first=%d last=%d",
        clip_start,
        period,
        frames,
        int(indices[0]),
        int(indices[-1]),
    )
    return video[:, indices, :, :]


def tensor_from_decoded_video(
    raw: np.ndarray,
    mean: np.ndarray | float,
    std: np.ndarray | float,
    frames: int = 32,
    period: int = 2,
    clip_start: int = 0,
    video_path: str | Path | None = None,
) -> torch.Tensor:
    """Normalize and sample one clip from an already decoded video.

    The returned shape is ``(1, 3, frames, 112, 112)``. After the batch axis,
    the axes are color channel, time, height, and width.
    """

    channels, frame_count, height, width = raw.shape
    LOGGER.info(
        "Decoded %s: channels=%d frames=%d height=%d width=%d dtype=%s",
        video_path,
        channels,
        frame_count,
        height,
        width,
        raw.dtype,
    )
    resized = resize_to_model_input(raw)
    normalized = normalize_video(resized, mean, std)
    clip = sample_clip(normalized, frames, period, clip_start)
    tensor = torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0)
    LOGGER.debug(
        "Model input shape (batch, channel, frames, height, width)=%s dtype=%s",
        tuple(tensor.shape),
        tensor.dtype,
    )
    return tensor


def preprocess_video(
    video_path: str | Path,
    mean: np.ndarray | float,
    std: np.ndarray | float,
    frames: int = 32,
    period: int = 2,
    clip_start: int = 0,
) -> torch.Tensor:
    """Decode one video and return a float32 clip of shape ``(1, 3, frames, 112, 112)``."""

    raw = load_video_rgb(video_path)
    return tensor_from_decoded_video(
        raw,
        mean,
        std,
        frames=frames,
        period=period,
        clip_start=clip_start,
        video_path=video_path,
    )


def read_normalization(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read mean and standard deviation saved from the training split.

    The file must declare ``role`` as ``echonet_dynamic_training_split``.
    A hand-written or test-set statistic is rejected so it cannot be used as
    if it were the pretrained model's preprocessing.
    """

    file_path = Path(path)
    with file_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    role = payload.get("role")
    if role != TRAINING_NORMALIZATION_ROLE:
        raise ValueError(
            f"{file_path} has role {role!r}. Preprocessing requires role "
            f"{TRAINING_NORMALIZATION_ROLE!r}, computed from the training split."
        )
    if payload.get("pixel_range") != "0-255":
        raise ValueError("Normalization pixel_range must be '0-255', matching loadvideo.")
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    return mean, std, payload


@contextmanager
def root_with_uppercase_splits(data_dir: str | Path) -> Iterator[Path]:
    """Yield a dataset root whose ``Split`` values are uppercase.

    ``echonet.datasets.Echo`` compares the ``Split`` column with
    ``split.upper()``. The line that would uppercase the column does not
    save its result, so lowercase names such as ``train`` produce an empty
    dataset. This yields the original directory when every value is already
    uppercase. Otherwise it yields a temporary directory with an uppercased
    ``FileList.csv`` and symlinks to ``Videos/`` and ``VolumeTracings.csv``.
    The pinned source in ``third_party/dynamic`` is left unchanged.
    """

    root = Path(data_dir)
    file_list = root / "FileList.csv"
    with file_list.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None or "Split" not in fieldnames:
            raise ValueError(
                f"{file_list} must contain a Split column. "
                "The official loader filters rows with that column."
            )
        rows = list(reader)
    if any(not row["Split"] for row in rows):
        raise ValueError(f"{file_list} contains an empty Split value.")
    if all(row["Split"] == row["Split"].upper() for row in rows):
        yield root
        return

    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        destination = temporary_root / "FileList.csv"
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                row["Split"] = row["Split"].upper()
                writer.writerow(row)
        (temporary_root / "Videos").symlink_to(root / "Videos", target_is_directory=True)
        (temporary_root / "VolumeTracings.csv").symlink_to(root / "VolumeTracings.csv")
        yield temporary_root


def write_training_normalization(
    data_dir: str | Path,
    destination: str | Path,
    seed: int = 0,
    samples: int = 128,
) -> Path:
    """Compute and save the official training-split mean and standard deviation.

    This calls ``echonet.utils.get_mean_and_std`` on ``split='train'`` after
    setting NumPy's seed, which is the order used by ``echonet video``.
    The official dataset class reads ``FileList.csv``, ``VolumeTracings.csv``,
    and ``Videos/`` before it returns a clip. Lowercase ``Split`` values are
    uppercased in a temporary view so the official comparison can see them.
    This function does not train the network, and it does not substitute
    numbers when those files are absent.
    """

    import os

    root = Path(data_dir)
    required = {
        "FileList.csv": (root / "FileList.csv").is_file(),
        "VolumeTracings.csv": (root / "VolumeTracings.csv").is_file(),
        "Videos/": (root / "Videos").is_dir(),
    }
    missing = [name for name, present in required.items() if not present]
    if missing:
        raise FileNotFoundError(
            f"{root} is missing {', '.join(missing)}. "
            "Training-split statistics need FileList.csv, VolumeTracings.csv, and Videos/. "
            "The official loader opens the tracing file while it reads the training split."
        )
    with root_with_uppercase_splits(root) as loader_root:
        np.random.seed(seed)
        dataset = echonet.datasets.Echo(root=str(loader_root), split="train")
        if len(dataset) == 0:
            raise ValueError(f"No training videos were found in {root}.")
        mean, std = echonet.utils.get_mean_and_std(dataset, samples=samples, num_workers=0)
    payload = {
        "role": TRAINING_NORMALIZATION_ROLE,
        "pixel_range": "0-255",
        "mean": [float(value) for value in np.atleast_1d(mean)],
        "std": [float(value) for value in np.atleast_1d(std)],
        "samples": samples,
        "seed": seed,
        "source": (
            "echonet.utils.get_mean_and_std on Echo(split='train'), "
            f"seed {seed}, {samples} videos"
        ),
    }
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the write atomic enough for a small JSON file.
    temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination_path)
    return destination_path
