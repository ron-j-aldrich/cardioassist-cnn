"""Corrupt one echocardiography video from a simulated 3-DOF probe pose.

The pose is ``(tx, ty, angle)``. Each value is clipped to [-1, 1]. The
neutral pose ``(0, 0, 0)`` leaves the video unchanged. Any other pose applies
the same effects, in this order:

1. Foreshortening, from ``angle``: the frame is compressed vertically and
   padded with black.
2. Off-axis translation, from ``tx`` and ``ty``: the frame shifts and the
   vacated edge is black.
3. Occlusion, from ``tx``: a black band on the right when ``tx`` is positive,
   and on the left when ``tx`` is negative.
4. Blur, from the pose severity.
5. Speckle, from the pose severity. The noise is fixed for a given pose, so
   the same pose always produces the same video.
6. Frame dropout, from the pose severity: selected frames become black.

``severity_label`` is the training target for a later quality model. It is
the root-mean-square of the three pose components, so it is 0 at the neutral
pose and 1 when every component is at the limit. It is not a prediction, and
these effects are not a physical ultrasound simulation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import cv2
import numpy as np

POSE_LIMIT = 1.0
MAX_SHIFT_FRACTION = 0.2
MAX_FORESHORTENING = 0.4
MAX_OCCLUSION_FRACTION = 0.3
MAX_BLUR_SIGMA = 2.0
MAX_SPECKLE_AMPLITUDE = 0.35
MAX_DROPOUT_FRACTION = 0.25


@dataclass(frozen=True)
class DegradationSettings:
    """The exact effects applied for one pose. This is the acquisition record."""

    tx: float
    ty: float
    angle: float
    severity_label: float
    shift_x_pixels: int
    shift_y_pixels: int
    vertical_scale: float
    occlusion_fraction: float
    blur_sigma: float
    speckle_amplitude: float
    dropped_frames: tuple[int, ...]

    def as_dict(self) -> dict:
        values = asdict(self)
        values["dropped_frames"] = list(self.dropped_frames)
        return values


def clip_pose(pose: Sequence[float]) -> np.ndarray:
    """Return three finite pose components clipped to [-1, 1]."""

    values = np.asarray(pose, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("Pose must be three finite numbers: tx, ty, and angle.")
    return np.clip(values, -POSE_LIMIT, POSE_LIMIT)


def severity_label(pose: Sequence[float]) -> float:
    """Training target in [0, 1] for this pose. Neutral is 0."""

    tx, ty, angle = clip_pose(pose)
    return float(np.sqrt((tx * tx + ty * ty + angle * angle) / 3.0))


def settings_for_pose(pose: Sequence[float], frame_count: int, height: int, width: int) -> DegradationSettings:
    """Map one pose onto pixel shifts, blur, speckle, and dropped frames."""

    if frame_count < 1 or height < 2 or width < 2:
        raise ValueError("Video must have at least one frame and a 2 by 2 frame size.")
    tx, ty, angle = (float(value) for value in clip_pose(pose))
    severity = severity_label((tx, ty, angle))
    return DegradationSettings(
        tx=tx,
        ty=ty,
        angle=angle,
        severity_label=severity,
        shift_x_pixels=int(round(tx * MAX_SHIFT_FRACTION * width)),
        shift_y_pixels=int(round(ty * MAX_SHIFT_FRACTION * height)),
        vertical_scale=1.0 - MAX_FORESHORTENING * abs(angle),
        occlusion_fraction=abs(tx) * MAX_OCCLUSION_FRACTION,
        blur_sigma=MAX_BLUR_SIGMA * severity,
        speckle_amplitude=MAX_SPECKLE_AMPLITUDE * severity,
        dropped_frames=dropout_indices(frame_count, severity),
    )


def dropout_indices(frame_count: int, severity: float) -> tuple[int, ...]:
    """Evenly spaced frames to remove. Severity 0 removes none."""

    count = int(round(float(severity) * MAX_DROPOUT_FRACTION * frame_count))
    count = max(0, min(count, frame_count))
    if count == 0:
        return ()
    if count == frame_count:
        return tuple(range(frame_count))
    positions = np.linspace(0, frame_count - 1, count)
    return tuple(sorted({int(round(position)) for position in positions}))


def degrade_video(video: np.ndarray, pose: Sequence[float]) -> tuple[np.ndarray, DegradationSettings]:
    """Return a corrupted copy of ``video`` and the settings that produced it.

    ``video`` is uint8 RGB with shape ``(channels=3, frames, height, width)``.
    The result has the same shape and dtype.
    """

    if video.ndim != 4 or video.shape[0] != 3 or video.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 video of shape (3, frames, height, width), got {video.dtype} {tuple(video.shape)}."
        )
    channels, frame_count, height, width = video.shape
    settings = settings_for_pose(pose, frame_count, height, width)
    frames = np.transpose(video, (1, 2, 3, 0)).copy()
    generator = np.random.default_rng(_speckle_seed(settings))
    for index in range(frame_count):
        frames[index] = _degrade_frame(frames[index], settings, generator)
    for index in settings.dropped_frames:
        frames[index] = 0
    degraded = np.ascontiguousarray(np.transpose(frames, (3, 0, 1, 2)))
    if channels != 3:
        raise ValueError("Video channel count changed during degradation.")
    return degraded, settings


def _speckle_seed(settings: DegradationSettings) -> int:
    payload = np.array([settings.tx, settings.ty, settings.angle], dtype=np.float64).tobytes()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _degrade_frame(
    frame: np.ndarray,
    settings: DegradationSettings,
    generator: np.random.Generator,
) -> np.ndarray:
    height, width = frame.shape[:2]
    image = frame
    if settings.vertical_scale != 1.0:
        new_height = max(1, int(round(height * settings.vertical_scale)))
        scaled = cv2.resize(image, (width, new_height), interpolation=cv2.INTER_LINEAR)
        letterboxed = np.zeros_like(image)
        top = (height - new_height) // 2
        letterboxed[top : top + new_height] = scaled
        image = letterboxed
    if settings.shift_x_pixels != 0 or settings.shift_y_pixels != 0:
        matrix = np.array(
            [[1.0, 0.0, settings.shift_x_pixels], [0.0, 1.0, settings.shift_y_pixels]],
            dtype=np.float32,
        )
        image = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    band = int(round(settings.occlusion_fraction * width))
    if band > 0:
        if settings.tx >= 0:
            image = image.copy()
            image[:, width - band :] = 0
        else:
            image = image.copy()
            image[:, :band] = 0
    if settings.blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=settings.blur_sigma)
    if settings.speckle_amplitude > 0:
        noise = generator.standard_normal(image.shape).astype(np.float32)
        speckled = image.astype(np.float32) * (1.0 + settings.speckle_amplitude * noise)
        image = np.clip(speckled, 0, 255).astype(np.uint8)
    return image
