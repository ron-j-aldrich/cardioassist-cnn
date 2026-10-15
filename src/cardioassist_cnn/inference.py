"""Run ejection-fraction inference on one echocardiography video.

The returned percentage is the model's regression output. It is a prediction
from an academic prototype, not a clinical measurement and not a measured
accuracy.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cardioassist_cnn.model import LoadedModel, load_ef_model, select_device
from cardioassist_cnn.preprocessing import (
    load_video_rgb,
    read_normalization,
    tensor_from_decoded_video,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """What a later robotics component can receive for one video."""

    status: str
    ejection_fraction_percent: float | None
    forward_seconds: float | None
    device: str
    model_id: str
    input_shape: tuple[int, ...] | None
    message: str


class EFPredictor:
    """Load the checkpoint once and run one or more videos through it."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        normalization_path: str | Path | None = None,
        device: str | None = "auto",
        clip_start: int = 0,
    ) -> None:
        selected = select_device(device)
        try:
            self.loaded = load_ef_model(checkpoint_path, selected)
        except Exception as exc:
            raise RuntimeError(f"Could not load the ejection-fraction checkpoint: {exc}") from exc
        self.clip_start = clip_start
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.normalization_source = ""
        if normalization_path is not None:
            self.mean, self.std, payload = read_normalization(normalization_path)
            self.normalization_source = str(payload.get("source", ""))

    def predict(self, video_path: str | Path) -> InferenceResult:
        """Preprocess one video and return the regression output."""

        common = {
            "device": self.loaded.device.type,
            "model_id": self.loaded.model_id,
        }
        try:
            decoded = load_video_rgb(video_path)
        except FileNotFoundError as exc:
            return InferenceResult(
                status="missing_video",
                ejection_fraction_percent=None,
                forward_seconds=None,
                input_shape=None,
                message=str(exc),
                **common,
            )
        except (ValueError, OSError) as exc:
            return InferenceResult(
                status="unreadable_video",
                ejection_fraction_percent=None,
                forward_seconds=None,
                input_shape=None,
                message=str(exc),
                **common,
            )
        return self.predict_decoded(decoded, video_path=video_path)

    def predict_decoded(self, decoded: np.ndarray, video_path: str | Path | None = None) -> InferenceResult:
        """Run one already decoded RGB video. The array is not read from disk again."""

        common = {
            "device": self.loaded.device.type,
            "model_id": self.loaded.model_id,
        }
        if self.mean is None or self.std is None:
            channels, frame_count, height, width = decoded.shape
            return InferenceResult(
                status="missing_normalization",
                ejection_fraction_percent=None,
                forward_seconds=None,
                input_shape=None,
                message=(
                    f"Decoded {frame_count} frames at {width}x{height}, channels={channels}. "
                    "Training-split mean and standard deviation are not available, "
                    "so no ejection fraction was calculated."
                ),
                **common,
            )
        try:
            tensor = tensor_from_decoded_video(
                decoded,
                self.mean,
                self.std,
                frames=self.loaded.frames,
                period=self.loaded.period,
                clip_start=self.clip_start,
                video_path=video_path,
            )
        except FileNotFoundError as exc:
            return InferenceResult(
                status="missing_video",
                ejection_fraction_percent=None,
                forward_seconds=None,
                input_shape=None,
                message=str(exc),
                **common,
            )
        except (ValueError, OSError) as exc:
            return InferenceResult(
                status="unreadable_video",
                ejection_fraction_percent=None,
                forward_seconds=None,
                input_shape=None,
                message=str(exc),
                **common,
            )

        try:
            percentage, seconds = self._run_forward(tensor)
        except RuntimeError as exc:
            return InferenceResult(
                status="forward_failed",
                ejection_fraction_percent=None,
                forward_seconds=None,
                input_shape=tuple(tensor.shape),
                message=str(exc),
                device=self.loaded.device.type,
                model_id=self.loaded.model_id,
            )
        return InferenceResult(
            status="ok",
            ejection_fraction_percent=percentage,
            forward_seconds=seconds,
            device=self.loaded.device.type,
            model_id=self.loaded.model_id,
            input_shape=tuple(tensor.shape),
            message=(
                "Regression output of the pretrained EchoNet-Dynamic model. "
                "Accuracy against ground truth has not been measured here."
            ),
        )


    def _run_forward(self, tensor: torch.Tensor) -> tuple[float, float]:
        """Run the clip, falling back to CPU if MPS fails."""

        try:
            return _forward(self.loaded, tensor)
        except RuntimeError as exc:
            if self.loaded.device.type == "cpu":
                raise RuntimeError(f"Forward pass failed on CPU: {exc}") from exc
            LOGGER.info("Forward pass failed on %s. Retrying on CPU.", self.loaded.device.type)
            self.loaded.module.to("cpu")
            self.loaded = LoadedModel(
                module=self.loaded.module,
                device=torch.device("cpu"),
                model_id=self.loaded.model_id,
                frames=self.loaded.frames,
                period=self.loaded.period,
                checkpoint_path=self.loaded.checkpoint_path,
            )
            try:
                return _forward(self.loaded, tensor)
            except RuntimeError as cpu_exc:
                raise RuntimeError(f"Forward pass failed on CPU: {cpu_exc}") from cpu_exc


def _forward(loaded: LoadedModel, tensor: torch.Tensor) -> tuple[float, float]:
    """Run one clip with gradients disabled and the model in evaluation mode."""

    loaded.module.eval()
    batch = tensor.to(loaded.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = loaded.module(batch)
        if loaded.device.type == "mps":
            torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    if tuple(output.shape) != (tensor.shape[0], 1):
        raise RuntimeError(f"Expected output shape (batch, 1), got {tuple(output.shape)}.")
    return float(output.detach().view(-1)[0].cpu()), elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict left-ventricular ejection fraction from one apical "
            "four-chamber echocardiography video."
        )
    )
    parser.add_argument("--video", required=True, help="Path to one video file.")
    parser.add_argument(
        "--checkpoint",
        default="models/r2plus1d_18_32_2_pretrained.pt",
        help="Official EchoNet-Dynamic EF checkpoint.",
    )
    parser.add_argument(
        "--normalization",
        default=None,
        help="JSON file with training-split mean and standard deviation.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--clip-start",
        type=int,
        default=0,
        help="First original frame of the clip. 0 is reproducible. The paper averages beats.",
    )
    parser.add_argument("--verbose", action="store_true", help="Log tensor dimensions.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns a process status code."""

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        predictor = EFPredictor(
            checkpoint_path=args.checkpoint,
            normalization_path=args.normalization,
            device=args.device,
            clip_start=args.clip_start,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    result = predictor.predict(args.video)
    print(f"Status: {result.status}")
    print(f"Device: {result.device}")
    print(f"Model: {result.model_id}")
    if result.input_shape is not None:
        print(
            "Input shape (batch, channel, frames, height, width): "
            f"{result.input_shape}"
        )
    if result.forward_seconds is not None:
        print(f"Forward time: {result.forward_seconds:.3f} s")
    if result.status == "ok" and result.ejection_fraction_percent is not None:
        print(f"Predicted EF: {result.ejection_fraction_percent:.2f}%")
    print(result.message)
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
