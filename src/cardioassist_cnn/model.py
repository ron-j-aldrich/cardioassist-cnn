"""Load the official EchoNet-Dynamic ejection-fraction checkpoint.

The architecture is torchvision's ``r2plus1d_18`` with its final layer
replaced by one linear output. That is the construction in
``echonet/utils/video.py``. The released checkpoint already contains the
trained one-output layer, so this module builds the same layer and then
loads every saved weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision

MODEL_ID = "echonet-dynamic-r2plus1d_18-v1.0.0"
EXPECTED_FRAMES = 32
EXPECTED_PERIOD = 2
EXPECTED_CHECKPOINT_BYTES = 250_572_083


@dataclass(frozen=True)
class LoadedModel:
    """A model kept in memory so repeated videos do not reload the weights."""

    module: torch.nn.Module
    device: torch.device
    model_id: str
    frames: int
    period: int
    checkpoint_path: Path


def select_device(requested: str | None = None) -> torch.device:
    """Choose MPS when this Mac can use it, and CPU otherwise.

    A caller can pass ``"cpu"`` or ``"mps"``. An unavailable MPS request falls
    back to CPU because this machine has no CUDA device.
    """

    if requested is None or requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device {requested!r}. Use 'auto', 'cpu', or 'mps'.")


def _state_dict_without_dataparallel(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Drop the ``module.`` prefix added by ``torch.nn.DataParallel``.

    The official training script wraps the model in DataParallel only when
    CUDA is available, and the released checkpoint was saved from that
    wrapper. The prefix names the wrapper. It is not part of R(2+1)D-18.
    """

    state = checkpoint["state_dict"]
    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def load_ef_model(checkpoint_path: str | Path, device: torch.device | None = None) -> LoadedModel:
    """Build R(2+1)D-18 and load the official ejection-fraction checkpoint.

    ``weights=None`` skips the Kinetics-400 download. The EchoNet checkpoint
    replaces every parameter, including the final layer.
    """

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    size = path.stat().st_size
    if size != EXPECTED_CHECKPOINT_BYTES:
        raise ValueError(
            f"Checkpoint size is {size} bytes. The v1.0.0 release asset is "
            f"{EXPECTED_CHECKPOINT_BYTES} bytes: {path}"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            "Expected the EchoNet training dictionary with a 'state_dict' entry."
        )
    frames = int(checkpoint["frames"])
    period = int(checkpoint["period"])
    if frames != EXPECTED_FRAMES or period != EXPECTED_PERIOD:
        raise ValueError(
            f"Checkpoint was trained with frames={frames}, period={period}. "
            f"This project expects frames={EXPECTED_FRAMES}, period={EXPECTED_PERIOD}."
        )

    model = torchvision.models.video.r2plus1d_18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 1)
    state = _state_dict_without_dataparallel(checkpoint)
    weight = state.get("fc.weight")
    if weight is None or tuple(weight.shape) != (1, in_features):
        raise ValueError(
            "The checkpoint output layer is not the one-value ejection-fraction head."
        )
    model.load_state_dict(state, strict=True)

    selected = device if device is not None else select_device("auto")
    model.to(selected)
    model.eval()
    return LoadedModel(
        module=model,
        device=selected,
        model_id=MODEL_ID,
        frames=frames,
        period=period,
        checkpoint_path=path,
    )
