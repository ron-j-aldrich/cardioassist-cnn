"""Compare this wrapper's clip with ``echonet.datasets.Echo`` on a synthetic file.

The mean and standard deviation below are inputs for that comparison.
They are not EchoNet-Dynamic training statistics, they are not saved, and
the checkpoint is not run.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import echonet
import numpy as np
import torch

from cardioassist_cnn.preprocessing import (
    load_video_rgb,
    root_with_uppercase_splits,
    tensor_from_decoded_video,
    write_training_normalization,
)

# Stand-in channel statistics. Shape (3,) matches get_mean_and_std's return.
FIXTURE_MEAN = np.array([11.0, 22.0, 33.0], dtype=np.float32)
FIXTURE_STD = np.array([2.0, 4.0, 5.0], dtype=np.float32)
FIXTURE_LABEL = 17.25


def _official_start(frame_count: int, seed: int = 0) -> int:
    """Reproduce the one ``np.random.choice`` call inside ``Echo.__getitem__``."""

    available = frame_count
    if available < 32 * 2:
        available = 32 * 2
    np.random.seed(seed)
    return int(np.random.choice(available - 31 * 2, 1)[0])


def _write_fixture(
    root: Path,
    frames: int,
    split: str,
    height: int = 112,
    width: int = 112,
) -> None:
    video = np.zeros((3, frames, height, width), dtype=np.uint8)
    for index in range(frames):
        video[0, index] = index % 200
        video[1, index] = 30
        video[2, index] = 90
        video[0, index, index % height, :] = 255
    videos = root / "Videos"
    videos.mkdir()
    echonet.utils.savevideo(str(videos / "clip.avi"), video, fps=10)
    (root / "FileList.csv").write_text(
        "FileName,EF,Split\n"
        f"clip,{FIXTURE_LABEL},{split}\n",
        encoding="utf-8",
    )
    (root / "VolumeTracings.csv").write_text(
        "FileName,X1,Y1,X2,Y2,Frame\n"
        "clip.avi,1,1,2,2,0\n"
        "clip.avi,1,1,2,2,1\n",
        encoding="utf-8",
    )


class OfficialClipMatchTest(unittest.TestCase):
    def _assert_matches_official(self, frames: int) -> None:
        random_state = np.random.get_state()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _write_fixture(root, frames=frames, split="TRAIN")
                with root_with_uppercase_splits(root) as prepared:
                    self.assertEqual(prepared, root)
                decoded = load_video_rgb(root / "Videos" / "clip.avi")
                dataset = echonet.datasets.Echo(
                    root=str(root),
                    split="train",
                    target_type="EF",
                    mean=FIXTURE_MEAN,
                    std=FIXTURE_STD,
                    length=32,
                    period=2,
                    clips=1,
                )
                self.assertEqual(len(dataset), 1)
                np.random.seed(0)
                official, label = dataset[0]
                start = _official_start(decoded.shape[1], seed=0)
                clip = tensor_from_decoded_video(
                    decoded,
                    FIXTURE_MEAN,
                    FIXTURE_STD,
                    frames=32,
                    period=2,
                    clip_start=start,
                    video_path=root / "Videos" / "clip.avi",
                )
                self.assertEqual(float(label), FIXTURE_LABEL)
                self.assertEqual(tuple(clip.shape), (1, 3, 32, 112, 112))
                self.assertEqual(clip.dtype, torch.float32)
                self.assertEqual(official.shape, (3, 32, 112, 112))
                self.assertEqual(official.dtype, np.float32)
                np.testing.assert_array_equal(clip[0].numpy(), official)
                self.assertFalse((root / "normalization.json").exists())
        finally:
            np.random.set_state(random_state)

    def test_long_clip_matches_official_loader(self) -> None:
        self._assert_matches_official(frames=70)

    def test_short_clip_pads_like_official_loader(self) -> None:
        self._assert_matches_official(frames=20)

    def test_lowercase_split_is_visible_through_the_temporary_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_fixture(root, frames=2, split="train", height=16, width=16)
            untouched = echonet.datasets.Echo(root=str(root), split="train")
            self.assertEqual(len(untouched), 0)
            with root_with_uppercase_splits(root) as prepared:
                self.assertNotEqual(prepared, root)
                visible = echonet.datasets.Echo(root=str(prepared), split="train")
                self.assertEqual(visible.fnames, ["clip.avi"])
            self.assertFalse((root / "normalization.json").exists())

    def test_missing_split_column_stops_before_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Videos").mkdir()
            (root / "FileList.csv").write_text("FileName,EF\nclip,17.25\n", encoding="utf-8")
            (root / "VolumeTracings.csv").write_text(
                "FileName,X1,Y1,X2,Y2,Frame\n",
                encoding="utf-8",
            )
            destination = root / "normalization.json"
            with self.assertRaises(ValueError):
                write_training_normalization(root, destination)
            self.assertFalse(destination.exists())
