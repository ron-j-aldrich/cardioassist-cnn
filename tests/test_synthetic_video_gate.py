"""Checks that can run before the EchoNet-Dynamic dataset exists.

The video written here is a solid color pattern. It is not an
echocardiogram, and these checks must not print an ejection fraction.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import echonet
import numpy as np

from cardioassist_cnn.inference import main
from cardioassist_cnn.preprocessing import (
    load_video_rgb,
    read_normalization,
    resize_to_model_input,
    sample_clip,
    write_training_normalization,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "r2plus1d_18_32_2_pretrained.pt"


def _solid_rgb(frames: int, height: int, width: int) -> np.ndarray:
    """RGB frames with red well above green and blue, shape (3, frames, height, width)."""

    video = np.zeros((3, frames, height, width), dtype=np.uint8)
    video[0] = 220
    video[1] = 10
    video[2] = 20
    return video


class SyntheticVideoGateTest(unittest.TestCase):
    def test_official_loader_decodes_a_synthetic_file(self) -> None:
        original = _solid_rgb(frames=6, height=48, width=64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.avi"
            echonet.utils.savevideo(str(path), original, fps=10)
            decoded = load_video_rgb(path)

        self.assertEqual(decoded.dtype, np.uint8)
        self.assertEqual(decoded.shape, (3, 6, 48, 64))
        self.assertGreater(float(decoded[0].mean()), 150.0)
        self.assertLess(float(decoded[1].mean()), 40.0)
        self.assertLess(float(decoded[2].mean()), 50.0)

    def test_resize_uses_cubic_only_when_the_frame_is_not_112(self) -> None:
        original = _solid_rgb(frames=2, height=48, width=64)
        resized = resize_to_model_input(original)
        self.assertEqual(resized.shape, (3, 2, 112, 112))
        self.assertTrue(np.all(resized[0] == 220))
        self.assertTrue(np.all(resized[1] == 10))
        self.assertTrue(np.all(resized[2] == 20))

        already = _solid_rgb(frames=2, height=112, width=112)
        self.assertIs(resize_to_model_input(already), already)

    def test_clip_sampling_follows_period_and_pads_after_the_array(self) -> None:
        video = np.arange(70, dtype=np.float32).reshape(1, 70, 1, 1)
        video = np.broadcast_to(video, (3, 70, 112, 112)).copy()
        clip = sample_clip(video, frames=32, period=2, clip_start=0)
        self.assertEqual(clip.shape, (3, 32, 112, 112))
        sampled_indices = clip[0, :, 0, 0]
        self.assertTrue(np.array_equal(sampled_indices, np.arange(0, 64, 2)))

        short = np.ones((3, 4, 112, 112), dtype=np.float32)
        padded = sample_clip(short, frames=32, period=2, clip_start=0)
        self.assertEqual(padded.shape, (3, 32, 112, 112))
        self.assertEqual(float(padded[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(padded[0, 1, 0, 0]), 1.0)
        self.assertEqual(float(padded[0, 2, 0, 0]), 0.0)

    def test_normalization_file_must_name_the_training_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalization.json"
            path.write_text(
                json.dumps(
                    {
                        "role": "hand_written",
                        "pixel_range": "0-255",
                        "mean": [0.0],
                        "std": [1.0],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_normalization(path)

            path.write_text(
                json.dumps(
                    {
                        "role": "echonet_dynamic_training_split",
                        "pixel_range": "0-1",
                        "mean": [0.0],
                        "std": [1.0],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                read_normalization(path)

    def test_training_statistics_stop_when_dataset_files_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError) as empty:
                write_training_normalization(root, root / "normalization.json")
            message = str(empty.exception)
            self.assertIn("FileList.csv", message)
            self.assertIn("VolumeTracings.csv", message)
            self.assertIn("Videos/", message)

            (root / "FileList.csv").write_text("FileName,Split\n", encoding="utf-8")
            (root / "Videos").mkdir()
            with self.assertRaises(FileNotFoundError) as partial:
                write_training_normalization(root, root / "normalization.json")
            self.assertIn("VolumeTracings.csv", str(partial.exception))
            self.assertFalse((root / "normalization.json").exists())

    def test_command_decodes_a_file_and_does_not_print_an_ef(self) -> None:
        self.assertTrue(CHECKPOINT.is_file(), f"Checkpoint missing: {CHECKPOINT}")
        original = _solid_rgb(frames=6, height=48, width=64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.avi"
            echonet.utils.savevideo(str(path), original, fps=10)
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "--video",
                        str(path),
                        "--checkpoint",
                        str(CHECKPOINT),
                        "--device",
                        "cpu",
                    ]
                )
        text = stdout.getvalue()
        self.assertEqual(code, 2)
        self.assertIn("Status: missing_normalization", text)
        self.assertIn("Decoded 6 frames at 64x48", text)
        self.assertNotIn("Predicted EF", text)
        self.assertNotIn("%", text)


if __name__ == "__main__":
    unittest.main()
