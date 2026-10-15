"""The severity model learns the pose label. It does not estimate ejection fraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cardioassist_cnn.quality import (
    ACCEPT_SEVERITY,
    QUALITY_FRAMES,
    SeverityNet,
    example_from_decoded,
    fit_examples,
    load_quality_model,
    passes_severity,
    predict_severity,
    quality_frame_indices,
    save_quality_checkpoint,
)


class QualityModelTest(unittest.TestCase):
    def test_frame_indices_include_the_ends(self) -> None:
        indices = quality_frame_indices(20, frames=8)
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 19)
        self.assertTrue(all(0 <= index < 20 for index in indices))

    def test_a_low_severity_passes_and_a_high_severity_does_not(self) -> None:
        self.assertTrue(passes_severity(0.0, ACCEPT_SEVERITY))
        self.assertTrue(passes_severity(ACCEPT_SEVERITY, ACCEPT_SEVERITY))
        self.assertFalse(passes_severity(ACCEPT_SEVERITY + 0.01, ACCEPT_SEVERITY))

    def test_neutral_pose_has_severity_zero(self) -> None:
        video = np.full((3, 12, 32, 32), 90, dtype=np.uint8)
        mean = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        std = np.array([2.0, 4.0, 5.0], dtype=np.float32)
        clip, label = example_from_decoded(video, (0.0, 0.0, 0.0), mean, std)
        self.assertEqual(label, 0.0)
        self.assertEqual(clip.shape, (3, QUALITY_FRAMES, 112, 112))
        self.assertEqual(clip.dtype, np.float32)

    def test_training_reduces_error_on_a_fixed_set(self) -> None:
        mean = np.array([32.0, 32.0, 32.0], dtype=np.float32)
        std = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        video = np.full((3, 16, 48, 64), 160, dtype=np.uint8)
        poses = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
        clips = []
        labels = []
        for pose in poses:
            clip, label = example_from_decoded(video, pose, mean, std)
            clips.append(clip)
            labels.append(label)
        model = SeverityNet()
        losses = fit_examples(model, clips, labels, steps=40, seed=0)
        self.assertLess(losses[-1], losses[0] * 0.5)
        self.assertTrue(all(0.0 < loss for loss in losses))

    def test_checkpoint_round_trip_keeps_the_threshold(self) -> None:
        model = SeverityNet()
        mean = np.zeros((3, QUALITY_FRAMES, 112, 112), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "severity.pt"
            save_quality_checkpoint(path, model, threshold=ACCEPT_SEVERITY)
            loaded = load_quality_model(path, device="cpu")
        self.assertEqual(loaded.threshold, ACCEPT_SEVERITY)
        self.assertEqual(loaded.device.type, "cpu")
        score = predict_severity(loaded, mean)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        with torch.no_grad():
            again = SeverityNet()
            again.load_state_dict(model.state_dict())
            expected = float(again(torch.from_numpy(mean).unsqueeze(0)))
        self.assertAlmostEqual(score, expected, places=5)


if __name__ == "__main__":
    unittest.main()
