"""Pose (0, 0, 0) must not change a video. Other poses apply a fixed corruption."""

from __future__ import annotations

import unittest

import numpy as np

from cardioassist_cnn.degradation import (
    degrade_video,
    dropout_indices,
    severity_label,
    settings_for_pose,
)


def _pattern(frames: int = 4, height: int = 112, width: int = 112) -> np.ndarray:
    generator = np.random.default_rng(0)
    return generator.integers(0, 256, size=(3, frames, height, width), dtype=np.uint8)


class DegradationTest(unittest.TestCase):
    def test_neutral_pose_is_unchanged(self) -> None:
        video = _pattern()
        degraded, settings = degrade_video(video, (0.0, 0.0, 0.0))
        self.assertTrue(np.array_equal(degraded, video))
        self.assertEqual(settings.severity_label, 0.0)
        self.assertEqual(settings.dropped_frames, ())
        self.assertEqual(degraded.dtype, np.uint8)
        self.assertEqual(degraded.shape, video.shape)

    def test_severity_is_the_root_mean_square_of_the_pose(self) -> None:
        self.assertEqual(severity_label((0, 0, 0)), 0.0)
        self.assertAlmostEqual(severity_label((1, 0, 0)), 1.0 / np.sqrt(3.0))
        self.assertAlmostEqual(severity_label((1, 1, 1)), 1.0)
        self.assertAlmostEqual(severity_label((2, 0, 0)), severity_label((1, 0, 0)))

    def test_positive_shift_moves_a_bright_block_right_and_occludes_that_edge(self) -> None:
        video = np.zeros((3, 1, 40, 40), dtype=np.uint8)
        video[:, 0, 10:20, 5:15] = 255
        degraded, settings = degrade_video(video, (1.0, 0.0, 0.0))
        self.assertEqual(settings.shift_x_pixels, 8)
        self.assertAlmostEqual(settings.occlusion_fraction, 0.3)
        right_edge = float(degraded[0, 0, :, -4:].mean())
        center = float(degraded[0, 0, :, 12:24].mean())
        self.assertLess(right_edge, center)
        original_column = int(np.argmax(video[0, 0].sum(axis=0)))
        degraded_column = int(np.argmax(degraded[0, 0].sum(axis=0)))
        self.assertGreater(degraded_column, original_column)

    def test_angle_adds_black_bars_above_and_below(self) -> None:
        video = np.full((3, 1, 40, 40), 255, dtype=np.uint8)
        degraded, settings = degrade_video(video, (0.0, 0.0, 1.0))
        self.assertAlmostEqual(settings.vertical_scale, 0.6)
        middle = float(degraded[0, 0, 20, 20])
        self.assertGreater(middle, float(degraded[0, 0, 0, 20]))
        self.assertGreater(middle, float(degraded[0, 0, -1, 20]))

    def test_full_pose_drops_frames_and_repeats_exactly(self) -> None:
        video = np.full((3, 20, 32, 32), 180, dtype=np.uint8)
        first, first_settings = degrade_video(video, (1.0, 1.0, 1.0))
        second, _ = degrade_video(video, (1.0, 1.0, 1.0))
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_settings.severity_label, 1.0)
        self.assertEqual(len(first_settings.dropped_frames), 5)
        for index in first_settings.dropped_frames:
            self.assertEqual(int(first[:, index].max()), 0)
        kept = next(index for index in range(20) if index not in first_settings.dropped_frames)
        self.assertGreater(int(first[:, kept].max()), 0)

    def test_dropout_count_follows_severity(self) -> None:
        self.assertEqual(dropout_indices(20, 0.0), ())
        self.assertEqual(len(dropout_indices(20, 1.0)), 5)

    def test_pose_must_be_three_finite_numbers(self) -> None:
        video = _pattern(frames=1, height=8, width=8)
        with self.assertRaises(ValueError):
            degrade_video(video, (0.0, 0.0))
        with self.assertRaises(ValueError):
            settings_for_pose((np.nan, 0.0, 0.0), frame_count=1, height=8, width=8)


if __name__ == "__main__":
    unittest.main()
