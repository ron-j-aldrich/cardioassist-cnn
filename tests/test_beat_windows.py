"""Checks for the beat window used by the official R script.

These arrays are not echocardiograms and they do not produce an ejection fraction.
"""

from __future__ import annotations

import unittest

import numpy as np

from cardioassist_cnn.segmentation import beat_clip_starts, systole_frames


class BeatWindowTest(unittest.TestCase):
    def test_long_video_subtracts_32_and_drops_a_start_past_the_end(self) -> None:
        starts = beat_clip_starts(np.array([10, 40, 200]), frame_count=180)
        self.assertEqual(starts, [0, 8])

    def test_short_video_uses_the_padded_start_range(self) -> None:
        starts = beat_clip_starts(np.array([0, 30]), frame_count=40)
        self.assertEqual(starts, [0, 0])

    def test_systole_is_a_valley_in_ventricle_size(self) -> None:
        size = np.concatenate(
            [
                np.full(40, 100.0),
                np.linspace(100.0, 20.0, 10, endpoint=False),
                np.linspace(20.0, 100.0, 10, endpoint=False),
                np.full(40, 100.0),
            ]
        )
        peaks = systole_frames(size)
        self.assertEqual(peaks.tolist(), [50])


if __name__ == "__main__":
    unittest.main()
