"""Arithmetic checks for the metric functions. These are not dataset scores."""

from __future__ import annotations

import math
import unittest

from cardioassist_cnn.evaluation import (
    coefficient_of_determination,
    mean_absolute_error,
    pearson_correlation,
    root_mean_squared_error,
)


class EvaluationMetricTest(unittest.TestCase):
    def test_known_pairs(self) -> None:
        labels = [0.0, 10.0]
        predictions = [0.0, 0.0]
        self.assertEqual(mean_absolute_error(labels, predictions), 5.0)
        self.assertEqual(root_mean_squared_error(labels, predictions), math.sqrt(50.0))
        self.assertEqual(coefficient_of_determination(labels, predictions), -1.0)

        perfect = [1.0, 2.0, 3.0]
        self.assertEqual(mean_absolute_error(perfect, perfect), 0.0)
        self.assertEqual(coefficient_of_determination(perfect, perfect), 1.0)
        self.assertEqual(pearson_correlation(perfect, perfect), 1.0)


if __name__ == "__main__":
    unittest.main()
