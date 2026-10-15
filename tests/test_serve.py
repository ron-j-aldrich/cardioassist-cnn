"""The robotics computer may only request a video inside the demo directory."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from cardioassist_cnn.inference import InferenceResult
from cardioassist_cnn.quality import LoadedQuality
from cardioassist_cnn.serve import PredictionServer, acquire_view, make_handler, resolve_video


class _Loaded:
    device = type("Device", (), {"type": "cpu"})()
    model_id = "test-model"


class _Predictor:
    def __init__(self) -> None:
        self.loaded = _Loaded()
        self.calls: list[Path] = []
        self.decoded_calls = 0
        self.mean = np.array([32.0, 32.0, 32.0], dtype=np.float32)
        self.std = np.array([50.0, 50.0, 50.0], dtype=np.float32)

    def predict(self, video_path: Path) -> InferenceResult:
        self.calls.append(Path(video_path))
        return self._result()

    def predict_decoded(self, decoded: np.ndarray, video_path: Path | None = None) -> InferenceResult:
        self.decoded_calls += 1
        return self._result()

    def _result(self) -> InferenceResult:
        return InferenceResult(
            status="ok",
            ejection_fraction_percent=61.54,
            forward_seconds=0.2,
            device="cpu",
            model_id="test-model",
            input_shape=(1, 3, 32, 112, 112),
            message="One frame-0 clip.",
        )


class _ConstantSeverity(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return torch.full((video.shape[0],), self.value)


def _quality(value: float) -> LoadedQuality:
    return LoadedQuality(
        module=_ConstantSeverity(value),
        device=torch.device("cpu"),
        model_id="test-quality",
        frames=8,
        threshold=0.25,
        checkpoint_path=Path("unused.pt"),
    )


class ServeContractTest(unittest.TestCase):
    def test_video_names_stay_inside_the_directory(self) -> None:
        root = Path("/tmp/cardioassist-demo-videos")
        with self.assertRaises(ValueError):
            resolve_video(root, "../secrets.avi")
        with self.assertRaises(ValueError):
            resolve_video(root, "/etc/passwd")

    def test_post_returns_the_robotics_fields(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "clip.avi"
            video.write_bytes(b"not-a-real-video")
            predictor = _Predictor()
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(PredictionServer(predictor, root)))
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            port = httpd.server_address[1]
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
                    health = json.load(response)
                self.assertEqual(health["status"], "ready")
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/predict",
                    data=json.dumps({"video": "clip.avi"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request) as response:
                    status = response.status
                    payload = json.load(response)
                self.assertEqual(status, 200)
                self.assertEqual(payload["ejection_fraction_percent"], 61.54)
                self.assertEqual(payload["model_id"], "test-model")
                self.assertEqual(predictor.calls, [video.resolve()])
                escaped = urllib.request.Request(
                    f"http://127.0.0.1:{port}/predict",
                    data=json.dumps({"video": "../clip.avi"}).encode("utf-8"),
                    method="POST",
                )
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(escaped)
                self.assertEqual(caught.exception.code, 400)
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_acquire_returns_ejection_fraction_only_after_the_view_passes(self) -> None:
        video = np.full((3, 8, 32, 32), 140, dtype=np.uint8)
        predictor = _Predictor()
        rejected = acquire_view(predictor, _quality(0.9), video, (1.0, 1.0, 1.0))
        self.assertEqual(rejected["status"], "rejected")
        self.assertFalse(rejected["passes"])
        self.assertIsNone(rejected["ejection_fraction_percent"])
        self.assertNotIn("severity_label", rejected)
        self.assertEqual(predictor.decoded_calls, 0)

        accepted = acquire_view(predictor, _quality(0.1), video, (0.0, 0.0, 0.0))
        self.assertEqual(accepted["status"], "accepted")
        self.assertTrue(accepted["passes"])
        self.assertEqual(accepted["ejection_fraction_percent"], 61.54)
        self.assertEqual(predictor.decoded_calls, 1)
        self.assertEqual(accepted["pose"], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
