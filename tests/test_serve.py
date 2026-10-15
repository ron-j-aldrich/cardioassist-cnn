"""The robotics computer may only request a video inside the demo directory."""

from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from cardioassist_cnn.inference import InferenceResult
from cardioassist_cnn.serve import PredictionServer, make_handler, resolve_video


class _Loaded:
    device = type("Device", (), {"type": "cpu"})()
    model_id = "test-model"


class _Predictor:
    def __init__(self) -> None:
        self.loaded = _Loaded()
        self.calls: list[Path] = []

    def predict(self, video_path: Path) -> InferenceResult:
        self.calls.append(Path(video_path))
        return InferenceResult(
            status="ok",
            ejection_fraction_percent=61.54,
            forward_seconds=0.2,
            device="cpu",
            model_id="test-model",
            input_shape=(1, 3, 32, 112, 112),
            message="One frame-0 clip.",
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


if __name__ == "__main__":
    unittest.main()
