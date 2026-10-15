"""HTTP service so another computer can request one ejection-fraction prediction.

ROS 2 on the robotics machine cannot see a process on this Mac by itself.
This service is the explicit connection. The caller sends the name of a
prerecorded video that already sits in the video directory. Gazebo probe
motion does not create the image. The response is one clip from frame 0,
which is the fast demo path. The beat-averaged test scores are a separate
measurement.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from cardioassist_cnn.inference import EFPredictor, InferenceResult

VIDEO_FIELDS = (
    "status",
    "ejection_fraction_percent",
    "forward_seconds",
    "device",
    "model_id",
    "message",
)


def resolve_video(video_dir: Path, name: str) -> Path:
    """Resolve ``name`` inside ``video_dir`` and reject paths that leave it."""

    if not isinstance(name, str) or name == "" or name.strip() != name:
        raise ValueError("The video name must be a file inside the video directory.")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("The video name must be a file inside the video directory.")
    root = video_dir.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("The video name must be a file inside the video directory.")
    if candidate == root or not candidate.is_file():
        raise FileNotFoundError(f"Video not found: {candidate}")
    return candidate


def result_payload(result: InferenceResult) -> dict:
    """The fields the robotics computer is meant to read."""

    return {field: getattr(result, field) for field in VIDEO_FIELDS}


class PredictionServer:
    """One loaded model, one request at a time."""

    def __init__(self, predictor: EFPredictor, video_dir: Path) -> None:
        self.predictor = predictor
        self.video_dir = Path(video_dir)
        self._lock = threading.Lock()

    def predict_named(self, name: str) -> tuple[int, dict]:
        try:
            path = resolve_video(self.video_dir, name)
        except ValueError as exc:
            return 400, {
                "status": "rejected_video",
                "ejection_fraction_percent": None,
                "forward_seconds": None,
                "device": self.predictor.loaded.device.type,
                "model_id": self.predictor.loaded.model_id,
                "message": str(exc),
            }
        except FileNotFoundError as exc:
            return 404, {
                "status": "missing_video",
                "ejection_fraction_percent": None,
                "forward_seconds": None,
                "device": self.predictor.loaded.device.type,
                "model_id": self.predictor.loaded.model_id,
                "message": str(exc),
            }
        with self._lock:
            result = self.predictor.predict(path)
        code = 200 if result.status == "ok" else 422
        return code, result_payload(result)


def make_handler(server: PredictionServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, code: int, payload: dict) -> None:
            body = (json.dumps(payload) + "\n").encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/health":
                self._send(404, {"status": "not_found", "message": "Use GET /health or POST /predict."})
                return
            self._send(
                200,
                {
                    "status": "ready",
                    "model_id": server.predictor.loaded.model_id,
                    "device": server.predictor.loaded.device.type,
                    "video_dir": str(server.video_dir),
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path != "/predict":
                self._send(404, {"status": "not_found", "message": "POST a video name to /predict."})
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw.decode("utf-8"))
                name = body["video"]
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
                self._send(
                    400,
                    {
                        "status": "bad_request",
                        "message": 'Send JSON {"video": "clip.avi"} for a file in the video directory.',
                    },
                )
                return
            code, payload = server.predict_named(name)
            self._send(code, payload)

    return Handler


def serve(
    predictor: EFPredictor,
    video_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """Listen until the process is stopped. The model stays loaded."""

    prediction_server = PredictionServer(predictor, video_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(prediction_server))
    print(f"Listening on http://{host}:{port}", flush=True)
    print(f"Video directory: {Path(video_dir).resolve()}", flush=True)
    print("GET /health    POST /predict  {\"video\": \"clip.avi\"}", flush=True)
    httpd.serve_forever()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Serve one frame-0 ejection-fraction prediction to another computer. "
            "The video must already be in the video directory."
        )
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--video-dir", default="videos")
    parser.add_argument("--checkpoint", default="models/r2plus1d_18_32_2_pretrained.pt")
    parser.add_argument("--normalization", default="models/normalization.json")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args(argv)
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    predictor = EFPredictor(
        checkpoint_path=args.checkpoint,
        normalization_path=args.normalization,
        device=args.device,
        clip_start=0,
    )
    serve(predictor, video_dir, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
