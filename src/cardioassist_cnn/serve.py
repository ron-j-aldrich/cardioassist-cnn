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

import numpy as np

from cardioassist_cnn.degradation import degrade_video
from cardioassist_cnn.inference import EFPredictor, InferenceResult
from cardioassist_cnn.preprocessing import load_video_rgb
from cardioassist_cnn.quality import (
    LoadedQuality,
    passes_severity,
    predict_severity,
    quality_clip_from_video,
)

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


def _pose_from_json(value) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Pose must be three numbers: tx, ty, and angle.")
    try:
        pose = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pose must be three numbers: tx, ty, and angle.") from exc
    if not all(np.isfinite(component) for component in pose):
        raise ValueError("Pose must be three numbers: tx, ty, and angle.")
    return pose[0], pose[1], pose[2]


def acquire_view(predictor: EFPredictor, quality: LoadedQuality, decoded: np.ndarray, pose) -> dict:
    """Score one degraded video. Ejection fraction is calculated only after the view passes.

    The reply does not include ``severity_label``. That number is the training
    target, and the controller is supposed to use the predicted score.
    """

    degraded, settings = degrade_video(decoded, pose)
    if predictor.mean is None or predictor.std is None:
        return {
            "status": "missing_normalization",
            "quality_score": None,
            "passes": False,
            "threshold": quality.threshold,
            "ejection_fraction_percent": None,
            "forward_seconds": None,
            "pose": [settings.tx, settings.ty, settings.angle],
            "device": predictor.loaded.device.type,
            "model_id": predictor.loaded.model_id,
            "quality_model_id": quality.model_id,
            "message": "Training-split normalization is not available, so the view was not scored.",
        }
    clip = quality_clip_from_video(degraded, predictor.mean, predictor.std, frames=quality.frames)
    score = predict_severity(quality, clip)
    accepted = passes_severity(score, quality.threshold)
    payload = {
        "status": "accepted" if accepted else "rejected",
        "quality_score": score,
        "passes": accepted,
        "threshold": quality.threshold,
        "ejection_fraction_percent": None,
        "forward_seconds": None,
        "pose": [settings.tx, settings.ty, settings.angle],
        "device": predictor.loaded.device.type,
        "model_id": predictor.loaded.model_id,
        "quality_model_id": quality.model_id,
        "message": (
            "Predicted severity is above the cutoff, so ejection fraction was not calculated."
            if not accepted
            else "Frame-0 ejection fraction of the degraded video after it passed the severity gate."
        ),
    }
    if not accepted:
        return payload
    result = predictor.predict_decoded(degraded)
    payload["ejection_fraction_percent"] = result.ejection_fraction_percent
    payload["forward_seconds"] = result.forward_seconds
    payload["device"] = result.device
    if result.status != "ok":
        payload["status"] = result.status
        payload["message"] = result.message
    return payload


class PredictionServer:
    """One loaded model, one request at a time."""

    def __init__(
        self,
        predictor: EFPredictor,
        video_dir: Path,
        quality: LoadedQuality | None = None,
    ) -> None:
        self.predictor = predictor
        self.video_dir = Path(video_dir)
        self.quality = quality
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

    def acquire_named(self, name: str, pose) -> tuple[int, dict]:
        if self.quality is None:
            return 503, {
                "status": "quality_model_unavailable",
                "message": "The severity model is not loaded.",
            }
        try:
            checked_pose = _pose_from_json(pose)
            path = resolve_video(self.video_dir, name)
            decoded = load_video_rgb(path)
        except ValueError as exc:
            return 400, {"status": "bad_request", "message": str(exc)}
        except FileNotFoundError as exc:
            return 404, {"status": "missing_video", "message": str(exc)}
        except OSError as exc:
            return 422, {"status": "unreadable_video", "message": str(exc)}
        with self._lock:
            payload = acquire_view(self.predictor, self.quality, decoded, checked_pose)
        code = 200 if payload["status"] in {"accepted", "rejected"} else 422
        return code, payload


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
                self._send(
                    404,
                    {"status": "not_found", "message": "Use GET /health, POST /predict, or POST /acquire."},
                )
                return
            health = {
                "status": "ready",
                "model_id": server.predictor.loaded.model_id,
                "device": server.predictor.loaded.device.type,
                "video_dir": str(server.video_dir),
                "quality_model_id": None if server.quality is None else server.quality.model_id,
                "threshold": None if server.quality is None else server.quality.threshold,
            }
            self._send(200, health)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path not in {"/predict", "/acquire"}:
                self._send(404, {"status": "not_found", "message": "POST a video name to /predict or /acquire."})
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
                        "message": 'Send JSON with "video" set to a file name in the video directory.',
                    },
                )
                return
            if path == "/predict":
                code, payload = server.predict_named(name)
            else:
                code, payload = server.acquire_named(name, body.get("pose"))
            self._send(code, payload)

    return Handler


def serve(
    predictor: EFPredictor,
    video_dir: Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    quality: LoadedQuality | None = None,
) -> None:
    """Listen until the process is stopped. The models stay loaded."""

    prediction_server = PredictionServer(predictor, video_dir, quality)
    httpd = ThreadingHTTPServer((host, port), make_handler(prediction_server))
    print(f"Listening on http://{host}:{port}", flush=True)
    print(f"Video directory: {Path(video_dir).resolve()}", flush=True)
    print('GET /health    POST /predict {"video": "clip.avi"}', flush=True)
    print('POST /acquire  {"video": "clip.avi", "pose": [0, 0, 0]}', flush=True)
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
    parser.add_argument("--quality-checkpoint", default="models/severity_cnn.pt")
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
    from cardioassist_cnn.quality import load_quality_model

    quality = load_quality_model(args.quality_checkpoint, device=args.device)
    serve(predictor, video_dir, host=args.host, port=args.port, quality=quality)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
