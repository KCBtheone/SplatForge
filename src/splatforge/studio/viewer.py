"""Dependency-light local viewer with live run state and render endpoints."""

from __future__ import annotations

import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

from .artifacts import load_checkpoint
from .cameras import Camera


class RunViewer:
    def __init__(self, root, device=None):
        self.root = Path(root).resolve()
        self.device = device
        self.pipeline = None
        self.mtime = None
        self.lock = threading.Lock()

    def state(self):
        path = self.root / "state.json"
        state = (
            json.loads(path.read_text())
            if path.is_file()
            else {"stage": "WAITING_FOR_RUN", "step": 0}
        )
        metrics = self.root / "metrics.jsonl"
        if metrics.is_file():
            with metrics.open("rb") as handle:
                handle.seek(max(0, metrics.stat().st_size - 65536))
                lines = handle.read().splitlines()
            rows = []
            for line in lines:
                try:
                    rows.append(json.loads(line))
                except (ValueError, UnicodeError):
                    continue
            state["history"] = rows[-200:]
            if rows:
                state.setdefault("loss", rows[-1]["loss"])
        export = self.root / "export.json"
        if export.is_file():
            state["export"] = json.loads(export.read_text())
        return state

    def cloud(self):
        checkpoint = self.root / "checkpoints" / "latest.pt"
        if not checkpoint.is_file():
            return {"points": [], "colors": [], "count": 0}
        payload = load_checkpoint(checkpoint)
        state = payload["pipeline"]["model"]
        points = state["params.means"].detach().numpy()
        color = (
            0.5 + 0.28209479177387814 * state["params.sh"][:, 0].detach().numpy()
        ).clip(0, 1)
        step = max(1, int(np.ceil(len(points) / 30000)))
        return {
            "points": points[::step].tolist(),
            "colors": color[::step].tolist(),
            "count": len(points),
            "preview_count": len(points[::step]),
            "checkpoint_step": payload["step"],
        }

    def mesh(self):
        from ..mesh.io import read_studio_mesh

        vertices, faces = read_studio_mesh(self.root / "mesh.ply")
        stride = max(1, int(np.ceil(len(faces) / 100000)))
        preview = vertices[faces[::stride]]
        normals = np.cross(preview[:, 1] - preview[:, 0], preview[:, 2] - preview[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
        lighting = 0.4 + 0.6 * np.abs(normals @ np.array([0.2, -0.5, 0.84]))
        colors = np.array([0.88, 0.72, 0.48])[None] * lighting[:, None]
        return {
            "points": preview.reshape(-1, 3).tolist(),
            "colors": np.repeat(colors, 3, axis=0).tolist(),
            "count": len(faces),
            "preview_count": len(preview),
            "triangles": True,
        }

    def render(self, index, mode):
        if mode not in ("rgb", "normal", "depth"):
            raise ValueError("unknown render mode")
        with self.lock:
            checkpoint = self.root / "checkpoints" / "latest.pt"
            modified = checkpoint.stat().st_mtime_ns
            if self.pipeline is None or modified != self.mtime:
                from .trainer import load_run

                self.pipeline, _ = load_run(self.root, device=self.device)
                self.mtime = modified
            if not 0 <= index < len(self.pipeline.data.frames):
                raise ValueError("view index is out of bounds")
            sample = self.pipeline.data.sample(index, self.pipeline.model.device)
            with torch.no_grad():
                output = self.pipeline.render(sample["camera"])
            image = {
                "rgb": output.rgb,
                "normal": output.normals * 0.5 + 0.5,
                "depth": output.depth,
            }[mode]
            if mode == "depth":
                image = image / image.max().clamp_min(1e-8)
            array = (
                (image.detach().cpu().clamp(0, 1).numpy() * 255)
                .round()
                .astype(np.uint8)
            )
            if array.shape[-1] == 1:
                array = array[..., 0]
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")
            return buffer.getvalue()

    def render_camera(self, payload):
        mode = payload.get("mode", "rgb")
        if mode not in ("rgb", "normal", "depth"):
            raise ValueError("unknown render mode")
        width = payload.get("width", 128)
        height = payload.get("height", 128)
        if (
            type(width) is not int
            or type(height) is not int
            or not 1 <= width <= 2048
            or not 1 <= height <= 2048
        ):
            raise ValueError("preview dimensions must be integer values in 1..2048")
        K = torch.tensor(payload["K"], dtype=torch.float32)
        matrix = torch.tensor(payload["world_to_camera"], dtype=torch.float32)
        if (
            K.shape != (3, 3)
            or matrix.shape != (4, 4)
            or not torch.isfinite(K).all()
            or not torch.isfinite(matrix).all()
        ):
            raise ValueError("preview camera matrices are invalid")
        if (
            K[0, 0] <= 0
            or K[1, 1] <= 0
            or not torch.allclose(matrix[3], torch.tensor([0.0, 0.0, 0.0, 1.0]))
        ):
            raise ValueError(
                "preview intrinsics or homogeneous camera transform are invalid"
            )
        with self.lock:
            checkpoint = self.root / "checkpoints" / "latest.pt"
            modified = checkpoint.stat().st_mtime_ns
            if self.pipeline is None or modified != self.mtime:
                from .trainer import load_run

                self.pipeline, _ = load_run(self.root, device=self.device)
                self.mtime = modified
            camera = Camera(K, matrix, width, height, "viewer-orbit").to(
                self.pipeline.model.device
            )
            with torch.no_grad():
                output = self.pipeline.render(camera)
            image = {
                "rgb": output.rgb,
                "normal": output.normals * 0.5 + 0.5,
                "depth": output.depth,
            }[mode]
            if mode == "depth":
                image = image / image.max().clamp_min(1e-8)
            array = (image.cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            if array.shape[-1] == 1:
                array = array[..., 0]
            buffer = io.BytesIO()
            Image.fromarray(array).save(buffer, format="PNG")
            return buffer.getvalue()


def make_server(root, *, host="127.0.0.1", port=8080, device=None):
    viewer = RunViewer(root, device)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/render-camera":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 65536:
                    raise ValueError("invalid camera request size")
                payload = json.loads(self.rfile.read(length))
                body = viewer.render_camera(payload)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, KeyError, RuntimeError, FileNotFoundError) as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path == "/":
                    body = files(__package__).joinpath("viewer.html").read_bytes()
                    content = "text/html; charset=utf-8"
                elif request.path in ("/viewer.js", "/viewer.css"):
                    body = files(__package__).joinpath(request.path[1:]).read_bytes()
                    content = (
                        "text/javascript; charset=utf-8"
                        if request.path.endswith(".js")
                        else "text/css; charset=utf-8"
                    )
                elif request.path == "/state":
                    body = json.dumps(viewer.state(), allow_nan=False).encode()
                    content = "application/json"
                elif request.path == "/cloud":
                    body = json.dumps(viewer.cloud(), allow_nan=False).encode()
                    content = "application/json"
                elif request.path == "/mesh":
                    body = json.dumps(viewer.mesh(), allow_nan=False).encode()
                    content = "application/json"
                elif request.path == "/render":
                    query = parse_qs(request.query)
                    body = viewer.render(
                        int(query.get("view", ["0"])[0]), query.get("mode", ["rgb"])[0]
                    )
                    content = "image/png"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(root, *, host="127.0.0.1", port=8080, device=None):
    torch.set_num_threads(min(4,torch.get_num_threads()))
    server = make_server(root, host=host, port=port, device=device)
    print(f"SplatForge viewer: http://{host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
