"""Local atlas server: static files + on-demand episode media.

    python -m motionscape serve <atlas_dir> --episodes <episodes.jsonl>

Endpoints:
    GET  /                      atlas/index.html
    GET  /data.json             slim payload (no media)
    GET  /meta/<id>.json        lazy per-episode detail
    GET  /clip/<id>.gif         generated on demand, cached under cache/clips
    GET  /api/info              server capability probe
    GET  /api/motifs            human motif annotations
    POST /api/motifs            save a human motif annotation (append-safe)

Why a server: thousands of source-video GIFs cannot be pre-rendered. The
server decodes one episode's frames when (and only when) a visitor selects
it, caching the result. Opening atlas/index.html statically still works —
the UI falls back to in-browser trajectory replay.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .schema import Episode
from .store import EpisodeStore


class AtlasServer:
    def __init__(self, atlas_dir: str | Path, episodes_path: str | Path | None = None):
        self.atlas_dir = Path(atlas_dir).resolve()
        self.episodes_path = Path(episodes_path) if episodes_path else None
        self._ep_by_id: dict[str, Episode] | None = None
        self._lock = threading.Lock()
        if self.episodes_path is None:
            # convenience: a run directory's atlas sits next to episodes.jsonl
            cand = self.atlas_dir.parent / "episodes.jsonl"
            if cand.exists():
                self.episodes_path = cand

    # ---- episode lookup (lazy full parse only on first media request) ----
    def _episodes(self) -> dict[str, Episode]:
        with self._lock:
            if self._ep_by_id is None:
                if not self.episodes_path or not self.episodes_path.exists():
                    self._ep_by_id = {}
                else:
                    self._ep_by_id = {e.episode_id: e for e in
                                      EpisodeStore.read_episodes(self.episodes_path)}
            return self._ep_by_id

    def episode(self, episode_id: str) -> Episode | None:
        return self._episodes().get(episode_id)

    def clip_path(self, episode_id: str) -> Path | None:
        """Generate (once) and return the cached GIF for an episode."""
        ep = self.episode(episode_id)
        if ep is None:
            return None
        cache = self.atlas_dir / "cache" / "clips"
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / f"{episode_id}.gif"
        if not out.exists():
            from .media import make_clip
            with self._lock:
                if not out.exists():
                    make_clip(ep, str(out))
        return out

    # ---- motif annotations ----
    def motif_annotations(self) -> dict:
        path = self.atlas_dir / "motif_annotations.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save_motif_annotation(self, motif: int, name: str, notes: str, annotator: str) -> dict:
        path = self.atlas_dir / "motif_annotations.json"
        data = self.motif_annotations()
        data[str(int(motif))] = {
            "name": name, "notes": notes, "annotator": annotator or "anonymous",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with self._lock:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data[str(int(motif))]


def make_handler(server: AtlasServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _file(self, rel: Path, ctype: str):
            if rel.is_file():
                self._send(200, rel.read_bytes(), ctype)
            else:
                self._json({"error": f"not found: {rel.name}"}, 404)

        def do_GET(self):
            p = urlparse(self.path).path
            if p in ("/", "/index.html"):
                return self._file(server.atlas_dir / "index.html", "text/html; charset=utf-8")
            if p == "/data.json":
                return self._file(server.atlas_dir / "data.json", "application/json")
            if p.startswith("/meta/"):
                eid = Path(unquote(p)).stem
                return self._file(server.atlas_dir / "meta" / f"{eid}.json", "application/json")
            if p.startswith("/clip/"):
                eid = Path(unquote(p)).stem
                try:
                    gif = server.clip_path(eid)
                except Exception as e:
                    return self._json({"error": str(e)}, 500)
                if gif is None:
                    return self._json({"error": "unknown episode"}, 404)
                return self._file(gif, "image/gif")
            if p == "/api/info":
                return self._json({"server": "motionscape-serve", "atlas": server.atlas_dir.name,
                                   "n_episodes_meta": len(server._episodes()),
                                   "clip_endpoint": True})
            if p == "/api/motifs":
                return self._json(server.motif_annotations())
            self._json({"error": "unknown path"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/motifs":
                return self._json({"error": "unknown path"}, 404)
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                saved = server.save_motif_annotation(
                    int(body["motif"]), str(body.get("name", ""))[:80],
                    str(body.get("notes", ""))[:500], str(body.get("annotator", ""))[:80])
                self._json({"saved": saved})
            except Exception as e:
                self._json({"error": str(e)}, 400)

    return Handler


def serve_atlas(atlas_dir: str | Path, episodes_path: str | Path | None = None,
                port: int = 8694) -> None:
    server = AtlasServer(atlas_dir, episodes_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(server))
    print(f"MOTIONSCAPE atlas: http://127.0.0.1:{port}/  (Ctrl+C to stop)")
    print(f"  atlas dir: {server.atlas_dir}")
    print(f"  episodes:  {server.episodes_path or '(media disabled — static mode)'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
