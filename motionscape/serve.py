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
    def __init__(self, atlas_dir: str | Path, episodes_path: str | Path | None = None,
                 reference_dir: str | Path | None = None):
        self.atlas_dir = Path(atlas_dir).resolve()
        self.episodes_path = Path(episodes_path) if episodes_path else None
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self._ep_by_id: dict[str, Episode] | None = None
        self._query_eps: dict[str, Episode] = {}      # from query.json (upload flow)
        self._ref_by_id: dict[str, Episode] | None = None
        self._job = {"running": False, "done": False, "error": None, "query_id": None}
        self._lock = threading.Lock()
        if self.episodes_path is None:
            # convenience: a run directory's atlas sits next to episodes.jsonl
            cand = self.atlas_dir.parent / "episodes.jsonl"
            if cand.exists():
                self.episodes_path = cand
        self._load_query_episodes()

    def _ref_episodes(self) -> dict[str, Episode]:
        """Reference index episodes (for /meta fallback of cross-dataset
        neighbors — the served atlas may not contain them)."""
        if self._ref_by_id is None:
            self._ref_by_id = {}
            if self.reference_dir:
                import glob as _glob
                cands = sorted(Path(self.reference_dir).glob(
                    "atlas_reference_v*/episodes.jsonl"))
                if cands:
                    for e in EpisodeStore.read_episodes(cands[-1]):
                        self._ref_by_id[e.episode_id] = e
            self._ref_by_id.update(self._query_eps)
        return self._ref_by_id

    def _reference_meta(self, episode_id: str):
        """Build a meta payload for a reference-only episode on demand."""
        ep = self._ref_episodes().get(episode_id)
        if ep is None:
            return None
        from .atlas import episode_meta
        return episode_meta(ep)

    def _load_query_episodes(self):
        """Query episodes (uploaded videos) can request clips too."""
        q = self.atlas_dir / "query.json"
        if q.exists():
            try:
                bundle = json.loads(q.read_text(encoding="utf-8"))
                self._query_eps = {d["episode_id"]: Episode.from_dict(d)
                                   for d in bundle.get("episode_dicts", [])}
            except Exception:
                self._query_eps = {}

    def run_query_job(self, video_bytes: bytes, filename: str,
                      detector_params: dict | None = None):
        """Find Similar upload: save video, run the query pipeline, write
        query.json into the atlas dir (the browser picks it up)."""
        try:
            from .query import run_query
            uploads = self.atlas_dir / "uploads"
            uploads.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            safe = "".join(c for c in Path(filename).name if c.isalnum() or c in "._-") or "video.mp4"
            vpath = uploads / f"{ts}_{safe}"
            vpath.write_bytes(video_bytes)
            qid = f"upload_{ts}"
            self._job = {"running": True, "done": False, "error": None, "query_id": qid}
            bundle = run_query(str(vpath), qid, self.reference_dir,
                               out_dir=self.atlas_dir, atlas_dir=self.atlas_dir,
                               detector_params=detector_params)
            self._load_query_episodes()
            self._job = {"running": False, "done": True,
                         "error": bundle.get("error"),
                         "query_id": qid,
                         "n_episodes": len(bundle.get("episodes", []))}
        except Exception as e:                     # pragma: no cover
            self._job = {"running": False, "done": True, "error": str(e),
                         "query_id": None}

    def save_query_rating(self, record: dict) -> dict:
        path = self.atlas_dir / "query_evaluations.jsonl"
        record = {**record, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        return record

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
        return self._episodes().get(episode_id) or self._query_eps.get(episode_id)

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
            if p == "/query.json":
                return self._file(server.atlas_dir / "query.json", "application/json")
            if p.startswith("/meta/"):
                eid = Path(unquote(p)).stem
                f = server.atlas_dir / "meta" / f"{eid}.json"
                if f.is_file():
                    return self._file(f, "application/json")
                m = server._reference_meta(eid)   # cross-dataset neighbor
                if m is not None:
                    return self._json(m)
                return self._json({"error": "not found"}, 404)
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
                                   "clip_endpoint": True,
                                   "reference": (str(server.reference_dir)
                                                 if server.reference_dir else None),
                                   "find_similar_upload": server.reference_dir is not None})
            if p == "/api/query-status":
                return self._json(server._job)
            if p == "/api/query":
                q = server.atlas_dir / "query.json"
                if q.exists():
                    return self._file(q, "application/json")
                return self._json({"no_query": True}, 404)
            if p == "/api/motifs":
                return self._json(server.motif_annotations())
            self._json({"error": "unknown path"}, 404)

        def do_POST(self):
            pth = urlparse(self.path).path
            if pth == "/api/find-similar":
                if server.reference_dir is None:
                    return self._json({"error": "server started without --reference"}, 400)
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                fname = self.headers.get("X-Filename", "upload.mp4")
                det = {}
                if self.headers.get("X-Detector-Params"):
                    try:
                        det = json.loads(self.headers["X-Detector-Params"])
                    except Exception:
                        det = {}
                threading.Thread(target=server.run_query_job,
                                 args=(body, fname, det), daemon=True).start()
                return self._json({"started": True})
            if pth == "/api/query-rate":
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    b = json.loads(self.rfile.read(n) or b"{}")
                    rec = server.save_query_rating(b)
                    return self._json({"saved": rec})
                except Exception as e:
                    return self._json({"error": str(e)}, 400)
            if pth != "/api/motifs":
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
                port: int = 8694, reference_dir: str | Path | None = None) -> None:
    server = AtlasServer(atlas_dir, episodes_path, reference_dir=reference_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(server))
    print(f"MOTIONSCAPE atlas: http://127.0.0.1:{port}/  (Ctrl+C to stop)")
    print(f"  atlas dir: {server.atlas_dir}")
    print(f"  episodes:  {server.episodes_path or '(media disabled — static mode)'}")
    print(f"  reference: {server.reference_dir or '(Find Similar upload disabled)'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
