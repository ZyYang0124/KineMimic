"""Annotation workbench: a fast human review interface for episodes.

    python -m kinemimic annotate <path/to/episodes.jsonl> [--store ROOT]

Layout: episode clip (cropped, zoomed to the animal) on the left; metadata
and the machine prediction on the right; classification controls at the
bottom. Every keystroke appends one record to the store's annotation log
immediately — no save button, no data loss — and advances to the next
episode. The log is append-only; corrections simply supersede.

Shortcuts (shown on screen at all times):
    S siler · A ant · O other spider · R other arthropod · U unknown
    X reject · F tracking failure · H severe occlusion · E edge effect
    T too short · B ambiguous taxon · Z undo · ←/→ navigate
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .annotation import (AnnotationRecord, append_record, apply_log, summary)
from .media import make_clip
from .schema import Episode
from .store import EpisodeStore

LABEL_KEYS = [("S", "siler", "Siler"), ("A", "ant", "Ant"),
              ("O", "other_spider", "Other spider"),
              ("R", "other_arthropod", "Other arthropod"), ("U", "unknown", "Unknown")]
QC_KEYS = [("X", "rejected", "Reject"), ("F", "tracking_failure", "Track fail"),
           ("H", "severe_occlusion", "Occlusion"), ("E", "edge_effect", "Edge"),
           ("T", "too_short", "Too short"), ("B", "ambiguous_taxon", "Ambiguous")]


def detect_store_root(episodes_path: Path) -> Path:
    """runs/<run_id>/episodes.jsonl -> store root (walk up to a dir that has
    runs/ or annotations/); otherwise the file's great-grandparent."""
    p = episodes_path.resolve()
    for anc in p.parents:
        if (anc / "annotations").exists() or (anc / "runs").exists():
            return anc
    return p.parents[2] if len(p.parents) > 2 else p.parent


class Workbench:
    def __init__(self, episodes_path: str | Path, store_root: str | Path | None = None):
        self.episodes_path = Path(episodes_path)
        self.store_root = Path(store_root) if store_root else detect_store_root(self.episodes_path)
        self.episodes: list[Episode] = EpisodeStore.read_episodes(self.episodes_path)
        apply_log(self.episodes, self.store_root)     # resume where the last session stopped
        self._lock = threading.Lock()

    def queue_entry(self, i: int) -> dict:
        ep = self.episodes[i]
        return {
            "i": i, "episode_id": ep.episode_id,
            "video_id": ep.source_video_id, "video_path": ep.source_video_path,
            "duration_s": round(ep.duration_s, 2), "n_frames": ep.n_frames,
            "frames": [ep.start_frame, ep.end_frame], "fps": ep.fps,
            "machine_label": ep.machine_label or ep.bio_label,
            "machine_confidence": round(ep.machine_confidence or ep.bio_label_confidence, 3),
            "machine_source": ep.machine_source or ep.bio_label_source,
            "human_label": ep.human_label, "annotation_status": ep.annotation_status,
            "annotator": ep.annotator,
            "site": ep.site_id or ep.environment.site, "date": ep.environment.date,
            "features": {k: round(float(v), 3) for k, v in
                         list(sorted(ep.trajectory_features.items()))[:17]},
            "ambiguity_events": len(ep.metadata.get("ambiguity_events", [])),
            "tracking_confidence": ep.metadata.get("tracking_confidence"),
            "n_interpolated_points": ep.metadata.get("n_interpolated_points", 0),
            "provenance_chain": ep.provenance_chain(),
        }

    def state(self) -> dict:
        with self._lock:
            eps = list(self.episodes)
        return {"queue": [self.queue_entry(i) for i in range(len(eps))],
                "summary": summary(eps), "store_root": str(self.store_root),
                "episodes_path": str(self.episodes_path)}

    def annotate(self, episode_id: str, human_label: str | None, qc_state: str,
                 annotator: str, note: str = "") -> dict:
        rec = AnnotationRecord(episode_id=episode_id, human_label=human_label,
                               qc_state=qc_state, annotator=annotator or "anonymous",
                               note=note)
        with self._lock:
            ep = next((e for e in self.episodes if e.episode_id == episode_id), None)
        append_record(self.store_root, rec)           # durable first (append-only)
        if ep is not None:
            from .annotation import apply_record
            apply_record(ep, rec)                      # then reflect in memory
        return {"saved": rec.to_dict(), "summary": summary(self.episodes)}

    def clip(self, episode_id: str, zoom: bool = True) -> Path | None:
        with self._lock:
            ep = next((e for e in self.episodes if e.episode_id == episode_id), None)
        if ep is None:
            return None
        cache = self.store_root / "cache" / "workbench"
        cache.mkdir(parents=True, exist_ok=True)
        out = cache / f"{episode_id}{'_zoom' if zoom else ''}.gif"
        if not out.exists():
            with self._lock:
                if not out.exists():
                    if zoom:
                        try:
                            from .media import video_clip_gif, clip_kind
                            if clip_kind(ep) == "video":
                                video_clip_gif(ep, str(out), width=420, crop_margin_px=90)
                            else:
                                make_clip(ep, str(out))
                            return out
                        except Exception:
                            return None
                    make_clip(ep, str(out))
        return out


TEMPLATE = r"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>KineMimic — Annotation Workbench</title><style>
 body{margin:0;background:#0a0f14;color:#d8e2ec;font:14px/1.5 system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}
 header{display:flex;align-items:baseline;gap:14px;padding:10px 16px;border-bottom:1px solid #16212d}
 header h1{font-size:15px;margin:0;color:#9fe8df;font-weight:500}
 #prog{font-size:13px;color:#8fa8bc} #prog b{color:#c8f2ec;font-size:16px}
 #counts{margin-left:auto;font-size:12px;color:#64798d;display:flex;gap:10px}
 .cnt{padding:1px 8px;border-radius:10px;background:#101a24}
 main{flex:1;display:flex;min-height:0}
 #left{flex:1;display:flex;align-items:center;justify-content:center;background:#05080c;position:relative}
 #clip{max-width:100%;max-height:100%}
 #idx{position:absolute;top:8px;left:12px;font-size:12px;color:#64798d}
 #right{width:330px;padding:14px;overflow-y:auto;border-left:1px solid #16212d;background:#0a1017}
 .muted{color:#64798d;font-size:12px} .k{color:#9fe8df;font-weight:600}
 table{border-collapse:collapse;font-size:12px;width:100%;margin-top:6px}
 td{border-bottom:1px solid #131d27;padding:2px 6px} td:last-child{text-align:right;color:#b8c8d8}
 #machine{margin:8px 0;padding:8px 10px;border:1px dashed #2a4a55;border-radius:8px;font-size:13px}
 #machine .warn{color:#e8a13c;font-size:11px;display:block;margin-top:3px}
 footer{padding:10px 16px;border-top:1px solid #16212d;display:flex;gap:14px;flex-wrap:wrap;background:#0a1017}
 .key{display:inline-flex;align-items:center;gap:6px;margin-right:10px;font-size:13px;cursor:pointer}
 .key b{display:inline-block;min-width:22px;text-align:center;padding:2px 5px;border:1px solid #2a4a55;border-radius:6px;background:#0e1a22;color:#9fe8df;font-size:12px}
 .key:hover b{background:#14303a}
 #annotator{background:#0e1a22;border:1px solid #2a4a55;color:#c8f2ec;border-radius:6px;padding:2px 8px;font-size:12px;width:120px}
 #toast{position:fixed;bottom:64px;left:50%;transform:translateX(-50%);background:#14303a;color:#c8f2ec;padding:6px 16px;border-radius:16px;font-size:13px;opacity:0;transition:opacity .3s;pointer-events:none}
 #skipnote{position:absolute;bottom:8px;left:12px;font-size:11px;color:#4a5a6a}
</style></head><body>
<header><h1>KineMimic · Annotation Workbench</h1>
 <div id="prog"><b>0</b> / 0</div>
 <div id="counts"></div></header>
<main><div id="left"><div id="idx"></div><img id="clip"><div id="skipnote">←/→ 导航 · Z 撤销 · 标注后自动进入下一条,即时持久化</div></div>
 <div id="right">
  <div id="idline" style="color:#9fe8df;font-size:13px"></div>
  <div id="machine"></div>
  <div class="muted">轨迹特征(观测,非标签)</div>
  <table id="feat"></table>
  <div class="muted" style="margin-top:8px">溯源</div>
  <div id="prov" class="muted" style="font-family:ui-monospace,monospace;font-size:10px;white-space:pre-wrap"></div>
 </div></main>
<footer>
 <span id="labelKeys"></span><span style="color:#2a3a4a">|</span><span id="qcKeys"></span>
 <span style="color:#2a3a4a">|</span>
 <span class="key"><b>Z</b>撤销</span>
 <span class="muted">annotator <input id="annotator" placeholder="your name"></span>
</footer>
<div id="toast"></div>
<script>
const LABELS=%LABEL_KEYS%, QCS=%QC_KEYS%;
let Q=[],S=0,cur=-1,pending=false;
const $=id=>document.getElementById(id);
function toast(t){const e=$('toast');e.textContent=t;e.style.opacity=1;clearTimeout(e._t);e._t=setTimeout(()=>e.style.opacity=0,1200);}
async function boot(){
 const st=await (await fetch('api/state')).json();
 STATE=st;
 Q=st.queue.filter(e=>e.annotation_status==='unreviewed').map(e=>e.i);
 if(!Q.length)Q=[...Array(st.summary.total).keys()];
 S=st.summary;$('annotator').value=localStorage.getItem('annotator')||'';
 renderSummary();
 if(Q.length)show(Q[0]);
}
function renderSummary(){
 $('prog').innerHTML=`<b>${S.reviewed.toLocaleString()}</b> / ${S.total.toLocaleString()}`;
 const bl=S.by_effective_label||{},bs=S.by_status||{};
 $('counts').innerHTML=
  `<span class="cnt" style="color:#e8a13c">ant ${bl.ant||0}</span>`+
  `<span class="cnt" style="color:#4fd1c5">siler ${bl.siler||0}</span>`+
  `<span class="cnt" style="color:#b794f4">other ${((bl.other_spider||0)+(bl.other_arthropod||0))}</span>`+
  `<span class="cnt" style="color:#718096">unknown ${bl.unknown||0}</span>`+
  `<span class="cnt" style="color:#e06c75">rejected ${bs.rejected||0}</span>`;
}
function show(i){
 cur=i;
 const st=STATE.queue[i];
 $('idx').textContent=`#${st.i+1} / ${STATE.summary.total}`;
 $('idline').textContent=`${st.episode_id} · ${st.video_id} · ${st.duration_s}s · frames ${st.frames[0]}–${st.frames[1]}`;
 $('machine').innerHTML=`机器预分类(非真值): <b>${st.machine_label}</b> (${st.machine_confidence.toFixed(2)})`+
   `<span class="warn">⚠ 运动特征参与了这个预测——拟态分析必须使用你的人工标注</span>`+
   (st.human_label?`<span style="color:#4fd1c5">已标注: ${st.human_label} (${st.annotation_status})</span>`:'')+
   (st.ambiguity_events?`<span class="warn">⚠ Association ambiguity: ${st.ambiguity_events} event(s) — check for identity contamination</span>`:'')+
   (st.n_interpolated_points?`<span class="warn">· ${st.n_interpolated_points} interpolated points</span>`:'');
 $('feat').innerHTML=Object.entries(st.features).map(([k,v])=>`<tr><td>${k}</td><td>${v}</td></tr>`).join('');
 $('prov').textContent=st.provenance_chain.join('\n');
 $('clip').src=`clip/${st.episode_id}.gif?zoom=1&r=${Date.now()}`;
 $('clip').onerror=()=>{$('clip').removeAttribute('src');$('idx').textContent+=' · (clip unavailable)';};
 pending=false;
}
let STATE;
async function send(label,qc){
 if(cur<0||pending)return;pending=true;
 const st=STATE.queue[cur];
 const r=await fetch('api/annotate',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({episode_id:st.episode_id,human_label:label,qc_state:qc,annotator:$('annotator').value})});
 const out=await r.json();
 if(out.saved){S=out.summary;renderSummary();toast(label?`${label} ✓`:`${qc} ✓`);next();}
 else pending=false;
}
function next(){if(!Q.length){toast('queue 完成 ✓');cur=-1;return;}Q.shift();if(Q.length)show(Q[0]);else toast('全部完成 ✓');}
function undo(){if(cur<0)return;Q.unshift(cur);toast('已回退(历史未被改写,可重新标注)');show(cur);}
document.addEventListener('keydown',ev=>{
 if(ev.target.tagName==='INPUT')return;
 const k=ev.key.toLowerCase();
 const lab=LABELS.find(x=>x[0].toLowerCase()===k);if(lab)return send(lab[1],'accepted');
 const qc=QCS.find(x=>x[0].toLowerCase()===k);if(qc)return send(null,qc[1]);
 if(k==='arrowright')next();
 if(k==='arrowleft'&&cur>0)show(cur-1);
 if(k==='z')undo();
 if(k==='?')toast('S/A/O/R/U 标注 · X/F/H/E/T/B QC · ←/→ 导航 · Z 撤销');
});
$('annotator').addEventListener('change',e=>localStorage.setItem('annotator',e.target.value));
$('labelKeys').innerHTML=LABELS.map(x=>`<span class="key" onclick="send('${x[1]}','accepted')"><b>${x[0]}</b>${x[2]}</span>`).join('');
$('qcKeys').innerHTML=QCS.map(x=>`<span class="key" onclick="send(null,'${x[1]}')"><b>${x[0]}</b>${x[2]}</span>`).join('');
boot().then(()=>{});
</script></body></html>"""


def _template() -> str:
    t = TEMPLATE.replace("%LABEL_KEYS%", json.dumps(LABEL_KEYS))
    return t.replace("%QC_KEYS%", json.dumps(QC_KEYS))


def make_handler(wb: Workbench):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = urlparse(self.path).path
            if p in ("/", "/index.html"):
                body = _template().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif p == "/api/state":
                self._json(wb.state())
            elif p.startswith("/clip/"):
                eid = Path(unquote(p)).stem
                zoom = "zoom=1" in (urlparse(self.path).query or "")
                try:
                    gif = wb.clip(eid, zoom=zoom)
                except Exception as e:
                    return self._json({"error": str(e)}, 500)
                if gif is None or not gif.exists():
                    return self._json({"error": "clip unavailable"}, 404)
                body = gif.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/gif")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "unknown path"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/annotate":
                return self._json({"error": "unknown path"}, 404)
            try:
                n = int(self.headers.get("Content-Length", 0))
                b = json.loads(self.rfile.read(n) or b"{}")
                out = wb.annotate(b["episode_id"], b.get("human_label"),
                                  b.get("qc_state", "accepted"),
                                  b.get("annotator", ""), b.get("note", ""))
                self._json(out)
            except Exception as e:
                self._json({"error": str(e)}, 400)

    return Handler


def run_workbench(episodes_path: str | Path, store_root: str | Path | None = None,
                  port: int = 8692) -> None:
    wb = Workbench(episodes_path, store_root)
    s = summary(wb.episodes)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(wb))
    print(f"KineMimic annotation workbench: http://127.0.0.1:{port}/")
    print(f"  episodes: {wb.episodes_path}  ({s['total']} total, {s['reviewed']} reviewed)")
    print(f"  log:      {wb.store_root / 'annotations' / 'annotations.jsonl'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped — every annotation was already persisted")
