"""Movement Murmuration: an explorable behavioral-space visualization.

A single self-contained HTML file (no server, no build step). Each
particle is one real movement episode positioned by its embedding.

Visual-property contract -- every channel is data, documented here and
in the page itself:

- position        = episode embedding (PCA of kinematic features)
- color           = biological label (ant / siler / unknown)
- flutter radius  = movement intermittency (speed_cv): stop-and-go
                    episodes visibly quiver, steady walkers glide
- trail           = the episode's real trajectory, replayed at true fps
- click           = full provenance chain: video, frames, models,
                    plus an animated replay of the original path
"""

from __future__ import annotations

import html
import json
from pathlib import Path

COLORS = {"ant": "#e8a13c", "siler": "#4fd1c5", "other_spider": "#b794f4",
          "other_arthropod": "#a0aec0", "unknown": "#718096"}

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>MOTIONSCAPE — Movement Murmuration</title>
<style>
 body{margin:0;background:#0b0f14;color:#d8e2ec;font:14px/1.5 system-ui,sans-serif}
 #stage{display:flex;height:100vh}
 #canvas{flex:1;display:block}
 #panel{width:340px;padding:18px;overflow-y:auto;border-left:1px solid #1f2a36;background:#0e141b}
 h1{font-size:16px;margin:0 0 4px;color:#9fe8df}
 .muted{color:#6b7f92;font-size:12px}
 .prov{font-family:ui-monospace,monospace;font-size:11px;color:#8fb0c8;white-space:pre-wrap;margin-top:8px;border-left:2px solid #2a3a4a;padding-left:8px}
 .legend{display:flex;gap:12px;font-size:12px;margin-bottom:12px}
 .dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px}
 table{border-collapse:collapse;font-size:12px;margin-top:8px}
 td{border-bottom:1px solid #1f2a36;padding:3px 8px}
 #replay{background:#05080c;border:1px solid #1f2a36;border-radius:6px;margin-top:10px}
</style>
</head>
<body>
<div id="stage">
 <canvas id="canvas"></canvas>
 <div id="panel">
  <h1>MOTIONSCAPE · Movement Murmuration</h1>
  <div class="legend" id="legend"></div>
  <div class="muted">每个粒子 = 一段真实运动片段（episode）。位置=行为空间嵌入；抖动幅度=运动间歇性(speed_cv)；颜色=类别。点击粒子查看溯源并回放轨迹。</div>
  <div id="detail"></div>
  <canvas id="replay" width="300" height="220"></canvas>
 </div>
</div>
<script>
const DATA = __DATA__;
const N = 200;                      // tame per-frame noise scale
const W = () => canvas.width, H = () => canvas.height;
const canvas = document.getElementById('canvas'), ctx = canvas.getContext('2d');
const replay = document.getElementById('replay'), rctx = replay.getContext('2d');
function fit(){canvas.width=innerWidth-340;canvas.height=innerHeight;}
addEventListener('resize',fit);fit();

let xs=DATA.episodes.map(e=>e.embedding[0]), ys=DATA.episodes.map(e=>e.embedding[1]);
const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
const sx=x=>0.08*W()+0.84*W()*(x-xmin)/(xmax-xmin+1e-9), sy=y=>0.9*H()-0.8*H()*(y-ymin)/(ymax-ymin+1e-9);
// downsample trajectories to <=300 pts for the embedded payload
let sel=-1, tFrame=0;

function draw(){
 ctx.fillStyle='#0b0f14';ctx.fillRect(0,0,W(),H());
 ctx.strokeStyle='rgba(60,80,100,.15)';
 for(let i=1;i<6;i++){ctx.beginPath();ctx.moveTo(W()*i/6,0);ctx.lineTo(W()*i/6,H());ctx.stroke();
   ctx.beginPath();ctx.moveTo(0,H()*i/6);ctx.lineTo(W(),H()*i/6);ctx.stroke();}
 tFrame++;
 DATA.episodes.forEach((e,i)=>{
   const flut=(e.speed_cv||0)*4;                       // flutter = intermittency
   const px=sx(e.embedding[0])+N*Math.sin(tFrame*0.013+i*2.1)*flut*0.02;
   const py=sy(e.embedding[1])+N*Math.cos(tFrame*0.011+i*1.7)*flut*0.02;
   e._px=px;e._py=py;
   ctx.beginPath();ctx.fillStyle=COLORS_[e.bio_label]||COLORS_.unknown;
   ctx.globalAlpha=i===sel?1:0.85;
   ctx.arc(px,py,i===sel?7:4,0,7);ctx.fill();ctx.globalAlpha=1;
 });
 if(sel>=0) replayPath();
 requestAnimationFrame(draw);
}
function replayPath(){                                // trail = real trajectory @ true fps
 const e=DATA.episodes[sel],tr=e.trajectory;
 rctx.fillStyle='#05080c';rctx.fillRect(0,0,300,220);
 const mx=Math.min(...tr.map(p=>p[0])),Mx=Math.max(...tr.map(p=>p[0]));
 const my=Math.min(...tr.map(p=>p[1])),My=Math.max(...tr.map(p=>p[1]));
 const P=p=>[20+260*(p[0]-mx)/(Mx-mx+1e-9),200-180*(p[1]-my)/(My-my+1e-9)];
 const n=Math.floor((tFrame%(e.fps*6))*tr.length/(e.fps*6));   // loop every 6 s of video time
 rctx.strokeStyle=COLORS_[e.bio_label];rctx.lineWidth=2;rctx.beginPath();
 for(let i=0;i<=n;i++){const q=P(tr[i]);i?rctx.lineTo(q[0],q[1]):rctx.moveTo(q[0],q[1]);}
 rctx.stroke();
 const h=P(tr[n]);rctx.beginPath();rctx.arc(h[0],h[1],5,0,7);rctx.fillStyle='#fff';rctx.fill();
}
canvas.addEventListener('click',ev=>{
 const r=canvas.getBoundingClientRect();let best=-1,bd=1e9;
 DATA.episodes.forEach((e,i)=>{const d=(ev.clientX-r.left-e._px)**2+(ev.clientY-r.top-e._py)**2;if(d<bd){bd=d;best=i;}});
 if(bd<400){sel=best;showDetail(DATA.episodes[best]);}
});
function showDetail(e){
 document.getElementById('detail').innerHTML=`
  <div class="prov">${htmlEscape(e.provenance_chain.join('\n'))}</div>
  <table>${Object.entries(e.trajectory_features).slice(0,10).map(([k,v])=>`<tr><td>${k}</td><td>${(+v).toFixed(3)}</td></tr>`).join('')}</table>
  <div class="muted" style="margin-top:8px">motif M${e.motif} · duration ${e.duration_s.toFixed(1)}s · label ${e.bio_label} (${(e.bio_label_confidence).toFixed(2)})</div>`;
}
const htmlEscape=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const COLORS_ = __COLORS__;
document.getElementById('legend').innerHTML=Object.entries(COLORS_).map(([k,c])=>`<span><span class="dot" style="background:${c}"></span>${k}</span>`).join('');
draw();
</script>
</body>
</html>"""


def export_murmuration(episodes, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def payload(ep):
        tr = ep.centroids_px
        if len(tr) > 300:
            idx = [round(i * (len(tr) - 1) / 299) for i in range(300)]
            tr = [tr[i] for i in idx]
        return {
            "episode_id": ep.episode_id,
            "source_video_id": ep.source_video_id,
            "embedding": ep.embedding or [0, 0],
            "bio_label": ep.bio_label,
            "bio_label_confidence": ep.bio_label_confidence,
            "motif": ep.motif,
            "fps": ep.fps,
            "duration_s": ep.duration_s,
            "speed_cv": ep.trajectory_features.get("speed_cv", 0.0),
            "trajectory": tr,
            "trajectory_features": ep.trajectory_features,
            "provenance_chain": ep.provenance_chain(),
        }

    data = json.dumps({"episodes": [payload(e) for e in episodes]}, ensure_ascii=False)
    doc = (_TEMPLATE.replace("__DATA__", data)
                    .replace("__COLORS__", json.dumps(COLORS)))
    out_path.write_text(doc, encoding="utf-8")
    return out_path
