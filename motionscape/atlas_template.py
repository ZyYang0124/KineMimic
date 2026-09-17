"""MOTIONSCAPE — The Murmur front-end (self-contained, no build step).

Visual contract (what every channel means — nothing is decoration):

- particle position : behavioral-space embedding (built WITHOUT species labels)
- drift path        : the episode's real sliding-window embedding path z1..zt
- flutter amplitude : movement intermittency (speed_cv)
- color             : hidden until "Reveal species"
- click             : the real episode — source video or faithful replay,
                      real time series, provenance back to frames

Science contract:

- nearest neighbors come from the original standardized feature space,
  never from 2-D screen distance;
- before Reveal, no species identity is shown anywhere;
- synthetic data would be labeled as such (this atlas is built by the same
  pipeline as real runs and carries its run provenance in Data & provenance).
"""

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>MOTIONSCAPE — The Murmur · An atlas of animal movement</title>
<style>
 :root{--bg:#06090e;--ink:#d8e2ec;--dim:#64798d;--line:#16212d;--accent:#9fe8df;--panel:#0a1017ee}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif;overflow:hidden}
 #galaxy{position:fixed;inset:0;display:block;cursor:crosshair}
 /* ---------- top bar ---------- */
 header{position:fixed;top:0;left:0;right:0;display:flex;align-items:center;gap:14px;
   padding:10px 18px;background:linear-gradient(#06090eee,#0609000);z-index:8;pointer-events:none}
 header>*{pointer-events:auto}
 #brand{font-size:15px;letter-spacing:1.5px;color:var(--accent)}
 #brand small{display:block;font-size:11px;letter-spacing:.3px;color:var(--dim);font-weight:400}
 #dataset{font-size:12px;color:var(--dim)}
 header .sp{flex:1}
 .btn{padding:7px 16px;border:1px solid #2a4a55;border-radius:18px;background:#0e1a22dd;
   color:var(--accent);font-size:13px;cursor:pointer;transition:background .3s}
 .btn:hover{background:#14303a}
 .btn.primary{border-color:#3b6a75}
 /* ---------- blind note ---------- */
 #blindnote{position:fixed;left:18px;bottom:16px;max-width:430px;font-size:12px;color:var(--dim);
   z-index:4;transition:opacity 1s;pointer-events:none}
 #blindnote b{color:#8fa8bc;font-weight:500}
 /* ---------- legend (post-reveal) ---------- */
 #legend{position:fixed;left:18px;bottom:52px;display:none;flex-direction:column;gap:4px;
   font-size:12px;color:var(--ink);z-index:4;background:#06090eaa;padding:8px 12px;border-radius:10px}
 #legend .dot{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:7px;vertical-align:-1px}
 /* ---------- right panel ---------- */
 #panel{position:fixed;top:0;right:0;bottom:0;width:410px;background:var(--panel);
   border-left:1px solid var(--line);backdrop-filter:blur(6px);transform:translateX(105%);
   transition:transform .35s cubic-bezier(.2,.8,.2,1);z-index:6;overflow-y:auto;padding:60px 16px 16px}
 #panel.open{transform:translateX(0)}
 h1{font-size:15px;margin:0;color:var(--accent);letter-spacing:.5px;font-weight:500}
 .muted{color:var(--dim);font-size:12px}
 .lbl{font-size:12px;padding:1px 9px;border-radius:10px;background:#1c2833;display:inline-block}
 .clip{width:100%;border-radius:8px;background:#04070b;margin:6px 0;display:block;min-height:60px}
 .nbr{display:flex;gap:10px;align-items:center;padding:6px;border-radius:8px;cursor:pointer;border:1px solid transparent}
 .nbr:hover{background:#101a24;border-color:#1f303d}
 .nbr img{width:96px;height:60px;object-fit:cover;border-radius:6px;background:#04070b}
 .nbr canvas{width:96px;height:60px;border-radius:6px;background:#04070b}
 .prov{font-family:ui-monospace,monospace;font-size:10px;color:#6d8ba3;white-space:pre-wrap;
   margin-top:8px;border-left:2px solid #1f2f3d;padding-left:8px}
 canvas.spark{width:100%;height:40px;background:#0c141c;border-radius:6px;margin:2px 0;display:block}
 .rowlab{font-size:10px;color:var(--dim);width:52px;display:inline-block;vertical-align:top;padding-top:12px}
 .rowwrap{display:flex;align-items:center}.rowwrap>div{flex:1}
 .split{display:flex;gap:10px}.split>div{flex:1;min-width:0}
 .bar{height:11px;border-radius:3px;display:inline-block;vertical-align:middle}
 .chip{padding:3px 10px;border-radius:12px;border:1px solid #22333f;background:#0d151d;font-size:12px;
   cursor:pointer;color:#8fa8bc;display:inline-block;margin:0 6px 6px 0}
 .chip.on{background:#1b3a44;color:#c8f2ec;border-color:#3b6a75}
 .fprow{display:flex;align-items:center;gap:8px;margin:4px 0;font-size:12px}
 .fprow .name{width:130px;color:#b8c8d8}
 .fprow .track{flex:1;height:11px;background:#101a24;border-radius:3px;position:relative}
 .fprow .fill{position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:#4fd1c5}
 .fprow.future .fill{background:#2a3a4a}
 .tabbar{display:flex;gap:6px;margin:10px 0;flex-wrap:wrap}
 .modeseg{display:flex;gap:0;margin-right:6px}
 .modeseg .btn{border-radius:0;border-right-width:0}
 .modeseg .btn:first-child{border-radius:18px 0 0 18px}
 .modeseg .btn:last-child{border-radius:0 18px 18px 0;border-right-width:1px}
 .modeseg .btn.on{background:#1b3a44;color:#c8f2ec}
 .scenewrap{position:relative;margin:6px 0}
 .scenewrap img,.scenewrap canvas.clipcv{width:100%;display:block;border-radius:8px;background:#04070b}
 .scenewrap canvas.overlay{position:absolute;inset:0;width:100%;height:100%}
 .ctxline{margin:6px 0;padding:7px 10px;border:1px solid #22404a;border-radius:8px;font-size:12px;color:#b8d8d2;background:#0b141c}
 table.cmp{border-collapse:collapse;font-size:12px;width:100%;margin:6px 0}
 table.cmp td,table.cmp th{border-bottom:1px solid #131d27;padding:4px 6px;text-align:right}
 table.cmp th:first-child,table.cmp td:first-child{text-align:left;color:#8fa8bc}
 /* ---------- find similar ---------- */
 .rate{display:flex;gap:4px;margin-top:4px}
 .rate span{font-size:10px;padding:2px 8px;border:1px solid #22333f;border-radius:10px;
   cursor:pointer;color:#64798d}
 .rate span:hover{color:#c8f2ec;background:#14303a}
 .ood{margin:8px 0;padding:9px 12px;border:1px solid #6b4a2a;border-radius:8px;
   background:#1a120a;color:#e8c9a0;font-size:12px}
 .bdrow{display:flex;align-items:center;gap:8px;margin:3px 0;font-size:12px}
 .bdrow .nm{width:120px;color:#b8c8d8}
 .bdrow .lv{width:76px;text-align:right;color:#9fe8df}
 #uploadBox{margin:10px 0;padding:12px;border:1px dashed #2a4a55;border-radius:10px;text-align:center}
 #uploadBox input{display:none}
 .repswitch{display:flex;gap:6px;margin:8px 0}
 input.txt,textarea.txt{width:100%;background:#0e1a22;border:1px solid #2a4a55;color:var(--ink);
   border-radius:6px;padding:5px 8px;font-size:12px;font-family:inherit}
 .fps{position:fixed;right:12px;bottom:10px;font-size:10px;color:#3a4a5a;z-index:4}
 /* ---------- hero ---------- */
 #hero{position:fixed;inset:0;background:var(--bg);display:flex;flex-direction:column;align-items:center;
   justify-content:center;z-index:20;transition:opacity 1.4s;cursor:pointer}
 #heroCv{width:min(560px,80vw);height:280px}
 #heroQ{font-size:clamp(22px,3.4vw,32px);color:#cfeee9;font-weight:300;letter-spacing:2px;
   margin:18px 0 6px;opacity:0;transition:opacity 1.6s;text-align:center}
 #heroSub{font-size:13px;color:var(--dim);opacity:0;transition:opacity 1.6s;text-align:center;max-width:560px;padding:0 20px}
 #heroHint{position:absolute;bottom:26px;font-size:11px;color:#3a4a5a}
 @media (prefers-reduced-motion: reduce){
   #panel{transition:none}
 }
</style>
</head>
<body>
<canvas id="galaxy"></canvas>

<header>
 <div id="brand">MOTIONSCAPE<small>an atlas of animal movement · The Murmur</small></div>
 <div id="dataset"></div>
 <div class="sp"></div>
 <div class="modeseg" id="modeseg" style="display:none">
  <button class="btn on" id="modeM">◉ Movement</button><button class="btn" id="modeI">⇄ Interaction</button>
 </div>
 <button class="btn" id="findBtn">⌕ Find Similar</button>
 <button class="btn" id="exploreBtn">☰ Explore</button>
 <button class="btn primary" id="revealBtn">✦ Reveal species</button>
</header>

<div id="blindnote"><b>1 particle = 1 real movement episode.</b> Position = behavioral
 similarity, drift = the episode's own path through movement space, quiver = intermittency.
 <span id="blindline">This space was built without knowing what any animal is — only how it moves.</span></div>

<div id="legend"></div>
<div class="fps" id="fps"></div>

<div id="panel"></div>

<div id="hero">
 <canvas id="heroCv" width="1120" height="560"></canvas>
 <div id="heroQ">How does a spider move like an ant?</div>
 <div id="heroSub">MOTIONSCAPE · The Murmur — thousands of real movements, one explorable space.<br>
   先看运动本身。物种身份,由你决定何时揭示。</div>
 <div id="heroHint">click to enter · 点击进入</div>
</div>

<script>
"use strict";
/* ================= data ================= */
let META=null, EP=[], IDX={}, serverMode=false;
let revealed=false, revealT=0, sel=-1, motifSel=-1, T=0;
let mode='movement', contextShade=false;
let QUERY=null, qSel=-1, qIntro=-1, uploadInfo=null;   // Find Similar state
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
const COLORS={ant:"#e8a13c",mimic:"#4fd1c5",siler:"#4fd1c5",other_spider:"#b794f4",
              other_arthropod:"#a0aec0",unknown:"#718096"};
const NAMES={ant:"Ant",mimic:"Mimic spider (Myrmarachne)",siler:"Siler",
             other_spider:"Other spider",other_arthropod:"Other arthropod",unknown:"Unknown"};
const HIDDEN="#56677a";
const cv=document.getElementById('galaxy'),cx=cv.getContext('2d');
const panel=document.getElementById('panel');
let DPR=Math.min(devicePixelRatio||1,1.5);
function fit(){cv.width=(innerWidth)*DPR;cv.height=innerHeight*DPR;cv.style.width=innerWidth+'px';cv.style.height=innerHeight+'px';}
addEventListener('resize',fit);fit();

fetch('data.json').then(r=>r.json()).then(d=>{boot(d);});
fetch('api/info').then(r=>r.ok?r.json():null).then(i=>{serverMode=!!i&&!!i.clip_endpoint;
  window.__info=i||null; refreshFindBtn();}).catch(()=>{});
fetch('query.json').then(r=>r.ok?r.json():null).then(q=>{if(q&&q.episodes){QUERY=q;
  qSel=q.qc.findIndex(c=>c.usable); if(qSel<0)qSel=0;
  refreshFindBtn(); startQueryIntro();}}).catch(()=>{});
function refreshFindBtn(){
  const b=document.getElementById('findBtn'); if(!b)return;
  b.textContent = QUERY? '⌕ Find Similar — your video' : '⌕ Find Similar';
}

function boot(d){
 META=d.meta;
 EP=d.episodes.map(e=>{
   const o={...e, nn:e.nn||[]};
   IDX[o.id]=o; return o;
 });
 scaleEmbeddings();
 document.getElementById('dataset').textContent=
   `${META.n_episodes.toLocaleString()} movements · ${META.hierarchy.n_videos} videos · ${META.hierarchy.n_sites} site(s) · build ${META.build_seconds}s`;
 if(META.interaction&&META.interaction.available){
   document.getElementById('modeseg').style.display='flex';
   document.getElementById('modeI').onclick=()=>setMode('interaction');
   document.getElementById('modeM').onclick=()=>setMode('movement');
 }
 hero();
 requestAnimationFrame(draw);
}
function setMode(m){
 mode=m;
 document.getElementById('modeM').classList.toggle('on',m==='movement');
 document.getElementById('modeI').classList.toggle('on',m==='interaction');
 document.getElementById('blindnote').innerHTML = m==='interaction'
   ? '<b>Interaction view.</b> Hollow Siler particles = <b>no ants nearby</b> during the episode; filled = ants within the analysis radius. Reveal species to see it. Click any particle for its full scene.'
   : '<b>1 particle = 1 real movement episode.</b> Position = behavioral similarity, drift = the episode\'s own path through movement space, quiver = intermittency. <span id="blindline">This space was built without knowing what any animal is — only how it moves.</span>';
}
/* embed PCA coords -> [0..1]^2 (padded), deterministic across builds */
function scaleEmbeddings(){
 let xs=Infinity,xa=-Infinity,ys=Infinity,ya=-Infinity;
 EP.forEach(e=>{const p=e.p[0]||e.e;xs=Math.min(xs,p[0]);xa=Math.max(xa,p[0]);ys=Math.min(ys,p[1]);ya=Math.max(ya,p[1]);});
 const sx=1/(xa-xs+1e-9), sy=1/(ya-ys+1e-9);
 EP.forEach((e,i)=>{
   e.base=e.p.map(q=>[0.06+(q[0]-xs)*sx*0.88, 0.94-(q[1]-ys)*sy*0.88]);
   e.phase=(hash(e.id)%1000)/1000;                       // per-episode drift phase
   e.period=Math.min(40,Math.max(9,(e.d||6)*2.2));       // loop ∝ real duration (clamped)
 });
}
function hash(s){let h=9;for(let i=0;i<s.length;i++)h=Math.imul(h^s.charCodeAt(i),387420489);return h>>>0;}

/* ================= Find Similar: query particles ================= */
function qHitById(qeid, rep){   // results are indexed over *usable* episodes
  const src = rep==='b' ? (QUERY&&QUERY.results_representation_b) : (QUERY&&QUERY.results);
  if(!src) return null;
  return (src.episode_hits||[]).find(h=>h.query_episode_id===qeid) || null;
}
function qNeighbors(qe){   // top-k neighbor ids for a query episode (physical space)
  const h=qHitById(qe.episode_id,'a');
  return h?h.neighbors.map(n=>n.episode_id):[];
}
function qNeighborsB(qe){
  const h=qHitById(qe.episode_id,'b');
  return h?h.neighbors.map(n=>n.episode_id):[];
}
function qPos(qe){
  // clamp into the canvas padding; outside-hull queries carry the OOD flag
  return [Math.min(0.97,Math.max(0.03,qe.atlas_position[0])),
          Math.min(0.97,Math.max(0.03,qe.atlas_position[1]))];
}
function startQueryIntro(){
  if(REDUCED){qIntro=-1;return;}
  qIntro=0;                       // 0: trajectory draws, 1: glide, -1: done
  qIntroT0=performance.now();
}
let qIntroT0=0;

/* ================= hero: four stages from one real trajectory ================= */
let heroStage=0;
function hero(){
 const h=document.getElementById('hero');
 if(REDUCED){h.remove();return;}
 const c=document.getElementById('heroCv'),g=c.getContext('2d');
 const tr=(META.hero&&META.hero.trajectory)||[];
 if(tr.length<2){h.remove();return;}
 const x0=Math.min(...tr.map(p=>p[0])),x1=Math.max(...tr.map(p=>p[0])),
       y0=Math.min(...tr.map(p=>p[1])),y1=Math.max(...tr.map(p=>p[1]));
 const P=p=>[80+(p[0]-x0)/(x1-x0+1e-9)*(c.width-160), 480-(p[1]-y0)/(y1-y0+1e-9)*400];
 let i=0;
 const t1=setTimeout(()=>document.getElementById('heroQ').style.opacity=1,900);
 const t2=setTimeout(()=>document.getElementById('heroSub').style.opacity=1,3200);
 const t3=setTimeout(()=>{heroStage=1;spawn();},2600);
 const t4=setTimeout(enter,6400);
 function spawn(){ /* stage 2→3: particles fade in at center, then glide home */
   EP.forEach((e,k)=>{e.heroDelay=k/Math.max(EP.length,1)*1.4;});
 }
 let born=0;
 (function tick(){
   if(heroStage===2)return;
   g.fillStyle='rgba(6,9,14,.28)';g.fillRect(0,0,c.width,c.height);
   if(i<tr.length-1)i+=Math.max(1,Math.round(tr.length/120));
   g.strokeStyle='#4fd1c5';g.lineWidth=3;g.beginPath();
   for(let k=0;k<=Math.min(i,tr.length-1);k++){const q=P(tr[k]);k?g.lineTo(q[0],q[1]):g.moveTo(q[0],q[1]);}
   g.stroke();
   if(heroStage>=1){ /* real trajectories dissolve into the flock */
     born=Math.min(EP.length,born+Math.ceil(EP.length/90));
     g.fillStyle='#56677a';
     const drawN=Math.min(born,2000);          // hero is an intro, not a benchmark
     for(let k=0;k<drawN;k++){const e=EP[k];e.heroDelay-=0.016;if(e.heroDelay>0)continue;
       const b=e.base[0];const u=Math.min(1,(1.4-Math.max(e.heroDelay,0))/1.4);
       const hx=c.width/2+(b[0]-0.5)*0.02, hy=c.height/2+(b[1]-0.5)*0.02;
       const gx=b[0]*c.width, gy=b[1]*c.height;
       g.globalAlpha=.5*u;g.beginPath();g.arc(hx+(gx-hx)*u,hy+(gy-hy)*u,3,0,7);g.fill();}
     g.globalAlpha=1;
   }
   requestAnimationFrame(tick);
 })();
 h.onclick=()=>{clearTimeout(t1);clearTimeout(t2);clearTimeout(t3);clearTimeout(t4);enter();};
 function enter(){heroStage=2;EP.forEach(e=>delete e.heroDelay);h.style.opacity=0;setTimeout(()=>h.remove(),1500);}
}

/* ================= The Murmur ================= */
function col(e,t){ // t: reveal progress 0..1
 if(t<=0)return HIDDEN;
 return mix(HIDDEN,COLORS[e.l]||COLORS.unknown,t);
}
function mix(a,b,t){const pa=parseInt(a.slice(1),16),pb=parseInt(b.slice(1),16);
 const r=(pa>>16)+(((pb>>16)-(pa>>16))*t),g=((pa>>8)&255)+((((pb>>8)&255)-((pa>>8)&255))*t),
       bl=(pa&255)+(((pb&255)-(pa&255))*t);
 return `rgb(${r|0},${g|0},${bl|0})`;}
function pos(e){
 const b=e.base; if(b.length<2||REDUCED)return b[0];
 const u=((T/60)/e.period+e.phase)%1;   // T ticks per frame; T/60 ≈ seconds
 const i=Math.min(Math.floor(u*(b.length-1)),b.length-2),f=u*(b.length-1)-i;
 return [b[i][0]+(b[i+1][0]-b[i][0])*f, b[i][1]+(b[i+1][1]-b[i][1])*f];
}
let frames=0,fpsT=performance.now();
function draw(){
 T+=REDUCED?0:1;
 if(revealT<(revealed?1:0))revealT=Math.min(revealT+(REDUCED?1:0.008),revealed?1:0);
 if(revealT>(revealed?1:0))revealT=Math.max(revealT-0.008,0);
 const W=cv.width,H=cv.height;
 cx.fillStyle='#06090e';cx.fillRect(0,0,W,H);
 cx.strokeStyle='rgba(50,70,90,.08)';
 for(let i=1;i<6;i++){cx.beginPath();cx.moveTo(W*i/6,0);cx.lineTo(W*i/6,H);cx.stroke();
   cx.beginPath();cx.moveTo(0,H*i/6);cx.lineTo(W,H*i/6);cx.stroke();}
 const drift=T>0;
 cx.fillStyle='#06090e';
 for(let i=0;i<EP.length;i++){
   const e=EP[i],p=drift?pos(e):e.base[0];
   const fl=REDUCED?0:(e.cv||0)*3.2;
   const x=p[0]*W+Math.sin(T*0.013+e.phase*40)*fl*DPR;
   const y=p[1]*H+Math.cos(T*0.011+e.phase*53)*fl*DPR;
   e.sx=x;e.sy=y;
   const inFocus=(motifSel<0||e.m===motifSel);
   const a=i===sel?1:(inFocus?0.82:0.07);
   cx.globalAlpha=a;
   const shade=(mode==='interaction'||contextShade)&&revealed&&e.ia&&
               (e.l==='siler'||e.l==='mimic');
   if(i===sel){cx.fillStyle='#ffffff';cx.beginPath();cx.arc(x,y,7*DPR,0,7);cx.fill();}
   else if(shade&&!e.ia.ha){          // hollow: no ants nearby (data, not style)
     cx.strokeStyle=col(e,revealT);cx.lineWidth=1.6*DPR;
     cx.beginPath();cx.arc(x,y,3.8*DPR,0,7);cx.stroke();
   }
   else{
     cx.fillStyle=i===sel?'#ffffff':col(e,revealT);
     cx.beginPath();cx.arc(x,y,(i===sel?7:3.6)*DPR,0,7);cx.fill();
   }
   if(i===sel){cx.strokeStyle=col(e,revealT);cx.lineWidth=2*DPR;
     cx.beginPath();cx.arc(x,y,11*DPR,0,7);cx.stroke();}
 }
 cx.globalAlpha=1;
 /* ---- query particles (Find Similar) ---- */
 if(QUERY){
   const hi=new Set(qNeighbors(QUERY.episodes[qSel]));
   const introU=REDUCED?1:Math.min(1,(performance.now()-qIntroT0)/2200);
   QUERY.episodes.forEach((qe,qi)=>{
     const [ux,uy]=qPos(qe);
     let x=ux*W,y=uy*H;
     if(qIntro>=0&&introU<1){        // glide from center to its real position
       const u=introU<0.55?0:((introU-0.55)/0.45);
       const ease=1-Math.pow(1-u,3);
       x=(W/2+(x-W/2)*ease); y=(H/2+(y-H/2)*ease);
     }
     qe._sx=x; qe._sy=y;
     const active=(qi===qSel);
     cx.globalAlpha=1;
     cx.strokeStyle='#ffffff'; cx.lineWidth=(active?3:2)*DPR;
     cx.beginPath(); cx.arc(x,y,(active?10:8)*DPR,0,7); cx.stroke();
     cx.fillStyle='#ffffff';
     cx.beginPath(); cx.arc(x,y,3*DPR,0,7); cx.fill();
     if(active){
       cx.setLineDash([4*DPR,4*DPR]); cx.strokeStyle='#ffffff77';
       cx.beginPath(); cx.arc(x,y,16*DPR,0,7); cx.stroke(); cx.setLineDash([]);
     }
   });
   // neighbor emphasis: reference particles near the active query glow
   EP.forEach((e,i)=>{
     if(hi.has(e.id)&&e.sx!==undefined){
       cx.globalAlpha=0.95; cx.strokeStyle='#fff'; cx.lineWidth=1.5*DPR;
       cx.beginPath(); cx.arc(e.sx,e.sy,6.5*DPR,0,7); cx.stroke();
     }
   });
   cx.globalAlpha=1;
 }
 frames++;const now=performance.now();
 if(now-fpsT>1500){document.getElementById('fps').textContent=
   `${Math.round(frames*1000/(now-fpsT))} fps · ${EP.length.toLocaleString()} particles`;frames=0;fpsT=now;}
 if(!document.hidden)requestAnimationFrame(draw);
}
document.addEventListener('visibilitychange',()=>{if(!document.hidden)requestAnimationFrame(draw);});

/* ---------- selection & region exploration ---------- */
cv.addEventListener('click',ev=>{
 if(document.getElementById('hero'))return;
 const r=cv.getBoundingClientRect(),mx=(ev.clientX-r.left)*DPR,my=(ev.clientY-r.top)*DPR;
 let best=-1,bd=1e9;
 for(let i=0;i<EP.length;i++){const e=EP[i];if(e.sx===undefined)continue;
   const d=(mx-e.sx)**2+(my-e.sy)**2;if(d<bd){bd=d;best=i;}}
 if(bd<(14*DPR)**2)select(best);
 else{ /* empty space: explore this region of movement space */
   const near=[];
   for(let i=0;i<EP.length;i++){const e=EP[i];if(e.sx===undefined)continue;
     const d=Math.hypot(mx-e.sx,my-e.sy);if(d<90*DPR)near.push([d,i]);}
   if(near.length>=3)showRegion(near.sort((a,b)=>a[0]-b[0]));
 }
});
function select(i){sel=i;fetchMeta(EP[i].id).then(m=>m&&showMovement(EP[i],m));openPanel();}
function showById(id){const i=EP.findIndex(x=>x.id===id);if(i>=0)select(i);}
addEventListener('keydown',ev=>{
 if(ev.target.tagName==='INPUT'||ev.target.tagName==='TEXTAREA')return;
 if(ev.key==='Escape'){sel=-1;panel.classList.remove('open');}
 if(ev.key.toLowerCase()==='r')toggleReveal();
 if(ev.key.toLowerCase()==='c'&&sel>=0)compare(sel);
});

/* ================= lazy meta ================= */
const metaCache={};
function fetchMeta(id){
 if(metaCache[id])return Promise.resolve(metaCache[id]);
 return fetch(`meta/${id}.json`).then(r=>r.json()).then(m=>{metaCache[id]=m;return m;})
        .catch(()=>null);
}

/* ================= movement detail ================= */
function labelChip(e){
 if(!revealed)return `<span class="lbl" style="color:#8fa8bc">movement</span>`;
 return `<span class="lbl" style="color:${COLORS[e.l]}">${NAMES[e.l]||e.l}</span>`;
}
function ctxText(c){
 if(!c)return null;
 if(!c.has_ant)return `Alone — no ants within ${Math.round(c.radius)} ${c.units} during this episode`;
 return `Ants nearby — mean ${c.n_ants_mean} within ${Math.round(c.radius)} ${c.units}`+
   (c.nearest_ant_dist_min!=null?` · nearest ${c.nearest_ant_dist_min} ${c.units} at closest`:'')+
   (c.ant_activity_mean!=null?` · local ant activity ${c.ant_activity_mean} ${c.units}/s`:'');
}
function showMovement(e,m){
 const nn=(m.neighbors||[]);
 const other=nn.find(n=>revealed&&n.label!==e.l)||null;
 const ct=ctxText(m.context);
 const contrast=nn.find(n=>{const ne=IDX[n.episode_id];
   return ne&&ne.ia&&m.context&&ne.ia.ha!==((m.context.has_ant)?1:0);});
 panel.innerHTML=`
  <div style="display:flex;align-items:center;gap:8px">
   ${labelChip(e)}
   <span class="muted">${e.d.toFixed(1)}s · motif M${e.m} · ${e.st!=='unreviewed'?'QC: '+e.st:'unreviewed'}</span>
   <span class="sp" style="flex:1"></span>
   <button class="btn" style="padding:3px 10px" onclick="closePanel()">esc</button></div>
  <div class="muted">${m.species_detail?'· '+m.species_detail+' · ':''}${m.video_id} · frames ${m.frames[0]}–${m.frames[1]}${m.sampling.site?' · '+esc(m.sampling.site):''}</div>
  ${ct?`<div class="ctxline">⇄ <b>Social context:</b> ${ct}</div>`:''}
  <div class="scenewrap"><img class="clip" id="mainclip" alt="movement clip"></div>
  <div class="muted">speed / turn / moving — real time series (replay below is the real trajectory, time-normalized)</div>
  <div id="rows">${sparkRow('speed','#4fd1c5')}${sparkRow('turn','#e8a13c')}${sparkRow('moving','#b794f4')}${
    m.scene?sparkRow('nearest ant','#e8a13c')+sparkRow('ants in radius','#b794f4'):''}</div>
  <div style="display:flex;gap:8px;margin:8px 0;flex-wrap:wrap">
    <button class="btn" id="playBtn" style="padding:4px 14px">⏸ pause</button>
    ${nn.length?`<button class="btn" onclick="compare()">⇄ compare nearest</button>`:''}
    ${contrast?`<button class="btn" onclick="compareWith('${contrast.episode_id}')">⇄ compare: same movement, ${contrast.ia&&contrast.ia.ha?'with':'without'} ants</button>`:''}
  </div>
  <div style="margin-top:6px;color:var(--accent)">Similar movements <span class="muted">— nearest in ${META.features.length}-D feature space, not screen distance</span></div>
  <div id="nnlist">${nn.map((n,k)=>{
     const ne=IDX[n.episode_id]||{};
     const name=revealed?`<span class="lbl" style="color:${COLORS[ne.l]}">${NAMES[ne.l]||ne.l}</span>`
                        :`<span class="lbl" style="color:#8fa8bc">Movement ${k+1}</span>`;
     const ctx2=revealed&&ne.ia?` · ${ne.ia.ha?'ants near':'no ants'}`:'';
     return `<div class="nbr" onclick="showById('${n.episode_id}')">
       <canvas width="192" height="120" data-traj="${n.episode_id}"></canvas>
       <div>${name}<div class="muted">similarity ${n.similarity.toFixed(2)} · ${(ne.d||0).toFixed(1)}s${ctx2}</div></div></div>`;
   }).join('')}</div>
  <div class="muted" style="margin-top:8px">Provenance — every claim traces back here</div>
  <div class="prov">${esc(m.provenance_chain.join('\n'))}</div>`;
 openPanel();
 // clip: real video when served; graceful in-browser replay otherwise
 const img=panel.querySelector('#mainclip');
 let mainCv=null;
 if(serverMode&&m.clip_kind==='video'){img.src=`clip/${e.id}.gif`;img.onerror=()=>{mainCv=replayFallback(img,m,e);};}
 else mainCv=replayFallback(img,m,e);
 drawSeries(m);
 if(m.scene){
   // full-scene overlay: focal + concurrent animals, from the same source video
   const ov=document.createElement('canvas');ov.className='overlay';ov.width=720;ov.height=400;
   panel.querySelector('.scenewrap').appendChild(ov);
   m._sceneCv=ov;
   if(revealed)markNeighbors(m);
 }
 startSyncReplay([m],[panel.querySelector('#rows')],panel.querySelector('#playBtn'),
   prog=>{if(m.scene&&m._sceneCv)drawScene(m._sceneCv,m,prog,revealed);});
 nn.forEach(n=>{const ne=IDX[n.episode_id];if(ne)fetchMeta(n.episode_id).then(nm=>{
    if(!nm)return;const c=panel.querySelector(`canvas[data-traj="${n.episode_id}"]`);
    if(c)drawStaticReplay(c,nm.trajectory||[],revealed?(COLORS[ne.l]||'#4fd1c5'):'#56677a');});});
}
function markNeighbors(m){
 // re-tint scene thumbnails/legend after reveal — handled by drawScene colors
 if(m._sceneCv)drawScene(m._sceneCv,m,0,true);
}
/* full-scene overlay: real trajectories of all concurrent animals,
   revealed up to the shared playhead. Colors = species (post-reveal). */
function drawScene(c,m,prog,showLabels){
 const g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 if(!m.trajectory||m.trajectory.length<2)return;
 let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
 m.trajectory.forEach(p=>{x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1]);});
 (m.scene.neighbors||[]).forEach(nb=>{
   (showLabels?([COLORS[nb.label]||'#56677a']):['#56677a']).forEach(col2=>{
     g.strokeStyle=col2;g.lineWidth=2;g.globalAlpha=.85;g.beginPath();
     let started=false;
     nb.frames.forEach((f,k)=>{
       const curf=m.frames[0]+prog*(m.frames[1]-m.frames[0]);
       if(f>curf)return;
       const px=16+(nb.xs[k]-x0)/(x1-x0+1e-9)*(c.width-32);
       const py=c.height-16-(nb.ys[k]-y0)/(y1-y0+1e-9)*(c.height-32);
       started?g.lineTo(px,py):g.moveTo(px,py);started=true;});
     g.stroke();g.globalAlpha=1;});
 });
 // focal trajectory up to playhead, white
 const curf=m.frames[0]+prog*(m.frames[1]-m.frames[0]);
 g.strokeStyle='#ffffff';g.lineWidth=3;g.beginPath();let st=false;
 m.trajectory.forEach((p,k)=>{const f=m.frames[0]+(k/(m.trajectory.length-1))*(m.frames[1]-m.frames[0]);
   if(f>curf)return;const px=16+(p[0]-x0)/(x1-x0+1e-9)*(c.width-32);
   const py=c.height-16-(p[1]-y0)/(y1-y0+1e-9)*(c.height-32);st?g.lineTo(px,py):g.moveTo(px,py);st=true;});
 g.stroke();
 const last=m.trajectory[Math.max(0,Math.min(m.trajectory.length-1,
   Math.round((curf-m.frames[0])/(m.frames[1]-m.frames[0])*(m.trajectory.length-1))))];
 if(last){const px=16+(last[0]-x0)/(x1-x0+1e-9)*(c.width-32);
   const py=c.height-16-(last[1]-y0)/(y1-y0+1e-9)*(c.height-32);
   g.fillStyle='#fff';g.beginPath();g.arc(px,py,5,0,7);g.fill();}
}
function replayFallback(img,m,e){
 // in-browser trajectory replay (the real path; used when no server media).
 // startSyncReplay picks this canvas up and animates it on the shared clock.
 const c=document.createElement('canvas');c.className='clip clipcv';c.width=720;c.height=400;
 img.replaceWith(c);
 return c;
}
function sparkRow(lab,c){
 return `<div class="rowwrap"><span class="rowlab">${lab}</span><div>
   <canvas class="spark" width="640" height="80" data-lab="${lab}" data-c="${c}"></canvas></div></div>`;}
function sparkArr(m,lab){
 if(lab==='nearest ant')return (m.scene&&m.scene.nearest_ant_dist||[]).map(v=>v==null?NaN:v);
 if(lab==='ants in radius')return (m.scene&&m.scene.n_ants_within||[]);
 return (m.series||{})[lab]||[0];}
function drawSeries(m){
 panel.querySelectorAll('canvas.spark').forEach(c=>{
   const arr=sparkArr(m,c.dataset.lab);c._arr=arr;
   const g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
   const vs=arr.filter(v=>v!=null&&isFinite(v));
   const mx=Math.max(...vs,1e-9);
   g.strokeStyle=c.dataset.c;g.lineWidth=2.5;g.beginPath();let st=false;
   arr.forEach((v,i)=>{if(v==null||!isFinite(v)){st=false;return;}
     const x=i/(arr.length-1||1)*c.width,y=c.height-6-(v/mx)*(c.height-12);
     st?g.lineTo(x,y):g.moveTo(x,y);st=true;});
   g.stroke();});
}
/* synchronized replay: shared playhead across trajectory canvas + sparklines */
function startSyncReplay(metas,rowsEls,playBtn,onFrame){
 let playing=true,t0=performance.now(),prog=0;
 const dur=Math.max(metas[0].duration_s,1)*1000;
 const rowsEl=rowsEls&&rowsEls[0];
 if(playBtn)playBtn.onclick=()=>{playing=!playing;
   if(playing)t0=performance.now()-prog*dur;
   playBtn.textContent=playing?'⏸ pause':'▶ play';};
 const mainCv=panel.querySelector('.clip canvas.clipcv')||panel.querySelector('.scenewrap .overlay');
 function frame(now){
   if(!rowsEl||!document.contains(rowsEl))return;   // panel replaced → stop
   if(playing)prog=((now-t0)/dur)%1;
   const trajCv=panel.querySelector('.clip canvas.clipcv');
   if(trajCv&&document.contains(trajCv))
     drawReplayProgress(trajCv,metas[0].trajectory||[],prog,revealed?(COLORS[metas[0].label]||'#4fd1c5'):'#4fd1c5');
   if(onFrame)onFrame(prog);
   panel.querySelectorAll('canvas.spark').forEach(c=>{
     const arr=c._arr||[0];const g=c.getContext('2d');
     const vs=arr.filter(v=>v!=null&&isFinite(v));
     const mx=Math.max(...vs,1e-9);g.clearRect(0,0,c.width,c.height);
     g.strokeStyle=c.dataset.c;g.lineWidth=2.5;g.beginPath();let st=false;
     arr.forEach((v,i)=>{if(v==null||!isFinite(v)){st=false;return;}
       const x=i/(arr.length-1||1)*c.width,y=c.height-6-(v/mx)*(c.height-12);
       st?g.lineTo(x,y):g.moveTo(x,y);st=true;});
     g.stroke();
     const px=prog*c.width;g.strokeStyle='#ffffff88';g.beginPath();g.moveTo(px,0);g.lineTo(px,c.height);g.stroke();});
   requestAnimationFrame(frame);
 }
 requestAnimationFrame(frame);
}
function drawStaticReplay(c,traj,color){
 const g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
 if(!traj||traj.length<2){g.fillStyle='#31404f';g.font='20px system-ui';g.fillText('no path',20,40);return;}
 drawReplayOn(c.getContext('2d'),c,traj,color,1);
}
function drawReplayOn(g,c,traj,color,prog){
 let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
 traj.forEach(p=>{x0=Math.min(x0,p[0]);x1=Math.max(x1,p[0]);y0=Math.min(y0,p[1]);y1=Math.max(y1,p[1]);});
 const P=p=>[16+(p[0]-x0)/(x1-x0+1e-9)*(c.width-32), c.height-16-(p[1]-y0)/(y1-y0+1e-9)*(c.height-32)];
 const n=Math.max(2,Math.floor(traj.length*prog));
 g.strokeStyle=color;g.lineWidth=3;g.beginPath();
 for(let k=0;k<n;k++){const q=P(traj[k]);k?g.lineTo(q[0],q[1]):g.moveTo(q[0],q[1]);}
 g.stroke();
 const h=P(traj[Math.min(n,traj.length)-1]);
 g.fillStyle='#fff';g.beginPath();g.arc(h[0],h[1],5,0,7);g.fill();
}
function drawReplayProgress(cvEl,traj,prog,color){
 const g=cvEl.getContext('2d');g.fillStyle='#04070b';g.fillRect(0,0,cvEl.width,cvEl.height);
 if(!traj||traj.length<2){g.fillStyle='#31404f';g.font='24px system-ui';
   g.fillText('trajectory unavailable',24,60);return;}
 drawReplayOn(g,cvEl,traj,color,prog);
}
window.closePanel=function(){panel.classList.remove('open');sel=-1;};
window.showById=showById;

/* ================= side-by-side synchronized comparison ================= */
window.compareWith=async function(targetId){
 if(sel<0)return;
 const e=EP[sel];const m=await fetchMeta(e.id);if(!m)return;
 const tid=targetId||((m.neighbors||[])[0]||{}).episode_id;
 if(!tid)return;
 const m2=await fetchMeta(tid);if(!m2)return;
 const e2=IDX[tid];
 const nb=((m.neighbors||[]).find(n=>n.episode_id===tid))||{similarity:0,dist:0};
 const nameOf=x=>revealed?`<span class="lbl" style="color:${COLORS[x.l]}">${NAMES[x.l]||x.l}</span>`
                         :`<span class="lbl" style="color:#8fa8bc">movement</span>`;
 const ctxOf=mm=>{const c=ctxText(mm.context);return c?`<div class="ctxline" style="font-size:11px">${c}</div>`:'';};
 panel.innerHTML=`
  <div style="display:flex;align-items:center;gap:8px">
    <button class="btn" style="padding:3px 10px" onclick="showById('${e.id}')">← back</button>
    <span class="muted">side-by-side · each at its own real pace · shared playhead</span></div>
  <div class="split" style="margin-top:10px">
   <div>${nameOf(e)}<div class="muted">${e.d.toFixed(1)}s</div><img class="clip" id="cmpA">${ctxOf(m)}</div>
   <div>${nameOf(e2)}<div class="muted">${e2.d.toFixed(1)}s</div><img class="clip" id="cmpB">${ctxOf(m2)}</div>
  </div>
  <div class="muted">A vs B — similarity ${(nb.similarity).toFixed(2)} in feature space (d=${nb.dist.toFixed(2)}).
   Watch the same playhead: where the speeds and paths agree, that is what “similar” means.</div>
  <div id="cmpRowsA"></div><div id="cmpRowsB"></div>
  <div style="display:flex;gap:8px;margin:8px 0"><button class="btn" id="cmpPlay" style="padding:4px 14px">⏸ pause</button></div>
  <div class="prov">A: ${esc(m.provenance_chain.join('\n'))}\n---\nB: ${esc(m2.provenance_chain.join('\n'))}</div>`;
 openPanel();
 if(serverMode&&m.clip_kind==='video'){panel.querySelector('#cmpA').src=`clip/${e.id}.gif`;}
 else{const img=panel.querySelector('#cmpA');img.onerror=()=>{};replayFallback(img,m,e);}
 if(serverMode&&m2.clip_kind==='video'){panel.querySelector('#cmpB').src=`clip/${e2.id}.gif`;}
 else{replayFallback(panel.querySelector('#cmpB'),m2,e2);}
 // sparkline rows for both
 for(const [mm,host] of [[m,'cmpRowsA'],[m2,'cmpRowsB']]){
   document.getElementById(host).innerHTML=sparkRow('speed','#4fd1c5')+sparkRow('moving','#b794f4');
   host&&document.getElementById(host).querySelectorAll('canvas.spark').forEach(c=>{
     const arr=(mm.series||{})[c.dataset.lab]||[0];c._arr=arr;
     const g=c.getContext('2d'),mx=Math.max(...arr,1e-9);
     g.strokeStyle=c.dataset.c;g.lineWidth=2.5;g.beginPath();
     arr.forEach((v,i)=>{const x=i/(arr.length-1||1)*c.width,y=c.height-6-(v/mx)*(c.height-12);
       i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();});
 }
 // shared playhead over both canvases
 const cA=document.createElement('canvas'),cB=document.createElement('canvas');
 [cA,cB].forEach(c=>{c.width=360;c.height=200;c.style.width='100%';c.style.background='#04070b';c.style.borderRadius='8px';});
 document.getElementById('cmpRowsA').prepend(cA);
 document.getElementById('cmpRowsB').prepend(cB);
 let playing=true,t0=performance.now(),pauseProg=0;
 const dur=Math.max(Math.min(m.duration_s,m2.duration_s),1)*1000;
 document.getElementById('cmpPlay').onclick=()=>{playing=!playing;
   if(playing)t0=performance.now()-pauseProg*dur;else pauseProg=0;};
 (function loop(now){
   if(!document.getElementById('cmpRowsA'))return;
   if(playing)pauseProg=((now-t0)/dur)%1;
   const prog=pauseProg;
   drawReplayProgress(cA,m.trajectory||[],prog,'#4fd1c5');
   drawReplayProgress(cB,m2.trajectory||[],prog,'#e8a13c');
   document.querySelectorAll('#cmpRowsA canvas.spark, #cmpRowsB canvas.spark').forEach(c=>{
     if(!c._base){c._base=document.createElement('canvas');c._base.width=c.width;c._base.height=c.height;
       c._base.getContext('2d').drawImage(c,0,0);}
     const g=c.getContext('2d');
     g.clearRect(0,0,c.width,c.height);g.drawImage(c._base,0,0);
     g.strokeStyle='#ffffff88';g.lineWidth=2;g.beginPath();
     g.moveTo(prog*c.width,0);g.lineTo(prog*c.width,c.height);g.stroke();});
   requestAnimationFrame(loop);
 })(performance.now());
};

/* ================= region ================= */
function showRegion(near){
 const ids=near.slice(0,50).map(x=>x[1]);
 const lab={};ids.forEach(i=>{const e=EP[i];lab[e.l]=(lab[e.l]||0)+1;});
 sel=-1;
 const rep=ids.slice(0,6);
 panel.innerHTML=`
  <div style="display:flex;align-items:center;gap:8px"><span style="color:var(--accent)">Region · ${ids.length} movements</span>
   <span class="sp" style="flex:1"></span><button class="btn" style="padding:3px 10px" onclick="closePanel()">esc</button></div>
  <div class="muted">You clicked a neighborhood of movement space — these episodes live here because their
   kinematics agree, ${revealed?'regardless of who they belong to:':'whatever they are.'}</div>
  ${revealed?`<div style="margin:8px 0">${Object.entries(lab).sort((a,b)=>b[1]-a[1]).map(([l,c])=>
     `<span class="lbl" style="color:${COLORS[l]};margin-right:6px">${NAMES[l]||l} ${c}</span>`).join('')}</div>`
   :`<div class="muted" style="margin:8px 0">Reveal species to see who lives here.</div>`}
  <div id="replist"></div>`;
 openPanel();
 const host=panel.querySelector('#replist');
 rep.forEach(async (i,k)=>{
   const e=EP[i],m=await fetchMeta(e.id);if(!m)return;
   const d=document.createElement('div');d.className='nbr';
   d.innerHTML=`<canvas width="192" height="120"></canvas>
     <div>${labelChip(e)}<div class="muted">${e.d.toFixed(1)}s · M${e.m} · ${(e.cv||0).toFixed(2)} cv</div></div>`;
   d.onclick=()=>select(i);host.appendChild(d);
   drawStaticReplay(d.querySelector('canvas'),m.trajectory||[],revealed?(COLORS[e.l]||'#4fd1c5'):'#56677a');
 });
}

window.compare=function(){return compareWith();};

/* ================= explore drawer ================= */
document.getElementById('exploreBtn').onclick=()=>{showExplore('dictionary');};
function tab(name,active){
 const names={dictionary:'☰ Motion Dictionary',fingerprint:'⌇ Mimicry fingerprint',
   river:'≈ Behavior River',interaction:'⇄ Interaction',data:'⌸ Data & provenance'};
 return `<span class="chip ${active?'on':''}" onclick="showExplore('${name}')">${names[name]}</span>`;}
window.showExplore=function(tabName){
 openPanel();
 renderExplore(tabName);
};
function openPanel(){panel.classList.add('open');}
function renderExplore(tabName){
 if(tabName==='dictionary')return renderDictionary();
 if(tabName==='fingerprint')return renderFingerprint();
 if(tabName==='river')return renderRiver();
 if(tabName==='interaction')return renderInteraction();
 if(tabName==='data')return renderData();
}
function tabbar(active){
 let t=tab('dictionary',active==='dictionary')+tab('fingerprint',active==='fingerprint')
      +tab('river',active==='river');
 if(META.interaction&&META.interaction.available)
   t+=tab('interaction',active==='interaction');
 return t+tab('data',active==='data');
}
function renderInteraction(){
 const I=META.interaction||{available:false};
 const cmp=I.comparison||{};
 const rows=(cmp.response||[]);
 panel.innerHTML=`
  <h1>Interaction — Siler × Ant</h1>
  <div class="tabbar">${tabbar('interaction')}</div>
  <div class="muted">Does Siler movement depend on the presence of ants?
   Contrasts below compare Siler episodes <b>with ants nearby</b> vs <b>without</b>
   (within the analysis radius), with an episode-shuffle null. These are
   <b>descriptive, predictive associations — not causal effects</b>; proximity is not interaction.</div>
  ${rows.length?`<table class="cmp"><tr><th>radius</th><th>n with/without</th>${rows[0].speed?'<th>speed Δ</th><th>p</th>':''}${rows[0].d_ant?'<th>D_ant Δ</th><th>p</th>':''}</tr>
   ${rows.map(r=>`<tr><td>${r.radius}${r.n_with!==undefined?'':''}</td><td>${r.n_with} / ${r.n_without}</td>
     ${r.speed?`<td>${r.speed.observed_median_diff}</td><td>${r.speed.p_perm}</td>`:''}
     ${r.d_ant?`<td>${r.d_ant.observed_median_diff}</td><td>${r.d_ant.p_perm}</td>`:''}</tr>`).join('')}
  </table>
  <div class="muted">Δ = median(with ants) − median(without). p = episode-shuffle permutation p.
   Distance-response across radii guards against one arbitrary threshold.</div>`
  :`<div class="muted">Not enough episodes in both contexts yet — annotate more Siler episodes (workbench) and re-run <b>motionscape interact</b>.</div>`}
  ${cmp.d_ant&&cmp.d_ant.note?`<div class="muted" style="margin-top:8px">${esc(cmp.d_ant.note)}</div>`:''}
  <div class="ctxline" style="margin-top:10px"><b>View:</b> Interaction mode shades Siler particles by their real social context —
   hollow = no ants nearby during the episode (from the same source video).</div>
  <button class="btn" id="shadeBtn" style="margin:6px 0">${contextShade?'◌ stop shading':'◌ shade Siler by ant context'}</button>
  <div class="prov">${esc(((I.provenance||{}).parameters||{}).labels_note||'')}
\npairs: ${I.n_pairs} · scene windows: ${I.n_scene_windows}${I.n_velocity_excluded?` · velocity-only episodes excluded: ${I.n_velocity_excluded}`:''}</div>`;
 panel.querySelector('#shadeBtn').onclick=function(){
   contextShade=!contextShade;
   this.textContent=contextShade?'◌ stop shading':'◌ shade Siler by ant context';
 };
}
function renderDictionary(){
 const counts={};EP.forEach(e=>counts[e.m]=(counts[e.m]||0)+1);
 const motifs=Object.keys(counts).map(Number).sort((a,b)=>a-b);
 const ann=(META.motif_annotations||{});
 const sel_m=motifSel>=0?motifSel:motifs[0];
 const reps=EP.filter(e=>e.m===sel_m);
 const lab={};reps.forEach(e=>lab[e.l]=(lab[e.l]||0)+1);
 const humanAnn=ann[String(sel_m)];
 panel.innerHTML=`
  <h1>Motion Dictionary</h1>
  <div class="muted">Motifs are discovered by the machine (k-means in feature space).
   They have numbers, not names — naming is a human act, after watching.</div>
  <div class="tabbar">${tabbar('dictionary')}</div>
  <div style="margin:6px 0">${motifs.map(m=>`<span class="chip ${m===sel_m?'on':''}" onclick="motifClick(${m})">M${m} · ${counts[m]}</span>`).join('')}</div>
  <div style="margin:10px 0 4px;color:var(--accent);font-size:15px">M${sel_m} ${humanAnn?`— “${esc(humanAnn.name)}” <span class="muted">(${esc(humanAnn.annotator)})</span>`:''}</div>
  <div style="display:flex;gap:6px;margin:8px 0">
   <input class="txt" id="motifName" placeholder="name this movement after watching…" value="${humanAnn?esc(humanAnn.name):''}">
   <button class="btn" id="motifSave" style="padding:4px 12px">save</button></div>
  <textarea class="txt" id="motifNotes" rows="2" placeholder="notes (optional)">${humanAnn?esc(humanAnn.notes||''):''}</textarea>
  <div class="muted" style="margin-top:6px">${reps.length} of ${EP.length} movements (${(reps.length/EP.length*100).toFixed(1)}%)
   ${revealed?`— occupancy: ${Object.entries(lab).sort((a,b)=>b[1]-a[1]).map(([l,c])=>`${NAMES[l]||l} ${(c/reps.length*100).toFixed(0)}%`).join(' · ')}`
             :'— <b>Reveal species</b> to see which animals do this.'}</div>
  ${revealed?Object.entries(lab).sort((a,b)=>b[1]-a[1]).map(([l,c])=>`
    <div class="fprow"><span class="name" style="color:${COLORS[l]}">${NAMES[l]||l}</span>
      <span class="track"><span class="fill" style="width:${(c/reps.length*100)|0}%;background:${COLORS[l]}"></span></span>
      <span class="muted">${(c/reps.length*100).toFixed(1)}%</span></div>`).join(''):''}
  <div id="replist" style="margin-top:8px"></div>`;
 const host=panel.querySelector('#replist');
 reps.slice(0,6).forEach(async e=>{
   const m=await fetchMeta(e.id);if(!m)return;
   const d=document.createElement('div');d.className='nbr';
   d.innerHTML=`<canvas width="192" height="120"></canvas><div>${labelChip(e)}<div class="muted">${e.d.toFixed(1)}s · ${(e.cv||0).toFixed(2)} cv</div></div>`;
   d.onclick=()=>select(EP.indexOf(e));host.appendChild(d);
   drawStaticReplay(d.querySelector('canvas'),m.trajectory||[],revealed?(COLORS[e.l]||'#4fd1c5'):'#56677a');
 });
 panel.querySelector('#motifSave').onclick=()=>{
   const name=panel.querySelector('#motifName').value.trim();
   const notes=panel.querySelector('#motifNotes').value.trim();
   const annotator=(localStorage.getItem('annotator')||'anonymous');
   if(serverMode){fetch('api/motifs',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({motif:sel_m,name,notes,annotator})}).then(r=>r.json()).then(o=>{
       if(o.saved){META.motif_annotations=META.motif_annotations||{};META.motif_annotations[String(sel_m)]=o.saved;
         toast('saved ✓');renderDictionary();}});
   }else{META.motif_annotations=META.motif_annotations||{};
     META.motif_annotations[String(sel_m)]={name,notes,annotator,timestamp:new Date().toISOString(),local:true};
     toast('saved locally — run motionscape serve to persist');renderDictionary();}
 };
}
window.motifClick=function(m){motifSel=(motifSel===m)?-1:m;renderDictionary();};
const FP_DIM_NAMES={speed_dynamics:"Speed dynamics",speed_intermittency:"Intermittency",
 stop_go_rhythm:"Stop–go rhythm",turning:"Turning dynamics",path_shape:"Trajectory geometry",
 trajectory_space:"Trajectory space"};
const FP_FUTURE=["Pose (foreleg-I ↔ antennae)","Leg-I dynamics","Motif occupancy","Behavioral grammar"];
function renderFingerprint(){
 const fp=(META.fingerprint&&META.fingerprint.fingerprint)||{};
 const prov=(META.fingerprint&&META.fingerprint.provenance)||{};
 panel.innerHTML=`
  <h1>Behavioral Mimicry Fingerprint</h1>
  <div class="muted">Per-dimension overlap (Bhattacharyya coefficient, 0–1) between Siler and ant
   episode distributions. <b style="color:#8fa8bc">Deliberately not one number</b> — mimicry is a profile, not a score.</div>
  <div class="tabbar">${tabbar('fingerprint')}</div>
  ${!revealed?`<div class="muted" style="margin:10px 0">This panel describes specific animals — press <b>Reveal species</b> first.</div>`:
  Object.entries(fp).length?Object.entries(fp).map(([d,v])=>`
    <div class="fprow"><span class="name">${FP_DIM_NAMES[d]||d}</span>
      <span class="track"><span class="fill" style="width:${(v*100)|0}%"></span></span>
      <span class="muted">${(+v).toFixed(2)}</span></div>`).join('')
   :'<div class="muted">Not enough reviewed episodes on both sides yet.</div>'}
  ${revealed?`<div class="muted" style="margin:10px 0">Not yet computed (next phases):</div>
    ${FP_FUTURE.map(f=>`<div class="fprow future"><span class="name">${f}</span>
      <span class="track"><span class="fill" style="width:0%"></span></span>
      <span class="muted">—</span></div>`).join('')}`:''}
  <div class="prov">n_siler=${prov.parameters?.n_siler??'—'} n_ant=${prov.parameters?.n_ant??'—'}
 model=${prov.model_name||'—'} @ ${prov.timestamp||'—'}</div>
  <div class="muted" style="margin-top:8px">2-D projection ≠ statistical evidence. Overlaps here summarize the
   same features the space was built from; formal tests belong in the analysis notebooks.</div>`;
}
function renderRiver(){
 panel.innerHTML=`
  <h1>Behavior River</h1>
  <div class="tabbar">${tabbar('river')}</div>
  ${!revealed?'<div class="muted" style="margin:10px 0">The river flows in species colors — press <b>Reveal species</b> first.</div>':''}
  <canvas id="river" width="760" height="240" style="width:100%;background:#0c141c;border-radius:8px"></canvas>
  <div class="muted">Species movement volume across the day (hour of day when timestamps exist).
   Rhythmic differences between groups are themselves testable questions.</div>`;
 if(revealed)drawRiver();
}
function drawRiver(){
 const withH=EP.filter(e=>e.hour!=null);
 const useH=withH.length>EP.length/2;
 const bins=24,keys=[...new Set(EP.map(e=>e.l))];
 const grid=Object.fromEntries(keys.map(k=>[k,new Array(bins).fill(0)]));
 EP.forEach((e,i)=>{const b=useH?Math.min(bins-1,Math.max(0,Math.floor(e.hour||0)))
   :Math.min(bins-1,Math.floor(i/EP.length*bins));grid[e.l][b]++;});
 const c=document.getElementById('river'),g=c.getContext('2d');
 const mx=Math.max(...keys.flatMap(k=>grid[k]),1);
 const rowH=(c.height-40)/keys.length;
 keys.forEach((k,ki)=>{g.fillStyle=COLORS[k]||COLORS.unknown;g.globalAlpha=.85;
   grid[k].forEach((v,b)=>{if(v>0)g.fillRect(b*(c.width/bins)+2,c.height-30-ki*rowH-rowH,
     c.width/bins-4,Math.max(v/mx*rowH,2));});});
 g.globalAlpha=1;g.fillStyle='#54687c';g.font='15px system-ui';g.textAlign='center';
 for(let b=0;b<bins;b+=4)g.fillText(useH?`${b}:00`:`#${b}`,b*(c.width/bins)+(c.width/bins)/2,c.height-8);
 keys.slice().reverse().forEach((k,ki)=>{g.fillStyle=COLORS[k];g.textAlign='left';
   g.fillText((NAMES[k]||k).split(' ')[0],8,20+ki*16);});
}
function renderData(){
 const h=META.hierarchy,tree=h.tree||{};
 panel.innerHTML=`
  <h1>Data & provenance</h1>
  <div class="tabbar">${tabbar('data')}</div>
  <div class="muted">Sampling hierarchy — episodes are <b>not</b> independent replicates. Statistics must
   respect site → session → video → episode (hierarchical bootstrap / mixed models).</div>
  <div style="margin:8px 0;font-size:12px">
   ${h.n_episodes} episodes · ${h.n_videos} videos · ${h.n_sessions} sessions · ${h.n_sites} site(s)</div>
  <div class="prov">${Object.entries(tree).map(([s,sess])=>
    `${s}\n`+Object.entries(sess).map(([ss,vids])=>
      `  ${ss}\n`+Object.entries(vids).map(([v,n])=>`    ${v} — ${n} eps`).join('\n')).join('\n')).join('\n')}</div>
  <div class="muted" style="margin-top:10px">Behavioral space</div>
  <div class="prov">${esc(META.embedding.model)} · blind to biological labels: ${META.embedding.blind_to_labels}
\n${esc(META.embedding.note)}</div>
  <div class="muted" style="margin-top:10px">Run provenance</div>
  <div class="prov">${esc(JSON.stringify(META.provenance,null,1).slice(0,900))}</div>
  <div class="muted" style="margin-top:10px">Atlas build</div>
  <div class="prov">MOTIONSCAPE v${META.motionscape_version} · built ${META.generated_utc} in ${META.build_seconds}s
\nserver media: ${serverMode?'on (clips generated on demand)':'off — run: python -m motionscape serve'}</div>`;
}
function toast(t){const e=document.createElement('div');e.textContent=t;
 e.style.cssText='position:fixed;bottom:70px;left:50%;transform:translateX(-50%);background:#14303a;color:#c8f2ec;padding:6px 16px;border-radius:16px;font-size:13px;z-index:30;transition:opacity .4s';
 document.body.appendChild(e);setTimeout(()=>e.style.opacity=0,1400);setTimeout(()=>e.remove(),1900);}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

/* ================= Find Similar ================= */
document.getElementById('findBtn').onclick=()=>showQuery();
const RATE_LABELS=[["very","Very similar"],["similar","Similar"],["weak","Weakly similar"],["not","Not similar"]];
function showQuery(){
 if(!QUERY&&!((window.__info||{}).find_similar_upload))
   { showQueryHelp(); return; }
 if(!QUERY){ showUpload(); return; }
 qRender();
}
function showQueryHelp(){
 panel.innerHTML=`<h1>Find Similar</h1>
  <div class="tabbar">${tabbar('data')}</div>
  <div class="muted">Drop in a video and see what moves like it.</div>
  <div class="prov">python -m motionscape find-similar VIDEO.mp4 --query-id my_query \
    --reference motionscape_runs/reference --atlas &lt;atlas_dir&gt;</div>
  <div class="muted" style="margin-top:8px">Then reload this page — your movement enters the atlas.
   Or start the server with --reference to upload from here.</div>`;
 openPanel();
}
function showUpload(){
 panel.innerHTML=`<h1>Find Similar</h1>
  <div class="muted">Upload a video: detection → episodes → behavior encoding →
   your movement enters the atlas and nearby movements light up.</div>
  <div id="uploadBox">
   <label class="btn" style="display:inline-block">Choose a video
    <input type="file" id="qfile" accept="video/*"></label>
   <div class="muted" style="margin-top:6px">processed locally — never leaves this machine</div>
  </div>
  <div id="qprog" class="muted"></div>
  <div class="muted" style="margin-top:8px">Tiny-animal footage: lower the detection threshold / min area.</div>
  <div style="display:flex;gap:6px;margin-top:8px">
    <label class="muted">threshold <input class="txt" id="dTh" value="26" style="width:60px"></label>
    <label class="muted">min area <input class="txt" id="dAr" value="2" style="width:60px"></label>
  </div>`;
 openPanel();
 const f=panel.querySelector('#qfile');
 f.onchange=()=>{
   const file=f.files[0]; if(!file)return;
   const det={threshold:parseInt(panel.querySelector('#dTh').value)||26,
              min_area:parseInt(panel.querySelector('#dAr').value)||2, morph_open_k:0};
   panel.querySelector('#qprog').textContent='processing… (detection → episodes → encoding)';
   fetch('api/find-similar',{method:'POST',body:file,
     headers:{'X-Filename':file.name,'Content-Type':'application/octet-stream',
              'X-Detector-Params':JSON.stringify(det)}}).then(r=>r.json()).then(o=>{
     if(!o.started)panel.querySelector('#qprog').textContent='upload failed: '+(o.error||'?');
   });
   const poll=setInterval(()=>{
     fetch('api/query-status').then(r=>r.json()).then(s=>{
       if(s.running)panel.querySelector('#qprog').textContent='processing…';
       else{clearInterval(poll);
         if(s.error)panel.querySelector('#qprog').textContent='failed: '+s.error;
         else{fetch('query.json').then(r=>r.json()).then(q=>{QUERY=q;qSel=0;
             refreshFindBtn();qRender();});}
       }});
   },1500);
 };
}
function rateRow(refId,rep){
 return `<div class="rate" data-ref="${refId}" data-rep="${rep}">`+
  RATE_LABELS.map(([k,l])=>`<span onclick="qRate('${refId}','${rep}','${k}',this)">${l}</span>`).join('')+`</div>`;
}
window.qRate=function(refId,rep,rating,el){
 const qe=QUERY.episodes[qSel];
 const rec={query_episode_id:qe.episode_id, reference_episode_id:refId,
            representation:rep, rating, query_id:QUERY.query_id};
 if(window.__info&&window.__info.server){
   fetch('api/query-rate',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify(rec)}).then(()=>{el.parentElement.style.opacity=.45;});
 } else {
   const k='ms_qrate'; const all=JSON.parse(localStorage.getItem(k)||'[]'); all.push(rec);
   localStorage.setItem(k,JSON.stringify(all)); el.parentElement.style.opacity=.45;
 }
};
function qMeta(qe){   // meta-like object for a query episode (panel + compare)
 return {episode_id:qe.episode_id, video_id:QUERY.query_id, clip_kind:'video',
   label:'query', duration_s:qe.duration_s, fps:qe.fps, frames:qe.frames,
   series:qe.series, trajectory:qe.trajectory, features:qe.features,
   sampling:{site:'your video',session:'',video:QUERY.query_id},
   provenance_chain:['query video: '+(QUERY.video_path||''),
     'encoder: '+((QUERY.encoder_provenance||{}).model_name||'?')+
       ' v'+((QUERY.encoder_provenance||{}).model_version||'?'),
     'reference atlas: '+QUERY.reference_atlas_version],
   neighbors:[]};
}
function qRender(){
 const r=QUERY.results, rb=QUERY.results_representation_b;
 const qe=QUERY.episodes[qSel];
 const qc=QUERY.qc[qSel];
 const hitA=qHitById(qe.episode_id,'a'), hitB=qHitById(qe.episode_id,'b');
 const bd=hitA&&hitA.breakdown;
 const bdNames={trajectory_geometry:'Trajectory geometry',speed_dynamics:'Speed dynamics',
   turning_dynamics:'Turning dynamics',stop_go_rhythm:'Stop–go rhythm',
   intermittency:'Intermittency'};
 panel.innerHTML=`
  <div style="display:flex;align-items:center;gap:8px">
   <span style="color:var(--accent)">⌕ Your video</span>
   <span class="muted">${QUERY.query_id} · ref ${QUERY.reference_atlas_version}</span>
   <span class="sp" style="flex:1"></span>
   <button class="btn" style="padding:3px 10px" onclick="closePanel()">esc</button></div>
  ${r.ood.any_flagged?`<div class="ood">⚠ ${esc(r.ood.message)} Physical-space distances are large
   (often units/calibration differences); see the shape-normalized view below.</div>`:''}
  <div class="muted" style="margin:8px 0">Your movement episodes — click one to place it in the atlas:</div>
  ${QUERY.episodes.map((e,i)=>{
    const c=QUERY.qc[i]; const act=i===qSel;
    return `<div class="nbr" style="${act?'background:#101a24;border-color:#3b6a75':''};${c&&!c.usable?'opacity:.45':''}"
      onclick="qSel=${i};qRender();">
      <div><span class="lbl" style="color:#8fa8bc">episode ${i+1}</span>
      <div class="muted">${e.duration_s.toFixed(1)}s${c&&c.warnings.length?' · '+c.warnings.join(', '):''}</div></div></div>`;}).join('')}
  ${(!qc.usable)?`<div class="ood">Low-confidence behavioral retrieval — this episode has quality issues
   (${esc(qc.warnings.join(', '))}). Neighbors are shown for exploration, not as evidence.</div>`:''}
  ${hitA&&hitA.neighbors.length?`
  <div style="margin-top:10px;color:var(--accent)">Closest movements — physical feature space</div>
  <div class="muted">nearest in ${META.features.length}-D interpretable feature space</div>
  ${hitA.neighbors.map((n,k)=>{
    const ne=IDX[n.episode_id]||{};
    return `<div class="nbr" data-nid="${n.episode_id}" onclick="showById('${n.episode_id}')">
      <canvas width="192" height="120" data-traj="${n.episode_id}"></canvas>
      <div><span class="lbl" style="color:${revealed?(COLORS[n.taxon]||'#888'):'#8fa8bc'}">${revealed?(NAMES[n.taxon]||n.taxon):'Movement '+(k+1)}</span>
      <div class="muted dur">similarity ${n.similarity.toFixed(2)}</div>
      ${rateRow(n.episode_id,'a')}</div></div>`;}).join('')}
  <button class="btn" style="margin-top:6px" onclick="compareQuery('a')">⇄ compare with closest (physical)</button>`:''}
  ${hitB&&hitB.neighbors.length?`
  <div style="margin-top:12px;color:var(--accent)">Closest movements — shape-normalized space</div>
  <div class="muted">movement SHAPE only (burst structure, rhythm); robust to unit and camera differences</div>
  ${hitB.neighbors.map((n,k)=>{
    const ne=IDX[n.episode_id]||{};
    return `<div class="nbr" data-nid="${n.episode_id}" onclick="showById('${n.episode_id}')">
      <canvas width="192" height="120" data-traj="${n.episode_id}"></canvas>
      <div><span class="lbl" style="color:${revealed?(COLORS[n.taxon]||'#888'):'#8fa8bc'}">${revealed?(NAMES[n.taxon]||n.taxon):'Movement '+(k+1)}</span>
      <div class="muted dur">similarity ${n.similarity.toFixed(2)}</div>
      ${rateRow(n.episode_id,'b')}</div></div>`;}).join('')}
  <button class="btn" style="margin-top:6px" onclick="compareQuery('b')">⇄ compare with closest (shape)</button>`:''}
  ${bd?`<div style="margin-top:12px;color:var(--accent)">Why similar? — real distance decomposition</div>
   ${Object.entries(bd).map(([g,v])=>`<div class="bdrow"><span class="nm">${bdNames[g]||g}</span>
     <span style="flex:1;height:9px;background:#101a24;border-radius:3px;position:relative">
     <span style="position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:#4fd1c5;width:${Math.max(4,100-Math.min(100,v.mean_standardized_diff*45))|0}%"></span></span>
     <span class="lv">${v.similarity}</span></div>`).join('')}
   <div class="muted">levels from mean standardized feature differences — never invented</div>`:''}
  ${r.motif_hits.length?`<div style="margin-top:12px;color:var(--accent)">Closest motifs</div>
   ${r.motif_hits.map(m=>`<div class="bdrow"><span class="nm">M${m.motif}</span>
     <span class="muted">similarity ${m.similarity.toFixed(2)}</span></div>`).join('')}`:''}
  ${(rb&&rb.taxa&&rb.taxa.length)?`<div style="margin-top:12px;color:var(--accent)">Behaviorally similar taxa <span class="muted">— not species identity</span></div>
   ${rb.taxa.map(tx=>`<div class="bdrow"><span class="nm">${esc(tx.taxon)}</span>
     <span style="flex:1;height:9px;background:#101a24;border-radius:3px;position:relative">
     <span style="position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:#e8a13c;width:${Math.min(100,(tx.score_corrected*200)|0)}%"></span></span>
     <span class="muted">n=${tx.n_reference_episodes} · score ${tx.score_corrected.toFixed(2)}</span></div>`).join('')}
   <div class="muted">sample-size-corrected (shrinkage); support counts shown</div>`:''}
  <div class="prov">reference ${QUERY.reference_atlas_version} · metric ${QUERY.metric} · k ${QUERY.k}

${esc(JSON.stringify((QUERY.provenance||{}).parameters||{}).slice(0,300))}</div>`;
 openPanel();
 (hitA?hitA.neighbors:[]).concat(hitB?hitB.neighbors:[]).forEach(n=>{
   fetchMeta(n.episode_id).then(nm=>{if(!nm)return;
     const col=revealed?(COLORS[n.taxon]||'#4fd1c5'):'#56677a';
     panel.querySelectorAll(`canvas[data-traj="${n.episode_id}"]`).forEach(c=>
       drawStaticReplay(c,nm.trajectory||[],col));
     panel.querySelectorAll(`.nbr[data-nid="${n.episode_id}"] .dur`).forEach(el=>{
       el.textContent=`similarity ${n.similarity.toFixed(2)} · ${nm.duration_s.toFixed(1)}s`;});});});
}
window.compareQuery=async function(rep){
 const qe=QUERY.episodes[qSel];
 const hit=qHitById(qe.episode_id,rep);
 if(!hit||!hit.neighbors.length)return;
 const qm=qMeta(qe);
 const m2=await fetchMeta(hit.neighbors[0].episode_id); if(!m2)return;
 const e2=IDX[hit.neighbors[0].episode_id];
 const nameA=`<span class="lbl" style="color:#ffffff">your video</span>`;
 const nameB=revealed?`<span class="lbl" style="color:${COLORS[e2.l]}">${NAMES[e2.l]||e2.l}</span>`
                     :`<span class="lbl" style="color:#8fa8bc">movement</span>`;
 panel.innerHTML=`
  <div style="display:flex;align-items:center;gap:8px">
    <button class="btn" style="padding:3px 10px" onclick="qRender()">← back</button>
    <span class="muted">side-by-side · shared playhead</span>
    <span class="sp" style="flex:1"></span>
    <button class="btn" style="padding:3px 10px" id="pbMode">normalized</button></div>
  <div class="split" style="margin-top:10px">
   <div>${nameA}<div class="muted">${qe.duration_s.toFixed(1)}s</div><img class="clip" id="cmpA"></div>
   <div>${nameB}<div class="muted">${e2.d.toFixed(1)}s</div><img class="clip" id="cmpB"></div>
  </div>
  <div class="muted">similarity ${hit.neighbors[0].similarity.toFixed(2)} (${rep==='a'?'physical':'shape-normalized'} space).
   “normalized”: both sides loop through their full clip together. “real-time”: each advances with the wall clock at its own fps.</div>
  <div id="cmpRowsA"></div><div id="cmpRowsB"></div>
  <div class="prov">your video: ${esc(qm.provenance_chain.join('\n'))}\n---\nref: ${esc(m2.provenance_chain.join('\n'))}</div>`;
 openPanel();
 replayFallback(panel.querySelector('#cmpA'),qm,{l:'query'});
 if(serverMode&&m2.clip_kind==='video'){panel.querySelector('#cmpB').src=`clip/${e2.id}.gif`;}
 else replayFallback(panel.querySelector('#cmpB'),m2,e2);
 for(const [mm,host] of [[qm,'cmpRowsA'],[m2,'cmpRowsB']]){
   document.getElementById(host).innerHTML=sparkRow('speed','#4fd1c5')+sparkRow('moving','#b794f4');
   document.getElementById(host).querySelectorAll('canvas.spark').forEach(c=>{
     c._arr=sparkArr(mm,c.dataset.lab);
     const g=c.getContext('2d'),vs=c._arr.filter(v=>v!=null&&isFinite(v));
     const mx=Math.max(...vs,1e-9);g.strokeStyle=c.dataset.c;g.lineWidth=2.5;g.beginPath();
     let st=false;c._arr.forEach((v,i)=>{if(v==null||!isFinite(v)){st=false;return;}
       const x=i/(c._arr.length-1||1)*c.width,y=c.height-6-(v/mx)*(c.height-12);
       st?g.lineTo(x,y):g.moveTo(x,y);st=true;});g.stroke();});
 }
 const cA=document.createElement('canvas'),cB=document.createElement('canvas');
 [cA,cB].forEach(c=>{c.width=360;c.height=200;c.style.width='100%';c.style.background='#04070b';
   c.style.borderRadius='8px';});
 document.getElementById('cmpRowsA').prepend(cA);
 document.getElementById('cmpRowsB').prepend(cB);
 let paused=false,t0=performance.now(),clock=0,mode='normalized';
 const durN=Math.max(Math.min(qe.duration_s,m2.duration_s),1)*1000;
 const fpsA=qm.fps||30, fpsB=m2.fps||30;
 const btn=document.getElementById('pbMode');
 btn.onclick=()=>{mode=mode==='normalized'?'real-time':'normalized';
   btn.textContent=mode;};
 cA.onclick=()=>{paused=!paused;};
 (function loop(now){
   if(!document.getElementById('cmpRowsA'))return;
   const dt=now-t0; t0=now; if(!paused)clock+=dt;
   let pA,pB;
   if(mode==='normalized'){pA=(clock/durN)%1;pB=(clock/durN)%1;}
   else{pA=(clock/(qe.duration_s*1000))%1;pB=(clock/(m2.duration_s*1000))%1;}
   drawReplayProgress(cA,qm.trajectory,pA,'#4fd1c5');
   drawReplayProgress(cB,m2.trajectory||[],pB,'#e8a13c');
   document.querySelectorAll('#cmpRowsA canvas.spark, #cmpRowsB canvas.spark').forEach(c=>{
     if(!c._base){c._base=document.createElement('canvas');c._base.width=c.width;
       c._base.height=c.height;c._base.getContext('2d').drawImage(c,0,0);}
     const g=c.getContext('2d');g.clearRect(0,0,c.width,c.height);
     g.drawImage(c._base,0,0);
     const prog=(c.closest('#cmpRowsA')?pA:pB)*c.width;
     g.strokeStyle='#ffffff88';g.lineWidth=2;g.beginPath();
     g.moveTo(prog,0);g.lineTo(prog,c.height);g.stroke();});
   requestAnimationFrame(loop);
 })(performance.now());
};

/* ================= Reveal species ================= */
const revealBtn=document.getElementById('revealBtn');
function toggleReveal(){
 revealed=!revealed;
 revealBtn.textContent=revealed?'◌ Hide species':'✦ Reveal species';
 const lg=document.getElementById('legend');
 if(revealed){
   lg.style.display='flex';
   lg.innerHTML='<div class="muted" style="max-width:300px;margin-bottom:4px">The space was built without knowing '+
     'what any animal is — only how it moves. Only now are names laid over it.</div>'+
     Object.entries(COLORS).filter(([k])=>k!=='mimic'||META.labels_summary.mimic)
       .filter(([k])=>(META.labels_summary[k]||0)>0)
       .map(([k,c])=>`<div><span class="dot" style="background:${c}"></span>${NAMES[k]||k} · ${(META.labels_summary[k]||0)}</div>`).join('');
 }else lg.style.display='none';
}
revealBtn.onclick=toggleReveal;
</script>
</body>
</html>"""
