"""HTML template for the Atlas (kept separate: pure frontend string)."""

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Murmur — The Atlas of Animal Movement</title>
<style>
 body{margin:0;background:#070b10;color:#d8e2ec;font:14px/1.5 system-ui,sans-serif;overflow:hidden}
 #stage{display:flex;height:100vh}
 #flock{flex:1;display:block;cursor:crosshair}
 #panel{width:400px;padding:16px;overflow-y:auto;border-left:1px solid #16212d;background:#0a1017}
 h1{font-size:17px;margin:0 0 2px;color:#9fe8df;letter-spacing:.5px}
 .muted{color:#64798d;font-size:12px}
 #revealBtn{margin:10px 0;padding:8px 14px;border:1px solid #2a4a55;border-radius:20px;background:#0e1a22;color:#9fe8df;font-size:13px;cursor:pointer;transition:all .4s}
 #revealBtn:hover{background:#14303a}
 .chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
 .chip{padding:3px 10px;border-radius:12px;border:1px solid #22333f;background:#0d151d;font-size:12px;cursor:pointer;color:#8fa8bc}
 .chip.on{background:#1b3a44;color:#c8f2ec;border-color:#3b6a75}
 .clip{width:100%;border-radius:8px;background:#000;margin:4px 0}
 .nbr{display:flex;gap:8px;align-items:center;padding:6px;border-radius:8px;cursor:pointer;border:1px solid transparent}
 .nbr:hover{background:#101a24;border-color:#1f303d}
 .nbr img{width:110px;border-radius:6px;background:#000}
 .lbl{font-size:12px;padding:1px 8px;border-radius:10px;background:#1c2833}
 .prov{font-family:ui-monospace,monospace;font-size:10px;color:#6d8ba3;white-space:pre-wrap;margin-top:8px;border-left:2px solid #1f2f3d;padding-left:8px}
 canvas.spark{width:100%;height:60px;background:#0c141c;border-radius:6px;margin:6px 0}
 #home{text-align:center;margin-top:40%}
</style>
</head>
<body>
<div id="stage">
 <canvas id="flock"></canvas>
 <div id="panel">
  <h1>Murmur · Atlas of Animal Movement</h1>
  <div class="muted">一个粒子 = 一段真实动物运动。位置=行为空间（沿滑窗嵌入轨迹漂移）；抖动=运动间歇性。运动本身排列，而非按物种分组。</div>
  <button id="revealBtn">✦ Reveal species</button>
  <div class="chips" id="motifChips"></div>
  <div id="detail"><div id="home"><div style="font-size:40px">͘·͘·͘·</div><div class="muted">Explore the flock — 点击任意粒子</div></div></div>
 </div>
</div>
<script>
const COLORS={ant:"#e8a13c",siler:"#4fd1c5",other_spider:"#b794f4",other_arthropod:"#a0aec0",unknown:"#718096"};
const HIDDEN="#5a6b7d";
let revealed=0, revealTarget=0;          // species color mix factor 0..1
let motifSel=-1, sel=-1, T=0;
const cv=document.getElementById('flock'),cx=cv.getContext('2d');
function fit(){cv.width=innerWidth-400;cv.height=innerHeight;}addEventListener('resize',fit);fit();

fetch('data.json').then(r=>r.json()).then(d=>init(d.episodes));
let EP=[];

function init(eps){
 EP=eps.map(e=>({...e}));
 // bounds from the union of all path points + episode embeddings
 let xs=[],ys=[];
 EP.forEach(e=>{e.path.forEach(p=>{xs.push(p[0]);ys.push(p[1]);});});
 const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 EP.forEach(e=>{e.px=e.path.map(p=>[0.07+(p[0]-xmin)/(xmax-xmin+1e-9)*0.86,
                                    0.92-(p[1]-ymin)/(ymax-ymin+1e-9)*0.84]);});
 // motif chips
 const counts={};EP.forEach(e=>counts[e.motif]=(counts[e.motif]||0)+1);
 const mc=document.getElementById('motifChips');
 Object.entries(counts).sort((a,b)=>a[0]-b[0]).forEach(([m,c])=>{
   const b=document.createElement('span');b.className='chip';b.textContent=`M${m} · ${c}`;
   b.onclick=()=>{motifSel=(motifSel==m)?-1:m;
     [...mc.children].forEach(x=>x.classList.remove('on'));
     if(motifSel==m)b.classList.add('on');showMotif(motifSel);};
   mc.appendChild(b);});
 requestAnimationFrame(draw);
}

function col(e){const c=COLORS[e.bio_label]||COLORS.unknown;
  return mix(HIDDEN,c,revealed);}
function mix(a,b,t){const pa=parseInt(a.slice(1),16),pb=parseInt(b.slice(1),16);
  const r=(pa>>16)+(((pb>>16)-(pa>>16))*t),g=((pa>>8)&255)+((((pb>>8)&255)-((pa>>8)&255))*t),
        bl=(pa&255)+(((pb&255)-(pa&255))*t);
  return `rgb(${r|0},${g|0},${bl|0})`;}

function pos(e){ // drift along the episode's real path through behavioral space
  if(e.px.length===1)return e.px[0];
  const u=(T*0.008)%1,i=Math.min(Math.floor(u*(e.px.length-1)),e.px.length-2),f=u*(e.px.length-1)-i;
  return [e.px[i][0]+(e.px[i+1][0]-e.px[i][0])*f, e.px[i][1]+(e.px[i+1][1]-e.px[i][1])*f];
}

function draw(){
 T++;
 if(revealed<revealTarget)revealed=Math.min(revealed+0.012,revealTarget);
 cx.fillStyle='#070b10';cx.fillRect(0,0,cv.width,cv.height);
 cx.strokeStyle='rgba(50,70,90,.12)';
 for(let i=1;i<6;i++){cx.beginPath();cx.moveTo(cv.width*i/6,0);cx.lineTo(cv.width*i/6,cv.height);cx.stroke();
   cx.beginPath();cx.moveTo(0,cv.height*i/6);cx.lineTo(cv.width,cv.height*i/6);cx.stroke();}
 EP.forEach((e,i)=>{
   const p=pos(e),fl=(e.speed_cv||0)*3;
   const x=p[0]*cv.width+Math.sin(T*0.013+i*2.1)*fl*1.1;
   const y=p[1]*cv.height+Math.cos(T*0.011+i*1.7)*fl*1.1;
   e.sx=x;e.sy=y;
   const inMotif=(motifSel<0||e.motif===motifSel);
   cx.globalAlpha=inMotif?(i===sel?1:0.85):0.12;
   cx.beginPath();cx.fillStyle=col(e);
   cx.arc(x,y,i===sel?7:4,0,7);cx.fill();
 });
 cx.globalAlpha=1;
 requestAnimationFrame(draw);
}

cv.addEventListener('click',ev=>{
 const r=cv.getBoundingClientRect();let best=-1,bd=1e9;
 EP.forEach((e,i)=>{const d=(ev.clientX-r.left-e.sx)**2+(ev.clientY-r.top-e.sy)**2;if(d<bd){bd=d;best=i;}});
 if(bd<500){sel=best;showMovement(EP[best]);}
});

// ---- Explore a movement ----
function showMovement(e){
 const pauseTxt=`⏸ ${e.n_pauses} 次停顿 · ${Math.round(e.frac_moving*100)}% 时间在移动`;
 document.getElementById('detail').innerHTML=`
  <div><span class="lbl" style="color:${COLORS[e.bio_label]}">${e.bio_label}</span>
   <span class="muted"> · ${e.duration_s.toFixed(1)}s · M${e.motif} · ${e.video_id} 帧 ${e.frames[0]}–${e.frames[1]}</span></div>
  <img class="clip" src="${e.clip}">
  <canvas class="spark" id="spk"></canvas>
  <div class="muted">${pauseTxt} — 速度曲线（真实数据）</div>
  <div style="margin-top:10px;color:#9fe8df">Most similar movements</div>
  <div class="muted">行为空间中距离最近的 ${e.neighbors.length} 段运动（同步循环播放）</div>
  ${e.neighbors.map(n=>`<div class="nbr" onclick="showById('${n.episode_id}')">
     <img src="${n.clip}"><div><span class="lbl" style="color:${COLORS[n.bio_label]}">${n.bio_label}</span>
     <div class="muted">d=${n.dist.toFixed(2)}</div></div></div>`).join('')}
  <div class="prov">${e.provenance_chain.join('\n')}</div>`;
 spark('spk',e.speed_series);
}
function showById(id){const i=EP.findIndex(x=>x.episode_id===id);if(i>=0){sel=i;showMovement(EP[i]);}}
function spark(id,arr){const c=document.getElementById(id);if(!c)return;
 c.width=c.clientWidth*2;c.height=120;const g=c.getContext('2d');
 const mx=Math.max(...arr,1e-9);
 g.strokeStyle='#4fd1c5';g.lineWidth=3;g.beginPath();
 arr.forEach((v,i)=>{const x=i/(arr.length-1||1)*c.width,y=c.height-8-(v/mx)*(c.height-16);
   i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();}

// ---- Explore a motif ----
function showMotif(m){
 if(m<0)return;
 const reps=EP.filter(e=>e.motif===m).sort((a,b)=>a.duration_s-b.duration_s).slice(0,6);
 const comp={};reps.forEach(e=>comp[e.bio_label]=(comp[e.bio_label]||0)+1);
 document.getElementById('detail').innerHTML=`
  <div style="color:#9fe8df;font-size:15px">Motif M${m}</div>
  <div class="muted">构成：${Object.entries(comp).map(([k,v])=>`${k}×${v}`).join(' · ')} — 代表性运动片段</div>
  ${reps.map(e=>`<div class="nbr" onclick="showById('${e.episode_id}')">
    <img src="${e.clip}"><div><span class="lbl" style="color:${COLORS[e.bio_label]}">${e.bio_label}</span>
    <div class="muted">${e.duration_s.toFixed(1)}s</div></div></div>`).join('')}`;
}

document.getElementById('revealBtn').onclick=function(){
 revealTarget=revealTarget?0:1;
 this.textContent=revealTarget?'◌ Hide species':'✦ Reveal species';
 this.style.background=revealTarget?'#14303a':'#0e1a22';
};
</script>
</body>
</html>"""
