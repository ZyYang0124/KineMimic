"""MOTIONSCAPE atlas front-end template (self-contained, no build step)."""

TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>MOTIONSCAPE — Explore how animals move</title>
<style>
 body{margin:0;background:#06090e;color:#d8e2ec;font:14px/1.5 system-ui,sans-serif;overflow:hidden}
 #stage{display:flex;height:100vh}
 #galaxy{flex:1;display:block;cursor:crosshair}
 #panel{width:420px;padding:16px;overflow-y:auto;border-left:1px solid #16212d;background:#0a1017}
 h1{font-size:17px;margin:0 0 2px;color:#9fe8df;letter-spacing:.5px}
 .muted{color:#64798d;font-size:12px}
 button.mbtn{margin:6px 6px 0 0;padding:6px 12px;border:1px solid #2a4a55;border-radius:16px;background:#0e1a22;color:#9fe8df;font-size:12px;cursor:pointer}
 button.mbtn:hover{background:#14303a}
 .chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
 .chip{padding:3px 10px;border-radius:12px;border:1px solid #22333f;background:#0d151d;font-size:12px;cursor:pointer;color:#8fa8bc}
 .chip.on{background:#1b3a44;color:#c8f2ec;border-color:#3b6a75}
 .clip{width:100%;border-radius:8px;background:#000;margin:4px 0}
 .nbr{display:flex;gap:8px;align-items:center;padding:6px;border-radius:8px;cursor:pointer;border:1px solid transparent}
 .nbr:hover{background:#101a24;border-color:#1f303d}
 .nbr img{width:110px;border-radius:6px;background:#000}
 .lbl{font-size:12px;padding:1px 8px;border-radius:10px;background:#1c2833}
 .prov{font-family:ui-monospace,monospace;font-size:10px;color:#6d8ba3;white-space:pre-wrap;margin-top:8px;border-left:2px solid #1f2f3d;padding-left:8px}
 canvas.spark{width:100%;height:44px;background:#0c141c;border-radius:6px;margin:2px 0}
 .rowlab{font-size:10px;color:#64798d;width:52px;display:inline-block}
 .split{display:flex;gap:8px}.split>div{flex:1}
 .bar{height:12px;border-radius:3px;display:inline-block;vertical-align:middle}
 #river{width:100%;height:120px;background:#0c141c;border-radius:8px;margin:8px 0;cursor:pointer}
 #hero{position:fixed;inset:0;background:#06090e;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:9;transition:opacity 1.2s}
 #hero h2{font-size:30px;color:#cfeee9;font-weight:300;letter-spacing:2px;margin:0}
 #home{text-align:center;margin-top:40%}
</style>
</head>
<body>
<div id="hero">
  <div id="heroTraj" style="width:520px;height:260px"></div>
  <h2>How does a spider move like an ant?</h2>
  <div class="muted" style="margin-top:10px">MOTIONSCAPE · 每个粒子 = 一段真实运动 · 点击进入运动宇宙</div>
</div>
<div id="stage">
 <canvas id="galaxy"></canvas>
 <div id="panel">
  <h1>MOTIONSCAPE · Movement Galaxy</h1>
  <div class="muted">粒子按运动相似性排列，沿真实滑窗嵌入轨迹漂移；抖动=运动间歇性。物种颜色默认隐藏——先看到行为世界本身。</div>
  <button class="mbtn" id="revealBtn">✦ Reveal species</button>
  <button class="mbtn" id="riverBtn">~ Behavior River</button>
  <div id="riverWrap" style="display:none"></div>
  <div class="chips" id="motifChips"></div>
  <div id="detail"><div id="home"><div style="font-size:38px">͘·͘·͘·</div><div class="muted">Explore the flock — 点击任意粒子</div></div></div>
 </div>
</div>
<script>
const COLORS={ant:"#e8a13c",mimic:"#4fd1c5",siler:"#4fd1c5",other_spider:"#b794f4",other_arthropod:"#a0aec0",unknown:"#718096"};
const NAMES={ant:"Ant",mimic:"Mimic spider",other_spider:"Other spider",siler:"Mimic spider",unknown:"Unknown"};
const HIDDEN="#5a6b7d";
let revealed=0, revealTarget=0, motifSel=-1, sel=-1, T=0;
const cv=document.getElementById('galaxy'),cx=cv.getContext('2d');
function fit(){cv.width=innerWidth-420;cv.height=innerHeight;}addEventListener('resize',fit);fit();

fetch('data.json').then(r=>r.json()).then(d=>init(d.episodes));
let EP=[];

function init(eps){
 EP=eps.map(e=>({...e}));
 let xs=[],ys=[];EP.forEach(e=>e.path.forEach(p=>{xs.push(p[0]);ys.push(p[1]);}));
 const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 EP.forEach(e=>{e.px=e.path.map(p=>[0.07+(p[0]-xmin)/(xmax-xmin+1e-9)*0.86,
                                    0.92-(p[1]-ymin)/(ymax-ymin+1e-9)*0.84]);});
 const mc=document.getElementById('motifChips');
 const counts={};EP.forEach(e=>counts[e.motif]=(counts[e.motif]||0)+1);
 Object.entries(counts).sort((a,b)=>a[0]-b[0]).forEach(([m,c])=>{
   const b=document.createElement('span');b.className='chip';b.textContent=`M${m} · ${c}`;
   b.onclick=()=>{m=+m;motifSel=(motifSel===m)?-1:m;
     [...mc.children].forEach(x=>x.classList.remove('on'));
     if(motifSel==m)b.classList.add('on');showMotif(motifSel);};
   mc.appendChild(b);});
 hero(EP.find(e=>e.bio_label==='mimic'||e.bio_label==='siler')||EP[0]);
 requestAnimationFrame(draw);
}

/* ---------- hero: one real trajectory grows, then enter the universe ---------- */
let heroDone=false;
function hero(e){
  const c=document.createElement('canvas');c.width=1040;c.height=520;
  c.style.width='520px';c.style.height='260px';
  document.getElementById('heroTraj').appendChild(c);
  const g=c.getContext('2d');
  const tr=e.trajectory; if(!tr||tr.length<2)return;
  const x0=Math.min(...tr.map(p=>p[0])),x1=Math.max(...tr.map(p=>p[0])),
        y0=Math.min(...tr.map(p=>p[1])),y1=Math.max(...tr.map(p=>p[1]));
  const P=p=>[60+(p[0]-x0)/(x1-x0+1e-9)*920, 460-(p[1]-y0)/(y1-y0+1e-9)*400];
  let i=0; const col=COLORS[e.bio_label];
  const tick=()=>{
    if(heroDone)return;
    g.fillStyle='rgba(6,9,14,.25)';g.fillRect(0,0,c.width,c.height);
    g.strokeStyle=col;g.lineWidth=3;g.beginPath();
    for(let k=0;k<=i;k++){const q=P(tr[k]);k?g.lineTo(q[0],q[1]):g.moveTo(q[0],q[1]);}
    g.stroke();
    if(i<tr.length-1){i+=Math.max(1,Math.round(tr.length/90));requestAnimationFrame(tick);}
    else{i=0;setTimeout(()=>{if(!heroDone)tick();},1200);}
  };
  tick();
  document.getElementById('hero').onclick=()=>{
    heroDone=true;
    const h=document.getElementById('hero');h.style.opacity=0;
    setTimeout(()=>h.remove(),1300);
  };
}

function col(e){return mix(HIDDEN,COLORS[e.bio_label]||COLORS.unknown,revealed);}
function mix(a,b,t){const pa=parseInt(a.slice(1),16),pb=parseInt(b.slice(1),16);
  const r=(pa>>16)+(((pb>>16)-(pa>>16))*t),g=((pa>>8)&255)+((((pb>>8)&255)-((pa>>8)&255))*t),
        bl=(pa&255)+(((pb&255)-(pa&255))*t);
  return `rgb(${r|0},${g|0},${bl|0})`;}
function pos(e){
  if(e.px.length===1)return e.px[0];
  const u=(T*0.008)%1,i=Math.min(Math.floor(u*(e.px.length-1)),e.px.length-2),f=u*(e.px.length-1)-i;
  return [e.px[i][0]+(e.px[i+1][0]-e.px[i][0])*f, e.px[i][1]+(e.px[i+1][1]-e.px[i][1])*f];
}
function draw(){
 T++;
 if(revealed<revealTarget)revealed=Math.min(revealed+0.012,revealTarget);
 cx.fillStyle='#06090e';cx.fillRect(0,0,cv.width,cv.height);
 cx.strokeStyle='rgba(50,70,90,.10)';
 for(let i=1;i<6;i++){cx.beginPath();cx.moveTo(cv.width*i/6,0);cx.lineTo(cv.width*i/6,cv.height);cx.stroke();
   cx.beginPath();cx.moveTo(0,cv.height*i/6);cx.lineTo(cv.width,cv.height*i/6);cx.stroke();}
 EP.forEach((e,i)=>{
   const p=pos(e),fl=(e.speed_cv||0)*3;
   const x=p[0]*cv.width+Math.sin(T*0.013+i*2.1)*fl*1.1;
   const y=p[1]*cv.height+Math.cos(T*0.011+i*1.7)*fl*1.1;
   e.sx=x;e.sy=y;
   const inM=(motifSel<0||e.motif===motifSel);
   cx.globalAlpha=inM?(i===sel?1:0.85):0.10;
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

/* ---------- Explore a movement: clip + speed/turn/moving rows ---------- */
function showMovement(e){
 const s=e.series||{speed:[0],turn:[0],moving:[0]};
 const nearestAnt=(e.neighbors||[]).find(n=>n.bio_label==='ant');
 document.getElementById('detail').innerHTML=`
  <div><span class="lbl" style="color:${COLORS[e.bio_label]}">${NAMES[e.bio_label]||e.bio_label}</span>
   <span class="muted">${e.species_detail?'· '+e.species_detail:''} · ${e.duration_s.toFixed(1)}s · M${e.motif} · ${e.video_id}
   ${e.clip_kind==='video'?' · 原始视频':' · 轨迹回放'}</span></div>
  <img class="clip" src="${e.clip}">
  <div class="muted">speed / turn / moving —— 真实时间序列</div>
  ${sparkRow('speed',s.speed,'#4fd1c5')}${sparkRow('turn',s.turn,'#e8a13c')}${sparkRow('moving',s.moving,'#b794f4')}
  ${nearestAnt?`<button class="mbtn" onclick="compare('${e.episode_id}')">⇄ Compare with nearest ant</button>`:''}
  <div style="margin-top:10px;color:#9fe8df">Most similar movements</div>
  ${(e.neighbors||[]).map(n=>`<div class="nbr" onclick="showById('${n.episode_id}')">
     <img src="${n.clip}"><div><span class="lbl" style="color:${COLORS[n.bio_label]}">${NAMES[n.bio_label]||n.bio_label}</span>
     <div class="muted">similarity ${n.similarity.toFixed(2)}</div></div></div>`).join('')}
  <div class="prov">${e.provenance_chain.join('\n')}</div>`;
 renderSparks(s);
}
function showById(id){const i=EP.findIndex(x=>x.episode_id===id);if(i>=0){sel=i;showMovement(EP[i]);}}
function sparkRow(lab,arr,c){
 return `<div><span class="rowlab">${lab}</span><canvas class="spark" width="760" height="88" data-lab="${lab}" data-c="${c}"></canvas></div>`;}
function renderSparks(s){
 document.querySelectorAll('canvas.spark').forEach(c=>{
  const arr=s[c.dataset.lab];if(!arr)return;
  const g=c.getContext('2d');const mx=Math.max(...arr,1e-9);
  g.strokeStyle=c.dataset.c;g.lineWidth=3;g.beginPath();
  arr.forEach((v,i)=>{const x=i/(arr.length-1||1)*c.width,y=c.height-6-(v/mx)*(c.height-12);
    i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();});
}

/* ---------- Compare with nearest ant: split view ---------- */
window.compare=function(id){
 const e=EP.find(x=>x.episode_id===id);
 const n=(e.neighbors||[]).find(x=>x.bio_label==='ant');if(!n)return;
 const o=EP.find(x=>x.episode_id===n.episode_id);if(!o)return;
 document.getElementById('detail').innerHTML=`
  <div class="split">
   <div><span class="lbl" style="color:${COLORS[e.bio_label]}">${NAMES[e.bio_label]}</span>
     <img class="clip" src="${e.clip}">
     ${sparkRow('speed',e.series.speed,COLORS[e.bio_label])}</div>
   <div><span class="lbl" style="color:${COLORS.ant}">Ant</span>
     <img class="clip" src="${o.clip}">
     ${sparkRow('speed',o.series.speed,COLORS.ant)}</div>
  </div>
  <div class="muted">并排对照：两段真实运动同源时间序列。相似度 ${n.similarity.toFixed(2)}（d=${n.dist.toFixed(2)}）。</div>
  <button class="mbtn" onclick="showById('${e.episode_id}')">← back</button>
  <div class="prov">${e.provenance_chain.join('\n')}\n--- nearest ant ---\n${o.provenance_chain.join('\n')}</div>`;
 renderSparks(e.series);renderSparks(o.series);
};

/* ---------- Motion Dictionary ---------- */
function showMotif(m){
 if(m<0)return;
 const reps=EP.filter(e=>e.motif===m);
 const comp={};reps.forEach(e=>comp[e.bio_label]=(comp[e.bio_label]||0)+1);
 const occ=Object.entries(comp).sort((a,b)=>b[1]-a[1]);
 document.getElementById('detail').innerHTML=`
  <div style="color:#9fe8df;font-size:15px">Motion M${m}</div>
  <div class="muted">机器发现的运动模式——先看代表片段，再命名它。</div>
  <div style="margin:8px 0">${occ.map(([k,v])=>`
    <div style="display:flex;align-items:center;gap:8px;margin:3px 0">
      <span class="lbl" style="color:${COLORS[k]};min-width:110px">${NAMES[k]||k}</span>
      <span class="bar" style="width:${(v/reps.length*180)|0}px;background:${COLORS[k]}"></span>
      <span class="muted">${(v/reps.length*100).toFixed(1)}% · ${v} 段</span></div>`).join('')}
  </div>
  <div class="muted">全部 ${EP.length} 段中此 motif 有 ${reps.length} 段（${(reps.length/EP.length*100).toFixed(1)}%）</div>
  ${reps.slice(0,6).map(e=>`<div class="nbr" onclick="showById('${e.episode_id}')">
    <img src="${e.clip}"><div><span class="lbl" style="color:${COLORS[e.bio_label]}">${NAMES[e.bio_label]||e.bio_label}</span>
    <div class="muted">${e.duration_s.toFixed(1)}s</div></div></div>`).join('')}`;
}

/* ---------- Behavior River ---------- */
document.getElementById('riverBtn').onclick=()=>{
 const w=document.getElementById('riverWrap');
 if(w.style.display==='none'){w.style.display='block';drawRiver(w);}else{w.style.display='none';w.innerHTML='';}
};
function drawRiver(w){
 const hasH=EP.filter(e=>e.hour!=null);
 const useH=hasH.length>EP.length/2;
 const bins=24, keys=[...new Set(EP.map(e=>e.bio_label))];
 const grid=Object.fromEntries(keys.map(k=>[k,new Array(bins).fill(0)]));
 EP.forEach((e,i)=>{
   const b=useH?Math.min(bins-1,Math.max(0,Math.floor(e.hour||0)))
              :Math.min(bins-1,Math.floor(i/EP.length*bins));
   grid[e.bio_label][b]++;});
 const c=document.createElement('canvas');c.id='river';c.width=820;c.height=240;c.style.height='120px';
 w.appendChild(c);
 const g=c.getContext('2d');
 const mx=Math.max(...keys.flatMap(k=>grid[k]),1);
 const rowH=(c.height-34)/keys.length;
 keys.forEach((k,ki)=>{
   g.fillStyle=COLORS[k]||COLORS.unknown;g.globalAlpha=.85;
   grid[k].forEach((v,b)=>{
     const h=Math.max(v/mx,0.0)*rowH;
     if(v>0)g.fillRect(b*(c.width/bins)+2, c.height-26-ki*rowH-rowH, c.width/bins-4, Math.max(h,2));
   });});
 g.globalAlpha=1;g.fillStyle='#54687c';g.font='15px system-ui';g.textAlign='center';
 for(let b=0;b<bins;b+=3)g.fillText(useH?`${b}:00`:`#${b}`,b*(c.width/bins)+(c.width/bins)/2,c.height-8);
 keys.slice().reverse().forEach((k,ki)=>{g.fillStyle=COLORS[k];g.textAlign='left';
   g.fillText((NAMES[k]||k).split(' ')[0],8,16+ki*16);});
 w.insertAdjacentHTML('beforeend',
  '<div class="muted">Behavior River — 各物种运动量沿'+(useH?'一天时刻':'录制顺序')+'的流动。ant / mimic / other 的节律差异本身即科学问题。</div>');
}

document.getElementById('revealBtn').onclick=function(){
 revealTarget=revealTarget?0:1;
 this.textContent=revealTarget?'◌ Hide species':'✦ Reveal species';
 this.style.background=revealTarget?'#14303a':'#0e1a22';
};
</script>
</body>
</html>"""
