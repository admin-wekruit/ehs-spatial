// r4b integrate (from r4/integrate f49a7a9's r4-integrate-shots.mjs): viewer screenshots of aimed clicks on a finished report served
// from a mirror (no replay: the final layers). JPEG (a nearly full disk); other ports than r4/integrate's (both may run).
// Each aim is {label, id} (the keyframe where the entity is biggest, the middle of its longest run: mvp-click-browser's aim) or
// {label, t, x, y} (video seconds, source pixels). Per aim: the page after the click, and what the card shows.
// Run from web/: node tests/r4b-viewer-shots.mjs ROOT REPORT OUT AIMS.json
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';

const web=path.resolve(new URL('..',import.meta.url).pathname),repo=path.dirname(web),[root,report,out,aimsFile]=process.argv.slice(2);
const VITE=5195,FAST=8815,python='/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python';
const {chromium}=createRequire(process.env.PLAYWRIGHT_FROM||'/Users/adam/Desktop/ontab/package.json')('playwright');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const aims=JSON.parse(fs.readFileSync(aimsFile,'utf8'));
fs.mkdirSync(out,{recursive:true});
// mvp-click-browser's page helpers: source pixel -> client point; an entity's biggest frame and the middle of its longest run
const toClient=([x,y])=>{const v=document.querySelector('.report-video-stage video'),b=v.getBoundingClientRect(),vw=v.videoWidth||1280,vh=v.videoHeight||720;
  const s=(v.style.objectFit==='cover'?Math.max:Math.min)(b.width/vw,b.height/vh);return [b.left+(b.width-vw*s)/2+x*s,b.top+(b.height-vh*s)/2+y*s];};
const aim=id=>{const p=window.__live.pick,d=p.data,[W,H]=d.source_wh,value=d.entities.indexOf(id);if(value<0)return null;let best=null;
  d.frames.forEach(f=>{const o=f.offset/p.unit;let at=0,count=0,run=null;
    for(let k=0;k<f.pairs;k++){const v=p.runs[2*(o+k)],len=p.runs[2*(o+k)+1];
      if(v===value){const row=Math.floor((at+len/2)/f.w),col=(at+len/2)%f.w;if(row>f.h*.08&&row<f.h*.92){count+=len;if(!run||len>run.len)run={len,row,col};}}
      at+=len;}
    if(run&&(!best||count>best.count))best={count,t:(f.t+f.t_end)/2,x:(run.col+.5)*W/f.w,y:(run.row+.5)*H/f.h};});
  return best;};
const server=spawn(python,['-m','fast_report.layers','serve',root,'--port',String(FAST)],{cwd:repo,stdio:'ignore'});
const vite=spawn(path.join(web,'node_modules/.bin/vite'),['--host','127.0.0.1','--port',String(VITE),'--strictPort'],{cwd:web,stdio:'ignore',
  env:{...process.env,FAST_PORT:String(FAST),VITE_CACHE_DIR:path.join(out,'.vite-cache')}});
try{
  for(let i=0;;i++){try{if((await fetch(`http://127.0.0.1:${VITE}/app.html`)).ok)break;}catch{}assert.ok(i<100,'vite did not start');await sleep(200);}
  const browser=await chromium.launch({args:['--use-angle=metal','--ignore-gpu-blocklist','--enable-gpu']});
  const page=await browser.newPage({viewport:{width:1600,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.addInitScript(()=>localStorage.setItem('panoptes.language','en'));
  await page.goto(`http://127.0.0.1:${VITE}/app.html#/live/${report}`);
  await page.waitForFunction(()=>window.__live?.pick&&/object_cards/.test(document.querySelector('.live-report-head span')?.textContent||''),null,{timeout:180000});
  await page.waitForFunction(()=>document.querySelector('.report-video-stage video')?.readyState>=2,null,{timeout:60000});
  await sleep(5000);  // the last cards version and the models land
  const rows=[];
  for(const a of aims){
    const at=a.id?await page.evaluate(aim,a.id):a;
    if(!at){rows.push({...a,error:'not in the pick map'});continue;}
    await page.evaluate(t=>window.dispatchEvent(new CustomEvent('panoptes:seek',{detail:t})),at.t);await sleep(500);
    const [cx,cy]=await page.evaluate(toClient,[at.x,at.y]);
    const box=await page.evaluate(()=>{const b=document.querySelector('.report-video-stage video').getBoundingClientRect(),s=document.querySelector('.report-video-stage').getBoundingClientRect();
      return [Math.max(b.left,s.left),Math.max(b.top,s.top),Math.min(b.right,s.right),Math.min(b.bottom,s.bottom)];});
    if(!(cx>=box[0]&&cy>=box[1]&&cx<box[2]&&cy<box[3])){rows.push({...a,error:'the point is off the visible video'});continue;}
    await page.mouse.click(cx,cy);await sleep(900);
    const card=await page.evaluate(()=>({title:(document.querySelector('.mvp-card h3')||document.querySelector('.live-report-card h3'))?.textContent||null,
      unknown:!!document.querySelector('.mvp-unknown'),text:(document.querySelector('.mvp-card')||document.querySelector('.mvp-unknown'))?.innerText?.slice(0,1500)||null}));
    const file=path.join(out,`${a.label}.jpg`);await page.screenshot({path:file,type:'jpeg',quality:82});
    rows.push({...a,t:at.t,x:Math.round(at.x),y:Math.round(at.y),...card,file:path.basename(file)});
  }
  fs.writeFileSync(path.join(out,'shots.json'),JSON.stringify({report,rows,errors},null,1));
  console.log(`r4b viewer shots: ${rows.filter(r=>r.file).length}/${aims.length} taken, page errors ${errors.length}`);
  await browser.close();
}finally{server.kill();vite.kill();}
