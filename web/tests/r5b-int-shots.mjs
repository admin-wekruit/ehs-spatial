// r5b integrate: viewer screenshots of one finished report served from a mirror (JPEG; the page is 1600 x 1100):
//   3d-<tag>.jpg        the 3D pane at the video camera of time --t (the 'follow the video camera' mode): one viewpoint for r4b's report
//                       in r4b's viewer (--web DIR: r4b's web folder) and for ours
//   overlay-<t>.jpg     ours: 'overlay on the video frame' at each --overlay time: the keyframe with every model drawn from its camera
//   click-<label>.jpg   ours: a video click aimed at an object (--click label=id,...; r4b-viewer-shots' aim: where it is biggest): the page
//                       with its card, the model inset and the part angles
// Run from web/: node tests/r5b-int-shots.mjs ROOT REPORT OUT --tag after --t 12.3 [--web DIR] [--overlay 5.2,14.1] [--click a=id,b=id]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';

const argv=process.argv.slice(2),option=(name,fallback)=>{const i=argv.indexOf(name);return i<0?fallback:argv[i+1];};
const [root,report,out]=argv,tag=option('--tag','after'),T=Number(option('--t','10'));
const web=path.resolve(option('--web',new URL('..',import.meta.url).pathname)),repo=path.resolve(new URL('../..',import.meta.url).pathname);
const VITE=5197,FAST=8817,python='/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python';
const {chromium}=createRequire(process.env.PLAYWRIGHT_FROM||'/Users/adam/Desktop/ontab/package.json')('playwright');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
fs.mkdirSync(out,{recursive:true});
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
  env:{...process.env,FAST_PORT:String(FAST),VITE_CACHE_DIR:path.join(path.dirname(out),'.vite-cache-'+tag)}});
try{
  for(let i=0;;i++){try{if((await fetch(`http://127.0.0.1:${VITE}/app.html`)).ok)break;}catch{}assert.ok(i<150,'vite did not start');await sleep(200);}
  const browser=await chromium.launch({args:['--use-angle=metal','--ignore-gpu-blocklist','--enable-gpu']});
  const page=await browser.newPage({viewport:{width:1600,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.addInitScript(()=>localStorage.setItem('panoptes.language','en'));
  await page.goto(`http://127.0.0.1:${VITE}/app.html#/live/${report}`);
  await page.waitForFunction(()=>window.__live?.pick&&/object_cards/.test(document.querySelector('.live-report-head span')?.textContent||''),null,{timeout:180000});
  await page.waitForFunction(()=>document.querySelector('.report-video-stage video')?.readyState>=2,null,{timeout:60000});
  const loaded=async()=>{for(let i=0;i<120;i++){const t=await page.evaluate(()=>document.querySelector('.live-report-tools small')?.textContent||'');const m=t.match(/(\d+)\/(\d+)/);if(m&&m[1]===m[2]&&i>8)return;await sleep(500);}};
  await loaded();
  const seek=async t=>{await page.evaluate(t=>window.dispatchEvent(new CustomEvent('panoptes:seek',{detail:t})),t);await sleep(700);};
  const pane=page.locator('.live-report-3d');
  const rows=[];
  await page.getByLabel('follow the video camera').check();
  await seek(T);await sleep(2500);
  await pane.screenshot({path:path.join(out,`3d-${tag}.jpg`),type:'jpeg',quality:85});rows.push({shot:`3d-${tag}.jpg`,t:T});
  const overlay=option('--overlay','');
  if(overlay){
    await page.getByLabel('overlay on the video frame').check();
    for(const t of overlay.split(',').map(Number)){
      await seek(t);await sleep(3000);
      const f=`overlay-${t.toFixed(1)}.jpg`;await pane.screenshot({path:path.join(out,f),type:'jpeg',quality:85});rows.push({shot:f,t});
    }
    await page.getByLabel('overlay on the video frame').uncheck();await sleep(800);
  }
  for(const pair of option('--click','').split(',').filter(Boolean)){
    const [label,id]=pair.split('=');
    const at=await page.evaluate(aim,id);
    if(!at){rows.push({label,id,error:'not in the pick map'});continue;}
    await seek(at.t);
    const [cx,cy]=await page.evaluate(toClient,[at.x,at.y]);
    await page.mouse.click(cx,cy);await sleep(2500);
    const card=await page.evaluate(()=>({title:(document.querySelector('.mvp-card h3')||document.querySelector('.live-report-card h3'))?.textContent||null,
      text:(document.querySelector('.mvp-card')||document.querySelector('.mvp-unknown'))?.innerText?.slice(0,2500)||null}));
    const f=`click-${label}.jpg`;await page.screenshot({path:path.join(out,f),type:'jpeg',quality:82});
    rows.push({shot:f,label,id,t:at.t,x:Math.round(at.x),y:Math.round(at.y),...card});
  }
  fs.writeFileSync(path.join(out,`shots-${tag}.json`),JSON.stringify({report,rows,errors},null,1));
  console.log(`r5b-int shots (${tag}): ${rows.filter(r=>r.shot).length} taken, page errors ${errors.length}`);
  await browser.close();
}finally{server.kill();vite.kill();}
