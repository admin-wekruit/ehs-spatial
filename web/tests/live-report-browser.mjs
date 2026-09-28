// Headless Chromium (GPU) on #/live/<id>: a recorded patch stream (fast_report.layers recording) is replayed at its recorded
// times through the local endpoint while the page is open. Screenshots as layers land; no asset is fetched twice; picking
// shows a card; following the video camera shows the splats; a second replay draws the same 3D picture.
// Run from web/: node tests/live-report-browser.mjs SRC REPORT OUT [--speed 1] [--times 5,21,31,70,125,160]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';

const web=path.resolve(new URL('..',import.meta.url).pathname),repo=path.dirname(web),argv=process.argv.slice(2);
const option=(name,fallback)=>{const i=argv.indexOf(name);return i<0?fallback:argv[i+1];};
const [src,report,out]=argv,speed=Number(option('--speed','1')),times=option('--times','5,21,31,70,125,160').split(',').map(Number),modal=argv.includes('--modal');
const python='/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python';
const {chromium}=createRequire(process.env.PLAYWRIGHT_FROM||'/Users/adam/Desktop/ontab/package.json')('playwright');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
fs.mkdirSync(out,{recursive:true});

const vite=spawn(path.join(web,'node_modules/.bin/vite'),['--host','127.0.0.1','--port','5173','--strictPort'],{cwd:web,stdio:'ignore'});
for(let i=0;;i++){try{if((await fetch('http://127.0.0.1:5173/app.html')).ok)break;}catch{}assert.ok(i<100,'vite did not start');await sleep(200);}
const browser=await chromium.launch({args:['--use-angle=metal','--ignore-gpu-blocklist','--enable-gpu']});
const results={src,report,speed,runs:[]};

async function replay(name,replaySpeed,shots){
  const page=await browser.newPage({viewport:{width:1600,height:1100}});
  const console_=[];page.on('console',m=>{if(m.type()==='error')console_.push(m.text());});
  await page.goto(`http://127.0.0.1:5173/app.html#/live/${name}`);await page.waitForSelector('.live-report');
  // --modal: the recording goes through a real Writer on the Modal Volume and comes back as run() would yield it.
  const command=modal?['modal-check',src,report,out,'--port','8793','--as',name,'--speed',String(replaySpeed)]:['serve',out,'--port','8793','--replay',src,report,'--as',name,'--speed',String(replaySpeed)];
  const server=spawn(python,['-m','fast_report.layers',...command],{cwd:repo,stdio:['ignore','pipe','inherit']});
  let done=false,log='';server.stdout.on('data',d=>{log+=d;if(String(d).includes('replay done'))done=true;});server.on('exit',()=>{done=true;});
  const start=Date.now(),frames=[];
  for(const t of shots){
    await sleep(Math.max(0,start+t*1000/replaySpeed-Date.now()));
    const file=path.join(out,`${name}-${String(t).padStart(3,'0')}s.png`);await page.screenshot({path:file});
    frames.push({t,file,layers:await page.locator('.live-report-head span').textContent(),load:await page.locator('.live-report-tools small').textContent()});
    console.log(JSON.stringify(frames.at(-1)));
  }
  while(!done)await sleep(200);
  // Settled: every asset of the open shot on the GPU (the tools line shows loaded/total).
  for(let i=0;i<300;i++){const [a,b]=(await page.locator('.live-report-tools small').textContent()).split('/');if(a&&a===b)break;await sleep(200);}
  await sleep(1500);
  const settled=await page.locator('.live-report-viewer').screenshot({path:path.join(out,`${name}-final-3d.png`)});
  await page.screenshot({path:path.join(out,`${name}-final.png`)});
  const stats=await page.evaluate(()=>({uploads:window.__live.uploads,ready:window.__live.ready,scenes:window.__live.scenes,errors:window.__live.errors}));
  const served=JSON.parse(fs.readFileSync(path.join(out,'reports',name,'served.json'),'utf8'));
  const written=JSON.parse(fs.readFileSync(path.join(out,'reports',name,'written.json'),'utf8'));
  // Per patch: sent and written on the container clock; served (first fetch) and drawn (its 3D assets on the GPU) on this one.
  const folder=path.join(out,'reports',name,'patches'),r=v=>v===undefined?null:Math.round(v*100)/100;
  const layers=fs.readdirSync(folder).sort().map(n=>JSON.parse(fs.readFileSync(path.join(folder,n),'utf8'))).filter(p=>p.layer!=='timing').map(p=>{
    const drawn=Object.values(p.blobs).map(b=>stats.ready['sha256:'+b.sha256]).filter(Boolean);
    return {seq:p.seq,layer:p.layer,version:p.version,sent_s:p.sent_s,written_s:written[p.seq],served_s:r(served[p.seq]-p.t0_unix),drawn_s:drawn.length?r(Math.max(...drawn)-p.t0_unix):null};});
  fs.writeFileSync(path.join(out,`${name}-layers.json`),JSON.stringify(layers,null,1));
  if(modal)fs.writeFileSync(path.join(out,`${name}-modal-stdout.jsonl`),log);
  return {page,server,frames,settled,stats,served,layers,console:console_};
}

const first=await replay(modal?`${report}-modal-${Date.now()/1000|0}`:`${report}-view-a`,speed,times);
const twice=Object.entries(first.stats.uploads).filter(([,n])=>n>1);
assert.deepEqual(twice,[],'an asset already on the GPU was fetched again: '+JSON.stringify(twice));
assert.deepEqual(first.stats.errors,[],'viewer load errors');
// Pick: click over the 3D view until an object's card shows.
const box=await first.page.locator('.live-report-viewer canvas').boundingBox();let picked=null;
for(let gy=.35;gy<=.75&&!picked;gy+=.1)for(let gx=.2;gx<=.8&&!picked;gx+=.1){
  await first.page.mouse.click(box.x+box.width*gx,box.y+box.height*gy);await sleep(250);
  const h=await first.page.locator('.live-report-card h3').count()?await first.page.locator('.live-report-card h3').textContent():null;if(h)picked={h,gx,gy};
}
assert.ok(picked,'a click in 3D selects something with a card');
await first.page.screenshot({path:path.join(out,`${report}-picked.png`)});
// Follow the video camera: the view stands on the path, where the splats draw.
await first.page.getByLabel(/物体框|boxes/).uncheck();await first.page.getByLabel(/跟随视频相机|follow the video camera/).check();await sleep(2500);
await first.page.screenshot({path:path.join(out,`${report}-follow.png`)});
const splatStats=await first.page.evaluate(()=>document.querySelector('.live-report-viewer canvas')?.toDataURL().length);
first.server.kill();await first.page.close();

// Once more, 4x faster (--modal: skipped, one paid run is enough): the settled 3D view must be the same picture.
const second=modal?null:await replay(`${report}-view-b`,speed*4,[]);
if(second){second.server.kill();await second.page.close();}
// Same picture: no pixel off by more than 8/255 (Metal rasterises the see-through boxes with 1/255 noise between runs).
const diff=second?await (async()=>{const page=await browser.newPage();const r=await page.evaluate(async([a,b])=>{
  const pixels=async s=>{const img=new Image();img.src='data:image/png;base64,'+s;await img.decode();const c=document.createElement('canvas');c.width=img.width;c.height=img.height;const g=c.getContext('2d');g.drawImage(img,0,0);return g.getImageData(0,0,c.width,c.height).data;};
  const [p,q]=[await pixels(a),await pixels(b)];if(p.length!==q.length)return {sizes:false};let max=0,over=0;for(let i=0;i<p.length;i++){const d=Math.abs(p[i]-q[i]);if(d>max)max=d;if(d>8)over++;}return {max,over};
},[first.settled.toString('base64'),second.settled.toString('base64')]);await page.close();return r;})():null;
const same=diff?diff.over===0&&diff.sizes!==false:null;
results.runs=[first,second].filter(Boolean).map(r=>({layers:r.layers,frames:r.frames,uploads:Object.keys(r.stats.uploads).length,maxUploads:Math.max(...Object.values(r.stats.uploads)),scenes:r.stats.scenes,errors:r.stats.errors,console:r.console}));
results.picked=picked;results.followCanvasDataURLLength=splatStats;results.sameFinal3D=same;results.final3DDiff=diff;
fs.writeFileSync(path.join(out,'browser-check.json'),JSON.stringify(results,null,1));
await browser.close();vite.kill();
assert.ok(same!==false,'two replays of one recording drew different 3D pictures');
console.log('live report browser check passed',JSON.stringify({picked,sameFinal3D:same,assets:results.runs[0].uploads}));
