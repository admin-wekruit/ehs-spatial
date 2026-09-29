// Headless Chromium on #/live/<id> with the click MVP layers (pick, object_cards, judgements): each recording is replayed through
// the local endpoint (fast), then 200 seeded clicks at random times and pixels measure pointer-down -> card in the DOM (p95 < 100 ms),
// three aimed clicks (an object with a check, a person or a second object, a miss) are screenshotted, the verdict filter keeps only
// its verdict, and an evidence thumbnail seeks the video to its keyframe.
// Run from web/: node tests/mvp-click-browser.mjs ROOT OUT [--reports a,b,c] [--clicks 200] [--speed 1]
//   At speed 1 the layers land at their recorded times, so the pick decodes while the 3D view is as busy as it would be live;
//   the clicks start once judgements v2 is in (models and the splat still arriving).
//   ROOT holds the recordings (scripts/mvp_viewer_fixture.py writes them) and receives the replays (no blob is copied: same root).
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';

const web=path.resolve(new URL('..',import.meta.url).pathname),repo=path.dirname(web),argv=process.argv.slice(2);
const option=(name,fallback)=>{const i=argv.indexOf(name);return i<0?fallback:argv[i+1];};
const [root,out]=argv,reports=option('--reports','me340-mvp-fixture,samsclub-mvp-fixture,walmart-mvp-fixture').split(','),n=Number(option('--clicks','200')),speed=option('--speed','1');
// --judged N: N aimed clicks on judged objects (one per check first, the biggest in the video), before the plain object / miss / person;
// --height: a taller viewport keeps the whole card (identity to judgements) in the screenshot
const nJudged=Number(option('--judged','1')),height=Number(option('--height','1100')),finalLayers=argv.includes('--final'),noServer=argv.includes('--no-server');
const python='/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python',VITE=Number(option('--vite-port','5183')),FAST=Number(option('--fast-port','8803'));
const {chromium}=createRequire(process.env.PLAYWRIGHT_FROM||'/Users/adam/Desktop/ontab/package.json')('playwright');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
fs.mkdirSync(out,{recursive:true});

const vite=spawn(path.join(web,'node_modules/.bin/vite'),['--host','127.0.0.1','--port',String(VITE),'--strictPort'],{cwd:web,stdio:'ignore',
  env:{...process.env,FAST_PORT:String(FAST),VITE_CACHE_DIR:path.join(out,'.vite-cache')}});
for(let i=0;;i++){try{if((await fetch(`http://127.0.0.1:${VITE}/app.html`)).ok)break;}catch{}assert.ok(i<100,'vite did not start');await sleep(200);}
const browser=await chromium.launch({args:['--use-angle=metal','--ignore-gpu-blocklist','--enable-gpu']});
const results={root,clicks:n,speed:Number(speed),videos:[]};

// In the page: source pixel -> client point through the video's fit (VideoView.toSource inverted).
const toClient=([x,y])=>{const v=document.querySelector('.report-video-stage video'),b=v.getBoundingClientRect(),vw=v.videoWidth||1280,vh=v.videoHeight||720;
  const s=(v.style.objectFit==='cover'?Math.max:Math.min)(b.width/vw,b.height/vh);return [b.left+(b.width-vw*s)/2+x*s,b.top+(b.height-vh*s)/2+y*s];};
// In the page: where an entity is biggest (its frame) and the middle of its longest run there; or a miss with depth.
const aim=([id,miss])=>{const p=window.__live.pick,d=p.data,[W,H]=d.source_wh,value=id?d.entities.indexOf(id):0;let best=null;
  d.frames.forEach((f,i)=>{const o=f.offset/p.unit;let at=0,count=0,run=null;
    for(let k=0;k<f.pairs;k++){const v=p.runs[2*(o+k)],len=p.runs[2*(o+k)+1];
      if(v===value){const row=Math.floor((at+len/2)/f.w),col=(at+len/2)%f.w;
        const ok=row>f.h*.08&&row<f.h*.92&&(!miss||row>f.h*.55&&p.depth&&p.depth[i*d.depth.w*d.depth.h+Math.floor(row*d.depth.h/f.h)*d.depth.w+Math.floor(col*d.depth.w/f.w)]>0);
        if(ok){count+=len;if(!run||len>run.len)run={len,row,col};}}
      at+=len;}
    if(run&&(!best||count>best.count))best={count,i,t:(f.t+f.t_end)/2,x:(run.col+.5)*W/f.w,y:(run.row+.5)*H/f.h};});
  return best;};

async function video(name){
  // mvp3: --no-server: a server is already up on FAST_PORT (click_ondemand.py view/measure --hold-s: the report container behind
  // its click route), the recording is served as it is (no replay); --ondemand 'report@t,x,y;...': one click there, on no entity
  const as=noServer?name:`${name}-check-${Date.now()/1000|0}`;
  const server=noServer?{kill(){}}:spawn(python,['-m','fast_report.layers','serve',root,'--port',String(FAST),'--replay',root,name,'--as',as,'--speed',speed],{cwd:repo,stdio:['ignore','pipe','inherit']});
  const page=await browser.newPage({viewport:{width:1600,height}});
  const errors=[];page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});page.on('pageerror',e=>errors.push(String(e)));
  await page.addInitScript(()=>localStorage.setItem('panoptes.language','en'));
  await page.goto(`http://127.0.0.1:${VITE}/app.html#/live/${as}`);await page.waitForSelector('.live-report');
  await page.waitForFunction(()=>window.__live?.pick&&document.querySelector('.live-report-head span')?.textContent.includes('judgements v2'),null,{timeout:300000});
  // --final: every judgements patch of the recording is in (mvp2/integrate: the last judge run is on the final names, cards v4)
  const nJudgements=fs.readdirSync(path.join(root,'reports',name,'patches')).filter(f=>f.endsWith('-judgements.json')).length;
  if(finalLayers)for(let i=0;;i++){
    const done=await page.evaluate(async n=>{const r=await fetch(location.hash.replace('#/live/','/fast/reports/')+'/patches?after=0').then(r=>r.json());
      return r.patches.filter(p=>p.layer==='judgements').length>=n;},nJudgements);
    if(done)break;assert.ok(i<600,'the final judgements never arrived');await sleep(1000);}
  await page.waitForFunction(()=>document.querySelector('.report-video-stage video')?.readyState>=2,null,{timeout:60000});
  const seekTo=async t=>{await page.evaluate(t=>window.dispatchEvent(new CustomEvent('panoptes:seek',{detail:t})),t);await sleep(30);};
  const box=await page.evaluate(()=>{const b=document.querySelector('.report-video-stage').getBoundingClientRect();return [b.left,b.top,b.width,b.height];});
  const click=async (x,y)=>{const [cx,cy]=await page.evaluate(toClient,[x,y]);
    assert.ok(cx>=box[0]&&cy>=box[1]&&cx<box[0]+box[2]&&cy<box[1]+box[3],'aimed point is off the visible video (object-fit cover crops it)');await page.mouse.click(cx,cy);};
  const info=await page.evaluate(()=>({pageErrors:window.__live.errors,readyFrames:[...window.__live.pick.ready].reduce((a,b)=>a+b,0),decodeMs:window.__live.pickDecodeMs,decodeSteps:window.__live.pickSteps,decodes:window.__live.pickDecodes,duration:document.querySelector('.report-video-stage video').duration,frames:window.__live.pick.data.frames.length}));

  // 200 seeded clicks: uniform time and a uniform point of the visible video (misses open the unknown-region card)
  let seed=1;const rnd=()=>(seed=(seed*16807)%2147483647)/2147483647;
  const before=await page.evaluate(()=>window.__live.clicks.length);let hits=0,misses=0;
  for(let i=0;i<n;i++){
    await seekTo(rnd()*info.duration*.98);await page.mouse.click(box[0]+rnd()*(box[2]-1),box[1]+rnd()*(box[3]-1));
    const unknown=await page.locator('.mvp-unknown').count();unknown?misses++:hits++;
  }
  const ms=await page.evaluate(b=>window.__live.clicks.slice(b),before);
  const q=p=>[...ms].sort((a,b)=>a-b)[Math.min(ms.length-1,Math.floor(p*ms.length))];
  assert.equal(ms.length,n,'every click committed a card');

  // Three aimed clicks, each screenshotted
  const layers=await page.evaluate(async()=>{const r=await fetch(location.hash.replace('#/live/','/fast/reports/')+'/patches?after=0').then(r=>r.json());
    const last=k=>r.patches.filter(p=>p.layer===k).at(-1);const j=last('judgements').data,c=last('object_cards');
    const cards=Array.isArray(c.data.cards)?c.data.cards:await fetch('/fast/blobs/'+c.blobs.cards.sha256).then(r=>r.json());return {rows:j.rows,cards};});
  const rank={FAIL:0,NEEDS_REVIEW:1,NO_DATA:2,PASS:3};
  // the aims: the most severe checked object, a plausible object with every side observed, a miss on the floor, a person;
  // among candidates, the one the video shows biggest
  const judged=layers.rows.filter(r=>r.subject.startsWith('obj-')&&r.evidence.some(e=>e.image)).sort((a,b)=>rank[a.verdict]-rank[b.verdict]);
  const biggest=async ids=>{let best=null;for(const id of ids.slice(0,25)){const a=await page.evaluate(aim,[id,false]);if(a&&(!best||a.count>best.count))best={...a,id};}return best?.id;};
  const plain=layers.cards.filter(c=>c.kind==='object'&&c.physical?.size_check?.status==='plausible'&&c.physical?.depth?.value!==undefined&&(c.views?.n||0)>=4).map(c=>c.id);
  const persons=layers.cards.filter(c=>c.kind==='person'&&c.rules?.length).map(c=>c.id);
  const picks=[];  // one subject per check (most severe verdict first), then the next biggest judged subjects
  for(const check of [...new Set(judged.map(r=>r.check))]){if(picks.length>=nJudged)break;
    // mvp2/integrate: the biggest subject among that check's most severe verdict (not the biggest of any verdict)
    const rows=judged.filter(r=>r.check===check&&!picks.includes(r.subject)),worst=Math.min(...rows.map(r=>rank[r.verdict]));
    const id=await biggest(rows.filter(r=>rank[r.verdict]===worst).map(r=>r.subject));if(id)picks.push(id);}
  while(picks.length<nJudged){const id=await biggest(judged.map(r=>r.subject).filter(s=>!picks.includes(s)));if(!id)break;picks.push(id);}
  const targets=[...picks,await biggest(plain),null,await biggest(persons)]
    .map((id,k)=>({id,k})).filter(({id,k})=>id||k===picks.length+1);
  const shots=[];
  for(const {id,k} of targets){
    const a=await page.evaluate(aim,[id,!id]);if(!a)continue;
    await seekTo(a.t);await click(a.x,a.y);await sleep(400);
    const picked=await page.evaluate(()=>({title:document.querySelector('.mvp-card h3')?.textContent||document.querySelector('.live-report-card h3')?.textContent,
      unknown:!!document.querySelector('.mvp-unknown'),chips:[...document.querySelectorAll('.mvp-card .mvp-chip')].map(e=>e.textContent)}));
    const file=path.join(out,`${name}-click-${k+1}.png`);await page.screenshot({path:file});
    shots.push({aimed:id||'miss',t:a.t,x:Math.round(a.x),y:Math.round(a.y),...picked,file});
    if(id)assert.ok(!picked.unknown,`aimed at ${id} but got the unknown region`);else assert.ok(picked.unknown,'aimed at a miss');
  }

  // mvp3: the on-demand card of a click on no entity (the report container segments and names that point)
  const odAim=(option('--ondemand','').split(';').find(x=>x.startsWith(name+'@'))||'').split('@')[1];
  if(odAim){
    const [t,x,y]=odAim.split(',').map(Number);await seekTo(t);await click(x,y);
    await page.waitForFunction(()=>document.querySelector('.mvp-ondemand')||/surface, not an object|On demand unavailable/.test(document.querySelector('.mvp-unknown')?.textContent||''),null,{timeout:15000});
    await sleep(300);
    const picked=await page.evaluate(()=>({title:document.querySelector('.mvp-ondemand h3')?.textContent||document.querySelector('.mvp-unknown')?.textContent?.slice(0,120),
      unknown:true,chips:[...document.querySelectorAll('.mvp-card .mvp-chip')].map(e=>e.textContent)}));
    const file=path.join(out,`${name}-click-ondemand.png`);await page.screenshot({path:file});
    shots.push({aimed:'on demand',t,x,y,...picked,file});
  }

  // Evidence thumbnail seeks the video to its keyframe
  let evidence=null;
  if(judged.length){
    await page.getByRole('tab',{name:/Objects/}).click();await page.locator('.mvp-filters input').fill(layers.cards.find(c=>c.id===judged[0].subject).identity.name);
    await page.locator('.mvp-list li button').first().click();await sleep(300);
    const thumb=page.locator('.mvp-judgements .mvp-thumb').first(),label=await thumb.locator('span').textContent();
    await seekTo(0);await thumb.click();await sleep(300);
    const now=await page.evaluate(()=>document.querySelector('.report-video-stage video').currentTime);
    evidence={label,videoTime:now};assert.ok(Math.abs(now-parseFloat(label))<.06,`evidence seek: ${label} vs ${now}`);
  }

  // Verdict filter: every row shown carries that verdict
  await page.getByRole('tab',{name:/Objects/}).click();await page.locator('.mvp-filters input').fill('');
  const counts=await page.evaluate(()=>Object.fromEntries([...document.querySelectorAll('.mvp-filters button')].map(b=>[b.dataset.v,Number(b.textContent.split(' ').at(-1))])));
  const which=['FAIL','NEEDS_REVIEW','PASS','NO_DATA'].find(v=>counts[v]>0);let filter=null;
  if(which){
    await page.locator(`.mvp-filters button[data-v="${which}"]`).click();await sleep(200);
    const chips=await page.locator('.mvp-list li .mvp-chip').allTextContents();
    filter={verdict:which,count:counts[which],shown:chips.length};
    assert.ok(chips.length===counts[which]&&chips.every(c=>c===which.replace('_',' ')),'verdict filter');
    await page.screenshot({path:path.join(out,`${name}-list-${which}.png`)});
  }
  const timing=await page.evaluate(()=>[...document.querySelectorAll('.live-report-timing table')[0].querySelectorAll('tbody tr')].map(r=>[...r.cells].map(c=>c.textContent)));
  // where a pick version's wait goes (mvp2): resource timing of every request in flight around each version's first chunk
  const network=await page.evaluate(()=>{const e=performance.getEntriesByType('resource'),r=x=>Math.round(x);
    return e.map(x=>({url:x.name.split('/').slice(-2).join('/').slice(0,40),start:r(x.startTime),queued:r((x.requestStart||x.fetchStart)-x.startTime),
      server:r(x.responseStart-(x.requestStart||x.fetchStart)),body:r(x.responseEnd-x.responseStart),kb:r((x.encodedBodySize||0)/1024)}));});
  server.kill();await page.close();
  const row={report:name,replay:as,pickDecodeMs:info.decodeMs,pickDecodeSteps:info.decodeSteps,pickDecodes:info.decodes,pickFrames:info.frames,pickReadyFrames:info.readyFrames,clicks:ms.length,hits,unknown:misses,network,
    latencyMs:{p50:q(.5),p95:q(.95),max:Math.max(...ms),mean:ms.reduce((a,b)=>a+b,0)/ms.length},shots,evidence,filter,counts,timing,errors:[...errors,...(info.pageErrors||[])]};
  console.log(JSON.stringify({report:name,latencyMs:row.latencyMs,decodeMs:row.pickDecodeMs,decodeSteps:row.pickDecodeSteps,hits,unknown:misses,filter,evidence,errors:errors.length}));
  return row;
}

for(const r of reports)results.videos.push(await video(r));
fs.writeFileSync(path.join(out,'click-check.json'),JSON.stringify(results,null,1));
await browser.close();vite.kill();
for(const v of results.videos){
  assert.ok(v.latencyMs.p95<100,`${v.report}: click p95 ${v.latencyMs.p95} ms`);
  // mvp2: every pick version serves clicks at the video's time < 300 ms after it lands (its first chunk); the rest follows
  for(const d of v.pickDecodes||[])assert.ok(d.ms<300,`${v.report}: pick seq ${d.seq} first chunk ${d.ms} ms (all ${d.all_ms} ms)`);
  assert.deepEqual(v.errors,[],`${v.report}: console errors`);
}
console.log('mvp click browser check passed');
