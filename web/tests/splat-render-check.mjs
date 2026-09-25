// Headless Chromium check of the splat layer inside the production viewer (Vite production build of tests/splat-page.html).
// Run: node tests/splat-render-check.mjs                 correctness on SwiftShader, device pixel ratio 1 and 2
//      node tests/splat-render-check.mjs --perf [N]      + N (default 2.5M) synthetic splats timed on this machine's GPU (ANGLE Metal)
//      node tests/splat-render-check.mjs --real DIR --out SHOTS
//                                                        + DIR/splats.splat rendered from DROID keyframe cameras, PNGs to SHOTS
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
import {build} from 'vite';

const web=path.resolve(new URL('..',import.meta.url).pathname),out='/private/tmp/claude-501/splat-check-build',argv=process.argv.slice(2);
const option=name=>{const i=argv.indexOf(name);return i<0?null:argv[i+1]||'';};
const {chromium}=createRequire(process.env.PLAYWRIGHT_FROM||'/Users/adam/Desktop/ontab/package.json')('playwright');
await build({root:web,configFile:false,base:'./',logLevel:'warn',build:{outDir:out,emptyOutDir:true,rollupOptions:{input:path.join(web,'tests/splat-page.html')}}});

// splat32 writer and deterministic synthetic scenes.
function random(seed){return ()=>{seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return ((t^t>>>14)>>>0)/4294967296;};}
const gauss=r=>Math.sqrt(-2*Math.log(1-r()))*Math.cos(2*Math.PI*r());
function write(buffer,i,p,s,rgba,q){
  const o=i*32,n=Math.hypot(...q);p.forEach((v,k)=>buffer.writeFloatLE(v,o+4*k));s.forEach((v,k)=>buffer.writeFloatLE(v,o+12+4*k));
  rgba.forEach((v,k)=>buffer.writeUInt8(Math.max(0,Math.min(255,Math.round(v))),o+24+k));q.forEach((v,k)=>buffer.writeUInt8(Math.max(0,Math.min(255,Math.round(v/n*128+128))),o+28+k));
}
const blobs=[[[-1,1.5,1],[225,35,35]],[[0,1.5,1],[35,200,60]],[[1,1.5,1],[40,70,230]]];
function blobFile(perBlob,seed){  // three coloured blobs, interleaved so that every prefix of the file holds all three
  const r=random(seed),buffer=Buffer.alloc(perBlob*3*32);
  for(let i=0;i<perBlob*3;i++){const [center,color]=blobs[i%3];write(buffer,i,center.map(v=>v+.06*gauss(r)),[0,1,2].map(()=>.015+.02*r()),[...color.map(v=>v+10*(r()-.5)),205],[gauss(r),gauss(r),gauss(r),gauss(r)]);}
  return buffer;
}
function* roomFile(count,seed){  // flat splats on the floor and walls of an 8 x 6 x 3 box, ~20-35 layers deep on screen; made while streamed
  const r=random(seed),turn=Math.SQRT1_2;
  const faces=[[48,p=>[p[0]*8-4,p[1]*6-3,0],[1,0,0,0]],[24,p=>[p[0]*8-4,-3,p[1]*3],[turn,turn,0,0]],[24,p=>[p[0]*8-4,3,p[1]*3],[turn,turn,0,0]],[18,p=>[-4,p[0]*6-3,p[1]*3],[turn,0,turn,0]],[18,p=>[4,p[0]*6-3,p[1]*3],[turn,0,turn,0]]];
  for(let start=0;start<count;start+=65536){
    const buffer=Buffer.alloc(Math.min(65536,count-start)*32);
    for(let i=0;i<buffer.length/32;i++){
      let pick=r()*132,face=0;while(pick>faces[face][0])pick-=faces[face++][0];
      const uv=[r(),r()],p=faces[face][1](uv),stripe=(Math.floor(uv[0]*12)+Math.floor(uv[1]*6))%2;
      write(buffer,i,p,[.008*Math.exp(.5*gauss(r)),.008*Math.exp(.5*gauss(r)),.002],[...[200,170,120].map(v=>v*(stripe?1:.55)+25*gauss(r)),150+105*r()],faces[face][2]);
    }
    yield buffer;
  }
}

// One long thin white splat turned 30 degrees about the view axis: its ellipse must lie along the predicted major axis.
const needle={center:[0,1.5,1.8],scale:[.25,.012,.012],q:[Math.cos(Math.PI/12),0,Math.sin(Math.PI/12),0]},needleRecord=Buffer.alloc(32);
write(needleRecord,0,needle.center,needle.scale,[255,255,255,255],needle.q);
const files={'/blobs.splat':Buffer.concat([blobFile(2000,1),needleRecord]),'/slow.splat':blobFile(13334,2)},types={'.html':'text/html','.js':'text/javascript','.css':'text/css'};
const realDir=option('--real'),realFile=realDir&&path.join(realDir,'splats.splat'),perfCount=Number(option('--perf'))||2500000;
const server=http.createServer((req,res)=>{
  const url=new URL(req.url,'http://check').pathname;
  if(url==='/slow.splat'){  // 256 KB every 300 ms: the page must draw before the download ends
    const body=files[url];let at=0;res.writeHead(200,{'content-type':'application/octet-stream','content-length':body.length});
    const tick=()=>{if(res.destroyed)return;res.write(body.subarray(at,at+=262144));if(at<body.length)setTimeout(tick,300);else res.end();};return tick();
  }
  if(url==='/stall.splat'){res.writeHead(200,{'content-type':'application/octet-stream','content-length':files['/blobs.splat'].length});res.flushHeaders();setTimeout(()=>res.end(files['/blobs.splat']),1500);return;}  // a download that stalls
  if(files[url]){res.writeHead(200,{'content-type':'application/octet-stream','content-length':files[url].length});return res.end(files[url]);}
  if(url==='/room.splat'){  // generated as it streams: nothing that size is held in memory or on disk
    res.writeHead(200,{'content-type':'application/octet-stream','content-length':perfCount*32});
    const chunks=roomFile(perfCount,3),pump=()=>{for(let c=chunks.next();!c.done;c=chunks.next())if(!res.write(c.value))return void res.once('drain',pump);res.end();};return pump();
  }
  if(url==='/real.splat'&&realFile){res.writeHead(200,{'content-type':'application/octet-stream','content-length':fs.statSync(realFile).size});return fs.createReadStream(realFile).pipe(res);}
  const file=path.join(out,path.normalize(url).replace(/^(\.\.[/\\])+/,''));
  if(!file.startsWith(out)||!fs.existsSync(file)||!fs.statSync(file).isFile()){res.writeHead(404);return res.end();}
  res.writeHead(200,{'content-type':types[path.extname(file)]||'application/octet-stream'});fs.createReadStream(file).pipe(res);
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const origin=`http://127.0.0.1:${server.address().port}`,problems=[];
async function open(browser,dpr,query,viewport={width:800,height:600}){
  const context=await browser.newContext({viewport,deviceScaleFactor:dpr}),page=await context.newPage();
  page.on('pageerror',error=>problems.push(String(error)));page.on('console',message=>{if(message.type()==='error')problems.push(message.text());});
  await page.goto(`${origin}/tests/splat-page.html?${query}`);await page.waitForFunction(()=>window.check);return page;
}
const check=(page,fn,arg)=>page.evaluate(fn,arg);
const dominant=(rgba,k)=>rgba[k]>110&&[0,1,2].every(j=>j===k||rgba[k]>rgba[j]+60);
const grey=rgba=>Math.max(...rgba.slice(0,3))-Math.min(...rgba.slice(0,3))<=12&&rgba[0]>40;

try {
  const browser=await chromium.launch({args:['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
  const points={red:[-1.08,1.5,1],green:[0,1.5,1],blue:[1,1.5,1],crate:[-1,1.8,1],wall:[2,2,2.2],hidden:[1.75,2.3,1.75],poster:[-2.25,2,2.05]};
  // The exact source-photo camera: skew-free K, non-square pixels, off-centre principal point, pixel centres.
  const exactFrame={cameraToWorld:[[1,0,0,.3],[0,0,1,-3.5],[0,-1,0,1.2],[0,0,0,1]],K:[[500,0,330.2],[0,520,250.7],[0,0,1]],width:640,height:480};
  async function clicks(page){  // real mouse clicks; the viewer's pick pass and room-surface fallback decide
    const picked={};
    for(const [name,p] of Object.entries({crate:[-1,1.8,1],wallInCrateBounds:[-1.2,2,1.2],green:[0,1.5,1]})){
      const q=await check(page,p=>check.at(p),p),before=(await check(page,()=>check.lastSelection())).count;
      await page.mouse.click(q.clientX,q.clientY);await page.waitForFunction(b=>check.lastSelection().count>b,before,{timeout:5000});
      picked[name]=(await check(page,()=>check.lastSelection())).entityId;
    }
    return picked;
  }
  for(const dpr of [1,2]){
    const page=await open(browser,dpr,'');
    await check(page,()=>check.mount());
    const none=await check(page,p=>check.snapshot(p),points);  // layer switched on, but no splats in this report
    const load=await check(page,url=>check.loadSplats(url,6001),origin+'/blobs.splat');
    const axes=await check(page,n=>check.ellipse(n.center,n.scale,n.q,1.5),needle);
    const on=await check(page,([p,css])=>check.snapshot(p,css),[points,{major:axes.major,majorOther:axes.majorOther,minor:axes.minor,minorOther:axes.minorOther}]),clicksOn=await clicks(page);
    await check(page,()=>check.setLayers({splats:false}));
    const off=await check(page,p=>check.snapshot(p),points),clicksOff=await clicks(page);
    await check(page,()=>check.setLayers({splats:true}));
    const s=on.samples,label=`DPR ${dpr}`;
    assert.equal(on.width,640*dpr,`${label}: canvas has device pixels`);
    assert.equal(load.stats.drawn,6001);
    const wall=off.samples.wall.rgba,lit=rgba=>rgba.slice(0,3).every((x,k)=>x>wall[k]+30),bare=rgba=>grey(rgba)&&Math.abs(rgba[0]-wall[0])<=8;
    assert.ok(lit(s.major.rgba)&&lit(s.majorOther.rgba)&&[s.minor,s.minorOther].every(p=>bare(p.rgba)),
      `${label}: the tilted splat's ellipse lies along the predicted major axis (sigma ${axes.sigmaMajor.toFixed(1)} x ${axes.sigmaMinor.toFixed(1)} px) `+JSON.stringify([s.major,s.majorOther,s.minor,s.minorOther].map(p=>p.rgba)));
    assert.ok(dominant(s.red.rgba,0)&&dominant(s.green.rgba,1)&&dominant(s.blue.rgba,2),`${label}: blobs where the camera projects them `+JSON.stringify(s));
    assert.deepEqual(s.crate.rgba,off.samples.crate.rgba,`${label}: a mesh behind a splat blob still draws on top of it`);
    assert.ok(grey(s.crate.rgba),`${label}: the crate, not the red blob, at the crate `+s.crate.rgba);
    assert.ok(bare(s.wall.rgba),`${label}: where no splat covers it the room surface shows under the splats (no hole) `+s.wall.rgba);
    assert.ok(bare(s.hidden.rgba),`${label}: a model behind the room surface stays hidden under splats (the room still writes depth) `+s.hidden.rgba);
    assert.ok(s.poster.rgba[0]>180&&s.poster.rgba[1]>180&&s.poster.rgba[2]<80,`${label}: a model lying on the room surface still shows `+s.poster.rgba);
    assert.ok(grey(off.samples.hidden.rgba),`${label}: splats off, the wall hides it as before `+off.samples.hidden.rgba);
    assert.ok(grey(off.samples.wall.rgba)&&grey(off.samples.green.rgba),`${label}: splats off shows the room surface again`);
    assert.equal(off.hash,none.hash,`${label}: switched off, the scene is pixel-identical to a report without splats`);
    assert.deepEqual(clicksOn,{crate:'crate',wallInCrateBounds:'crate',green:null},`${label}: picking with splats shown `+JSON.stringify(clicksOn));
    assert.deepEqual(clicksOff,clicksOn,`${label}: picking is the same with the layer off`);
    console.log(`PASS ${label}: blobs at projected points, mesh on top, room surface under the splats (no holes) with its depth and picks (model behind the wall hidden, poster on it shown), off = no-splat report, tilted splat oriented as predicted (loaded 6001 in ${Math.round(load.ms)} ms)`);
    // Follow -> drag -> follow: a drag must not rewrite the camera the viewer was given, and giving it again returns there.
    const held=await check(page,()=>check.holdCamera()),box=await page.locator('canvas').boundingBox();
    await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();
    for(let i=1;i<=15;i++)await page.mouse.move(box.x+box.width/2+i*10,box.y+box.height/2+i*4);
    await page.mouse.up();
    const back=await check(page,()=>check.returnToHeld());
    assert.ok(back.untouched,`${label}: a drag must not rewrite the camera object the viewer was given`);
    assert.notEqual(back.dragged,held,`${label}: the drag moved the view`);
    assert.equal(back.hash,held,`${label}: the same camera again returns to the same image (follow -> drag -> follow)`);
    console.log(`PASS ${label}: follow -> drag -> follow returns to the given camera (the caller's object is never rewritten)`);
    if(dpr===1){  // a stalled download keeps the room surface, and so do drawn splats that miss it; a stale GL error elsewhere is not ours
      await check(page,()=>check.staleError());await check(page,([url,n])=>check.startSplats(url,n,'stall'),[origin+'/stall.splat',6001]);await page.waitForTimeout(600);
      const stalled=await check(page,p=>check.snapshot(p),{wall:points.wall}),stats=await check(page,()=>check.stats());
      assert.equal(stats.drawn,0,'the download is stalled: nothing drawn yet');
      assert.ok(grey(stalled.samples.wall.rgba),'the room surface stays until splats are drawn '+stalled.samples.wall.rgba);
      await check(page,url=>check.loadSplats(url,6001,'stall'),origin+'/stall.splat');
      const loaded=(await check(page,p=>check.snapshot(p),{wall:points.wall})).samples.wall.rgba;
      assert.ok(grey(loaded),'...and stays under them where no splat covers it '+loaded);
      console.log('PASS: a stalled download keeps the room surface until splats are drawn; a stale GL error from other code does not fail the load');
    }
    await check(page,f=>check.setCamera(f),exactFrame);
    const exact=await check(page,p=>check.snapshot(p),{green:points.green,blue:points.blue,red:[-1.08,1.5,1]});
    for(const name of ['green','blue']){const [k,q]=await check(page,p=>[check.kPixel(p),check.at(p)],points[name]);assert.ok(Math.hypot(k[0]-q.x,k[1]-q.y)<.01,'K and the viewer agree');}
    assert.ok(dominant(exact.samples.green.rgba,1)&&dominant(exact.samples.blue.rgba,2)&&dominant(exact.samples.red.rgba,0),`${label}: exact K camera `+JSON.stringify(exact.samples));
    console.log(`PASS ${label}: exact source-photo camera (K) puts the blobs at K's pixels`);
    if(dpr===1){
      const slow=await check(page,url=>check.loadSplats(url,40002,'slow',[0,1.5,1]),origin+'/slow.splat');
      assert.ok(slow.partial&&slow.partial.drawn>0&&slow.partial.loaded<slow.partial.total,'drawn before the download ended '+JSON.stringify(slow.partial));
      assert.ok(dominant(slow.partial.rgba,1),'the first records already draw the blob '+JSON.stringify(slow.partial));
      assert.equal(slow.stats.drawn,40002);
      const steps=[...new Set(slow.log.map(l=>l.drawn))];
      console.log(`PASS: streaming: drawn counts ${steps.join(' -> ')} over ${Math.round(slow.ms)} ms; green at ${slow.partial.drawn}/${slow.partial.total} records`);
    }
    await page.context().close();
    if(dpr===1){
      // With the source photo on, an exact (source-photo) camera draws no splats: they would cover the photo.
      const photo=await open(browser,1,'photo=1');await check(photo,()=>check.mount());await check(photo,url=>check.loadSplats(url,6001),origin+'/blobs.splat');
      await check(photo,f=>check.setCamera(f),exactFrame);const shot=(await check(photo,p=>check.snapshot(p),{green:points.green})).samples.green.rgba;
      assert.ok(!dominant(shot,1),'with the source photo on, an exact camera draws no splats '+shot);await photo.context().close();
      // A layer whose worker cannot start frees every GL object it made, and the viewer reports it.
      const leak=await open(browser,1,'leak=1');await check(leak,()=>check.mount());await check(leak,url=>check.loadSplats(url,6001),origin+'/blobs.splat');
      const counts=await check(leak,url=>check.leakCheck(url,6001),origin+'/blobs.splat');
      assert.ok(counts.failed,'the viewer reports the failed layer');assert.deepEqual(counts.after,counts.before,'a failed layer frees its GL objects '+JSON.stringify(counts));
      await leak.context().close();
      console.log('PASS: no splats over the source photo; a layer that cannot start frees its GL objects and reports splat_load_failed');
      // Splats drawn: an unselected object's photo-coloured observed surface lies under the splats (showing where they are
      // thin) and still takes the click; the selected one, moving objects' surfaces and models draw on top; splats off, as before.
      const objs=await open(browser,1,'scene=objects');await check(objs,()=>check.mount());await check(objs,url=>check.loadSplats(url,6001),origin+'/blobs.splat');
      const P={mug:[2.1,1.6,.4],walker:[-2.1,1.6,.4],crate:[-1,1.8,1]},seen=(await check(objs,p=>check.snapshot(p),P)).samples;
      assert.ok(seen.mug.rgba[0]>180&&seen.mug.rgba[1]>80&&seen.mug.rgba[2]<60,'splats drawn: an unselected object surface shows under them where none covers it '+seen.mug.rgba);
      assert.ok(seen.walker.rgba[1]>150&&seen.walker.rgba[2]>150&&seen.walker.rgba[0]<60,'a moving object surface stays visible '+seen.walker.rgba);
      assert.ok(grey(seen.crate.rgba)&&seen.crate.rgba[0]>150,'models stay visible '+seen.crate.rgba);
      const before=(await check(objs,()=>check.lastSelection())).count;await objs.mouse.click(seen.mug.clientX,seen.mug.clientY);
      await objs.waitForFunction(b=>check.lastSelection().count>b,before,{timeout:5000});
      assert.equal((await check(objs,()=>check.lastSelection())).entityId,'mug','its surface still takes the click');
      await check(objs,()=>check.select('mug'));const selected=(await check(objs,p=>check.snapshot(p),{mug:P.mug})).samples.mug.rgba;
      assert.ok(selected[0]>120&&selected[2]<120,'the selected object surface is drawn, highlighted '+selected);
      await check(objs,()=>check.select(null));await check(objs,()=>check.setLayers({splats:false}));
      const plain=(await check(objs,p=>check.snapshot(p),{mug:P.mug})).samples.mug.rgba;
      assert.ok(plain[0]>200&&plain[1]>80&&plain[1]<150&&plain[2]<60,'splats off: the surface is drawn as before '+plain);
      await objs.context().close();
      console.log('PASS: splats drawn: unselected object surfaces lie under the splats (still pickable); selected, moving and model surfaces on top');
    }
    // World-sized points (pointSizeNative): no gaps up close, round, 1-16 device px; without it today's fixed 2 px.
    const pts=await open(browser,dpr,'scene=points');
    await check(pts,()=>check.mount());await check(pts,c=>check.setCamera(c),{eye:[0,-4,0],target:[0,0,0],up:[0,0,1],fov:Math.PI/3});
    const probe=await check(pts,()=>check.pointProbes()),p=(await check(pts,css=>check.snapshot({},css),probe.css)).samples,bg='17,27,33,255';
    const red=q=>q.rgba[0]>200&&q.rgba[1]<60&&q.rgba[2]<60,green=q=>q.rgba[1]>200&&q.rgba[0]<60,blue=q=>q.rgba[2]>200&&q.rgba[0]<60;
    for(const i of [3,10,16]){
      assert.ok(red(p[`aBetween${i}`]),`${label}: world-sized points cover the gaps of their grid `+p[`aBetween${i}`].rgba);
      assert.equal(p[`bBetween${i}`].rgba.join(),bg,`${label}: without pointSizeNative points keep the fixed size (gaps stay)`);
      assert.ok(green(p[`bPoint${i}`]),`${label}: fixed-size points still drawn `+p[`bPoint${i}`].rgba);
    }
    assert.ok(blue(p.nearCentre)&&blue(p.nearInside)&&blue(p.nearInsideDown),`${label}: a near point is clamped to 16 device px `+JSON.stringify([p.nearCentre,p.nearInside,p.nearInsideDown].map(q=>q.rgba)));
    assert.ok(p.nearOutside.rgba.join()===bg&&p.nearDiagonal.rgba.join()===bg,`${label}: ... and round, not larger (a 16 px square would cover the diagonal) `+JSON.stringify([p.nearOutside.rgba,p.nearDiagonal.rgba]));
    // One device pixel, antialiased by the canvas's multisampling: brighter than the background in one to four pixels.
    const farLit=Object.keys(p).filter(k=>k.startsWith('far')&&p[k].rgba[0]+p[k].rgba[1]+p[k].rgba[2]>17+27+33+60).length;
    assert.ok(farLit>=1&&farLit<=4,`${label}: a far point is clamped up to 1 device px (${farLit} lit)`);
    const picks={};
    for(const name of ['aBetween10','bBetween10','bPoint10']){
      const [x,y]=probe.css[name],box=await pts.locator('canvas').boundingBox(),before=(await check(pts,()=>check.lastSelection())).count;
      await pts.mouse.click(box.x+x,box.y+y);await pts.waitForFunction(b=>check.lastSelection().count>b,before,{timeout:5000});picks[name]=(await check(pts,()=>check.lastSelection())).entityId;
    }
    assert.deepEqual(picks,{aBetween10:'gridA',bBetween10:null,bPoint10:'gridB'},`${label}: point picking follows the drawn points `+JSON.stringify(picks));
    await check(pts,()=>check.setLayers({splats:true}));await check(pts,url=>check.loadSplats(url,6001),origin+'/blobs.splat');
    const withSplats=(await check(pts,css=>check.snapshot({},css),{nearCentre:probe.css.nearCentre})).samples.nearCentre;
    assert.ok(blue(withSplats),`${label}: the room's point cloud stays visible with splats shown `+withSplats.rgba);
    console.log(`PASS ${label}: world-sized points (no gaps, round, clamped 1-16 px), fixed-size fallback, point picking, room points kept with splats`);
    await pts.context().close();
  }
  await browser.close();
  if(argv.includes('--perf')||realFile){
    const gpu=await chromium.launch({args:['--use-angle=metal','--ignore-gpu-blocklist','--enable-gpu']});
    const pct=(a,p)=>{const s=[...a].sort((x,y)=>x-y);return s.length?+s[Math.min(s.length-1,Math.floor(p*s.length))].toFixed(2):NaN;};
    if(argv.includes('--perf')){
      for(const dpr of [1,2]){
        const page=await open(gpu,dpr,'w=1280&h=720&scene=empty&timer=1',{width:1300,height:740});
        await check(page,()=>check.mount());
        await check(page,()=>check.setCamera({eye:[0,-2.4,1.6],target:[0,.5,1.2],up:[0,0,1],fov:Math.PI/3}));
        const load=await check(page,([url,n])=>check.loadSplats(url,n),[origin+'/room.splat',perfCount]);
        const run=await check(page,()=>check.orbit(240));
        assert.equal(run.drawn,perfCount,'at rest every splat is drawn again');
        const renderer=await check(page,()=>{const gl=document.querySelector('canvas').getContext('webgl2'),e=gl.getExtension('WEBGL_debug_renderer_info');return gl.getParameter(e.UNMASKED_RENDERER_WEBGL);});
        console.log(`PERF DPR ${dpr} (${1280*dpr}x${720*dpr}, ${renderer}): ${perfCount} splats streamed+uploaded in ${Math.round(load.ms)} ms; `+
          `splat render GPU ms while orbiting median ${pct(run.gpuMs,.5)} p95 ${pct(run.gpuMs,.95)} (n=${run.gpuMs.length}); at rest, all ${run.drawn}: ${run.restMs.map(v=>v.toFixed(1)).join(', ')} ms; `+
          `frame interval ms median ${pct(run.intervals,.5)} p95 ${pct(run.intervals,.95)}; worker sort ms median ${pct(run.sorts,.5)} max ${pct(run.sorts,1)} (${run.sorts.length} sorts in ${run.intervals.length} frames)`);
        await page.context().close();
      }
    }
    if(realFile){
      const info=JSON.parse(fs.readFileSync(path.join(realDir,'splats.json'),'utf8')),shots=option('--out')||'/private/tmp/claude-501/splat-shots';
      const npz=option('--cameras')||'/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/droid-me340-165-171/prediction.npz';
      const clip=JSON.parse(fs.readFileSync(option('--clip')||'/Users/adam/Desktop/panoptes-public/research-notes/phase2/data/clips/me340-165/clip.json','utf8'));
      const indices=(option('--frames')||'0,300,600,898').split(',').map(Number);
      const poses=JSON.parse(execFileSync('python3',['-c','import sys,json,numpy as np;p=np.load(sys.argv[1])["poses_c2w"];print(json.dumps([p[int(i)].tolist() for i in sys.argv[2].split(",")]))',npz,indices.join(',')]).toString());
      const f=1.5*clip.K[0],K=[[f,0,639.5],[0,1.5*clip.K[1],359.5],[0,0,1]];  // the 1280x720 frames: 1.5x the 640x480 crop, centred
      fs.mkdirSync(shots,{recursive:true});
      const page=await open(gpu,1,`w=1280&h=720&scene=empty&frame=${encodeURIComponent(info.coordinateFrameId)}`,{width:1300,height:740});
      await check(page,()=>check.mount());
      await check(page,f=>check.setCamera(f),{cameraToWorld:poses[0],K,width:1280,height:720});
      const load=await check(page,([url,n])=>check.loadSplats(url,n),[origin+'/real.splat',info.count]);
      for(const [n,pose] of poses.entries()){
        await check(page,f=>check.setCamera(f),{cameraToWorld:pose,K,width:1280,height:720});
        const file=path.join(shots,`splats-frame-${String(indices[n]).padStart(3,'0')}.png`);await page.locator('canvas').screenshot({path:file});console.log('SHOT',file);
      }
      console.log(`REAL: ${info.count} splats from ${realFile} loaded in ${Math.round(load.ms)} ms`);
      await page.context().close();
    }
    await gpu.close();
  }
  assert.deepEqual(problems,[],'page errors');
} finally { server.close(); fs.rmSync(out,{recursive:true,force:true}); }
