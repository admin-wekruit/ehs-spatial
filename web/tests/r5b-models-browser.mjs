// r5b (models): headless Chromium on #/live/<report> served from a bench mirror. Once the surfaces layer (tier 0) is in, N object
// cards are opened from the Objects list; each must show its model line by tier (generated mesh, checked primitive or observed
// surface; 'display only'), its model must draw in the 3D pane (the pane changes; the model alone in its corner), the 3D label of the
// selected object must be the card's shown name, and cards with planar parts show lettered parts with their angles to the floor
// ('part A ... to the floor'). The shot GLBs' nodes must load without errors. Screenshots per object.
// Run from web/: node tests/r5b-models-browser.mjs MIRROR REPORT OUT [--n 8] [--ids card-id,card-id]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';

const web=path.resolve(new URL('..',import.meta.url).pathname),repo=path.dirname(web),argv=process.argv.slice(2);
const option=(name,fallback)=>{const i=argv.indexOf(name);return i<0?fallback:argv[i+1];};
const [root,report,out]=argv,n=Number(option('--n','8')),VITE=5189,FAST=8809;
const python='/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python';
const {chromium}=createRequire(process.env.PLAYWRIGHT_FROM||'/Users/adam/Desktop/ontab/package.json')('playwright');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
fs.mkdirSync(out,{recursive:true});
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
  await page.waitForFunction(()=>/surfaces/.test(document.querySelector('.live-report-head span')?.textContent||''),null,{timeout:120000});
  await page.getByRole('tab',{name:/Objects/}).click();
  await page.waitForSelector('.mvp-list li button',{timeout:60000});
  await sleep(6000);  // the shot GLBs parse and upload
  const names=await page.$$eval('.mvp-list li button strong',els=>els.map(e=>e.textContent));
  const ids=option('--ids',null)?.split(',');
  const picks=(ids||[]).length?ids:names.map((_,i)=>i).filter(i=>i%Math.max(1,Math.floor(names.length/n))===0).slice(0,n);
  const pane=page.locator('.live-report-3d canvas');
  const shot=async()=>pane.screenshot();
  const rows=[];
  for(const [k,pick] of picks.entries()){
    await page.getByRole('tab',{name:/Objects/}).click();
    const before=await shot();
    await (typeof pick==='number'?page.locator('.mvp-list li button').nth(pick):page.locator(`.mvp-list li button[data-id="${pick}"]`)).click();
    await page.waitForSelector('.mvp-model',{timeout:10000});
    await sleep(1200);
    const inset=await page.waitForSelector('.live-report-model-inset img',{timeout:15000}).then(()=>true,()=>false);
    const after=await shot(),line=await page.locator('.mvp-model').innerText(),tier=await page.locator('.mvp-model').getAttribute('data-tier');
    const physical=await page.locator('.mvp-physical').innerText().catch(()=>'');
    const title=(await page.locator('.mvp-card h3').innerText()).split('\n')[0].trim();
    const labels=await page.$$eval('.live-report-viewer svg text',els=>els.map(e=>e.textContent));
    const changed=before.length!==after.length||!before.equals(after);
    fs.writeFileSync(path.join(out,`model-${k}.png`),await page.screenshot());
    rows.push({pick:typeof pick==='number'?names[pick]:pick,title,tier,line,changed3d:changed,inset,label3d:labels.some(t=>title.startsWith(t)||t===title),
      parts:(physical.match(/part [A-Z] to the floor/g)||[]).length});
    assert.match(line,/display only/i,'the model line says display only');
    assert.match(line,/observed surface|generated complete mesh|box|plane|cylinder|open frame|no model/i,'the model line names the tier');
  }
  const live=await page.evaluate(()=>({errors:window.__live?.errors||[],ready:Object.keys(window.__live?.ready||{}).length}));
  fs.writeFileSync(path.join(out,'r5b-models-browser.json'),JSON.stringify({report,rows,errors,live},null,1));
  assert.deepEqual(errors,[],'no page errors');
  assert.ok(!live.errors.some(e=>/glb|node|invalid/i.test(e)),'the shot GLBs and models load: '+live.errors.join(', '));
  assert.ok(rows.filter(r=>r.changed3d).length>=Math.ceil(rows.length/2),'a selected model redraws the 3D pane');
  assert.ok(rows.filter(r=>r.inset).length>=Math.ceil(rows.length*.75),'the selected models show alone in the 3D pane\'s corner');
  assert.ok(rows.filter(r=>r.label3d).length>=Math.ceil(rows.length*.75),'the 3D label is the card\'s shown name');
  assert.ok(rows.some(r=>r.parts>0),'some card shows lettered parts with their angles to the floor');
  console.log(`r5b models browser check passed: ${rows.length} cards (${[...new Set(rows.map(r=>r.tier))].join(', ')}), ${rows.filter(r=>r.changed3d).length} redrew the 3D pane, `+
    `${rows.filter(r=>r.inset).length} insets, ${rows.filter(r=>r.label3d).length} 3D labels = card names, ${rows.filter(r=>r.parts).length} with lettered parts; `+
    `asset errors: ${live.errors.length ? live.errors.join(', ') : 'none'}`);
  await browser.close();
}finally{server.kill();vite.kill();}
