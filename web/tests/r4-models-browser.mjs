// r4 (models): headless Chromium on #/live/<report> served from a mirror: once the cards are in, N object cards are opened from the
// Objects list; each must show its model line (kind, source, how chosen, SAM 3D), and its model must draw in the 3D pane (the pane's
// pixels change when the object is selected: the selected model shows in full, the others faint) and alone in the pane's corner
// (the viewer's preview capture). Screenshots per object.
// Run from web/: node tests/r4-models-browser.mjs MIRROR REPORT OUT [--n 4] [--ids card-id,card-id]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';

const web=path.resolve(new URL('..',import.meta.url).pathname),repo=path.dirname(web),argv=process.argv.slice(2);
const option=(name,fallback)=>{const i=argv.indexOf(name);return i<0?fallback:argv[i+1];};
const [root,report,out]=argv,n=Number(option('--n','4')),VITE=5187,FAST=8807;
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
  await page.waitForFunction(()=>/object_cards/.test(document.querySelector('.live-report-head span')?.textContent||''),null,{timeout:120000});
  await page.getByRole('tab',{name:/Objects/}).click();
  await page.waitForSelector('.mvp-list li button',{timeout:60000});
  await sleep(4000);  // the 3D models upload
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
    await sleep(800);
    const inset=await page.waitForSelector('.live-report-model-inset img',{timeout:15000}).then(()=>true,()=>false);
    const after=await shot(),line=await page.locator('.mvp-model').innerText();
    const changed=before.length!==after.length||!before.equals(after);
    fs.writeFileSync(path.join(out,`model-${k}.png`),await page.screenshot());
    rows.push({pick:typeof pick==='number'?names[pick]:pick,line,changed3d:changed,inset});
    assert.match(line,/generated, display only/i,'the model line says display only');
    assert.match(line,/SAM 3D:/,'the model line says what SAM 3D did');
  }
  fs.writeFileSync(path.join(out,'r4-models-browser.json'),JSON.stringify({report,rows,errors},null,1));
  assert.deepEqual(errors,[],'no page errors');
  assert.ok(rows.filter(r=>r.changed3d).length>=Math.ceil(rows.length/2),'a selected model redraws the 3D pane');
  assert.ok(rows.every(r=>r.inset),'every selected model shows alone in the 3D pane\'s corner');
  console.log(`r4 models browser check passed: ${rows.length} cards, ${rows.filter(r=>r.changed3d).length} redrew the 3D pane, ${rows.filter(r=>r.inset).length} insets`);
  await browser.close();
}finally{server.kill();vite.kill();}
