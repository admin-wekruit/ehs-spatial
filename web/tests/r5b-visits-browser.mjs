// r5b (visits): the visits panel on a finished revisit report served from a mirror. Opens the report, the Visits tab, checks
// the tab's count against the change rows, opens the first change's evidence (its JPEG decodes), picks visit A (the site map's
// ghosts join the 3D scene) and B off and on, and saves JPEG screenshots. Ports of its own (other viewer checks may run).
// Run from web/: node tests/r5b-visits-browser.mjs ROOT REPORT OUT
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';

const web=path.resolve(new URL('..',import.meta.url).pathname),repo=path.dirname(web),[root,report,out]=process.argv.slice(2);
const VITE=5197,FAST=8817,python='/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python';
const {chromium}=createRequire(process.env.PLAYWRIGHT_FROM||'/Users/adam/Desktop/ontab/package.json')('playwright');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
fs.mkdirSync(out,{recursive:true});
const server=spawn(python,['-m','fast_report.layers','serve',root,'--port',String(FAST)],{cwd:repo,stdio:'ignore'});
const vite=spawn(path.join(web,'node_modules/.bin/vite'),['--host','127.0.0.1','--port',String(VITE),'--strictPort'],{cwd:web,stdio:'ignore',
  env:{...process.env,FAST_PORT:String(FAST),VITE_CACHE_DIR:path.join(out,'.vite-cache')}});
try{
  for(let i=0;;i++){try{if((await fetch(`http://127.0.0.1:${VITE}/app.html`)).ok)break;}catch{}assert.ok(i<100,'vite did not start');await sleep(200);}
  const browser=await chromium.launch({args:['--use-angle=metal','--ignore-gpu-blocklist','--enable-gpu']});
  const page=await browser.newPage({viewport:{width:1600,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.addInitScript(()=>localStorage.setItem('panoptes.language','en'));
  await page.goto(`http://127.0.0.1:${VITE}/app.html#/live/${report}`);
  await page.waitForFunction(()=>[...document.querySelectorAll('.mvp-tabs button')].some(b=>/^Visits/.test(b.textContent)),null,{timeout:180000});
  await page.waitForFunction(()=>window.__live?.pick,null,{timeout:120000});
  await sleep(3000);
  await page.evaluate(()=>[...document.querySelectorAll('.mvp-tabs button')].find(b=>/^Visits/.test(b.textContent)).click());
  await page.waitForSelector('.visits-panel');
  const tabCount=await page.evaluate(()=>Number(([...document.querySelectorAll('.mvp-tabs button')].find(b=>/^Visits/.test(b.textContent)).textContent.match(/\((\d+)\)/)||[])[1]));
  const rows=await page.evaluate(()=>document.querySelectorAll('.visits-row').length);
  assert.equal(rows,tabCount,'the tab counts the change rows the panel lists');
  const scenes0=await page.evaluate(()=>window.__live.scenes);
  await page.screenshot({path:path.join(out,'visits-panel.jpg'),type:'jpeg',quality:82});
  let evidence=null;
  if(rows){
    await page.evaluate(()=>document.querySelector('.visits-row > button').click());
    await page.waitForFunction(()=>{const i=document.querySelector('.visits-evidence img');return i&&i.complete&&i.naturalWidth>0;},null,{timeout:20000}).catch(()=>{});
    evidence=await page.evaluate(()=>{const i=document.querySelector('.visits-evidence img');return i?{w:i.naturalWidth,h:i.naturalHeight,text:document.querySelector('.visits-evidence').innerText.slice(0,400)}:null;});
    await sleep(800);
    await page.screenshot({path:path.join(out,'visits-change.jpg'),type:'jpeg',quality:82});
  }
  const pick=async label=>page.evaluate(l=>[...document.querySelectorAll('.visits-pick button')].find(b=>b.textContent.startsWith(l)).click(),label);
  await pick('A');await sleep(2500);
  const scenesA=await page.evaluate(()=>window.__live.scenes);
  assert.ok(scenesA>scenes0,'picking visit A redraws the scene (its ghosts)');
  const pressed=await page.evaluate(()=>[...document.querySelectorAll('.visits-pick button')].map(b=>b.getAttribute('aria-pressed')));
  assert.deepEqual(pressed,['true','true'],'both visits shown');
  await page.screenshot({path:path.join(out,'visits-a-and-b.jpg'),type:'jpeg',quality:82});
  await pick('B');await sleep(1500);
  await page.screenshot({path:path.join(out,'visits-a-only.jpg'),type:'jpeg',quality:82});
  const result={report,tabCount,rows,evidence,scenes:[scenes0,scenesA],pressedAfterB:await page.evaluate(()=>[...document.querySelectorAll('.visits-pick button')].map(b=>b.getAttribute('aria-pressed'))),errors};
  fs.writeFileSync(path.join(out,'visits-browser.json'),JSON.stringify(result,null,1));
  assert.equal(errors.length,0,'no page errors: '+errors.join(' | '));
  if(rows)assert.ok(evidence&&evidence.w>0,'the first change shows its evidence frames');
  console.log(`r5b visits browser check passed: ${rows} change rows (tab ${tabCount}), evidence ${evidence?evidence.w+'x'+evidence.h:'none'}, scenes ${scenes0} -> ${scenesA}`);
  await browser.close();
}finally{server.kill();vite.kill();}
