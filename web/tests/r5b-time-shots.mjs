// r5b time: viewer screenshots of a report's time layer (from r4b-viewer-shots.mjs: a finished report served from a mirror, JPEG,
// own ports): the objects list with 'with a change' and every card's state at the video's time, a changed card's timeline (bar,
// state at t, values per interval) and the 3D pane at each time. Checks the scrubber: at each time the card's line in the list and
// its card say the state of its timeline's interval there.
// Run from web/: node tests/r5b-time-shots.mjs ROOT REPORT OUT CARD T1 [T2 ...]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';

const web=path.resolve(new URL('..',import.meta.url).pathname),repo=path.dirname(web),[root,report,out,card,...times]=process.argv.slice(2);
const VITE=5197,FAST=8817,python='/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python';
const {chromium}=createRequire(process.env.PLAYWRIGHT_FROM||'/Users/adam/Desktop/ontab/package.json')('playwright');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
fs.mkdirSync(out,{recursive:true});
const server=spawn(python,['-m','fast_report.layers','serve',root,'--port',String(FAST)],{cwd:repo,stdio:'ignore'});
const vite=spawn(path.join(web,'node_modules/.bin/vite'),['--host','127.0.0.1','--port',String(VITE),'--strictPort'],{cwd:web,stdio:'ignore',
  env:{...process.env,FAST_PORT:String(FAST),VITE_CACHE_DIR:process.env.VITE_CACHE_DIR||path.join(out,'.vite-cache')}});
const rows=[];
try{
  for(let i=0;;i++){try{if((await fetch(`http://127.0.0.1:${VITE}/app.html`)).ok)break;}catch{}assert.ok(i<100,'vite did not start');await sleep(200);}
  const browser=await chromium.launch({args:['--use-angle=metal','--ignore-gpu-blocklist','--enable-gpu']});
  const page=await browser.newPage({viewport:{width:1600,height:1100}});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  await page.addInitScript(()=>localStorage.setItem('panoptes.language','en'));
  await page.goto(`http://127.0.0.1:${VITE}/app.html#/live/${report}`);
  await page.waitForFunction(()=>/object_cards/.test(document.querySelector('.live-report-head span')?.textContent||''),null,{timeout:180000});
  await sleep(5000);
  await page.evaluate(()=>{[...document.querySelectorAll('button[role=tab]')].find(b=>b.textContent.startsWith('Objects')).click();});
  await sleep(300);
  const changedOnly=()=>page.evaluate(()=>{const i=[...document.querySelectorAll('.mvp-filters label')].find(l=>l.textContent.includes('with a change'))?.querySelector('input');
    if(i&&!i.checked)i.click();});
  for(const t of times.map(Number)){
    await changedOnly();
    await page.evaluate(t=>window.dispatchEvent(new CustomEvent('panoptes:seek',{detail:t})),t);
    await sleep(2500);
    const list=await page.evaluate(()=>[...document.querySelectorAll('.mvp-list li button')].map(b=>({id:b.dataset.id,state:b.querySelector('.mvp-state')?.textContent})));
    await page.screenshot({path:path.join(out,`list-${t}.jpg`),type:'jpeg',quality:80});
    await page.evaluate(id=>document.querySelector(`button[data-id="${id}"]`)?.click(),card);
    await sleep(1500);
    const sec=await page.evaluate(()=>{const s=[...document.querySelectorAll('.mvp-block')].find(b=>b.querySelector('h4')?.textContent==='Time');
      if(!s)return null;s.scrollIntoView({block:'start'});const d=s.querySelector('details');if(d)d.open=true;return {state:s.dataset.stateAt,text:s.innerText.slice(0,400)};});
    await sleep(500);
    await page.screenshot({path:path.join(out,`card-${card}-${t}.jpg`),type:'jpeg',quality:80});
    const mine=list.find(x=>x.id===card);
    rows.push({t,card,list_state:mine?.state??null,card_state:sec?.state??null,changed_cards:list.length,card_text:sec?.text});
    if(mine&&sec)assert.equal(mine.state,sec.state,`at ${t} s the list and the card agree`);
    await page.evaluate(()=>{[...document.querySelectorAll('button[role=tab]')].find(b=>b.textContent.startsWith('Objects')).click();});
    await sleep(300);
  }
  assert.deepEqual(errors,[],'no page errors');
  await browser.close();
}finally{server.kill();vite.kill();}
fs.writeFileSync(path.join(out,'shots.json'),JSON.stringify({report,card,rows},null,1));
console.log(JSON.stringify(rows.map(r=>({t:r.t,list:r.list_state,card:r.card_state,changed:r.changed_cards})),null,1));
