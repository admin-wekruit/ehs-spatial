// NODE_PATH=<Playwright node_modules> node tests/check_workspace_selection.cjs [origin]
// Read-only real report check: selection before load and after iframe reload.
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const origin = process.argv[2] || 'http://127.0.0.1:8792';
(async () => {
  const browser = await chromium.launch({args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
  let release = () => {};
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1000}});
    const errors=[]; page.on('pageerror',error=>errors.push(error.message));
    const detail = await (await page.request.get(origin+'/api/reports/user-bor1-02')).json();
    const candidate=detail.evidence.candidates.find(c=>c.label==='robotic arm'&&c.frame_id==='frame_0001'&&c.inventory_indices.length);
    assert(candidate,'real report must expose its measured robot evidence');
    const gate=new Promise(resolve=>release=resolve); let paused=true;
    await page.route('**/report/user-bor1-02?embedded=1**',async route=>{if(paused)await gate;await route.continue();});
    await page.goto(origin+'/reports/user-bor1-02',{waitUntil:'domcontentloaded'});
    await page.locator('[data-tab="objects"]').click();
    await page.locator(`.candidate[data-id="${candidate.id}"]`).click();
    paused=false; release();
    await page.locator('[data-tab="inspection"]').click();
    async function check() {
      const reportLocator=page.frameLocator('#inspection-frame');
      await reportLocator.locator('#v3d').scrollIntoViewIfNeeded();
      const report=await (await page.locator('#inspection-frame').elementHandle()).contentFrame();
      await report.waitForFunction(inv=>JSON.stringify([...document.querySelector('.linked').bus.sel])===JSON.stringify(inv),candidate.inventory_indices);
      const viewer=await (await report.locator('#v3d').elementHandle()).contentFrame();
      await viewer.waitForFunction(inv=>typeof selectedInv==='function'&&JSON.stringify(selectedInv())===JSON.stringify(inv),candidate.inventory_indices);
      assert.equal(await report.locator('.photo-frame').inputValue(),candidate.frame_id);
      assert(await report.locator('.icad polygon.on').count()>0,'CAD must highlight the same selection');
      return report;
    }
    await check();
    await page.evaluate(()=>{const frame=document.getElementById('inspection-frame');frame.src=frame.src+'&selection-reload=1';});
    await page.waitForFunction(()=>document.getElementById('inspection-frame').contentWindow.location.href.includes('selection-reload=1'));
    await check();
    await page.route('**/workspace-check-view/**',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><title>Test scene</title>'}));
    await page.evaluate(()=>{current.playgrounds=[{id:'a',title:'Scene A',url:'/workspace-check-view/a'},{id:'b',title:'Scene B',url:'/workspace-check-view/b-v1'}];renderDetail(false);});
    await page.locator('[data-tab="playgrounds"]').click();
    await page.locator('#playground-links button').nth(1).click();
    assert.equal(await page.locator('#playground-frame').getAttribute('src'),'/workspace-check-view/b-v1');
    await page.evaluate(()=>{current.playgrounds[1].url='/workspace-check-view/b-v2';renderDetail(false);});
    assert.equal(await page.locator('#playground-frame').getAttribute('src'),'/workspace-check-view/b-v2','completed generation refreshes the selected scene revision');
    await page.selectOption('#language','en');
    assert.equal(await page.locator('#playground-frame').getAttribute('src'),'/workspace-check-view/b-v2','language changes preserve the chosen playground');
    const other=detail.evidence.candidates.find(c=>c.id!==candidate.id&&c.frame_id==='frame_0001'&&c.inventory_indices.length);
    const playground=await (await page.locator('#playground-frame').elementHandle()).contentFrame();
    await playground.evaluate(ids=>{window.received=[];window.addEventListener('message',event=>received.push(event.data));parent.postMessage({type:'panoptes:ready',objectIds:ids},location.origin);},[candidate.id,other.id]);
    await playground.waitForFunction(id=>received.some(m=>m.type==='panoptes:select'&&m.objectId===id),candidate.id);
    await page.locator('[data-tab="objects"]').click();
    await page.locator(`.candidate[data-id="${other.id}"]`).click();
    await playground.waitForFunction(id=>received.some(m=>m.type==='panoptes:select'&&m.objectId===id),other.id);
    await playground.evaluate(id=>parent.postMessage({type:'panoptes:selection',objectId:id},location.origin),candidate.id);
    await page.waitForFunction(id=>selectedCandidateId===id,candidate.id);
    assert.equal(await page.locator('#photo-frame').inputValue(),candidate.frame_id);
    await page.evaluate(id=>window.dispatchEvent(new MessageEvent('message',{origin:location.origin,source:window,data:{type:'panoptes:selection',objectId:id}})),other.id);
    assert.equal(await page.evaluate(()=>selectedCandidateId),candidate.id,'only the active playground window may select evidence');
    await playground.evaluate(()=>received=[]);
    await page.evaluate(ids=>selectCandidate(current.evidence.candidates.find(c=>!ids.includes(c.id))),[candidate.id,other.id]);
    await playground.waitForFunction(()=>received.some(m=>m.type==='panoptes:select'&&m.objectId===null));
    await page.evaluate(id=>{const c=current.evidence.candidates.find(c=>c.id===id);c.generation={status:'stale',viewer_url:'/workspace-check-view/previous'};renderCandidates();selectCandidate(c);},candidate.id);
    await page.locator('[data-tab="objects"]').click();
    assert.equal(await page.locator('#generated-object').getAttribute('href'),'/workspace-check-view/previous');
    assert.equal(await page.locator('#generated-object').textContent(),'Open previous 3D result ↗');
    assert(await page.locator('#generate-object').isEnabled(),'an earlier source revision permits explicit revalidation');
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({passed:true,selectionBeforeReportLoad:true,selectionAfterReportReload:true,playgroundRevisionRefresh:true,playgroundBidirectionalSelection:true,staleResultLink:true,inv:candidate.inventory_indices,linkedViews:['source photo','CAD','3D']}));
  } finally {release();await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
