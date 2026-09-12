// Actual WebGL smoke over cached, derived viewers. No provider calls.
// node tests/check_viewer_reference.cjs http://127.0.0.1:8894
const assert=require('node:assert/strict');
const {chromium,webkit}=require('/Users/adam/.codex/skills/gstack/node_modules/playwright');
const base=process.argv[2]||'http://127.0.0.1:8894';
(async()=>{for(const [name,engine,mobile] of [['chromium',chromium,false],['webkit',webkit,true]]){
 const browser=await engine.launch({headless:true,...(!mobile?{args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']}:{})});
 try{const page=await browser.newPage({viewport:mobile?{width:390,height:844}:{width:1280,height:900},isMobile:mobile,hasTouch:mobile});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/report/*/surface/*',route=>route.continue({url:route.request().url().replace('/report/','/runs/')}));
 for(const [rid,inv,surface] of [['b61ed56896064063bb845605b233609e',25,false],['user-bor1-02',null,true]]){
  await page.goto(`${base}/runs/${rid}/viewer.html`,{waitUntil:'load'});
  await page.waitForFunction(()=>typeof DATA!=='undefined'&&(!SURFACE||surfaceView!==null));
  await page.selectOption('[data-language-select]','zh');
  const target=inv??await page.evaluate(()=>SURFACE.supported_inv[0]);
  await page.locator('#objectlist').evaluate(el=>el.open=true);
  await page.locator(`[data-inv="${target}"]`).first().click();
  await page.locator('#stage').evaluate(el=>el.scrollIntoView({block:'center',behavior:'instant'}));
  const box=page.locator('#selection-reference');
  await page.waitForFunction(()=>document.querySelectorAll('#selection-reference [data-axis]').length>=3);
  assert.equal(await box.getAttribute('data-reference-frame'),surface?'surface-native':'estimated-floor');
  assert(await box.locator('[data-bbox-edge]').count()>=12);
  assert.equal(await box.evaluate(el=>getComputedStyle(el).pointerEvents),'none');
  const info=page.locator('#selection-reference-info'),height=await info.getAttribute('data-height');
  if(surface){assert.equal(height,null);assert.match(await info.innerText(),/地面高度与实测倾角未知/);}
  else{assert(+height>0);assert.equal(await box.locator('[data-height-line]').count(),1);assert.match(await info.innerText(),/模型估计尺度/);}
  const before=await box.innerHTML();
  await page.selectOption('[data-language-select]','en');
  await page.waitForFunction(()=>document.getElementById('none').textContent==='Clear selection');
  assert.equal(await page.locator('#all').innerText(),'Select all');
  assert.equal(await page.locator('#reset').innerText(),'Reset');
  assert.match(await page.locator('#hud').innerText(),surface?/Interior model/:/Measurement point cloud/);
  assert.match(await page.locator(`[data-inv="${target}"] .meta`).first().innerText(),/Height .*Camera distance/s);
  assert.match(await info.innerText(),/measured tilt unknown/);
  assert.equal(await info.getAttribute('data-height'),height);
  const canvas=page.locator(surface?'#surface-c':'#c');
  // Real pointer orbit: overlay projection must update without editing geometry.
  const rect=await canvas.boundingBox();
  await page.mouse.move(rect.x+rect.width*.5,rect.y+rect.height*.5);
  await page.mouse.down();await page.mouse.move(rect.x+rect.width*.5+25,rect.y+rect.height*.5+12,{steps:3});await page.mouse.up();
  assert.notEqual(await box.innerHTML(),before);
  assert.equal(await info.getAttribute('data-height'),height);
  if(surface){
   await page.locator('#point-mode').click();
   assert.equal(await box.getAttribute('data-reference-frame'),'estimated-floor');
   assert.deepEqual(await page.evaluate(()=>selectedInv()),[target]);
   await page.locator('#surface-mode').click();
   assert.equal(await box.getAttribute('data-reference-frame'),'surface-native');
   assert.equal(await info.getAttribute('data-height'),null);
  }
  await page.locator('#stage').evaluate(el=>el.scrollIntoView({block:'center',behavior:'instant'}));
  await page.screenshot({path:`/tmp/panoptes-reference-${name}-${surface?'surface':'point'}.png`});
  await page.locator('#none').click();
  assert.equal(await box.locator('[data-selection-key]').count(),0);
  assert(await info.isHidden());
  console.log(JSON.stringify({engine:name,run:rid,mode:surface?'native-surface':'floor-point-cloud',inv:target,height,status:'passed'}));
 }
 assert.deepEqual(errors,[]);
 }finally{await browser.close();}
}})().catch(e=>{console.error(e);process.exit(1);});
