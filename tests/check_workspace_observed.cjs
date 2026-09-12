// NODE_PATH=<Playwright node_modules> node tests/check_workspace_observed.cjs [origin]
// Exercises the real chat handler and generic observed-scene UI with network-bound test evidence.
const assert=require('node:assert/strict');
const {chromium,webkit}=require('playwright');
const origin=process.argv[2]||'http://127.0.0.1:8792';
const run='generic-workcell-check',object='object_added_by_agent';
const measure={version:1,status:'available',dimensions_native:{height:2,width:3,depth:4},
 scale:{status:'estimated',source:'moge_anchor',m_per_native:.5},
 orientation:{planar_slope:{status:'available',value_deg:30,reason:null},
 principal_axis_tilt:{status:'unavailable',value_deg:null,reason:'No stable dominant axis'}}};
const candidate={id:object,label:'new workcell panel',frame_id:'frame_0001',inventory_indices:[],
 mask:{status:'available',resolution:'original',bbox:[20,10,80,90]},geometry:{status:'supported'},
 generation:{status:'pending_validation'},measurements:measure,
 spatial:{status:'ready',revision:'revision2',viewer_url:'/published/workspace-observed-check/scene?object='+object}};
const detail={run_id:run,title:'New workcell',phase:'done',has_report:true,inspection_url:'/report/'+run+'?embedded=1',download_url:'/report/'+run+'/file',
 evidence:{frames:[{frame_id:'frame_0001',width:100,height:100,url:'/api/reports/'+run+'/images/frame_0001'}],candidates:[]},playgrounds:[],revision:'revision1'};
async function runBrowser(engine){
 const browser=await({chromium,webkit}[engine]).launch(engine==='chromium'?{args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']}:{});
 try{
  const page=await browser.newPage(engine==='webkit'?{viewport:{width:390,height:844},isMobile:true,hasTouch:true}:{viewport:{width:1440,height:1100}}),errors=[];
  let added=false,posts=0;page.on('pageerror',error=>errors.push(error.message));await page.emulateMedia({reducedMotion:'reduce'});
  await page.route('**/reports/'+run,route=>route.fulfill({path:require('node:path').join(__dirname,'../ehs_spatial/static/workspace.html'),contentType:'text/html'}));
  await page.route('**/workspace-assets/**',route=>{const pathname=new URL(route.request().url()).pathname;return route.fulfill({path:require('node:path').join(__dirname,'../ehs_spatial/static',pathname.slice('/workspace-assets/'.length)),contentType:pathname.endsWith('.css')?'text/css':'application/javascript'});});
  await page.route('**/api/reports/'+run,route=>{const data=structuredClone(detail);if(added){data.evidence.candidates=[candidate];data.playgrounds=[{id:'observed',title:{zh:'工位物体与范围',en:'Workcell objects and bounds'},url:'/published/workspace-observed-check/scene?revision=revision2'}];data.spatial={status:'ready',viewer_url:data.playgrounds[0].url};data.revision='revision2';}return route.fulfill({json:data});});
  await page.route('**/api/session',route=>route.fulfill({json:{can_write:true,requires_access:false}}));
  await page.route('**/api/chat/'+run,route=>route.fulfill({json:[]}));
  await page.route('**/api/agent',route=>{const body=route.request().postDataJSON();assert.equal(body.run_id,run);assert.equal(body.action,'add_object');assert.equal(body.frame_id,'frame_0001');assert.deepEqual(body.box,[20,10,80,90]);added=true;posts++;return route.fulfill({json:{reply:'Saved source evidence',changed:posts===1,refresh:{updated:true},candidate_ids:[object]}});});
  await page.route('**/api/reports/'+run+'/images/frame_0001',route=>route.fulfill({contentType:'image/svg+xml',body:'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="#aabbcc"/></svg>'}));
  await page.route('**/report/'+run+'?*',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><body>Existing inspection report</body>'}));
  await page.route('**/published/workspace-observed-check/**',route=>{
   return route.fulfill({contentType:'text/html',body:`<!doctype html><body><p>Shared scene</p><script>window.received=[];addEventListener('message',e=>received.push(e.data));parent.postMessage({type:'panoptes:ready',objectIds:['${object}']},location.origin);</script>`});
  });
  await page.goto(origin+'/reports/'+run);await page.waitForFunction(()=>typeof chatReady!=='undefined'&&chatReady);
  await page.locator('#agent-action').selectOption('add_object');await page.locator('#object-label').fill('new workcell panel');
  await page.evaluate(()=>{box=[20,10,80,90];drawRect(box);});
  await page.evaluate(async()=>{await document.getElementById('report-chat').connect.handler({messages:[{role:'user',text:'Add the panel inside this workcell'}]},{onResponse:async message=>{if(message.error)throw Error(message.error);}});});
  assert.equal(posts,1);assert.equal(await page.locator('.candidate[aria-pressed=true]').getAttribute('data-id'),object);
  // Retrying already-saved evidence repairs a previous stale report even when
  // the segmentation itself has not changed.
  await page.evaluate(async()=>{current.evidence.candidates=[];box=[20,10,80,90];await document.getElementById('report-chat').connect.handler({messages:[{role:'user',text:'Add this saved panel again'}]},{onResponse:async message=>{if(message.error)throw Error(message.error);}});});
  assert.equal(posts,2);assert.equal(await page.locator('.candidate[aria-pressed=true]').getAttribute('data-id'),object);
  assert.equal(await page.locator('#selection-box').getAttribute('width'),'60');assert(await page.locator('#objects').isVisible());
  assert.deepEqual(await page.locator('.measurement-values strong').allTextContents(),['1.000 m','1.500 m','2.000 m']);
  assert((await page.locator('#object-measurements').textContent()).includes('30.0°'));assert((await page.locator('#object-measurements').textContent()).includes('估计米制'));
  assert(await page.locator('#open-spatial').isEnabled());assert(await page.locator('#generated-object').isHidden(),'observed scene must not depend on a generated model');
  await page.locator('#open-spatial').click();const frame=await(await page.locator('#playground-frame').elementHandle()).contentFrame();
  await frame.waitForFunction(id=>window.received?.some(m=>m.type==='panoptes:select'&&m.objectId===id),object);
  assert.equal(await page.locator('#playground-frame').getAttribute('src'),origin+'/published/workspace-observed-check/scene?revision=revision2');
  await page.selectOption('#language','en');await page.locator('[data-tab=objects]').click();
  assert((await page.locator('#object-measurements').textContent()).includes('Estimated metres'));assert((await page.locator('#object-measurements').textContent()).includes('Principal-axis tilt from vertical'));
  await page.evaluate(()=>{const c=current.evidence.candidates[0];c.measurements.scale={status:'uncalibrated',m_per_native:null};c.spatial={status:'unavailable',reason:'No connected surface'};selectCandidate(c);});
  assert.deepEqual(await page.locator('.measurement-values strong').allTextContents(),['2.000','3.000','4.000']);assert(await page.locator('#open-spatial').isDisabled());
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));assert.deepEqual(errors,[]);
  console.log(JSON.stringify({engine,passed:true,agentAutoSelect:true,observedWithoutGeneration:true,sourceDimensions:true,uncalibratedUnits:true,language:true}));
 }finally{await browser.close();}
}
(async()=>{for(const engine of(process.env.ENGINES||'chromium,webkit').split(','))await runBrowser(engine);})().catch(error=>{console.error(error);process.exitCode=1;});
