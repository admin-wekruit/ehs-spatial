// NODE_PATH=<Playwright node_modules> node tests/check_workspace_site.cjs [origin]
// Browser contract check for hosting the existing workspace beneath a static site path.
const assert=require('node:assert/strict');
const path=require('node:path');
const {chromium,webkit}=require('playwright');
const origin=process.argv[2]||'http://127.0.0.1:8792',root=origin+'/panoptes-workcell-report/',apiOrigin='https://workspace-api.example.invalid';
const run='generic-site-check',object='generic-panel';
const candidate={id:object,label:'Workcell panel',frame_id:'frame_0001',inventory_indices:[4],mask:{status:'available',resolution:'original',bbox:[10,10,80,90]},geometry:{status:'supported'},generation:{status:'done',viewer_url:'/published/viewer.html?scene=/api/generated/scene.json&object='+object},spatial:{status:'ready'},measurements:{status:'available',dimensions_native:{height:2,width:3,depth:4},scale:{status:'uncalibrated',m_per_native:null},orientation:{}}};
const detail={run_id:run,title:'Generic workcell',phase:'done',has_report:true,inspection_url:'/report/'+run+'?embedded=1',download_url:'/report/'+run+'/file',revision:'r1',evidence:{frames:[{frame_id:'frame_0001',width:100,height:100,url:'/api/reports/'+run+'/images/frame_0001'}],candidates:[candidate]},playgrounds:[{id:'observed',title:{zh:'观测空间',en:'Observed scene'},url:'/published/viewer.html?scene=/api/observed/scene.json'},{id:object,title:'Generated panel',url:candidate.generation.viewer_url}]};
async function check(engine){
 const browser=await({chromium,webkit}[engine]).launch();
 try{
  const page=await browser.newPage(engine==='webkit'?{viewport:{width:390,height:844},isMobile:true,hasTouch:true}:{viewport:{width:1440,height:1000}}),errors=[];let uploads=0,accessWrites=0,inspectionRequests=0;
  page.setDefaultTimeout(10000);page.on('pageerror',e=>{errors.push(e.message);console.error('pageerror',e.message);});
  await page.route(origin+'/**',route=>{
   const u=new URL(route.request().url());
   if(u.pathname.endsWith('/reports.html'))return route.fulfill({path:path.join(__dirname,'../ehs_spatial/static/workspace.html'),contentType:'text/html'});
   if(u.pathname==='/workspace-assets/site-config.js')return route.fulfill({body:'window.panoptesSiteConfig='+JSON.stringify({apiOrigin,siteRoot:root,workspaceRoot:root+'reports.html',workspaceAssets:origin+'/workspace-assets/'})+';',contentType:'application/javascript'});
   if(u.pathname.startsWith('/workspace-assets/'))return route.fulfill({path:path.join(__dirname,'../ehs_spatial/static',u.pathname.slice('/workspace-assets/'.length)),contentType:u.pathname.endsWith('.css')?'text/css':'application/javascript'});
   if(u.pathname.endsWith('/viewer.html'))return route.fulfill({contentType:'text/html',body:`<body><p>Same website scene</p><script>window.received=[];addEventListener('message',e=>received.push(e.data));parent.postMessage({type:'panoptes:ready',objectIds:['${object}']},location.origin);</script>`});
   return route.continue();
  });
  await page.route(apiOrigin+'/**',route=>{
   const req=route.request(),url=new URL(req.url()),headers={'Access-Control-Allow-Origin':origin,'Access-Control-Allow-Headers':'Authorization,Content-Type','Access-Control-Allow-Methods':'GET,POST,OPTIONS'};
   const result=data=>route.fulfill({json:data,headers});if(req.method()==='OPTIONS')return route.fulfill({status:204,headers});
   if(url.pathname==='/api/session'){if(req.method()==='POST'){assert.equal(req.postDataJSON().token,'test-access-code');accessWrites++;return result({can_write:true,session_token:'test-session-token'});}return result({can_write:req.headers().authorization==='Bearer test-session-token',requires_access:true});}
   if(url.pathname==='/api/reports'){if(req.method()==='POST'){assert.equal(req.headers().authorization,'Bearer test-session-token');uploads++;return result({run_id:'fresh-workcell',url:'/reports/fresh-workcell'});}return result({reports:[{run_id:run,title:'Generic workcell',phase:'done',report_url:'/reports/'+run,image_count:1}]});}
   if(url.pathname==='/api/reports/'+run||url.pathname==='/api/reports/fresh-workcell')return result(detail);
   if(url.pathname.startsWith('/api/chat/'))return result([]);
   if(url.pathname.endsWith('/images/frame_0001'))return route.fulfill({contentType:'image/svg+xml',headers,body:'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="#9aa"/></svg>'});
   if(url.pathname.startsWith('/report/')){if(!url.pathname.endsWith('/file'))inspectionRequests++;return route.fulfill({contentType:'text/html',headers,body:'<body>Original full inspection report<script>window.received=[];addEventListener("message",e=>received.push(e.data));</script></body>'});}
   throw Error('Unexpected API '+req.method()+' '+url.pathname);
  });
  await page.goto(root+'reports.html?run='+run+'&tab=objects');await page.waitForSelector('.candidate',{state:'attached'});assert(await page.locator('#objects').isVisible());assert.equal(await page.locator('.candidate[aria-pressed=true]').count(),0);assert.equal(inspectionRequests,0,'objects deep link must not fetch the hidden inspection');
  await page.goto(root+'reports.html?run='+run+'&object='+object);await page.waitForSelector('.candidate',{state:'attached'}).catch(async e=>{console.error(await page.locator('body').innerText());throw e;});
  assert.equal(await page.locator('.candidate[aria-pressed=true]').getAttribute('data-id'),object);
  assert(await page.locator('#playgrounds').isVisible());assert.equal(new URL(page.url()).searchParams.get('object'),object);
  const scene=await(await page.locator('#playground-frame').elementHandle()).contentFrame();await scene.waitForFunction(id=>received.some(m=>m.type==='panoptes:select'&&m.objectId===id),object);
  assert((await page.locator('#playground-frame').getAttribute('src')).startsWith(root+'viewer.html?'));
  assert((await page.locator('#generated-object').getAttribute('href')).startsWith(root+'viewer.html?'));
  assert.equal(inspectionRequests,0,'object playground must not fetch the hidden inspection');await page.locator('#language').selectOption('en');assert.equal(inspectionRequests,0);
  await page.locator('[data-tab=inspection]').click();await page.frameLocator('#inspection-frame').getByText('Original full inspection report').waitFor();assert.equal(inspectionRequests,1);const inspection=await(await page.locator('#inspection-frame').elementHandle()).contentFrame();await inspection.waitForFunction(()=>received.some(m=>m.type==='panoptes:report-selection'&&m.inv[0]===4));assert.equal(await page.locator('.candidate[aria-pressed=true]').getAttribute('data-id'),object);assert.equal(await page.locator('html').getAttribute('lang'),'en');
  await page.locator('[data-tab=objects]').click();await page.locator('#generated-object').click();assert.equal(new URL(page.url()).pathname,new URL(root+'reports.html').pathname);assert.equal(new URL(page.url()).searchParams.get('playground'),object);assert((await page.locator('#playground-frame').getAttribute('src')).includes('/api/generated/scene.json'));
  await page.reload();await page.waitForSelector('.candidate[aria-pressed=true]',{state:'attached'});assert((await page.locator('#playground-frame').getAttribute('src')).includes('/api/generated/scene.json'));
  await page.waitForFunction(()=>document.getElementById('source-photo').getAttribute('href')?.startsWith('blob:'));
  assert.equal(inspectionRequests,1,'reloading the generated playground must leave inspection unloaded');assert.equal(await page.locator('#inspection-frame').getAttribute('src'),null);
  const download=page.waitForEvent('download');await page.locator('#download').click();assert.equal((await download).suggestedFilename(),'generic-site-check-report.html');
  await page.locator('#unlock-workspace').click();await page.locator('#access-code').fill('test-access-code');await page.locator('#access-submit').click();await page.waitForFunction(()=>window.panoptesSession?.canWrite);assert.equal(accessWrites,1);
  assert.equal(await page.evaluate(()=>localStorage.getItem('panoptes.session.https://workspace-api.example.invalid')),'test-session-token');assert.equal(new URL(page.url()).origin,origin);
  await page.locator('#new-report').click();await page.locator('#photos').setInputFiles({name:'photo.png',mimeType:'image/png',buffer:Buffer.from('test upload data')});await page.locator('#submit-upload').click();await page.waitForURL(root+'reports.html?run=fresh-workcell');assert.equal(uploads,1);await page.frameLocator('#inspection-frame').getByText('Original full inspection report').waitFor();assert.equal(inspectionRequests,2,'a normal report visit still opens the full inspection');
  await page.goto(root+'reports.html');await page.waitForSelector('.report-card');assert.equal(await page.locator('.report-card').getAttribute('href'),root+'reports.html?run='+run);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));assert.deepEqual(errors,[]);
  console.log(JSON.stringify({engine,passed:true,websiteNavigation:true,sessionHeader:true,imagesViaAuthenticatedFetch:true,samePageSpatialSelection:true,reportFrame:true,lazyInspection:true,objectsDeepLink:true,uploadRedirect:true}));
 }finally{await browser.close();}
}
(async()=>{for(const engine of(process.env.ENGINES||'chromium,webkit').split(','))await check(engine);})().catch(e=>{console.error(e);process.exitCode=1;});
