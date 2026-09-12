// NODE_PATH=/Users/adam/.codex/skills/gstack/node_modules SITE_CLIENT_BROWSERS=chromium,webkit node tests/check_site_client.cjs
// Real separate HTTP origins; all API responses are fixtures, never real providers.
const assert=require('node:assert/strict'),http=require('node:http'),fs=require('node:fs'),path=require('node:path');
const {chromium,webkit}=require('playwright');
const clientFile=path.join(__dirname,'../ehs_spatial/static/site-client.js');
const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jN1EAAAAASUVORK5CYII=','base64');
const token='test-short-lived-session-not-an-access-code',traffic=[];
let siteOrigin,apiOrigin;
const listen=server=>new Promise(resolve=>server.listen(0,'127.0.0.1',()=>resolve('http://127.0.0.1:'+server.address().port)));
function send(res,body,type='text/html'){res.setHeader('Content-Type',type);res.end(body);}
const api=http.createServer((req,res)=>{
 traffic.push({url:apiOrigin+req.url,authorization:req.headers.authorization||null});
 if(req.headers.origin===siteOrigin){res.setHeader('Access-Control-Allow-Origin',siteOrigin);res.setHeader('Access-Control-Allow-Headers','Authorization,Content-Type');res.setHeader('Access-Control-Allow-Methods','GET,POST,OPTIONS');}
 if(req.method==='OPTIONS'){res.statusCode=200;return res.end();}
 if(req.headers.authorization!=='Bearer '+token){res.statusCode=401;return send(res,'{"detail":"access required"}','application/json');}
 if(req.url==='/api/reports/redirect'){res.writeHead(302,{Location:siteOrigin+'/site/redirect-target'});return res.end();}
 if(req.url==='/api/reports/private/images/frame_0001'||req.url.endsWith('/aligned.png'))return send(res,png,'image/png');
 if(req.url.endsWith('/surface/surface.glb')||req.url.endsWith('/surface/face-inv.bin'))return send(res,Buffer.from([1,2,3,4]),'application/octet-stream');
 if(req.url.endsWith('/bad.json')){res.statusCode=404;return send(res,'missing');}
 if(req.url==='/report/private/viewer?failure=1'){res.statusCode=404;return send(res,'missing viewer');}
 if(req.url.startsWith('/report/private/viewer'))return send(res,`<body><script id="surface-data" type="application/json">{"asset_url":"/report/private/surface/surface.glb","face_map_url":"/report/private/surface/face-inv.bin"}</script><script src="/workspace-assets/surface-probe.js"></script></body>`);
 if(req.url.startsWith('/report/private')){const letter=new URL(req.url,apiOrigin).searchParams.get('letter')||'A';return setTimeout(()=>send(res,`<body><h1>${letter}</h1>${letter==='L'?'<div style="height:4000px" aria-hidden="true"></div>':''}<iframe title="native" src="/report/private/viewer${letter==='E'?'?failure=1':''}"></iframe><script>parent.postMessage({fixture:'report',origin:location.origin,letter:'${letter}'},location.origin)</script></body>`),letter==='B'?180:5);}
 if(req.url.endsWith('/metrics.html'))return send(res,'<body><h1>Private metric</h1><img id="alignment" src="aligned.png"><a id="return" href="index.html">Return</a><a id="record" href="comparisons.json">Record</a><a id="bad" href="bad.json">Missing record</a></body>');
 return send(res,'{"ok":true}','application/json');
});
const site=http.createServer((req,res)=>{
 traffic.push({url:siteOrigin+req.url,authorization:req.headers.authorization||null});
 if(req.url==='/site/site-client.js')return send(res,fs.readFileSync(clientFile),'application/javascript');
 if(req.url==='/site/workspace-assets/surface-probe.js')return send(res,`(async()=>{const data=JSON.parse(document.getElementById('surface-data').textContent);const lengths=await Promise.all(Object.values(data).map(u=>fetch(u).then(r=>r.arrayBuffer()).then(b=>b.byteLength)));parent.postMessage({fixture:'native',origin:location.origin,lengths},location.origin)})();`,'application/javascript');
 if(req.url.startsWith('/site/reports.html'))return send(res,`<!doctype html><body><iframe id="report"></iframe><script>window.messages=[];window.revoked=[];window.originalFetch=fetch;addEventListener('message',e=>messages.push({origin:e.origin,data:e.data}));const revoke=URL.revokeObjectURL.bind(URL);URL.revokeObjectURL=u=>{revoked.push(u);revoke(u)};window.panoptesSiteConfig=${JSON.stringify({apiOrigin,siteRoot:siteOrigin+'/site/',workspaceRoot:siteOrigin+'/site/reports.html',workspaceAssets:siteOrigin+'/site/workspace-assets/'})};</script><script src="site-client.js"></script></body>`);
 return send(res,'static');
});
(async()=>{
 apiOrigin=await listen(api);siteOrigin=await listen(site);const failures=[];
 try{for(const name of (process.env.SITE_CLIENT_BROWSERS||'chromium').split(',')){
  const engine={chromium,webkit}[name];if(!engine)throw Error('Unknown browser '+name);
  const executablePath=process.env[name.toUpperCase()+'_EXECUTABLE'];const browser=await engine.launch({timeout:15000,...(executablePath?{executablePath}:{})});const context=await browser.newContext(),page=await context.newPage();
  try{
   await page.goto(siteOrigin+'/site/reports.html?run=private');
   await page.evaluate(t=>panoptesSite.setSession(t),token);
   assert.equal(await page.evaluate(()=>fetch===originalFetch),true);
   const routes=await page.evaluate(()=>({id:panoptesSite.reportId(),report:panoptesSite.pageURL('/reports/private'),published:panoptesSite.pageURL('/published/viewer.html?scene=components/scene.json'),metric:panoptesSite.pageURL('/api/reports/private/generations/object_a/assets/metrics.html')}));
   assert.equal(routes.id,'private');assert.equal(routes.report,siteOrigin+'/site/reports.html?run=private');
   assert.equal(routes.published,siteOrigin+'/site/viewer.html?scene=components/scene.json');
   assert.equal(new URL(routes.metric).pathname,'/site/report-artifact.html');assert.equal(new URL(routes.metric).searchParams.get('source'),'/api/reports/private/generations/object_a/assets/metrics.html');
   await page.evaluate(async()=>{
    if(!(await panoptesSite.fetch('/api/agent',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'})).ok)throw Error('Bearer transport failed');
    await panoptesSite.fetch('/site/static',{headers:{Authorization:'Bearer must-not-leak'}});
    for(const url of ['https://evil.invalid/api/reports/private','/outside']){try{await panoptesSite.fetch(url);throw Error('untrusted URL accepted');}catch(e){if(e.message==='untrusted URL accepted')throw e;}}
    let blocked=false;try{await panoptesSite.fetch('/api/reports/redirect')}catch(e){blocked=true;}if(!blocked)throw Error('API redirect accepted');
    const blob=await panoptesSite.blobURL('/api/reports/private/images/frame_0001');if(!blob.startsWith('blob:'+location.origin))throw Error('Private image not browser-local');
   });
   const extra=await context.newPage();await extra.goto(siteOrigin+'/site/reports.html?run=private');assert.equal(await extra.evaluate(async()=>(await panoptesSite.fetch('/api/reports/private')).status),200);await extra.close();
   const nativeRequests=()=>traffic.filter(x=>x.url.startsWith(apiOrigin)&&(/\/report\/private\/viewer/.test(x.url)||x.url.includes('/surface/'))).length;
   const beforeOffscreen=nativeRequests();
   await page.evaluate(()=>panoptesSite.reportFrame(document.getElementById('report'),'/report/private?letter=L'));
   await page.waitForFunction(()=>messages.some(x=>x.data.fixture==='report'&&x.data.letter==='L'));
   // Give IntersectionObserver its rendering turns; no network or fixed-size mesh is needed to prove the gate.
   await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
   await page.waitForTimeout(200);
   assert.equal(nativeRequests(),beforeOffscreen,'Offscreen legacy viewer must not request HTML, GLB or face-map data');
   assert.equal(await page.evaluate(()=>document.getElementById('report').contentDocument.querySelector('iframe').getAttribute('src')),null);
   await page.evaluate(()=>document.getElementById('report').contentDocument.querySelector('iframe').scrollIntoView());
   await page.waitForFunction(()=>document.getElementById('report').contentWindow.frames[0]?.document?.getElementById('surface-data')?.textContent.includes('blob:'));
   assert.ok(nativeRequests()>beforeOffscreen,'Visible legacy viewer must load after scrolling into view');
   const lazyFrame=page.frames().find(f=>f.parentFrame()===page.frames().find(x=>x.parentFrame()===page.mainFrame()));
   assert.equal(await lazyFrame.evaluate(()=>location.origin),siteOrigin);
   await page.evaluate(()=>panoptesSite.reportFrame(document.getElementById('report'),'/report/private?letter=A'));
   await page.waitForFunction(()=>messages.some(x=>x.data.fixture==='report'&&x.data.letter==='A'));
   await page.waitForFunction(()=>document.getElementById('report').contentWindow.frames[0]?.document?.getElementById('surface-data')?.textContent.includes('blob:'));
   await page.waitForFunction(()=>document.getElementById('report').contentWindow.frames[0]?.document?.querySelector('script[src]')?.src.includes('/site/workspace-assets/'));
   const native=page.frames().find(f=>f.parentFrame()===page.frames().find(x=>x.parentFrame()===page.mainFrame()));
   assert.equal(await native.evaluate(()=>location.origin),siteOrigin);
   assert.ok((await native.evaluate(()=>JSON.parse(document.getElementById('surface-data').textContent))).asset_url.startsWith('blob:'+siteOrigin));
   const old=await page.locator('#report').getAttribute('src');
   await page.evaluate(async()=>{const f=document.getElementById('report');await Promise.all([panoptesSite.reportFrame(f,'/report/private?letter=B'),panoptesSite.reportFrame(f,'/report/private?letter=C')]);});
   await page.waitForFunction(()=>document.getElementById('report').contentDocument.querySelector('h1')?.textContent==='C');
   if(!await page.evaluate(url=>revoked.includes(url),old))failures.push(name+': visible blob A leaked when B pending then C replaced it');
   await page.evaluate(()=>panoptesSite.reportFrame(document.getElementById('report'),'/api/reports/private/generations/object_a/assets/metrics.html'));
   await page.waitForFunction(()=>document.getElementById('report').contentDocument?.querySelector('h1')?.textContent==='Private metric');
   const metricFrame=page.frames().find(f=>f.parentFrame()===page.mainFrame());
   const img=await metricFrame.locator('#alignment').evaluate(e=>({src:e.src,loaded:e.complete&&e.naturalWidth>0}));
   if(!img.src.startsWith('blob:'+siteOrigin)||!img.loaded)failures.push(name+': private metric relative image did not resolve through authenticated blob fetch: '+img.src);
   const link=await metricFrame.locator('#return').getAttribute('href');
   const returned=new URL(link,siteOrigin);const returnedScene=returned.searchParams.get('scene');
   if(returned.pathname!=='/site/viewer.html'||!returnedScene||new URL(returnedScene,apiOrigin).pathname!=='/api/reports/private/generations/object_a/assets/scene.json')failures.push(name+': private metric return must target the same candidate scene in current site viewer: '+link);
   const archived=await page.evaluate(()=>panoptesSite.pageURL('/api/reports/private/generations/object_a/revisions/revision_123/assets/index.html'));
   const archivedScene=new URL(archived).searchParams.get('scene');
   if(!archivedScene||new URL(archivedScene,apiOrigin).pathname!=='/api/reports/private/generations/object_a/revisions/revision_123/assets/scene.json')failures.push(name+': viewer navigation lost historical asset revision');
   const downloaded=page.waitForEvent('download');await metricFrame.locator('#record').click();assert.equal((await downloaded).suggestedFilename(),'comparisons.json');
   await metricFrame.locator('#bad').click();await metricFrame.getByRole('alert').filter({hasText:'Download HTTP 404'}).waitFor();
   await page.evaluate(()=>panoptesSite.reportFrame(document.getElementById('report'),'/report/private?letter=E'));
   await page.waitForFunction(()=>document.getElementById('report').contentDocument?.querySelector('[role=alert]')?.textContent==='Report HTTP 404');
   console.log(name+': routing, credential boundary, redirects, private blobs, real Blob origin, offscreen network gate and nested surface transport passed');
  }catch(e){failures.push(name+': '+e.stack);}finally{await browser.close();}
 }
 assert.ok(!traffic.some(x=>x.url.includes(token)),'Session token appeared in URL');
 assert.ok(!traffic.some(x=>x.url.startsWith(siteOrigin)&&x.authorization),'Bearer credential leaked to static origin');
 assert.ok(!traffic.some(x=>x.url.endsWith('/redirect-target')),'API redirect was followed');
 if(failures.length)throw Error(failures.join('\n'));
 console.log('PASS: selected browser engines; zero live API/provider calls');
 }finally{api.close();site.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
