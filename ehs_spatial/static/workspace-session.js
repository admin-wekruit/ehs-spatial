'use strict';
window.panoptesSession = (()=>{
  const I=window.panoptesI18n,site=window.panoptesSite;
  I.add({'access.button':{zh:'启用编辑',en:'Enable editing'},'access.title':{zh:'打开编辑工作区',en:'Open editing workspace'},'access.copy':{zh:'公开报告可直接浏览。上传和 Agent 操作使用工作区访问码。',en:'Published reports are open to browse. Uploads and agent actions require workspace access.'},'access.label':{zh:'工作区访问码',en:'Workspace access code'},'access.submit':{zh:'进入',en:'Continue'},'access.required':{zh:'请输入工作区访问码以继续。',en:'Enter the workspace access code to continue.'},'access.failed':{zh:'访问码无效，请重试。',en:'Invalid access code. Try again.'}});
  const $=id=>document.getElementById(id),dialog=$('access-dialog');
  let canWrite=false,accessError=false;
  function render(){for(const [id,key] of [['unlock-workspace','button'],['access-title','title'],['access-copy','copy'],['access-label','label'],['access-submit','submit']])$(id).textContent=I.t('access.'+key);if(accessError)$('access-status').textContent=I.t('access.failed');}
  render();window.addEventListener('panoptes:language-change',render);
  const ready=site.fetch('/api/session').then(async r=>{if(!r.ok)throw new Error('Could not read workspace access');const s=await r.json();canWrite=s.can_write;$('unlock-workspace').hidden=canWrite;return canWrite;});
  $('unlock-workspace').onclick=()=>dialog.showModal();$('close-access').onclick=()=>dialog.close();
  $('access-form').onsubmit=async e=>{e.preventDefault();$('access-submit').disabled=true;try{const r=await site.fetch('/api/session',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token:$('access-code').value})});if(!r.ok)throw new Error(I.t('access.failed'));const session=await r.json();site.setSession(session.session_token);$('access-code').value='';location.reload();}catch(error){accessError=true;$('access-status').textContent=I.t('access.failed');$('access-submit').disabled=false;}};
  return {ready,get canWrite(){return canWrite;},async requireWrite(){await ready;if(!canWrite){dialog.showModal();throw new Error(I.t('access.required'));}}};
})();
