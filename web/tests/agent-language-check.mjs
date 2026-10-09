// Drive the actual agent and public-feedback send handlers without network/model calls.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
import ts from 'typescript';
import React from 'react';
import {LANGUAGES,translate} from '../src/translate.ts';
import * as core from '../src/core.ts';

const require=createRequire(import.meta.url),code=ts.transpileModule(fs.readFileSync(new URL('../src/AgentPanel.tsx',import.meta.url),'utf8'),{
  compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,jsx:ts.JsxEmit.ReactJSX},
}).outputText;
const slots=[],effects=[],requests=[],chat={};let cursor=0,language='en';
const hooks={
  createElement(type,props,...children){if(type==='deep-chat')props.ref.current=chat;return React.createElement(type,props,...children);},
  useRef(initial){return slots[cursor++]??={current:initial};},
  useState(initial){const i=cursor++;if(!Object.hasOwn(slots,i))slots[i]=initial;return [slots[i],value=>{slots[i]=typeof value==='function'?value(slots[i]):value;}];},
  useEffect(effect,deps){const i=cursor++,previous=slots[i];if(!previous||deps.some((value,j)=>!Object.is(value,previous.deps[j])))effects.push(()=>{previous?.cleanup?.();slots[i]={deps,cleanup:effect()};});},
};
globalThis.customElements={whenDefined:async()=>{}};
const module={exports:{}};
new Function('require','module','exports',code)(name=>{
  if(name==='react')return hooks;
  if(name==='deep-chat')return {};
  if(name==='./i18n')return {useI18n:()=>({language,t:(key,params)=>translate(language,key,params)})};
  if(name==='./core')return core;
  if(name==='./App')return {ErrorNotice:()=>null};
  if(name==='./api')return {
    ApiError:class extends Error {},id:()=> 'request',
    feedbackSession:async()=>({conversationId:'conversation',capability:'feedback-capability'}),
    request:async(url,options)=>{
      if(options?.method!=='POST')return {items:[]};
      requests.push({url,body:options.body});
      return url.endsWith('/feedback')?{...options.body,status:'saved',assistantMessage:null,errorCode:null}:
        {id:'turn',status:'succeeded',response:{kind:'answer',message:'saved'}};
    },
  };
  return require(name);
},module,module.exports);
const revision={id:'revision',document:{entities:[{id:'object',observationRefs:[]}],observations:[]}};
async function render(props){
  for(let i=0;i<8;i++){
    cursor=0;const node=module.exports.AgentPanel(props);node.type(node.props);
    for(const effect of effects.splice(0))effect();
    await new Promise(resolve=>setImmediate(resolve));
  }
}
for(const feedback of [false,true]){
  slots.length=0;effects.length=0;
  const props={projectId:'project',revision,branch:{id:'branch'},entityId:'object',observationId:null,box:null,canWrite:true,onApply:()=>{},
    ...(feedback?{feedbackPublicationId:'publication'}:{})};
  for(language of [...LANGUAGES,'en']){
    await render(props);
    assert.equal(chat.textInput.disabled,false,'send handler is ready');
    const responses=[];
    await chat.connect.handler({messages:[{role:'user',text:'Inspect the object'}]},{onResponse:async response=>responses.push(response)});
    const sent=requests.at(-1);
    assert.equal(sent.body.language,language,`${feedback?'public feedback':'agent'} preserves the selected language`);
    assert.equal(sent.url,feedback?'/api/publications/publication/entities/object/feedback':'/api/projects/project/agent-turns');
    assert.equal(responses.length,1);
    assert.ok(responses[0].text);
  }
}
assert.equal(requests.length,8);
console.log('PASS: actual agent and public-feedback handlers preserve en -> zh -> nl -> en');
