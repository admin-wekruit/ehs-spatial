/* Explicitly authored UI translations only. Evidence and user content opt out with data-i18n-ignore. */
(() => {
  'use strict';
  const storageKey='panoptes.language', valid=value=>value==='zh'||value==='en';
  // srcdoc inherits its creator's origin even though its URL is about:srcdoc.
  const messageOrigin=window.origin;
  let language='zh';
  try { const saved=localStorage.getItem(storageKey); if(valid(saved))language=saved; } catch {}
  const entries=new Map(), literal=new Map(), templates=[], textRecords=new WeakMap(), attributeRecords=new WeakMap();
  const escape=value=>value.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  const interpolate=(text,params)=>text.replace(/\{(\w+)\}/g,(match,key)=>Object.hasOwn(params,key)?String(params[key]):match);
  function add(catalog) {
    for(const [key,value] of Object.entries(catalog)) {
      const pair=typeof value==='string'?{zh:key,en:value}:value;
      if(!pair||typeof pair.zh!=='string'||typeof pair.en!=='string')throw Error('Invalid UI translation: '+key);
      entries.set(key,pair);literal.set(pair.zh,pair.en);
    }
    templates.length=0;
    for(const [zh,en] of literal) {
      const names=[...zh.matchAll(/\{(\w+)\}/g)].map(m=>m[1]);
      if(names.length)templates.push({specificity:zh.replace(/\{\w+\}/g,'').length,pattern:new RegExp('^'+zh.split(/\{\w+\}/).map(escape).join('(.*?)')+'$'),names,en});
    }
    templates.sort((a,b)=>b.specificity-a.specificity);
    if(document.readyState!=='loading')refresh();
  }
  function translate(text,depth=0) {
    if(language==='zh'||depth>5)return text;
    const clean=text.trim();let result=literal.get(clean);
    if(result===undefined)for(const rule of templates) {
      const match=clean.match(rule.pattern);if(!match)continue;
      result=interpolate(rule.en,Object.fromEntries(rule.names.map((name,i)=>[name,translate(match[i+1],depth+1)])));break;
    }
    if(result===undefined&&text.includes('\n'))return text.split('\n').map(line=>translate(line,depth+1)).join('\n');
    return result===undefined?text:text.slice(0,text.indexOf(clean))+result+text.slice(text.indexOf(clean)+clean.length);
  }
  const t=(key,params={})=>interpolate(entries.get(key)?.[language]??translate(key),params);
  const ignored=node=>(node.nodeType===1?node:node.parentElement)?.closest('script,style,pre,code,textarea,.raw,[contenteditable],[data-i18n-ignore],[translate="no"]');
  const attributes=['aria-label','title','alt','placeholder'];
  function update(node) {
    if(ignored(node))return;
    if(node.nodeType===3) {
      const previous=textRecords.get(node), current=node.nodeValue;
      const source=previous&&current===previous.rendered?previous.source:current;
      const rendered=translate(source);textRecords.set(node,{source,rendered});
      if(current!==rendered)node.nodeValue=rendered;
    } else if(node.nodeType===1) {
      let records=attributeRecords.get(node);if(!records){records={};attributeRecords.set(node,records);}
      for(const key of attributes)if(node.hasAttribute(key)) {
        const current=node.getAttribute(key),previous=records[key],source=previous&&current===previous.rendered?previous.source:current,rendered=translate(source);
        records[key]={source,rendered};if(current!==rendered)node.setAttribute(key,rendered);
      }
    }
  }
  function apply(root) {
    update(root);if(root.nodeType!==1&&root.nodeType!==9)return;
    const walker=document.createTreeWalker(root,NodeFilter.SHOW_ELEMENT|NodeFilter.SHOW_TEXT,{acceptNode:node=>ignored(node)?NodeFilter.FILTER_REJECT:NodeFilter.FILTER_ACCEPT});
    while(walker.nextNode())update(walker.currentNode);
  }
  function refresh(){document.documentElement.lang=language==='zh'?'zh-CN':'en';apply(document.documentElement);document.querySelectorAll('[data-language-select]').forEach(select=>select.value=language);}
  const related=source=>source===parent&&parent!==window||[...document.querySelectorAll('iframe')].some(frame=>frame.contentWindow===source);
  function broadcast(){if(messageOrigin==='null')return;const message={type:'panoptes:language',language};if(parent!==window)parent.postMessage(message,messageOrigin);document.querySelectorAll('iframe').forEach(frame=>frame.contentWindow?.postMessage(message,messageOrigin));}
  function setLanguage(next){if(!valid(next))return false;const changed=next!==language;language=next;try{localStorage.setItem(storageKey,next);}catch{}refresh();if(changed){broadcast();window.dispatchEvent(new CustomEvent('panoptes:language-change',{detail:{language}}));}return true;}
  window.panoptesI18n={t,add,setLanguage,apply,get language(){return language;}};
  add(window.panoptesTranslations||{});delete window.panoptesTranslations;
  window.addEventListener('storage',event=>{if(event.key===storageKey&&valid(event.newValue))setLanguage(event.newValue);});
  window.addEventListener('message',event=>{
    if(messageOrigin==='null'||event.origin!==messageOrigin||!related(event.source)||!event.data||typeof event.data!=='object')return;
    if(event.data.type==='panoptes:language'&&valid(event.data.language))setLanguage(event.data.language);
    if(event.data.type==='panoptes:language-request')event.source.postMessage({type:'panoptes:language',language},messageOrigin);
  });
  function init(){
    const style=document.createElement('style');style.textContent='.language-switch{display:inline-flex;align-items:center;gap:5px;flex-shrink:0;max-width:100%}.language-switch select{font:inherit;font-size:12px;padding:6px 8px;border:1px solid currentColor;border-radius:5px;background:transparent;color:inherit;max-width:100%;cursor:pointer}.language-switch select:focus-visible{outline:2px solid currentColor;outline-offset:3px}.language-switch option{background:Canvas;color:CanvasText}';document.head.append(style);
    document.querySelectorAll('[data-language-switch]').forEach(slot=>{slot.classList.add('language-switch');const select=document.createElement('select');select.dataset.languageSelect='';select.setAttribute('aria-label','语言 / Language');select.innerHTML='<option value="zh">中文</option><option value="en">English</option>';select.addEventListener('change',()=>setLanguage(select.value));slot.append(select);});
    refresh();
    new MutationObserver(records=>{for(const record of records){if(record.type==='childList')record.addedNodes.forEach(apply);else update(record.target);}}).observe(document.documentElement,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:attributes});
    document.addEventListener('load',event=>{if(messageOrigin!=='null'&&event.target.tagName==='IFRAME')event.target.contentWindow?.postMessage({type:'panoptes:language',language},messageOrigin);},true);
    if(messageOrigin!=='null'&&parent!==window)parent.postMessage({type:'panoptes:language-request'},messageOrigin);
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init,{once:true});else init();
})();
