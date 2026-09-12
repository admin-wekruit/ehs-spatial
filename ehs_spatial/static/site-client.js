'use strict';
// One frontend on both static hosting and the application server. Only API traffic crosses origins.
window.panoptesSite = (() => {
  const config = window.panoptesSiteConfig;
  const apiOrigin = new URL(config.apiOrigin).origin;
  const siteRoot = new URL(config.siteRoot);
  const key = 'panoptes.session.' + apiOrigin;
  const blobs = new Map(), frames = new WeakMap();
  const apiPath = path => /^\/(api\/(?:session|reports|agent|chat|review)(?:\/|$)|report\/[^/]+(?:\/|$))/.test(path);
  function resolve(value, base = location.href) {
    const raw = String(value);
    if (!raw || raw.includes('\\') || raw.startsWith('//')) throw Error('Invalid resource URL');
    let url = new URL(raw, base);
    if (raw.startsWith('/') && apiPath(url.pathname)) url = new URL(url.pathname + url.search + url.hash, apiOrigin);
    if (url.username || url.password || !['http:', 'https:'].includes(url.protocol)) throw Error('Invalid resource URL');
    return url;
  }
  function trusted(url) {
    return url.origin === apiOrigin && apiPath(url.pathname) || url.origin === siteRoot.origin && url.pathname.startsWith(siteRoot.pathname)
      || url.origin === location.origin && url.pathname.startsWith(new URL(config.workspaceAssets).pathname);
  }
  async function request(value, options = {}) {
    const url = resolve(value);
    if (!trusted(url)) throw Error('Resource is outside this workspace');
    const headers = new Headers(options.headers);
    // Never forward session credentials to static assets, URLs supplied by a model, or redirects.
    if (url.origin === apiOrigin && apiPath(url.pathname)) {
      const token = localStorage.getItem(key);
      if (token) headers.set('Authorization', 'Bearer ' + token);
    } else headers.delete('Authorization');
    return fetch(url.href, {...options, headers, credentials: url.origin === location.origin ? 'same-origin' : 'omit', redirect: 'error'});
  }
  function assetURL(value, base = location.href) {
    const url = resolve(value, base);
    const backendAsset = url.origin === apiOrigin && (/^\/api\/reports\/[^/]+\//.test(url.pathname) || /^\/report\/[^/]+\/surface\//.test(url.pathname));
    if (!backendAsset && !(url.origin === siteRoot.origin && url.pathname.startsWith(siteRoot.pathname))) throw Error('Scene asset is outside this workspace');
    return url.href;
  }
  function pageURL(value) {
    const raw = String(value);
    const url = resolve(raw, raw.startsWith('/') ? apiOrigin : location.href);
    if (url.origin === apiOrigin) {
      const report = url.pathname.match(/^\/reports(?:\/([^/]+))?\/?$/);
      if (report) {
        const target = new URL(config.workspaceRoot);
        if (report[1]) {
          if (target.pathname.endsWith('.html')) target.searchParams.set('run', decodeURIComponent(report[1]));
          else target.pathname += '/' + report[1];
        }
        for (const [key, value] of url.searchParams) target.searchParams.set(key, value);
        target.hash = url.hash;
        return target.href;
      }
      if (url.pathname.startsWith('/published/')) return new URL(url.pathname.slice('/published/'.length) + url.search + url.hash, siteRoot).href;
      if (/^\/api\/reports\/[^/]+\/.+\.html$/.test(url.pathname)) {
        if (/\/(index|viewer)\.html$/.test(url.pathname)) {
          const target = new URL('viewer.html', siteRoot);
          target.searchParams.set('scene', new URL('scene.json', url).pathname);
          for (const [key, value] of url.searchParams) if (key !== 'scene') target.searchParams.set(key, value);
          return target.href;
        }
        const target = new URL('report-artifact.html', siteRoot);
        target.searchParams.set('source', url.pathname + url.search);
        return target.href;
      }
    }
    if (url.origin === siteRoot.origin && url.pathname.startsWith(siteRoot.pathname)) return url.href;
    throw Error('Page is outside this website');
  }
  async function blobURL(value) {
    const url = resolve(value);
    if (!trusted(url)) throw Error('Image is outside this workspace');
    if (!(url.origin === apiOrigin && apiPath(url.pathname))) return url.href;
    if (!blobs.has(url.href)) blobs.set(url.href, request(url.href).then(async response => {
      if (!response.ok) throw Error('Resource HTTP ' + response.status);
      return URL.createObjectURL(await response.blob());
    }).catch(error => { blobs.delete(url.href); throw error; }));
    return blobs.get(url.href);
  }
  async function download(url, filename) {
    const response = await request(url);
    if (!response.ok) throw Error('Download HTTP ' + response.status);
    const objectURL = URL.createObjectURL(await response.blob());
    const link = document.createElement('a');
    link.href = objectURL; link.download = filename; link.click();
    setTimeout(() => URL.revokeObjectURL(objectURL), 60000);
  }
  async function reportFrame(iframe, source) {
    const ticket = {};
    frames.get(iframe)?.observer?.disconnect();
    frames.set(iframe, ticket);
    const response = await request(source);
    if (!response.ok) throw Error('Report HTTP ' + response.status);
    const doc = new DOMParser().parseFromString(await response.text(), 'text/html');
    const base = doc.createElement('base'); base.href = siteRoot.href; doc.head.prepend(base);
    for (const script of doc.querySelectorAll('script[src^="/workspace-assets/"]')) script.src = new URL(script.getAttribute('src').slice('/workspace-assets/'.length), config.workspaceAssets).href;
    for (const link of doc.querySelectorAll('link[href^="/workspace-assets/"]')) link.href = new URL(link.getAttribute('href').slice('/workspace-assets/'.length), config.workspaceAssets).href;
    await Promise.all([...doc.querySelectorAll('img[src],image[href]')].map(async image => {
      const attribute = image.tagName.toLowerCase() === 'image' ? 'href' : 'src', value = image.getAttribute(attribute);
      if (!value || /^(data:|blob:)/.test(value)) return;
      image.setAttribute(attribute, await blobURL(assetURL(value, resolve(source).href)));
    }));
    // Saved viewer geometry stays byte-identical; only its protected asset addresses become browser-local URLs.
    const manifest = doc.getElementById('surface-data');
    if (manifest) {
      const data = JSON.parse(manifest.textContent);
      if (data) {
        await Promise.all(['asset_url', 'face_map_url'].map(async key => {
          if (data[key] && !data[key].startsWith('data:')) data[key] = await blobURL(assetURL(data[key], resolve(source).href));
        }));
        manifest.textContent = JSON.stringify(data).replace(/</g, '\\u003c');
      }
    }
    for (const child of doc.querySelectorAll('iframe[src]')) {
      const url = resolve(child.getAttribute('src'), resolve(source).href);
      if (url.origin === apiOrigin && /^\/report\/[^/]+\/viewer$/.test(url.pathname)) {
        child.dataset.siteReport = url.href; child.removeAttribute('src');
        child.srcdoc = '<p role="status">Loading 3D… / 正在加载 3D…</p>';
      }
    }
    for (const anchor of doc.querySelectorAll('a[href^="/reports"],a[href^="/published/"]')) anchor.href = pageURL(anchor.getAttribute('href'));
    for (const anchor of doc.querySelectorAll('a[href]')) {
      const value = anchor.getAttribute('href');
      if (!value || value.startsWith('#')) continue;
      const url = resolve(value, resolve(source).href);
      if (url.origin === apiOrigin && /^\/api\/reports\/[^/]+\//.test(url.pathname)) {
        if (url.pathname.endsWith('.html')) anchor.href = pageURL(url.href);
        else { anchor.dataset.siteDownload = assetURL(url.href); anchor.href = '#download'; anchor.removeAttribute('target'); }
      }
    }
    if (frames.get(iframe) !== ticket) return;
    const previousURL = iframe.getAttribute('src');
    const objectURL = URL.createObjectURL(new Blob(['<!doctype html>' + doc.documentElement.outerHTML], {type: 'text/html'}));
    ticket.url = objectURL;
    iframe.addEventListener('load', () => {
      if (frames.get(iframe) !== ticket) return;
      if (previousURL?.startsWith('blob:')) URL.revokeObjectURL(previousURL);
      for (const anchor of iframe.contentDocument.querySelectorAll('a[data-site-download]')) anchor.addEventListener('click', event => {
        event.preventDefault(); download(anchor.dataset.siteDownload, new URL(anchor.dataset.siteDownload).pathname.split('/').pop()).catch(error => {
          const p = iframe.contentDocument.createElement('p'); p.setAttribute('role', 'alert'); p.textContent = error.message; anchor.after(p);
        });
      });
      // A saved report can contain a large legacy model below the fold. Fetch
      // it when its pane is visible so it cannot block login and active scenes.
      ticket.observer = new iframe.contentWindow.IntersectionObserver(entries => {
        for (const entry of entries) if (entry.isIntersecting && entry.intersectionRatio >= .25) {
          const child = entry.target; ticket.observer.unobserve(child);
          reportFrame(child, child.dataset.siteReport).catch(error => {
            const p = iframe.contentDocument.createElement('p'); p.setAttribute('role', 'alert'); p.textContent = error.message; child.replaceWith(p);
          });
        }
      }, {threshold: .25});
      iframe.contentDocument.querySelectorAll('iframe[data-site-report]').forEach(child => ticket.observer.observe(child));
    }, {once: true});
    // Blob documents retain the website origin, including for selection messages and Safari.
    iframe.removeAttribute('srcdoc'); iframe.src = objectURL;
  }
  return {fetch: request, assetURL, pageURL, blobURL, download, reportFrame,
    reportId: () => new URLSearchParams(location.search).get('run') || (location.pathname.match(/^\/reports\/([^/]+)$/)?.[1] ? decodeURIComponent(location.pathname.split('/')[2]) : null),
    setSession(token) { if (typeof token !== 'string' || !token || token.length > 4096) throw Error('Invalid session'); localStorage.setItem(key, token); },
  };
})();
