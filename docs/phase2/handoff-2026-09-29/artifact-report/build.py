import json, os, shutil, sys
WRAP = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>{}</body></html>'
R = os.path.dirname(os.path.abspath(__file__))
TITLES = {'me340': 'ME340 车间 3D 报告', 'samsclub': "Sam's Club 3D 报告", 'walmart': 'Walmart 3D 报告'}
links = json.load(open(f'{R}/links.json')) if os.path.exists(f'{R}/links.json') else {}
tpl = open(f'{R}/index.html', encoding='utf-8').read()
nav = {k: {'label': {'me340': 'ME340 车间', 'samsclub': "Sam's Club", 'walmart': 'Walmart'}[k], 'url': links.get(k, '#')} for k in TITLES}
for k, title in TITLES.items():
    page = tpl.replace('{{TITLE}}', title).replace('{{KEY}}', k).replace('{{LINKS}}', json.dumps(nav, ensure_ascii=False))
    art = page.replace('{{GLB_JSON}}', 'true')
    open(f'{R}/{k}/index.html', 'w', encoding='utf-8').write(art)  # claude.ai artifact page (the skeleton is added at publish)
    open(f'{R}/{k}/test.html', 'w', encoding='utf-8').write(WRAP.format(art))
    # GitHub Pages copy: plain .glb, relative links between the three, a full HTML document
    pnav = {kk: dict(v, url=f'../{kk}/') for kk, v in nav.items()}
    pages = tpl.replace('{{TITLE}}', title).replace('{{KEY}}', k).replace('{{LINKS}}', json.dumps(pnav, ensure_ascii=False)).replace('{{GLB_JSON}}', 'false')
    d = f'{R}/pages/video/{k}'; os.makedirs(d, exist_ok=True)
    open(f'{d}/index.html', 'w', encoding='utf-8').write(WRAP.format(pages))
    for f in os.listdir(f'{R}/{k}'):
        if f.endswith('.glb') or f in ('data.json', 'video.mp4'):
            shutil.copy(f'{R}/{k}/{f}', f'{d}/{f}')
open(f'{R}/pages/video/index.html', 'w', encoding='utf-8').write(WRAP.format(open(f'{R}/landing.html', encoding='utf-8').read()))
print('built', list(TITLES))
