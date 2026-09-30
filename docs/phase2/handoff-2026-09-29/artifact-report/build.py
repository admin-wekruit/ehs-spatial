import json, os, sys
R = os.path.dirname(os.path.abspath(__file__))
TITLES = {'me340': 'ME340 车间 3D 报告', 'samsclub': "Sam's Club 3D 报告", 'walmart': 'Walmart 3D 报告'}
links = json.load(open(f'{R}/links.json')) if os.path.exists(f'{R}/links.json') else {}
tpl = open(f'{R}/index.html', encoding='utf-8').read()
nav = {k: {'label': {'me340': 'ME340 车间', 'samsclub': "Sam's Club", 'walmart': 'Walmart'}[k], 'url': links.get(k, '#')} for k in TITLES}
for k, title in TITLES.items():
    page = tpl.replace('{{TITLE}}', title).replace('{{KEY}}', k).replace('{{LINKS}}', json.dumps(nav, ensure_ascii=False))
    open(f'{R}/{k}/index.html', 'w', encoding='utf-8').write(page)
    open(f'{R}/{k}/test.html', 'w', encoding='utf-8').write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + page + '</body></html>')
print('built', list(TITLES))
