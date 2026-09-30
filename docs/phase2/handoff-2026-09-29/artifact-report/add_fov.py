import json, glob, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export import VIDEOS, RUNS, latest_patches
for key, (run, rep, title) in VIDEOS.items():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), key, 'data.json')
    if not os.path.exists(p): continue
    d = json.load(open(p)); P = latest_patches(f'{RUNS}/{run}/mirror/reports/{rep}')
    shots = {int(s['index']): s for s in P['cameras']['data']['shots']}
    for c in d['cameras']:
        s = shots[c['shot']]; K = s['K'][0] if isinstance(s['K'][0][0], list) else s['K']
        c['fov_y'] = round(math.degrees(2 * math.atan(s['wh'][1] / 2 / K[1][1])), 2)
    json.dump(d, open(p, 'w'), ensure_ascii=False, separators=(',', ':'))
    print(key, [c['fov_y'] for c in d['cameras']])
