"""Interactive per-run report — the full linked experience ported from
the test-set report generator: click an object in the PHOTO and it
highlights; click plan polygons; multi-select for the distance matrix;
detection checklist, verdict gloss, reprojection check, refinement
gallery. The 3D viewer iframe embeds lazily only when the run has a
compact viewer; everything else is self-contained data URIs, so the file
opens offline anywhere."""

import base64, io, json, re
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from .providers.sam3 import decode_coco_rle

_CSS = '\n:root{--paper:#F5F7F6;--surface:#FFF;--ink:#1B2327;--muted:#5A676F;--line:#DAE0DE;--accent:#0E7490;--accent-ink:#0A5B71;\n--pass:#178A50;--pass-bg:#E4F3EA;--warn:#A9740E;--warn-bg:#F7EEDC;--fail:#C13B3B;--fail-bg:#F9E7E5;--insuff:#5D6B76;--insuff-bg:#E9EDEF;--card:#FCFDFC}\n@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--paper:#101619;--surface:#161D21;--ink:#E7ECEC;--muted:#9AA7AC;--line:#263136;--accent:#41BBD3;--accent-ink:#6ED0E3;\n--pass:#41BE7E;--pass-bg:#12301F;--warn:#DFA83D;--warn-bg:#33280F;--fail:#E4706A;--fail-bg:#3A1B18;--insuff:#93A2AC;--insuff-bg:#232D33;--card:#141B1F}}\n:root[data-theme="dark"]{--paper:#101619;--surface:#161D21;--ink:#E7ECEC;--muted:#9AA7AC;--line:#263136;--accent:#41BBD3;--accent-ink:#6ED0E3;\n--pass:#41BE7E;--pass-bg:#12301F;--warn:#DFA83D;--warn-bg:#33280F;--fail:#E4706A;--fail-bg:#3A1B18;--insuff:#93A2AC;--insuff-bg:#232D33;--card:#141B1F}\n*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.7 \'Noto Sans SC\',\'IBM Plex Sans\',sans-serif}\n.wrap{max-width:1020px;margin:0 auto;padding:44px 22px 90px}\nh1{font-family:\'IBM Plex Sans\',\'Noto Sans SC\',sans-serif;font-size:clamp(24px,4vw,34px);margin:0 0 6px;text-wrap:balance}\nh4{font-family:\'IBM Plex Sans\',sans-serif;margin:20px 0 8px}\n.eyebrow{font-family:\'IBM Plex Mono\',monospace;font-size:11.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:12px}\n.eyebrow b{color:var(--accent-ink)}\n.lede{color:var(--muted);max-width:47em;margin:0 0 6px}\n.mono{font-family:\'IBM Plex Mono\',monospace;font-variant-numeric:tabular-nums}\n.pill{display:inline-block;padding:1px 9px;border-radius:3px;font-size:12.5px;font-weight:600;white-space:nowrap}\n.pill.fail{background:var(--fail-bg);color:var(--fail)}.pill.warn{background:var(--warn-bg);color:var(--warn)}\n.pill.insuff{background:var(--insuff-bg);color:var(--insuff)}.pill.pass{background:var(--pass-bg);color:var(--pass)}\n.legend{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:14px 18px;margin:20px 0}\n.legend div{margin:4px 0}\ndetails.case{background:var(--surface);border:1px solid var(--line);border-radius:5px;margin:12px 0;overflow:hidden}\ndetails.case summary{display:flex;align-items:center;gap:14px;padding:10px 16px;cursor:pointer;list-style:none}\ndetails.case summary::-webkit-details-marker{display:none}\ndetails.case summary img{width:96px;height:64px;object-fit:cover;border-radius:3px;border:1px solid var(--line)}\ndetails.case[open] summary{border-bottom:1px solid var(--line)}\n.cid{font-weight:600}\n.cmeta{margin-left:auto;color:var(--muted);font-size:12.5px}\n.body{padding:16px 18px}\n.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin:10px 0}\nfigure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:4px;overflow:hidden}\nfigure img{max-width:100%;display:block;cursor:zoom-in}\nfigcaption{padding:8px 12px;font-size:12.5px;color:var(--muted)}\nfigcaption b{color:var(--ink)}\n.policy{border:1px solid var(--line);border-radius:4px;padding:10px 12px;margin:8px 0;background:var(--card)}\n.pname{font-weight:600;margin-left:6px}\n.reason{margin:6px 0 0;padding-left:10px;border-left:3px solid var(--line);font-size:13.5px}\n.reason.vio{border-left-color:var(--fail)}\n.reason .raw{color:var(--muted);font-size:11.5px;overflow-x:auto}\n.tblwrap{overflow-x:auto;border:1px solid var(--line);border-radius:4px}\ntable{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--surface)}\nth,td{padding:7px 11px;border-bottom:1px solid var(--line);text-align:left}\nth{font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}\n.hint{font-size:11.5px;color:var(--muted);margin-top:14px;overflow-x:auto}\n.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}\n.tile{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:12px 14px}\n.tile .v{font-family:\'IBM Plex Mono\',monospace;font-size:22px;font-weight:600}\n.tile .l{font-size:12px;color:var(--muted)}\n#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:flex;align-items:center;justify-content:center;z-index:50;cursor:zoom-out}\n#lb[hidden]{display:none}\n#lb img{max-width:96vw;max-height:94vh}\na{color:var(--accent-ink)}\n'


def uri(path, maxw=760, q=68):
    im = Image.open(path).convert('RGB')
    if im.width > maxw: im = im.resize((maxw, round(im.height*maxw/im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b,'JPEG',quality=q,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
def thumb(p): return uri(p,340,66)
def srcdoc(p): return open(p,encoding='utf-8').read().replace('&','&amp;').replace('"','&quot;')
PALETTE = ["#c1121f","#1d4ed8","#047857","#b45309","#6d28d9","#0e7490","#9d174d","#4d7c0f","#7c2d12","#334155"]
ZH = {'FAIL':('fail','FAIL 违规'),'NEEDS_REVIEW':('warn','NEEDS REVIEW 待复核'),'INSUFFICIENT_EVIDENCE':('insuff','证据不足'),'PASS':('pass','PASS')}
LBL = {'industrial robot arm':'机械臂','robotic arm':'机械臂','emergency stop button':'急停按钮','safety fence':'安全围栏',
       'safety light':'信号灯','warning sign':'警示牌','step ladder':'梯子','portable work platform':'移动平台',
       'safety sensor':'安全传感器(光幕)','worker':'工人','floor marking':'地面标线','control panel':'控制柜',
       'control box':'控制箱','hard hat':'安全帽','fire extinguisher':'灭火器','floor mat':'地垫','cable':'线缆','signboard':'指示牌','sign':'指示牌',
       'indicator light':'指示灯','door interlock switch':'门联锁开关','sloped surface':'斜坡挡板'}
def gloss(w):
    if 'no entity matching subject labels' in w: return '场景中没有该规则要测量的对象类别——无对象可量，不是漏检'
    if 'within the' in w and 'error budget' in w: return '测量值落在单目误差带内，拒绝确认——需复核或多角度补拍'
    if 'fence segments merged' in w: return '多段围栏合并为一个边界 hull 后测距'
    if 'fragment(s) discarded' in w: return '部分围栏碎片未过证据门'
    if 'reduced capture' in w: return '单张照片：证据冗余降低，判定带加宽'
    if 'not evaluable' in w or 'unsupported' in w: return '该政策编译期即声明不支持（故意保留的拒绝样本）'
    return ''
POLICY_ZH = {'p01-movable-equipment-clearance':'p01 · 移动设备-围栏间距 ≥0.6 m','p05-pallet-max-height':'p05 · 堆放最大高度','p02-robot-guarding-separation':'p02 · 机器人围护间距（不支持样本）'}
CASE_NOTES = {
 'real-clean-03': '<div class="reason" style="margin:10px 0;border-left-color:var(--pass)"><b>PASS</b>：平台-围栏间距过带。</div>',
 'gen-07': '<div class="reason" style="margin:10px 0"><b>违规样本</b>：工人在条纹区内。</div>',
 'gen-08': '<div class="reason" style="margin:10px 0"><b>违规样本</b>：梯子靠围栏。</div>',
 'gen-01': '<div class="reason" style="margin:10px 0"><b>补测已回灌</b>：光幕柱×2 + 左右侧板×3。围栏族矩形强制贴车间主轴（直角回归）。</div>',
}

def snap_rect(footprint, theta_deg):
    if theta_deg is None: return None
    pts=np.asarray(footprint,float)
    if len(pts)<3: return None
    t=np.radians(theta_deg)
    rot=np.array([[np.cos(-t),-np.sin(-t)],[np.sin(-t),np.cos(-t)]])
    q=pts@rot.T
    x0,x1=np.percentile(q[:,0],2),np.percentile(q[:,0],98)
    y0,y1=np.percentile(q[:,1],2),np.percentile(q[:,1],98)
    c=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
    back=c@np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]]).T
    return [[round(float(x),3),round(float(y),3)] for x,y in back]

def case_objects(rid):
    run = Path('runs')/rid
    inv = json.loads((run/'inventory'/'inventory.json').read_text())
    objs=[dict(o) for o in inv['objects'] if not o.get('off_plan_reason') and o['label'] not in ('floor','factory floor','ceiling','concrete floor','ceiling structure')]
    theta=inv.get('manhattan_theta_deg')
    rf = run/'refinements.json'
    if rf.exists():
        seen={o['refine_slug'] for o in objs if o.get('refine_slug')}
        for item in json.loads(rf.read_text()):
            slug=f"{item['label'].replace(' ','_')}_{'_'.join(str(b) for b in item['box'])}"
            if slug in seen: continue
            seen.add(slug)
            if 'footprint_xy' not in item: continue
            objs.append({'label':item['label'],'instance':1000+len(seen),'height_m':item['height_m'],
                         'size_m':item.get('extent_m','0x0'),'camera_dist_m':item['camera_dist_m'],
                         'centroid_xy':item['centroid_xy'],'footprint':item['footprint_xy'],
                         'footprint_method':'refine','tilt_deg':None,'_slug':slug,
                         'rect_snapped':snap_rect(item['footprint_xy'], theta),
                         'footprint_area_m2':1.0})
    return inv, objs

def mask_index_png(rid, objs):
    run = Path('runs')/rid
    frame_dir = sorted((run/'geometry'/'frames').iterdir())[0]
    valid = np.load(frame_dir/'valid_mask.npy')
    H, W = valid.shape
    with Image.open(next((run/'input').glob('image_*'))) as full: FW,FH = full.size
    # decode every mask first, then paint largest-first so the smallest mask
    # wins every overlapping pixel (a big fence refinement must not swallow
    # the slope's click region)
    layers = []
    for i, o in enumerate(objs):
        try:
            if o.get('footprint_method') == 'refine':
                cache = run/'refinements'/f"{o['_slug']}.json"
                if not cache.exists(): continue
                r = json.loads(cache.read_text()); rles = r.get('rle') or []
                if isinstance(rles, str): rles = [rles]
                if not rles: continue
                best = int(np.argmax(r.get('scores') or [1.0]*len(rles)))
                m = decode_coco_rle(rles[best], height=FH, width=FW).astype(np.uint8)
                m = np.asarray(Image.fromarray(m*255).resize((W, H))) > 127
            elif o.get('refine_slug'):
                cache = run/'refinements'/f"{o['refine_slug']}.json"
                if not cache.exists(): continue
                r = json.loads(cache.read_text()); rles = r.get('rle') or []
                if isinstance(rles, str): rles = [rles]
                if not rles: continue
                best = int(np.argmax(r.get('scores') or [1.0]*len(rles)))
                m = decode_coco_rle(rles[best], height=FH, width=FW).astype(np.uint8)
                m = np.asarray(Image.fromarray(m*255).resize((W, H))) > 127
            else:
                slug = re.sub(r'[^a-z0-9]+','_',o['label']).strip('_')
                cache = run/'inventory'/'sam'/f"frame_0001__{slug}.json"
                if not cache.exists(): continue
                r = json.loads(cache.read_text()); rles = r.get('rle') or []
                if isinstance(rles, str): rles = [rles]
                if o['instance'] >= len(rles): continue
                m = np.zeros((H, W), bool)
                for member in o.get('merged_instances') or [o['instance']]:
                    m |= decode_coco_rle(rles[member], height=H, width=W).astype(bool)
        except Exception:
            continue
        if m.any(): layers.append((int(m.sum()), i, m))
    idx = np.zeros((H, W), np.uint8)
    # perspective-aware ownership: paint FAR structures first so the NEAR
    # one wins every overlapping pixel (a panel behind a rail must not
    # claim the rail's face); equal-distance ties: bigger first so the
    # smaller object stays clickable
    def order_key(t):
        px_count, i, _ = t
        d = objs[i].get('camera_dist_m') or 0
        return (-d, -px_count)
    for _, i, m in sorted(layers, key=order_key):
        idx[m] = i+1
    b = io.BytesIO(); Image.fromarray(idx, mode='L').save(b,'PNG',optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(b.getvalue()).decode(), W, H

def plan_fragment(rid, inv, objs):
    theta = inv.get('manhattan_theta_deg')
    if not objs: return None
    walls = inv.get('walls', [])
    xs=[0.0]+[x for o in objs for x,_ in o['footprint']]; ys=[0.0]+[y for o in objs for _,y in o['footprint']]
    minx,maxx=min(xs)-1,max(xs)+1; miny,maxy=min(ys)-1,max(ys)+1
    W,H=560,440; s=min(W/(maxx-minx),H/(maxy-miny))
    def px(p): return (round((p[0]-minx)*s,1), round(H-(p[1]-miny)*s,1))
    # distances and regions use the projected thin rect when a method
    # produced one — the raw hull of a see-through structure IS the smear
    polys=[Polygon(o['rect_snapped']) if (o.get('footprint_method') and o.get('rect_snapped'))
           else Polygon(o['footprint']) for o in objs]
    n=len(objs); gaps=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            g=round(polys[i].distance(polys[j]),2); gaps[i][j]=gaps[j][i]=g
    counts={}
    for o in objs: counts[o['label']]=counts.get(o['label'],0)+1
    per={}; disp=[]
    for o in objs:
        per[o['label']]=per.get(o['label'],0)+1
        base=LBL.get(o['label'],o['label'])
        name=(f"{base} #{per[o['label']]}" if counts[o['label']]>1 else base)
        if o.get('footprint_method')=='refine' or o.get('refine_slug'): name='补测·'+name
        disp.append(name)
    meta=[{"label":f"{o['label']} #{i+1}","zh":disp[i],"h":o['height_m'],"tilt":o.get('tilt_deg'),
           "size":o['size_m'],"d":o['camera_dist_m'],"method":o.get('footprint_method','hull'),"ns":1 if o.get('normal_split') else 0,
           "cx":px(tuple(o['centroid_xy']))[0],"cy":px(tuple(o['centroid_xy']))[1]} for i,o in enumerate(objs)]
    svg=[f'<svg viewBox="0 0 {W} {H}" style="width:100%;background:var(--card);border:1px solid var(--line);border-radius:4px">']
    for gx in range(int(minx)-1,int(maxx)+2):
        a,b=px((gx,miny)),px((gx,maxy)); svg.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="var(--line)" stroke-width="0.5"/>')
    for gy in range(int(miny)-1,int(maxy)+2):
        a,b=px((minx,gy)),px((maxx,gy)); svg.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="var(--line)" stroke-width="0.5"/>')
    for w in walls:
        a,b=px(tuple(w['start'])),px(tuple(w['end'])); svg.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="var(--ink)" stroke-width="5" opacity="0.7"/>')
    svg.append('<g class="iplines"></g>')
    for i,o in enumerate(objs):
        c=PALETTE[i%len(PALETTE)]
        coords = o.get('rect_snapped')
        if coords: coords = coords + [coords[0]]
        else:
            rect=polys[i].minimum_rotated_rectangle
            coords=list(rect.exterior.coords) if rect.geom_type=='Polygon' else o['footprint']+[o['footprint'][0]]
        if any((x!=x or y!=y) for x,y in coords): coords=o['footprint']+[o['footprint'][0]]
        bpts=' '.join(f'{x},{y}' for x,y in (px(p) for p in coords))
        hpts=' '.join(f'{x},{y}' for x,y in (px(p) for p in o['footprint']))
        cx,cy=px(tuple(o['centroid_xy']))
        dash=''
        if o.get('footprint_method')=='contact-edge': dash=' stroke-dasharray="7,3"'
        if o.get('footprint_method')=='near-cluster': dash=' stroke-dasharray="1,3"'
        if o.get('footprint_method')=='guard-line': dash=' stroke-dasharray="12,3"'
        if o.get('footprint_method')=='refine': dash=' stroke-dasharray="2,3"'
        if not o.get('footprint_method'):
            svg.append(f'<polygon points="{hpts}" fill="none" stroke="{c}" stroke-width="0.8" opacity="0.5"/>')
        svg.append(f'<polygon class="ip" data-i="{i}" points="{bpts}" fill="{c}" fill-opacity="0.22" stroke="{c}" stroke-width="2"{dash} style="cursor:pointer"/>')
        svg.append(f'<text x="{cx}" y="{cy}" font-size="11" fill="{c}" text-anchor="middle" style="pointer-events:none;font-family:IBM Plex Mono,monospace">{i+1}</text>')
    cam=px((0,0)); svg.append(f'<circle cx="{cam[0]}" cy="{cam[1]}" r="6" fill="#c1121f"/><text x="{cam[0]+9}" y="{cam[1]+4}" font-size="10" fill="#c1121f" style="font-family:IBM Plex Mono,monospace">CAM</text>')
    svg.append('</svg>')
    legend=' '.join(f'<span class="iplegend" data-i="{i}" style="cursor:pointer"><span style="display:inline-block;width:9px;height:9px;background:{PALETTE[i%len(PALETTE)]};border-radius:2px;margin-right:4px"></span>{i+1}.{disp[i]}</span>' for i,o in enumerate(objs))
    payload=json.dumps({"meta":meta,"gaps":gaps},ensure_ascii=False).replace('&','&amp;').replace('"','&quot;')
    theta_note = f' · 主轴 {theta}°' if theta is not None else ''
    return (f'<div class="iplan" data-run="{rid}" data-pack="{payload}">'+''.join(svg)+
            f'<div style="margin:6px 0;font-size:12.5px">{legend} <button class="ipclear">清空选择</button><span style="color:var(--muted)">{theta_note} · 长虚线=接触边投影 · 点虚线=近簇裁剪 · 疏长虚线=共线对齐 · 短虚线=人工补测 · 灰粗线=墙面走向(主轴输入)</span></div>'
            f'<div class="ipinfo" style="border:1px solid var(--line);border-radius:4px;padding:8px 12px;font-size:13.5px;color:var(--muted)">点<b>原图</b>、平面图或 3D 三向联动；连点多个出间距矩阵+连线</div></div>')


def build_case(rid):
    run=Path('runs')/rid
    a=json.loads((run/'assessment.json').read_text()); s=json.loads((run/'scene.json').read_text()); p=json.loads((run/'policies.json').read_text())
    status=a.get('status'); cls,zh=ZH[status]
    inp=next((run/'input').glob('image_*')); ov=run/'evidence'/'frame_0001_overlay.png'; fp=run/'inventory'/'floor_plan.png'; dr=run/'depth_render.png'
    inv, objs = case_objects(rid)
    plan = plan_fragment(rid, inv, objs) or ''
    nobj = len(objs)
    base_uri = uri(inp, 1100, 76)
    photo=''
    if objs:
        idx_uri, mw, mh = mask_index_png(rid, objs)
        photo=(f'<h4>原图点选 · 点照片里的物体直接高亮（含人工补测）</h4>'
               f'<div class="iphoto" data-mw="{mw}" data-mh="{mh}">'
               f'<canvas style="width:100%;display:block;border:1px solid var(--line);border-radius:4px;cursor:crosshair"></canvas>'
               f'<img class="ipbase" src="{base_uri}" hidden><img class="ipidx" src="{idx_uri}" hidden></div>')
    ents={}
    for e in s.get('entities',[]): ents.setdefault(e['label'],[]).append(e.get('height_m'))
    ent_rows=''.join(f'<tr><td>{LBL.get(k,k)}</td><td class="mono">×{len(v)}</td><td class="mono">{", ".join(f"{h:.2f}" for h in v if h is not None)} m</td></tr>' for k,v in ents.items()) or '<tr><td colspan=3>无词表实体过证据门</td></tr>'
    pol_rows=''
    for r in p.get('results',[]):
        st=r['status'] if isinstance(r['status'],str) else str(r['status'])
        pcls,pzh=ZH.get(st,('insuff',st))
        reasons=''.join(f'<div class="reason"><b>{gloss(w) or "备注"}</b><div class="mono raw">{w}</div></div>' for w in (r.get('warnings') or [])[:3])
        vio=''.join(f'<div class="reason vio"><b>违规：测得 {v.get("measured")} m，阈值 {v.get("threshold")} m</b><div class="mono raw">{v.get("subject_id")} ↔ {v.get("object_id")}</div></div>' for v in (r.get('violations') or [])[:3])
        pol_rows+=f'<div class="policy"><span class="pill {pcls}">{pzh}</span> <span class="pname">{POLICY_ZH.get(r["policy_id"],r["policy_id"])}</span>{vio}{reasons}</div>'
    depth_chips=' '.join(f'<span class="iplegend" data-i="{i}" style="cursor:pointer;font-size:11.5px"><span style="display:inline-block;width:8px;height:8px;background:{PALETTE[i%len(PALETTE)]};border-radius:2px;margin-right:3px"></span>{i+1}</span>' for i in range(nobj))
    figs=f'<div class="figs"><figure><img src="{uri(ov)}"><figcaption><b>evidence overlay</b>（判定链 mask）</figcaption></figure>'
    if dr.exists(): figs+=f'<figure><img src="{uri(dr)}"><figcaption><b>深度渲染</b> · 点编号选中<br>{depth_chips}</figcaption></figure>'
    if fp.exists(): figs+=f'<figure><img src="{uri(fp)}"><figcaption><b>测量平面图（CAD 版，全实例）</b></figcaption></figure>'
    figs+='</div>'
    vs=run/'viewer_small.html'
    viewer=f'<h4>交互 3D（照片色 · 按实例）</h4><iframe class="v3d" srcdoc="{srcdoc(vs)}" style="width:100%;height:460px;border:1px solid var(--line);border-radius:4px;display:block;background:#0d1114" title="{rid} 3D"></iframe>' if vs.exists() else ''
    refine_html=''
    det_html=''
    det_path=run/'detection'/'detections.json'
    if det_path.exists():
        env=json.loads(det_path.read_text())
        CAT_META={'A':('感知防护 SENSING/AOPD','#39c5cf'),'B':('控制防护 CONTROL','#f25c8a'),'C':('防护罩/围护 GUARDS','#4ad07a'),'D':('阻挡与引导 IMPEDING','#e8b93c'),'E':('信息标识 INFO','#c9a0ff')}
        ov=run/'detection'/'overlay.png'
        from PIL import Image as PImage
        import io as io_mod
        with PImage.open(ov) as im_:
            im_=im_.convert('RGB'); im_.thumbnail((1150,1150))
            b=io_mod.BytesIO(); im_.save(b,'JPEG',quality=82)
        ov_uri='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
        groups={}
        for d in env['detections']:
            if 'rle' not in d: continue
            groups.setdefault(d['category'],[]).append(d)
        legend_html=''
        for cat in 'ABCDE':
            if cat not in groups: continue
            name,color=CAT_META[cat]
            rows=' '.join(f'<span style="display:inline-block;margin:2px 10px 2px 0;font-size:12.5px"><b style="color:{color}">#{d["number"]}</b> {d["zh"]} <span style="color:var(--muted)">· SAM {d["sam_score"]}{(" · "+d["iso"]) if d.get("iso") else ""}</span></span>' for d in groups[cat])
            legend_html+=f'<div style="margin:4px 0"><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:6px"></span><b style="font-size:12.5px">{name}</b><div style="margin-left:16px">{rows}</div></div>'
        n_found=sum(len(v) for v in groups.values())
        missing=env.get('missing',[])
        miss_html=('<div style="margin:6px 0;font-size:12.5px;color:var(--fail)"><b>未见/需现场核实：</b>'+ '、'.join(m['zh'] for m in missing)+'</div>') if missing else '<div style="margin:6px 0;font-size:12.5px;color:var(--pass)"><b>清单全部检出</b>（缺失清单为空）</div>'
        det_html=(f'<h4>装置检测清单（taxonomy 检测层 · VLM 出框+裁剪自检 → SAM box-prompt）</h4>'
                  f'<figure><img src="{ov_uri}" style="max-width:100%"><figcaption>{n_found} 项检出 · 编号=下方图例</figcaption></figure>'
                  f'{legend_html}{miss_html}')
    reproj_html=''
    rp=run/'inventory'/'reprojection.json'
    if rp.exists() and (run/'inventory'/'reprojection.png').exists():
        rj=json.loads(rp.read_text())
        from PIL import Image as PImage2
        import io as io_mod2
        with PImage2.open(run/'inventory'/'reprojection.png') as im2:
            im2=im2.convert('RGB'); im2.thumbnail((1000,1000))
            b2=io_mod2.BytesIO(); im2.save(b2,'JPEG',quality=70)
        def _dv(s):
            v=s.get('mean_dv_frac')
            return '—' if v is None else f'{round(v*100,1)}%'
        rows=' · '.join(f"#{s.get('instance')} {_dv(s)}" for s in rj['scores'])
        reproj_html=(f'<h4>回投验证（平面矩形基线投回照片 · 红点应压在结构接地线上 · 偏差=图高占比）</h4>'
                     f'<figure><img src="data:image/jpeg;base64,{base64.b64encode(b2.getvalue()).decode()}" style="max-width:100%"><figcaption>围栏族偏差：{rows}</figcaption></figure>')
    rf=run/'refinements.json'
    if rf.exists():
        items=json.loads(rf.read_text()); refine_html='<h4>人工框选补测（--apply 回灌判定 · 工作台"补测"tab 可自助）</h4><div class="figs">'
        seen=set()
        for it in items:
            slug=f"{it['label'].replace(' ','_')}_{'_'.join(str(b) for b in it['box'])}"
            if slug in seen: continue
            seen.add(slug)
            if 'height_m' not in it or not (run/'refinements'/(slug+'.png')).exists(): continue
            refine_html+=f'<figure><img src="{uri(run/"refinements"/(slug+".png"))}"><figcaption><b>{LBL.get(it["label"],it["label"])}</b> · SAM {it["sam_score"]} · 高 {it["height_m"]} m · {it["extent_m"]} m · 距相机 {it["camera_dist_m"]} m</figcaption></figure>'
        refine_html+='</div>'
    return (f'''<details class="case"><summary><img src="{thumb(inp)}"><span class="cid mono">{rid}</span>
    <span class="pill {cls}">{zh}</span><span class="cmeta mono">scale {round(s.get("scale_factor",0),2)} · {len(s.get("entities",[]))} 实体</span></summary>
      <div class="body">{CASE_NOTES.get(rid,'')}{det_html}{photo}{plan}{viewer}{figs}
    <h4>判定明细</h4>{pol_rows}
    <h4>实体测量</h4>
    <div class="tblwrap"><table><tr><th>物体</th><th>数量</th><th>高度</th></tr>{ent_rows}</table></div>
    {reproj_html}{refine_html}
    <p class="hint mono">全量 3D：runs/{rid}/viewer.html · 自助补测：工作台 补测 tab（或 scripts/refine_region.py --apply）</p>
      </div></details>''')


_TAIL_JS = '\n</div>\n<div id="lb" hidden><img alt=""></div>\n<script>\n(function(){\nvar lb=document.getElementById(\'lb\'),im=lb.querySelector(\'img\');\ndocument.querySelectorAll(\'figure img\').forEach(function(el){el.addEventListener(\'click\',function(){im.src=el.src;lb.hidden=false})});\nlb.addEventListener(\'click\',function(){lb.hidden=true;im.src=\'\'});\ndocument.addEventListener(\'keydown\',function(e){if(e.key===\'Escape\')lb.hidden=true});\nvar PAL=["#c1121f","#1d4ed8","#047857","#b45309","#6d28d9","#0e7490","#9d174d","#4d7c0f","#7c2d12","#334155"];\nfunction hex2rgb(hx){return [parseInt(hx.slice(1,3),16),parseInt(hx.slice(3,5),16),parseInt(hx.slice(5,7),16)];}\nvar plans=[];\ndocument.querySelectorAll(\'.iplan\').forEach(function(w){\n  var pack=JSON.parse(w.getAttribute(\'data-pack\')),meta=pack.meta,gaps=pack.gaps;\n  var info=w.querySelector(\'.ipinfo\'),linesG=w.querySelector(\'.iplines\');\n  var body=w.closest(\'.body\'),frame=body?body.querySelector(\'iframe.v3d\'):null;\n  var photo=body?body.querySelector(\'.iphoto\'):null;\n  var selected=[];\n  var P={w:w,meta:meta,frame:frame,selected:selected};\n  var idxData=null,pcv=null,pbase=null,mw=0,mh=0;\n  if(photo){\n    pcv=photo.querySelector(\'canvas\');pbase=photo.querySelector(\'.ipbase\');\n    var pidx=photo.querySelector(\'.ipidx\');\n    mw=+photo.dataset.mw;mh=+photo.dataset.mh;\n    var ready=function(){\n      try{\n        var oc=document.createElement(\'canvas\');oc.width=mw;oc.height=mh;\n        var octx=oc.getContext(\'2d\');octx.drawImage(pidx,0,0);\n        idxData=octx.getImageData(0,0,mw,mh).data;\n        pcv.width=pbase.naturalWidth;pcv.height=pbase.naturalHeight;\n        photoDraw();\n      }catch(err){}\n    };\n    var pending=0;\n    [pbase,pidx].forEach(function(el){if(!el.complete)pending++;});\n    if(pending===0)ready();\n    else [pbase,pidx].forEach(function(el){\n      if(!el.complete)el.addEventListener(\'load\',function(){if(--pending===0)ready();});\n    });\n  }\n  P.photoClick=function(e){\n    if(!idxData||!pcv)return;\n    var r=pcv.getBoundingClientRect();\n    var x=Math.floor((e.clientX-r.left)/r.width*mw);\n    var y=Math.floor((e.clientY-r.top)/r.height*mh);\n    var v=idxData[(y*mw+x)*4];\n    if(v>0)pick(v-1);\n  };\n  P.pcv=pcv;\n  function photoDraw(){\n    if(!pcv||!idxData)return;\n    var ctx=pcv.getContext(\'2d\');\n    ctx.clearRect(0,0,pcv.width,pcv.height);\n    ctx.drawImage(pbase,0,0,pcv.width,pcv.height);\n    if(!selected.length)return;\n    var oc=document.createElement(\'canvas\');oc.width=mw;oc.height=mh;\n    var octx=oc.getContext(\'2d\');var imd=octx.createImageData(mw,mh);\n    for(var p=0;p<mw*mh;p++){var v=idxData[p*4];\n      if(v>0&&selected.indexOf(v-1)>=0){var c=hex2rgb(PAL[(v-1)%PAL.length]);\n        imd.data[p*4]=c[0];imd.data[p*4+1]=c[1];imd.data[p*4+2]=c[2];imd.data[p*4+3]=150;}}\n    octx.putImageData(imd,0,0);\n    ctx.drawImage(oc,0,0,pcv.width,pcv.height);\n  }\n  function fmt(i){var m=meta[i];\n    var t=(m.tilt!==null&&m.tilt!==undefined)?(\'，倾角 \'+Math.round(m.tilt)+\'°\'):\'\';\n    var mm=m.method===\'contact-edge\'?\' · 接触边投影\':(m.method===\'refine\'?\' · 人工补测\':(m.method===\'near-cluster\'?\' · 近簇裁剪\':(m.method===\'guard-line\'?\' · 共线对齐\':\'\')));\n    if(m.ns)mm+=\' · 法线分割\';\n    return \'<b>\'+m.zh+\'</b> · 高 \'+m.h.toFixed(2)+\' m\'+t+\' · \'+m.size+\' m · 距相机 \'+m.d.toFixed(2)+\' m\'+mm;}\n  function render(notify){\n    w.querySelectorAll(\'polygon.ip\').forEach(function(pg){var on=selected.indexOf(+pg.dataset.i)>=0;\n      pg.setAttribute(\'fill-opacity\',on?\'0.55\':\'0.12\');pg.setAttribute(\'stroke-width\',on?\'3.5\':\'1.2\');});\n    while(linesG.firstChild)linesG.removeChild(linesG.firstChild);\n    if(selected.length===0){info.innerHTML=\'点<b>原图</b>、平面图或 3D 三向联动；连点多个出间距矩阵+连线\';info.style.color=\'var(--muted)\';photoDraw();return;}\n    var html=selected.map(fmt).join(\'<br>\');\n    if(selected.length>1){\n      html+=\'<table style="margin-top:6px;border-collapse:collapse;font-size:12.5px"><tr><th></th>\'+selected.map(function(i){return \'<th style="padding:2px 8px">\'+meta[i].zh+\'</th>\'}).join(\'\')+\'</tr>\';\n      selected.forEach(function(i){html+=\'<tr><th style="padding:2px 8px;text-align:left">\'+meta[i].zh+\'</th>\'+selected.map(function(j){return \'<td style="padding:2px 8px;text-align:center" class="mono">\'+(i===j?\'—\':gaps[i][j].toFixed(2)+\' m\')+\'</td>\'}).join(\'\')+\'</tr>\';});\n      html+=\'</table>\';\n      for(var a=0;a<selected.length;a++)for(var b=a+1;b<selected.length;b++){\n        var i=selected[a],j=selected[b];\n        var ln=document.createElementNS(\'http://www.w3.org/2000/svg\',\'line\');\n        ln.setAttribute(\'x1\',meta[i].cx);ln.setAttribute(\'y1\',meta[i].cy);ln.setAttribute(\'x2\',meta[j].cx);ln.setAttribute(\'y2\',meta[j].cy);\n        ln.setAttribute(\'stroke\',\'#047857\');ln.setAttribute(\'stroke-width\',\'2\');ln.setAttribute(\'stroke-dasharray\',\'5,4\');linesG.appendChild(ln);\n        var tx=document.createElementNS(\'http://www.w3.org/2000/svg\',\'text\');\n        tx.setAttribute(\'x\',(meta[i].cx+meta[j].cx)/2);tx.setAttribute(\'y\',(meta[i].cy+meta[j].cy)/2-4);\n        tx.setAttribute(\'font-size\',\'11\');tx.setAttribute(\'fill\',\'#047857\');tx.setAttribute(\'text-anchor\',\'middle\');\n        tx.style.fontFamily=\'IBM Plex Mono,monospace\';tx.textContent=gaps[i][j].toFixed(2)+\' m\';linesG.appendChild(tx);}\n    }\n    info.innerHTML=html;info.style.color=\'var(--ink)\';\n    photoDraw();\n    if(notify!==false&&selected.length){var last=selected[selected.length-1];\n      if(frame&&frame.contentWindow)frame.contentWindow.postMessage({type:\'ehs-select\',label:meta[last].label},\'*\');}\n  }\n  function pick(i,notify){var k=selected.indexOf(i);\n    if(k>=0)selected.splice(k,1);else selected.push(i);\n    if(selected.length>4)selected.shift();render(notify);}\n  P.pick=pick;P.render=render;\n  w.querySelectorAll(\'polygon.ip\').forEach(function(pg){pg.addEventListener(\'click\',function(){pick(+pg.dataset.i)})});\n  document.querySelectorAll(\'.iplegend\').forEach(function(){});\n  (body||w).querySelectorAll(\'.iplegend\').forEach(function(l){l.addEventListener(\'click\',function(){pick(+l.dataset.i)})});\n  var clr=w.querySelector(\'.ipclear\');\n  if(clr)clr.addEventListener(\'click\',function(){selected.length=0;render(false);});\n  plans.push(P);\n});\ndocument.addEventListener(\'click\',function(e){\n  var t=e.target;\n  if(!(t&&t.tagName===\'CANVAS\'))return;\n  plans.forEach(function(pl){if(pl.pcv===t&&pl.photoClick)pl.photoClick(e);});\n},true);\naddEventListener(\'message\',function(e){\n  var m=e.data;if(!m||m.type!==\'ehs-picked\')return;\n  plans.forEach(function(pl){\n    if(!pl.frame||pl.frame.contentWindow!==e.source)return;\n    if(m.label===null){pl.selected.length=0;pl.render(false);return;}\n    var idx=-1;\n    pl.meta.forEach(function(mt,i){if(idx<0&&(mt.label===m.label||mt.label.indexOf(m.label)>=0||m.label.indexOf(mt.label)>=0))idx=i;});\n    if(idx>=0){pl.selected.length=0;pl.pick(idx,false);}\n  });\n});\n})();\n</script>'



def build_interactive_report(run_ids, out_path=None, title="Panoptes"):
    cards = [build_case(rid) for rid in run_ids]
    head = (
        '<meta charset="utf-8"><title>' + title + '</title>'
        '<style>' + _CSS + '</style><body><div class="wrap">'
        '<h1 style="font-size:22px">' + title + '</h1>'
    )
    html = head + "\n".join(cards) + _TAIL_JS
    if out_path is not None:
        Path(out_path).write_text(html, encoding="utf-8")
    return html


def build_interactive_run_report(run_id):
    """One run, full interactivity; returns the output path."""
    out = Path("runs") / run_id / "report.html"
    build_interactive_report([run_id], out, title="Panoptes · " + run_id)
    return out
