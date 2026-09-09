"""Interactive per-run report — the full linked experience ported from
the test-set report generator: click an object in the PHOTO and it
highlights; click plan polygons; multi-select for the distance matrix;
detection checklist, verdict gloss, reprojection check, refinement
gallery. The 3D viewer iframe embeds lazily only when the run has a
compact viewer; everything else is self-contained data URIs, so the file
opens offline anywhere."""

import base64, html, io, json, re, sys
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from .providers.sam3 import decode_coco_rle

# Reuse the same inventory renderer as the app; no model calls during report builds.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scene_inventory import inventory_plan_objects, ensure_floor_plan, _refine_mask
from .taxonomy import TAXONOMY

_CSS = '\n:root{--paper:#F5F7F6;--surface:#FFF;--ink:#1B2327;--muted:#5A676F;--line:#DAE0DE;--accent:#0E7490;--accent-ink:#0A5B71;\n--pass:#178A50;--pass-bg:#E4F3EA;--warn:#A9740E;--warn-bg:#F7EEDC;--fail:#C13B3B;--fail-bg:#F9E7E5;--insuff:#5D6B76;--insuff-bg:#E9EDEF;--card:#FCFDFC}\n@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--paper:#101619;--surface:#161D21;--ink:#E7ECEC;--muted:#9AA7AC;--line:#263136;--accent:#41BBD3;--accent-ink:#6ED0E3;\n--pass:#41BE7E;--pass-bg:#12301F;--warn:#DFA83D;--warn-bg:#33280F;--fail:#E4706A;--fail-bg:#3A1B18;--insuff:#93A2AC;--insuff-bg:#232D33;--card:#141B1F}}\n:root[data-theme="dark"]{--paper:#101619;--surface:#161D21;--ink:#E7ECEC;--muted:#9AA7AC;--line:#263136;--accent:#41BBD3;--accent-ink:#6ED0E3;\n--pass:#41BE7E;--pass-bg:#12301F;--warn:#DFA83D;--warn-bg:#33280F;--fail:#E4706A;--fail-bg:#3A1B18;--insuff:#93A2AC;--insuff-bg:#232D33;--card:#141B1F}\n*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.7 \'Noto Sans SC\',\'IBM Plex Sans\',sans-serif}\n.wrap{max-width:1020px;margin:0 auto;padding:44px 22px 90px}\nh1{font-family:\'IBM Plex Sans\',\'Noto Sans SC\',sans-serif;font-size:clamp(24px,4vw,34px);margin:0 0 6px;text-wrap:balance}\nh4{font-family:\'IBM Plex Sans\',sans-serif;margin:20px 0 8px}\n.eyebrow{font-family:\'IBM Plex Mono\',monospace;font-size:11.5px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:12px}\n.eyebrow b{color:var(--accent-ink)}\n.lede{color:var(--muted);max-width:47em;margin:0 0 6px}\n.mono{font-family:\'IBM Plex Mono\',monospace;font-variant-numeric:tabular-nums}\n.pill{display:inline-block;padding:1px 9px;border-radius:3px;font-size:12.5px;font-weight:600;white-space:nowrap}\n.pill.fail{background:var(--fail-bg);color:var(--fail)}.pill.warn{background:var(--warn-bg);color:var(--warn)}\n.pill.insuff{background:var(--insuff-bg);color:var(--insuff)}.pill.pass{background:var(--pass-bg);color:var(--pass)}\n.legend{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:14px 18px;margin:20px 0}\n.legend div{margin:4px 0}\ndetails.case{background:var(--surface);border:1px solid var(--line);border-radius:5px;margin:12px 0;overflow:hidden}\ndetails.case summary{display:flex;align-items:center;gap:14px;padding:10px 16px;cursor:pointer;list-style:none}\ndetails.case summary::-webkit-details-marker{display:none}\ndetails.case summary img{width:96px;height:64px;object-fit:cover;border-radius:3px;border:1px solid var(--line)}\ndetails.case[open] summary{border-bottom:1px solid var(--line)}\n.cid{font-weight:600}\n.cmeta{margin-left:auto;color:var(--muted);font-size:12.5px}\n.body{padding:16px 18px}\n.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin:10px 0}\nfigure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:4px;overflow:hidden}\nfigure img{max-width:100%;display:block;cursor:zoom-in}\nfigcaption{padding:8px 12px;font-size:12.5px;color:var(--muted)}\nfigcaption b{color:var(--ink)}\n.policy{border:1px solid var(--line);border-radius:4px;padding:10px 12px;margin:8px 0;background:var(--card)}\n.pname{font-weight:600;margin-left:6px}\n.reason{margin:6px 0 0;padding-left:10px;border-left:3px solid var(--line);font-size:13.5px}\n.reason.vio{border-left-color:var(--fail)}\n.reason .raw{color:var(--muted);font-size:11.5px;overflow-x:auto}\n.tblwrap{overflow-x:auto;border:1px solid var(--line);border-radius:4px}\ntable{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--surface)}\nth,td{padding:7px 11px;border-bottom:1px solid var(--line);text-align:left}\nth{font-family:\'IBM Plex Mono\',monospace;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}\n.hint{font-size:11.5px;color:var(--muted);margin-top:14px;overflow-x:auto}\n.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}\n.tile{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:12px 14px}\n.tile .v{font-family:\'IBM Plex Mono\',monospace;font-size:22px;font-weight:600}\n.tile .l{font-size:12px;color:var(--muted)}\n#lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:flex;align-items:center;justify-content:center;z-index:50;cursor:zoom-out}\n#lb[hidden]{display:none}\n#lb img{max-width:96vw;max-height:94vh}\na{color:var(--accent-ink)}\n'
# 2×2 linked-selection block: photo | 3D / CAD | plan
_CSS += '''
.lbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:12.5px;margin:6px 0 10px}
.lbar .lnames span{display:inline-block;padding:1px 8px;border-radius:3px;background:var(--accent);color:#fff;cursor:pointer;margin:2px 4px 2px 0}
.linked{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:999px){.linked{grid-template-columns:1fr}}
.lcell{min-width:0}.lcell h4{margin:0 0 6px;font-size:13.5px}
.icad{position:relative;border:1px solid var(--line);border-radius:4px;overflow:hidden;background:#fff}
.icad img{display:block;width:100%}
.icad svg{position:absolute;inset:0;width:100%;height:100%}
.icad polygon{fill:transparent;stroke:none;cursor:pointer}
.icad polygon:hover{fill:rgba(14,116,144,.18)}
.icad polygon.on{fill:rgba(14,116,144,.3);stroke:#0E7490;stroke-width:6}
.linked polygon[role="button"]:focus{outline:none;stroke:#0E7490;stroke-width:4;stroke-dasharray:5,3}
'''


def uri(path, maxw=760, q=68):
    im = Image.open(path).convert('RGB')
    if im.width > maxw: im = im.resize((maxw, round(im.height*maxw/im.width)), Image.LANCZOS)
    b = io.BytesIO(); im.save(b,'JPEG',quality=q,optimize=True)
    return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
def thumb(p): return uri(p,340,66)
def srcdoc(p): return open(p,encoding='utf-8').read().replace('&','&amp;').replace('"','&quot;')
def esc(x): return html.escape('' if x is None else str(x), quote=False)
def _json(path, default):
    try: return json.loads(Path(path).read_text(encoding='utf-8'))
    except Exception: return default
def _empty(section, title, msg='无'):
    return f'<h4 data-section="{section}">{title}</h4><p class="hint">{msg}</p>'
def _fig(path, cap, maxw=1000):
    try: return f'<figure><img src="{uri(path, maxw, 70)}"><figcaption>{cap}</figcaption></figure>'
    except Exception: return ''
TAX_ZH = {t.item_id: t.zh for t in TAXONOMY}
METHOD_ZH = {'contact-edge':'接触边投影','near-cluster':'近簇裁剪','guard-line':'共线对齐','refine':'人工补测'}
SIDE_ZH = {'u_min':'u− 边','u_max':'u+ 边','v_min':'v− 边','v_max':'v+ 边'}
DECISION_ZH = {'confirmed':'确认机器判定','overridden':'推翻机器判定'}
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

def case_objects(rid):
    inventory = _json(Path('runs')/rid/'inventory'/'inventory.json', {})
    return inventory, inventory_plan_objects(inventory)


def mask_index_png(rid, objs, frame_id='frame_0001'):
    """Native canonical grid; RGB encodes inventory index + 1, independent of list order."""
    run = Path('runs')/rid
    with Image.open(run/'geometry'/'frames'/frame_id/'canonical.png') as image:
        width, height = image.size
    layers, issues = [], []
    for obj in objs:
        if obj.get('frame', 'frame_0001') != frame_id:
            continue
        try:
            if obj.get('refine_slug'):
                mask = _refine_mask(run, obj["refine_slug"], height, width, frame_id, expected_sha256=obj.get("refine_mask_sha256"))
                if mask is None: raise ValueError("缺少该对象的源掩码")
            else:
                slug = re.sub(r'[^a-z0-9]+', '_', obj['label']).strip('_')
                response = _json(run/'inventory'/'sam'/f'{frame_id}__{slug}.json', {})
                rles = response.get('rle') or []
                if isinstance(rles, str): rles = [rles]
                mask = np.zeros((height, width), bool)
                for member in obj.get('merged_instances') or [obj['instance']]:
                    mask |= decode_coco_rle(rles[member], height=height, width=width).astype(bool)
            if mask.any(): layers.append((obj, mask))
        except (ValueError, TypeError, KeyError, IndexError, OSError) as error:
            issues.append({'inv': obj['inv'], 'reason': str(error) or '缺少该机位掩码'})
    index = np.zeros((height, width), np.uint32)
    # Farther masks paint first; smaller masks win equal-depth overlaps.
    for obj, mask in sorted(layers, key=lambda pair: (-(pair[0].get('camera_dist_m') or 0), -int(pair[1].sum()))):
        if not 0 <= obj['inv'] < 16777215: raise ValueError('Invalid inventory index')
        index[mask] = obj['inv']+1
    rgb = np.stack([index & 255, (index >> 8) & 255, (index >> 16) & 255], -1).astype('uint8')
    buffer = io.BytesIO(); Image.fromarray(rgb).save(buffer, 'PNG', optimize=True)
    visible = [int(value)-1 for value in np.unique(index) if value]
    masks = {}
    for obj, mask in layers:
        encoded = io.BytesIO(); Image.fromarray(mask.astype('uint8')*255).save(encoded, 'PNG', optimize=True)
        masks[obj['inv']] = 'data:image/png;base64,'+base64.b64encode(encoded.getvalue()).decode()
    return 'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode(), width, height, visible, issues, masks


def photo_fragment(rid, objs):
    frames = sorted((Path('runs')/rid/'geometry'/'frames').glob('frame_*/canonical.png'))
    if not frames: return '<p class="hint">无（无可点选实体或缺少几何帧）</p>'
    options, panels = [], []
    for number, canonical in enumerate(frames, 1):
        frame_id = canonical.parent.name
        index_uri, width, height, visible, issues, masks = mask_index_png(rid, objs, frame_id)
        options.append(f'<option value="{frame_id}">机位 {number} · {frame_id}</option>')
        attrs = html.escape(json.dumps({'clickable_inv': visible, 'mask_inv': list(masks), 'issues': issues}), quote=True)
        panels.append(f'<div class="iphoto" data-frame="{frame_id}" data-mw="{width}" data-mh="{height}" data-photo="{attrs}"'+(' hidden' if number>1 else '')+'>'
                      f'<canvas style="width:100%;display:block;border:1px solid var(--line);border-radius:4px;cursor:crosshair"></canvas>'
                      f'<img class="ipbase" src="data:image/png;base64,{base64.b64encode(canonical.read_bytes()).decode()}" hidden>'
                      f'<img class="ipidx" src="{index_uri}" hidden>'+''.join(f'<img class="ipmask" data-mask-inv="{inv}" src="{mask_uri}" hidden>' for inv, mask_uri in masks.items())+'</div>')
    return ('<label>照片机位 <select class="photo-frame">'+''.join(options)+'</select></label>'
            '<p class="photo-status hint" role="status"></p>'+''.join(panels))


def disp_names(objs):
    counts={}
    for o in objs: counts[o['label']]=counts.get(o['label'],0)+1
    per={}; disp=[]
    for o in objs:
        per[o['label']]=per.get(o['label'],0)+1
        base=LBL.get(o['label'],o['label'])
        name=(f"{base} #{per[o['label']]}" if counts[o['label']]>1 else base)
        if o.get('footprint_method')=='refine' or o.get('refine_slug'): name='补测·'+name
        disp.append(name)
    return disp

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
    disp=disp_names(objs)
    meta=[{"inv":o['inv'],"frame":o.get("frame", "frame_0001"),"label":f"{o['label']} #{i+1}","zh":disp[i],"h":o['height_m'],"tilt":o.get('tilt_deg'),
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
        label = html.escape(f'交互平面：{disp[i]} · {o.get("frame", "frame_0001")} · inv {o["inv"]}', quote=True)
        svg.append(f'<polygon class="ip" data-inv="{o["inv"]}" role="button" tabindex="0" aria-label="{label}" aria-pressed="false" points="{bpts}" fill="{c}" fill-opacity="0.22" stroke="{c}" stroke-width="2"{dash} style="cursor:pointer"/>')
        svg.append(f'<text x="{cx}" y="{cy}" font-size="11" fill="{c}" text-anchor="middle" style="pointer-events:none;font-family:IBM Plex Mono,monospace">{i+1}</text>')
    cam=px((0,0)); svg.append(f'<circle cx="{cam[0]}" cy="{cam[1]}" r="6" fill="#c1121f"/><text x="{cam[0]+9}" y="{cam[1]+4}" font-size="10" fill="#c1121f" style="font-family:IBM Plex Mono,monospace">CAM</text>')
    svg.append('</svg>')
    legend=' '.join(f'<span class="iplegend" data-inv="{o["inv"]}" style="cursor:pointer"><span style="display:inline-block;width:9px;height:9px;background:{PALETTE[i%len(PALETTE)]};border-radius:2px;margin-right:4px"></span>{i+1}.{disp[i]}</span>' for i,o in enumerate(objs))
    payload=json.dumps({"meta":meta,"gaps":gaps},ensure_ascii=False).replace('&','&amp;').replace('"','&quot;')
    theta_note = f' · 主轴 {theta}°' if theta is not None else ''
    return (f'<div class="iplan" data-run="{rid}" data-pack="{payload}">'+''.join(svg)+
            f'<div style="margin:6px 0;font-size:12.5px">{legend} <span style="color:var(--muted)">{theta_note} · 长虚线=接触边投影 · 点虚线=近簇裁剪 · 疏长虚线=共线对齐 · 短虚线=人工补测 · 灰粗线=墙面走向(主轴输入)</span></div>'
            f'<div class="ipinfo" style="border:1px solid var(--line);border-radius:4px;padding:8px 12px;font-size:13.5px;color:var(--muted)">{_IDLE}</div></div>')

_IDLE = '点<b>原图</b>、3D、CAD 或平面图任一格，四格联动；连点多个出间距矩阵+连线'

def cad_fragment(run):
    """floor_plan.png with a transparent SVG hit-area per object (from the
    renderer's floor_plan_map.json sidecar), rebuilt from inventory when stale."""
    ensure_floor_plan(run)
    fp = run/'inventory'/'floor_plan.png'
    if not fp.exists(): return '<p class="hint">无（inventory/floor_plan.png 缺失）</p>'
    m = _json(run/'inventory'/'floor_plan_map.json', None)
    overlay = ''
    if isinstance(m, dict) and m.get('objects'):
        polygons = []
        for obj in m['objects']:
            if obj.get('inv') is None or len(obj.get('polygon') or []) < 3: continue
            label = html.escape(f'CAD 平面图：{obj["label"]} · inv {obj["inv"]}', quote=True)
            points = ' '.join(f'{x},{y}' for x,y in obj['polygon'])
            polygons.append(f'<polygon data-inv="{obj["inv"]}" role="button" tabindex="0" aria-label="{label}" aria-pressed="false" points="{points}"><title>{obj["legend"]}. {esc(obj["label"])}</title></polygon>')
        overlay = f'<svg viewBox="0 0 {m["width"]} {m["height"]}" preserveAspectRatio="xMidYMid meet">{"".join(polygons)}</svg>'
    return f'<div class="icad"><img src="{uri(fp, 1600, 80)}" alt="CAD floor plan">{overlay}</div>'


def _disp_line(d):
    if not d: return '无（尚未审核）'
    dec = DECISION_ZH.get(d.get('decision'), d.get('decision') or '—')
    if d.get('overridden_status'): dec += f' → {esc(d["overridden_status"])}'
    return (f'<b>{esc(d.get("reviewer") or "—")}</b> · {dec} · {esc(d.get("reason") or "无理由")} · '
            f'<span class="mono">{esc(d.get("created_at") or d.get("time") or "—")}</span>')

def cell_rect_html(inv):
    cr = inv.get('cell_rect')
    nb = [o for o in inv.get('objects', []) if o.get('outside_cell')]
    nb_html = ('<div style="margin-top:6px"><b>邻 cell 结构</b>（在矩形外，不参与本 cell 判定）：' +
               '、'.join(f'{LBL.get(o["label"],o["label"])} #{o.get("instance")}（{esc(o.get("frame") or "—")} · 距相机 {o.get("camera_dist_m")} m）' for o in nb) + '</div>') if nb else '<div style="margin-top:6px"><b>邻 cell 结构</b>：无</div>'
    if not cr: return f'<div class="legend"><b>cell 矩形约束</b>：无（inventory 未产出 cell_rect）{nb_html}</div>'
    rows = ''
    for k in ('u_min', 'u_max', 'v_min', 'v_max'):
        sd = (cr.get('sides') or {}).get(k)
        if sd: rows += f'<tr><td>{SIDE_ZH[k]}</td><td class="mono">{sd.get("offset")} m</td><td class="mono">{sd.get("support_m")} m</td><td>{esc(", ".join(sd.get("sources") or []))}</td></tr>'
        else: rows += f'<tr><td>{SIDE_ZH[k]}</td><td colspan=3><span class="pill insuff">开口</span></td></tr>'
    size = cr.get('size_m')
    size_txt = f'{size[0]} × {size[1]} m（闭合）' if isinstance(size, (list, tuple)) and len(size) == 2 else '未闭合（存在开口边）'
    return (f'<div class="legend"><b>cell 矩形约束</b> · Manhattan 角度 <span class="mono">{cr.get("theta_deg")}°</span> · {size_txt}'
            f'<div class="tblwrap" style="margin-top:6px"><table><tr><th>边</th><th>偏移</th><th>支撑长度</th><th>来源</th></tr>{rows}</table></div>{nb_html}</div>')

def _chat_text(v):
    if isinstance(v, dict): return v.get('content') or v.get('text') or ''
    return '' if v is None else str(v)

def chat_html(run):
    cp = run / 'chat.jsonl'
    rows = ''
    for line in (cp.read_text(encoding='utf-8').splitlines() if cp.exists() else []):
        try: e = json.loads(line)
        except Exception: continue
        kind = e.get('type')
        if kind not in ('chat_turn', 'agent_turn'): continue
        t = esc(e.get('ts') or e.get('created_at') or e.get('time') or '—')
        tag = f'<span class="pill insuff">{esc(e.get("intent"))}</span> ' if kind == 'agent_turn' else '<span class="pill insuff">ask</span> '
        chg = ' <span class="pill warn">已改动 run</span>' if e.get('changed') else ''
        q = _chat_text(e.get('question') or e.get('user')); a_ = e.get('answer') or e.get('assistant')
        facts = (a_.get('fact_ids') if isinstance(a_, dict) else None) or e.get('fact_ids') or []
        rows += (f'<div class="policy"><div><b>用户</b> {tag}<span class="mono" style="color:var(--muted);font-size:12px">{t}</span>{chg}</div>'
                 f'<div style="white-space:pre-wrap">{esc(q)}</div>'
                 f'<div class="reason"><b>Agent</b><div style="white-space:pre-wrap">{esc(_chat_text(a_))}</div>'
                 + (f'<div class="mono raw">facts: {esc(", ".join(map(str, facts)))}</div>' if facts else '') + '</div></div>')
    return '<h4 data-section="chat">Agent 对话记录（本 run 上的问答 / 纠错 / 阈值调整，附时间）</h4>' + (rows or '<p class="hint">无对话记录</p>')


def build_case(rid):
    run=Path('runs')/rid
    a=_json(run/'assessment.json',{}); s=_json(run/'scene.json',{}); p=_json(run/'policies.json',{}); man=_json(run/'manifest.json',{})
    status=a.get('status'); cls,zh=ZH.get(status,('insuff','无判定'))
    inp=next((run/'input').glob('image_*'),None) if (run/'input').exists() else None
    ov=run/'evidence'/'frame_0001_overlay.png'; fp=run/'inventory'/'floor_plan.png'; dr=run/'depth_render.png'
    inv, objs = case_objects(rid)
    plan = plan_fragment(rid, inv, objs) or ''
    nobj = len(objs)
    # 1. 头部 —— review.json is the ReviewDisposition the app writes; disposition.json kept as fallback name
    disp=_json(run/'review.json',None) or _json(run/'disposition.json',None)
    header=(f'<div class="legend" data-section="header"><div><b>Run</b> <span class="mono">{rid}</span> <span class="pill {cls}">{zh}</span>'
            f' · 采集层 {esc(man.get("capture_tier") or "—")} · 操作员 {esc(man.get("operator") or "—")}</div>'
            f'<div><b>审核结论</b> {_disp_line(disp)}</div>'
            f'<div><b>分析版本</b> <span class="mono">{esc(inv.get("analysis_version") or "—")}</span> · <b>创建时间</b> <span class="mono">{esc(man.get("created_at") or "—")}</span></div>'
            f'<div class="hint">四联动使用当前库存测量；判定明细沿用已保存的分析结果，重建视图不会重新评定规则。</div></div>')
    # Photo and hit map use the exact same canonical pixels for each evidence frame.
    photo = photo_fragment(rid, objs)
    # 8. 实体测量 + 尺度来源
    ents={}
    for e in s.get('entities',[]): ents.setdefault(e['label'],[]).append(e.get('height_m'))
    ent_rows=''.join(f'<tr><td>{LBL.get(k,k)}</td><td class="mono">×{len(v)}</td><td class="mono">{", ".join(f"{h:.2f}" for h in v if h is not None)} m</td></tr>' for k,v in ents.items()) or '<tr><td colspan=3>无词表实体过证据门</td></tr>'
    conf=s.get('scale_confidence'); sf=s.get('scale_factor')
    scale_html=(f'<div class="legend"><b>尺度来源与置信</b>：<span class="mono">{esc(s.get("scale_source") or "—")}</span>'
                f' · 置信 <span class="mono">{f"{conf:.2f}" if isinstance(conf,(int,float)) else "—"}</span>'
                f' · 尺度因子 <span class="mono">{f"{sf:.3f}" if isinstance(sf,(int,float)) else "—"}</span></div>')
    names=disp_names(objs)
    obj_rows=''.join(f'<tr><td>{i+1}. {names[i]}</td><td class="mono">{o.get("height_m")}</td><td class="mono">{o.get("size_m")}</td><td class="mono">{o.get("camera_dist_m")}</td><td>{METHOD_ZH.get(o.get("footprint_method"), o.get("footprint_method") or "hull")}</td></tr>' for i,o in enumerate(objs))
    obj_tbl=(f'<div class="tblwrap" style="margin-top:10px"><table><tr><th>物体（平面编号）</th><th>高 m</th><th>尺寸 m</th><th>距相机 m</th><th>足迹方法</th></tr>{obj_rows}</table></div>') if obj_rows else ''
    # 2. 判定明细
    pol_rows=''
    adjusted={sp.get('policy_id') for sp in (p.get('specs') or []) if sp.get('adjusted_for_run')}
    for r in p.get('results',[]):
        st=r['status'] if isinstance(r['status'],str) else str(r['status'])
        pcls,pzh=ZH.get(st,('insuff',st))
        reasons=''.join(f'<div class="reason"><b>{gloss(w) or "备注"}</b><div class="mono raw">{w}</div></div>' for w in (r.get('warnings') or [])[:3])
        vio=''.join(f'<div class="reason vio"><b>违规：测得 {v.get("measured")} m，阈值 {v.get("threshold")} m</b><div class="mono raw">{v.get("subject_id")} ↔ {v.get("object_id")}</div></div>' for v in (r.get('violations') or [])[:3])
        adj=' <span class="pill warn">阈值已按本 run 调整</span>' if r.get('policy_id') in adjusted else ''
        pol_rows+=f'<div class="policy"><span class="pill {pcls}">{pzh}</span> <span class="pname">{POLICY_ZH.get(r["policy_id"],r["policy_id"])}</span>{adj}{vio}{reasons}</div>'
    pol_rows = pol_rows or '<p class="hint">无判定结果</p>'
    # 3. VLM 枚举与去向
    phrases=_json(run/'inventory'/'phrases.json',None)
    if not isinstance(phrases,list): phrases=inv.get('phrases') or []
    unres=_json(run/'inventory'/'unresolved.json',[]) or []
    reason_by={u.get('phrase'):u.get('reason') for u in unres if isinstance(u,dict)}
    counts_all={}
    for o in inv.get('objects',[]): counts_all[o['label']]=counts_all.get(o['label'],0)+1
    ph_rows=''
    for ph in phrases:
        n=counts_all.get(ph,0)
        fate=f'<span class="pill pass">{n} 实例</span>' if n else f'<span class="pill insuff">0 实例</span> {esc(reason_by.get(ph) or "未落地：unresolved 无记录")}'
        ph_rows+=f'<tr><td>{esc(ph)}</td><td>{LBL.get(ph,"")}</td><td>{fate}</td></tr>'
    phr_html=('<h4 data-section="phrases">VLM 枚举与去向（短语 → 实例数 / unresolved 原因）</h4>'
              + (f'<div class="tblwrap"><table><tr><th>VLM 短语</th><th>中文</th><th>去向</th></tr>{ph_rows}</table></div>' if ph_rows else '<p class="hint">无（缺少 inventory/phrases.json）</p>')
              + f'<p class="hint">枚举必有交代：每个 VLM 枚举短语要么落地为实例，要么在 unresolved 中记录原因。当前 {len(phrases)} 短语 · {len(unres)} 条 unresolved。</p>')
    # 10. 证据图集
    depth_chips=' '.join(f'<span class="iplegend" data-inv="{o["inv"]}" style="cursor:pointer;font-size:11.5px"><span style="display:inline-block;width:8px;height:8px;background:{PALETTE[i%len(PALETTE)]};border-radius:2px;margin-right:3px"></span>{i+1}</span>' for i,o in enumerate(objs))
    gal=''.join(_fig(f, f'<b>{f.stem}</b> evidence overlay（判定链 mask）') for f in (sorted((run/'evidence').glob('*.png')) if (run/'evidence').exists() else []))
    if dr.exists(): gal+=_fig(dr, f'<b>深度渲染</b> · 点编号选中<br>{depth_chips}')
    # plan_view.png (fence-only hull, camera at origin) is a pipeline intermediate, not a product view
    for f_,cap in ((run/'cloud_perspective.png','<b>点云透视</b>'),(run/'cloud_topdown.png','<b>点云顶视</b>'),(fp,'<b>测量平面图（CAD 版，全实例）</b>')):
        if f_.exists(): gal+=_fig(f_, cap)
    figs='<h4 data-section="gallery">证据图集（每帧 evidence overlay · 深度渲染 · 点云透视/顶视 · 平面图）</h4>' + (f'<div class="figs">{gal}</div>' if gal else '<p class="hint">无</p>')
    # 7. 交互 3D
    vs=run/'viewer.html'
    viewer=f'<iframe id="v3d" class="v3d" srcdoc="{srcdoc(vs)}" style="width:100%;height:460px;border:1px solid var(--line);border-radius:4px;display:block;background:#0d1114" title="{rid} 3D"></iframe>' if vs.exists() else '<p class="hint">无（viewer.html 缺失）</p>'
    # 4. 装置检测清单
    refine_html=''
    det_html=''
    det_path=run/'detection'/'detections.json'
    if det_path.exists():
        env=json.loads(det_path.read_text())
        CAT_META={'A':('感知防护 SENSING/AOPD','#39c5cf'),'B':('控制防护 CONTROL','#f25c8a'),'C':('防护罩/围护 GUARDS','#4ad07a'),'D':('阻挡与引导 IMPEDING','#e8b93c'),'E':('信息标识 INFO','#c9a0ff'),'F':('物料/载具 PAYLOAD','#f0a35e')}
        ov=run/'detection'/'overlay.png'
        from PIL import Image as PImage
        import io as io_mod
        ov_fig=''
        if ov.exists():
            with PImage.open(ov) as im_:
                im_=im_.convert('RGB'); im_.thumbnail((1150,1150))
                b=io_mod.BytesIO(); im_.save(b,'JPEG',quality=82)
            ov_fig='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode()
        groups={}
        for d in env.get('detections',[]):
            if 'rle' not in d: continue
            groups.setdefault(d['category'],[]).append(d)
        legend_html=''
        for cat in 'ABCDEF':
            if cat not in groups: continue
            name,color=CAT_META[cat]
            rows=' '.join(f'<span style="display:inline-block;margin:2px 10px 2px 0;font-size:12.5px"><b style="color:{color}">#{d["number"]}</b> {d["zh"]} <span style="color:var(--muted)">· SAM {d["sam_score"]}{(" · "+d["iso"]) if d.get("iso") else ""}</span></span>' for d in groups[cat])
            legend_html+=f'<div style="margin:4px 0"><span style="display:inline-block;width:10px;height:10px;background:{color};border-radius:2px;margin-right:6px"></span><b style="font-size:12.5px">{name}</b><div style="margin-left:16px">{rows}</div></div>'
        n_found=sum(len(v) for v in groups.values())
        missing=env.get('missing',[])
        miss_html=('<div style="margin:6px 0;font-size:12.5px;color:var(--fail)"><b>未见/需现场核实：</b>'+ '、'.join(m['zh'] for m in missing)+'</div>') if missing else '<div style="margin:6px 0;font-size:12.5px;color:var(--pass)"><b>清单全部检出</b>（缺失清单为空）</div>'
        rej=env.get('rejected') or []
        rej_html=('<div style="margin:6px 0;font-size:12.5px"><b style="color:var(--warn)">拒绝项（VLM 出框但裁剪自检未通过，不计入检出）：</b><ul style="margin:4px 0 0 18px;padding:0">'
                  + ''.join(f'<li><b>{esc(r.get("zh") or TAX_ZH.get(r.get("item_id"), r.get("item_id")))}</b> <span class="mono" style="color:var(--muted)">{esc(r.get("item_id"))}</span> — {esc(r.get("reason"))}</li>' for r in rej)
                  + '</ul></div>') if rej else '<div style="margin:6px 0;font-size:12.5px;color:var(--muted)"><b>拒绝项：</b>无</div>'
        det_html=(f'<h4 data-section="detections">装置检测清单（taxonomy 检测层 · VLM 出框+裁剪自检 → SAM box-prompt）</h4>'
                  + (f'<figure><img src="{ov_fig}" style="max-width:100%"><figcaption>{n_found} 项检出 · 编号=下方图例</figcaption></figure>' if ov_fig else '')
                  + f'{legend_html}{miss_html}{rej_html}')
    det_html = det_html or _empty('detections','装置检测清单','无（detection/detections.json 缺失）')
    # 9. 回投验证
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
        reproj_html=(f'<h4 data-section="reprojection">回投验证（平面矩形基线投回照片 · 红点应压在结构接地线上 · 偏差=图高占比）</h4>'
                     f'<figure><img src="data:image/jpeg;base64,{base64.b64encode(b2.getvalue()).decode()}" style="max-width:100%"><figcaption>围栏族偏差：{rows}</figcaption></figure>')
    reproj_html = reproj_html or _empty('reprojection','回投验证','无（inventory/reprojection.* 缺失）')
    # 11. 人工框选补测
    rf=run/'refinements.json'
    if rf.exists():
        items=json.loads(rf.read_text()); refine_html='<h4 data-section="refinements">人工框选补测（--apply 回灌判定 · 工作台"补测"tab 可自助）</h4><div class="figs">'
        seen=set()
        for it in items:
            slug=f"{it['label'].replace(' ','_')}_{'_'.join(str(b) for b in it['box'])}"
            if slug in seen: continue
            seen.add(slug)
            if 'height_m' not in it or not (run/'refinements'/(slug+'.png')).exists(): continue
            refine_html+=f'<figure><img src="{uri(run/"refinements"/(slug+".png"))}"><figcaption><b>{LBL.get(it["label"],it["label"])}</b> · SAM {it["sam_score"]} · 高 {it["height_m"]} m · {it["extent_m"]} m · 距相机 {it["camera_dist_m"]} m</figcaption></figure>'
        refine_html+='</div>'
    refine_html = refine_html or _empty('refinements','人工框选补测','无补测记录')
    # 13. Review 记录
    if disp:
        rev_rows=''.join(f'<tr><td>{k}</td><td>{esc(disp.get(f))}</td></tr>' for k,f in (('审核员','reviewer'),('决定','decision'),('推翻为','overridden_status'),('理由','reason'),('时间','created_at'),('run','run_id')) if disp.get(f) is not None)
        rev_html=f'<h4 data-section="review">Review 记录（审核员 · 决定 · 理由 · 时间）</h4><div class="tblwrap"><table>{rev_rows}</table></div>'
    else: rev_html=_empty('review','Review 记录','无审核记录（review.json 缺失）')
    # 14. 附录
    warn_li=''.join(f'<li class="mono" style="font-size:12.5px">{esc(w)}</li>' for w in (s.get('warnings') or [])) or '<li>无</li>'
    unres_li=''.join(f'<li><b>{esc(u.get("phrase") if isinstance(u,dict) else u)}</b> — {esc(u.get("reason")) if isinstance(u,dict) else ""}</li>' for u in unres) or '<li>无</li>'
    cam_h=man.get('camera_height_m') or s.get('camera_height_m')
    spec_li=''.join(f'<li><span class="mono">{esc(sp.get("policy_id"))}</span> · {esc(sp.get("predicate"))} {esc(sp.get("threshold"))} {esc(sp.get("unit") or "")}'
                    + (' <span class="pill warn">阈值已按本 run 调整</span>' if sp.get('adjusted_for_run') else '')
                    + (f' · <span style="color:var(--muted)">不支持：{esc(sp["unsupported_reason"])}</span>' if sp.get('unsupported_reason') else '') + '</li>' for sp in (p.get('specs') or [])) or '<li>无</li>'
    prov_rows=''.join(f'<tr><td class="mono">{esc(k)}</td><td class="mono">{esc(v)}</td></tr>' for k,v in (man.get('providers') or {}).items()) or '<tr><td colspan=2>无</td></tr>'
    appendix=(f'<h4 data-section="appendix">附录（warnings 全量 · unresolved 全量 · run 参数）</h4>'
              f'<div class="legend"><b>scene warnings（{len(s.get("warnings") or [])}）</b><ul style="margin:4px 0 0 18px;padding:0">{warn_li}</ul></div>'
              f'<div class="legend"><b>unresolved（{len(unres)}）</b><ul style="margin:4px 0 0 18px;padding:0">{unres_li}</ul></div>'
              f'<div class="legend"><b>run 参数</b><div>相机高度：<span class="mono">{f"{cam_h} m" if cam_h is not None else "—（未持久化）"}</span></div>'
              f'<div>policy 集：<ul style="margin:4px 0 0 18px;padding:0">{spec_li}</ul></div>'
              f'<div>后端来源：<div class="tblwrap" style="margin-top:4px"><table>{prov_rows}</table></div></div></div>')
    # 5–7. 2×2 linked block: photo | 3D / CAD | plan — one selection bus, key = inv.
    # Wrapper is data-section-group (not data-section): the 14 section markers stay 14.
    linked=(f'<h4>物体联动 · 原图 / 交互 3D / CAD 平面图 / 交互平面 — 点一个物体，四格同时选中（多选出间距矩阵）</h4>'
            f'<div class="lbar"><button class="lall">全选</button><button class="lclear">全不选</button><span class="lcount">已选 0 个</span><span class="lnames"></span></div>'
            f'<div class="linked" data-section-group="linked">'
            f'<div class="lcell" data-section="photo"><h4>原图点选（按机位）</h4>{photo}</div>'
            f'<div class="lcell" data-section="viewer"><h4>交互 3D（照片色 · 按实例）</h4>{viewer}</div>'
            f'<div class="lcell" data-subsection="cad"><h4>CAD 平面图（全实例 · 悬停看名称）</h4>{cad_fragment(run)}</div>'
            f'<div class="lcell" data-section="plan"><h4>交互平面（点选出距离矩阵）</h4>{plan or "<p class=hint>无平面对象</p>"}</div>'
            f'</div>{cell_rect_html(inv)}')
    sf_txt=round(sf,2) if isinstance(sf,(int,float)) else '—'
    return (f'''<details class="case"><summary>{f'<img src="{thumb(inp)}">' if inp is not None else ''}<span class="cid mono">{rid}</span>
    <span class="pill {cls}">{zh}</span><span class="cmeta mono">scale {sf_txt} · {nobj} 联动物体 · {len(s.get("entities",[]))} 判定实体</span></summary>
      <div class="body">{header}{linked}{CASE_NOTES.get(rid,'')}
    <h4 data-section="verdicts">判定明细</h4>{pol_rows}
    {phr_html}{det_html}
    <h4 data-section="measurements">实体测量（物体 / 数量 / 高 / 尺寸 / 距相机 / 足迹方法）</h4>{scale_html}
    <div class="tblwrap"><table><tr><th>物体</th><th>数量</th><th>高度</th></tr>{ent_rows}</table></div>{obj_tbl}
    {reproj_html}{figs}{refine_html}{chat_html(run)}{rev_html}{appendix}
    <p class="hint mono">全量 3D：runs/{rid}/viewer.html · 自助补测：工作台 补测 tab（或 scripts/refine_region.py --apply）</p>
      </div></details>''')


_TAIL_JS = '\n' + '''</div>
<div id="lb" hidden><img alt=""></div>
<script>
(function(){
var lb=document.getElementById('lb'),im=lb.querySelector('img');
document.querySelectorAll('figure img').forEach(function(el){el.addEventListener('click',function(){im.src=el.src;lb.hidden=false})});
lb.addEventListener('click',function(){lb.hidden=true;im.src=''});
document.addEventListener('keydown',function(e){if(e.key==='Escape')lb.hidden=true});
var PAL=["#c1121f","#1d4ed8","#047857","#b45309","#6d28d9","#0e7490","#9d174d","#4d7c0f","#7c2d12","#334155"];
function hex2rgb(hx){return [parseInt(hx.slice(1,3),16),parseInt(hx.slice(3,5),16),parseInt(hx.slice(5,7),16)];}
var IDLE='__IDLE__';
// one selection bus per linked block; key = inv (index in inventory.json objects)
document.querySelectorAll('.linked').forEach(function(L){
  var ip=L.querySelector('.iplan');if(!ip)return;
  var body=L.closest('.body')||L;
  var pack=JSON.parse(ip.getAttribute('data-pack')),meta=pack.meta,gaps=pack.gaps;
  var pos={};meta.forEach(function(m,i){pos[m.inv]=i;});
  var bus={sel:new Set(),cbs:[],
    select:function(inv,o){o=o||{};if(!(inv in pos))return;
      if(this.sel.has(inv)){if(o.toggle!==false)this.sel.delete(inv);}
      else{this.sel.add(inv);}
      this.emit(o.source);},
    set:function(list,o){this.sel=new Set(list.filter(function(v){return Number.isInteger(v)&&v in pos;}));this.emit((o||{}).source);},
    on:function(cb){this.cbs.push(cb);},
    emit:function(src){var arr=Array.from(this.sel);this.cbs.forEach(function(cb){cb(arr,src);});}};
  L.bus=bus;
  // anything carrying data-inv inside this card toggles: plan polygons, legend chips, CAD hit-areas, toolbar chips, depth chips
  body.addEventListener('click',function(e){var t=e.target.closest?e.target.closest('[data-inv]'):null;if(t)bus.select(+t.getAttribute('data-inv'));});
  body.addEventListener('keydown',function(e){var t=e.target.closest?e.target.closest('polygon[data-inv][role="button"]'):null;
    if(t&&(e.key==='Enter'||e.key===' ')){e.preventDefault();bus.select(+t.getAttribute('data-inv'));}});
  bus.on(function(sel){L.querySelectorAll('polygon[data-inv][role="button"]').forEach(function(pg){
    pg.setAttribute('aria-pressed',sel.indexOf(+pg.getAttribute('data-inv'))>=0?'true':'false');});});
  // toolbar
  var bar=body.querySelector('.lbar');
  if(bar){bar.querySelector('.lall').addEventListener('click',function(){bus.set(meta.map(function(m){return m.inv;}));});
    bar.querySelector('.lclear').addEventListener('click',function(){bus.set([]);});
    bus.on(function(sel){bar.querySelector('.lcount').textContent='已选 '+sel.length+' 个';
      bar.querySelector('.lnames').innerHTML=sel.map(function(v){return '<span data-inv="'+v+'" title="点击取消">'+meta[pos[v]].zh+' ×</span>';}).join('');});}
  // CAD overlay
  bus.on(function(sel){L.querySelectorAll('.icad polygon').forEach(function(pg){pg.classList.toggle('on',sel.indexOf(+pg.getAttribute('data-inv'))>=0);});});
  // Each photo and mask use its own canonical grid and inventory IDs.
  var photos=Array.from(L.querySelectorAll('.iphoto')),framePicker=L.querySelector('.photo-frame');
  var activePhoto=photos[0],previousSelection=new Set();
  function pixelInv(data,p){return data[p*4]+(data[p*4+1]<<8)+(data[p*4+2]<<16)-1;}
  function photoDraw(){
    if(!activePhoto)return;
    var item=activePhoto,cv=item.querySelector('canvas'),base=item.querySelector('.ipbase');
    var data=item.indexData,w=+item.dataset.mw,h=+item.dataset.mh;
    if(!data||!base.naturalWidth)return;
    var ctx=cv.getContext('2d');ctx.clearRect(0,0,w,h);ctx.drawImage(base,0,0,w,h);
    if(bus.sel.size){
      var oc=document.createElement('canvas');oc.width=w;oc.height=h;
      var octx=oc.getContext('2d'),overlay=octx.createImageData(w,h);
      bus.sel.forEach(function(inv){var mask=(item.masks||{})[inv];if(!mask)return;
        var c=hex2rgb(PAL[pos[inv]%PAL.length]);
        for(var p=0;p<w*h;p++)if(mask[p*4]){
          overlay.data[p*4]=c[0];overlay.data[p*4+1]=c[1];overlay.data[p*4+2]=c[2];overlay.data[p*4+3]=150;}
      });
      octx.putImageData(overlay,0,0);ctx.drawImage(oc,0,0);
    }
  }
  function photoStatus(){
    if(!activePhoto)return;
    var evidence=JSON.parse(activePhoto.dataset.photo),sel=Array.from(bus.sel);
    var missing=sel.filter(function(inv){return evidence.mask_inv.indexOf(inv)<0;});
    var text=sel.length?'当前机位可见 '+(sel.length-missing.length)+' / '+sel.length+' 个已选对象':'此机位可点选 '+evidence.clickable_inv.length+' 个对象';
    if(missing.length)text+='；当前不可见：'+missing.map(function(inv){return meta[pos[inv]].zh+'（'+meta[pos[inv]].frame+'）';}).join('、');
    if(evidence.issues.length)text+='；'+evidence.issues.length+' 个对象的掩码不可用';
    L.querySelector('.photo-status').textContent=text;
  }
  function showPhoto(item){
    if(!item)return;
    activePhoto=item;photos.forEach(function(p){p.hidden=p!==item;});
    if(framePicker)framePicker.value=item.dataset.frame;
    photoDraw();photoStatus();
  }
  photos.forEach(function(item){
    var cv=item.querySelector('canvas'),base=item.querySelector('.ipbase'),index=item.querySelector('.ipidx');
    var w=+item.dataset.mw,h=+item.dataset.mh;cv.width=w;cv.height=h;
    function ready(){
      if(!base.complete||!base.naturalWidth||!index.complete||!index.naturalWidth)return;
      var oc=document.createElement('canvas');oc.width=w;oc.height=h;
      var ctx=oc.getContext('2d');ctx.drawImage(index,0,0);item.indexData=ctx.getImageData(0,0,w,h).data;
      if(item===activePhoto)photoDraw();
    }
    base.addEventListener('load',ready);index.addEventListener('load',ready);ready();
    item.masks={};item.querySelectorAll('.ipmask').forEach(function(image){
      function maskReady(){if(!image.complete||!image.naturalWidth)return;
        var oc=document.createElement('canvas');oc.width=w;oc.height=h;var ctx=oc.getContext('2d');ctx.drawImage(image,0,0);
        item.masks[+image.dataset.maskInv]=ctx.getImageData(0,0,w,h).data;if(item===activePhoto)photoDraw();}
      image.addEventListener('load',maskReady);maskReady();
    });
    cv.addEventListener('click',function(e){
      if(!item.indexData)return;
      var rect=cv.getBoundingClientRect(),x=Math.floor((e.clientX-rect.left)/rect.width*w),y=Math.floor((e.clientY-rect.top)/rect.height*h);
      if(x<0||x>=w||y<0||y>=h)return;
      var inv=pixelInv(item.indexData,y*w+x);if(inv>=0)bus.select(inv,{source:'photo'});
    });
  });
  if(framePicker)framePicker.addEventListener('change',function(){showPhoto(photos.find(function(p){return p.dataset.frame===framePicker.value;}));});
  bus.on(function(sel){
    var added=sel.filter(function(inv){return !previousSelection.has(inv);});
    if(added.length){var focus=added[added.length-1],frameId=meta[pos[focus]].frame;
      showPhoto(photos.find(function(p){return p.dataset.frame===frameId;}));}
    previousSelection=new Set(sel);photoDraw();photoStatus();
  });
  // interactive plan + distance matrix
  var info=ip.querySelector('.ipinfo'),linesG=ip.querySelector('.iplines');
  function fmt(i){var m=meta[i];
    var t=(m.tilt!==null&&m.tilt!==undefined)?('，倾角 '+Math.round(m.tilt)+'°'):'';
    var mm=m.method==='contact-edge'?' · 接触边投影':(m.method==='refine'?' · 人工补测':(m.method==='near-cluster'?' · 近簇裁剪':(m.method==='guard-line'?' · 共线对齐':'')));
    if(m.ns)mm+=' · 法线分割';
    return '<b>'+m.zh+'</b> · 高 '+m.h.toFixed(2)+' m'+t+' · '+m.size+' m · 距相机 '+m.d.toFixed(2)+' m'+mm;}
  bus.on(function(sel){
    var selected=sel.map(function(v){return pos[v];});
    ip.querySelectorAll('polygon.ip').forEach(function(pg){var on=sel.indexOf(+pg.getAttribute('data-inv'))>=0;
      pg.setAttribute('fill-opacity',on?'0.55':'0.12');pg.setAttribute('stroke-width',on?'3.5':'1.2');});
    while(linesG.firstChild)linesG.removeChild(linesG.firstChild);
    if(selected.length===0){info.innerHTML=IDLE;info.style.color='var(--muted)';return;}
    var html=selected.map(fmt).join('<br>');
    if(selected.length>1){
      html+='<table style="margin-top:6px;border-collapse:collapse;font-size:12.5px"><tr><th></th>'+selected.map(function(i){return '<th style="padding:2px 8px">'+meta[i].zh+'</th>'}).join('')+'</tr>';
      selected.forEach(function(i){html+='<tr><th style="padding:2px 8px;text-align:left">'+meta[i].zh+'</th>'+selected.map(function(j){return '<td style="padding:2px 8px;text-align:center" class="mono">'+(i===j?'—':gaps[i][j].toFixed(2)+' m')+'</td>'}).join('')+'</tr>';});
      html+='</table>';
      for(var a=0;a<selected.length;a++)for(var b=a+1;b<selected.length;b++){
        var i=selected[a],j=selected[b];
        var ln=document.createElementNS('http://www.w3.org/2000/svg','line');
        ln.setAttribute('x1',meta[i].cx);ln.setAttribute('y1',meta[i].cy);ln.setAttribute('x2',meta[j].cx);ln.setAttribute('y2',meta[j].cy);
        ln.setAttribute('stroke','#047857');ln.setAttribute('stroke-width','2');ln.setAttribute('stroke-dasharray','5,4');linesG.appendChild(ln);
        var tx=document.createElementNS('http://www.w3.org/2000/svg','text');
        tx.setAttribute('x',(meta[i].cx+meta[j].cx)/2);tx.setAttribute('y',(meta[i].cy+meta[j].cy)/2-4);
        tx.setAttribute('font-size','11');tx.setAttribute('fill','#047857');tx.setAttribute('text-anchor','middle');
        tx.style.fontFamily='IBM Plex Mono,monospace';tx.textContent=gaps[i][j].toFixed(2)+' m';linesG.appendChild(tx);}
    }
    info.innerHTML=html;info.style.color='var(--ink)';
  });
  // A ready/load handshake replays selections made before the embedded viewer loaded.
  var frame=L.querySelector('iframe.v3d');
  if(frame){
    var expectedOrigin=window.origin||window.location.origin,targetOrigin=expectedOrigin==='null'?'*':expectedOrigin;
    function sendSelection(){if(frame.contentWindow)frame.contentWindow.postMessage({type:'panoptes:select',inv:Array.from(bus.sel),exclusive:true},targetOrigin);}
    bus.on(function(sel,src){if(src!=='viewer')sendSelection();});
    frame.addEventListener('load',sendSelection);
    addEventListener('message',function(e){var m=e.data;
      if(e.source!==frame.contentWindow||e.origin!==expectedOrigin||!m)return;
      if(m.type==='panoptes:ready'){sendSelection();return;}
      if(m.type==='panoptes:selected'&&Array.isArray(m.inv))bus.set(m.inv,{source:'viewer'});
    });
  }
  bus.emit();
});
})();
</script>
'''.replace('__IDLE__', _IDLE)



def build_interactive_report(run_ids, out_path=None, title="Panoptes"):
    cards = [build_case(rid) for rid in run_ids]
    if len(cards) == 1: cards[0] = cards[0].replace('<details class="case">', '<details class="case" open>', 1)
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
