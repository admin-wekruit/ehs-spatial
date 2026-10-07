"""Build (refresh) a four-view report's measurement layer from the generic checks: per-object confidence, pipeline stage facts,
photo measurements against the field values, button dimensions, the MoGe-3 scale cross-check (on-prem CLI).

Port of research-notes/cell030-sept-pipeline-2026-10-05/{confidence.py,refresh.py}: same logic; every input is an argument or
comes from a per-report config (configs/workcell-layers/{090,030}.json), whose paths use ${NAME} roots: ${DATA} = the report
inputs directory (--data DIR: NNN/view.json = the publication view, 030/layer-patch.json + its asset files), others with
--root NAME=DIR. Idempotent. The e-stop scale (nativeToMeters) and the e-stop's host entities are guarded: the build asserts
the layer's scale equals the config's, and neither the scale nor any fact of a host entity (one with a scale / reference-object
/ e-stop recheck fact, or the config's button host) other than its confidence, pipeline and check facts changed. Every input
is read before anything is written; a configured input that does not exist fails the build; the layer is
built and guarded in memory, then OUT.tmp (and each model-patch asset as NAME.tmp next to OUT) is written and renamed into
place; on any failure the .tmp files are removed and OUT and its assets stay as they were.

  python scripts/workcell_layer_build.py --config configs/workcell-layers/090.json --layer LAYER.json --out OUT.json --data DIR \
      --root NOTES=research-notes --root RUNS=panoptes-serving/outputs/candidate-evaluation
  python scripts/workcell_layer_build.py --self-test

Config keys (entity ids in full, matched exactly): scale, photos, inputs {view, run, shape [first existing], floor,
planeStereo, maskTransfer, identity, moge, lowerEdge}, checkKeys {planeStereo, maskTransfer, identity}, fieldValues
[{entityId, part (the lower_edge target's part label or null), fieldCm (null = shown for reference), labelZh}], button {host,
text}, moge {host, text}; optional valueMFromText [ids], appendToMeasured {id: {unless, text}}, scaleInterval {estopScale,
key}, modelPatch {entity, patch, dropFactKinds, facts}, corrected {id: [finding]}, shapeOverride {id: shape row}. The 现场对照
facts are generated from the lower_edge check (scripts/workcell_checks/lower_edge.py results, inputs.lowerEdge) and
fieldValues: one fact per entity, its first entry with a field value first; values with +-sigma, graded 可信 (sigma <= 1 cm)
or 偏大 (up to lower_edge's 2.5 cm gate, with what dominates sigma), and the scale / floor terms and flags (grazed lines
excluded, edge on the model's own bottom, photos without Pi3X, segmentation corrections); no value with its reason (and the
estimate +- sigma when it was too uncertain); the model's lowest point only as 'model lower part missing' where it ends > 5 cm
above the field edge (the confidence cap's rule; never differenced with a part's edge). A model-path value that agrees with
the field value within max(2 cm, 2 sigma) verifies the object's confidence ('与现场值吻合'); one off by more than max(3 cm,
3 sigma) caps it at low. A fresh layer (no facts, no scale) is filled from the config's scale.
"""
import argparse
import glob
import json
import os
import shutil
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- confidence.py (write() edits in memory) ---------------------------------------------------------------------------------------------
LEVEL = {'high': '高', 'medium': '中', 'low': '低', 'unverified': '未验证'}
ORDER = ['unverified', 'low', 'medium', 'high']
HIGH_OFFSET_M, HIGH_NCC, WEAK_IOU, STEP = .03, .3, .6, .0101


def source_ids(entity):
    return [l.get('sourceRecordId') for l in entity.get('lineage') or [] if isinstance(l, dict) and l.get('operation') == 'offline_import' and l.get('sourceRecordId')]


def cap(level, ceiling):
    return ORDER[min(ORDER.index(level), ORDER.index(ceiling))]


def assess(doc, shape_rows, comparisons, photo_names, corrected=(), extra=None):
    """extra: {entityId: [{'stage': str, 'text': str, 'reason': str|None, 'missing': str|None, 'cap': level|None,
    'verifies': bool, 'contradicts': bool, 'photos': [index0]}]} from the extra checks; 'verifies' = an independent photo check
    that confirms the model where the shape check could not judge; 'contradicts' = one that shows it wrong (level low);
    'photos' = photos that gained an accepted mask."""
    images = {c['imageId']: i for i, c in enumerate(doc['cameras'])}
    obs = {o['id']: o for o in doc['observations']}
    shape = {r['entityId']: r for r in shape_rows if r['model'] == 'displayed'}
    out = {}
    for e in doc['entities']:
        if e.get('sourceContext') or e.get('visible') is False or not e.get('activeModelRepresentationId'):
            continue
        rep = next((r for r in e['representations'] if r['id'] == e['activeModelRepresentationId']), {})
        found = (extra or {}).get(e['id'], [])
        masked = sorted({images[obs[o]['imageId']] for o in e.get('observationRefs') or [] if o in obs and obs[o]['imageId'] in images})
        transferred = sorted({k for f in found for k in f.get('photos') or []} - set(masked))
        photos = sorted(set(masked) | set(transferred))
        comp = next((comparisons[s] for s in source_ids(e) if s in comparisons), None)
        ious = {v['frame_id']: v['generated_refined']['visible_iou'] for v in comp['views']} if comp else {}
        r = shape.get(e['id']); verdict = r['verdict'] if r else None
        seen = sorted({k for i, j, *_ in (r or {}).get('pairsAtModel') or [] for k in (i, j)} | set(photos))
        reasons, missing = [], []
        if len(seen) < 2:
            missing.append('只有 1 张照片看到它：需要另一角度的照片')
        elif len(photos) < 2:
            missing.append('照片 ' + '、'.join(str(k + 1) for k in seen if k not in photos) + ' 也看得到它但没有掩码：可补掩码（不需补拍）')
        offset = r['depthChangeM'] if r else None
        if verdict == 'ok':
            reasons.append(f"照片在模型位置最一致（偏移 {offset*100:+.0f} cm，一致度 {r['nccAtModel']:.2f}）")
        elif verdict == 'depth_off':
            reasons.append(f"照片最一致的位置偏离模型 {offset*100:+.0f} cm"); missing.append('深度可能有误：需要复核或补拍侧面')
        elif verdict == 'ambiguous':
            reasons.append('两个深度同样吻合'); missing.append('需要第三个角度区分深度（或纹理是重复图案）')
        elif verdict in ('inconclusive', None):
            reasons.append('照片间共同可见的表面不足'); missing.append('被遮挡或纹理不足：需要能看到它的另一角度照片')
        weak = [f for f, v in ious.items() if v < WEAK_IOU]
        if ious:
            reasons.append('摆放轮廓 IoU ' + ' / '.join(f'{v:.2f}' for v in ious.values()))
        if weak:
            missing.append('摆放轮廓吻合度偏低（' + ', '.join(f'{ious[f]:.2f}' for f in weak) + '）')
        strong = verdict == 'ok' and (abs(r['bestScale'] - 1) <= STEP or abs(offset) <= HIGH_OFFSET_M) and r['nccAtModel'] >= HIGH_NCC
        if verdict == 'ok' and not strong:
            if r['nccAtModel'] < HIGH_NCC:
                missing.append(f"照片一致度 {r['nccAtModel']:.2f} 低于 {HIGH_NCC}：深度证据弱")
            else:
                missing.append(f"照片最一致的深度与模型差 {offset*100:+.0f} cm（超过一个扫描步长）：深度需复核")
        verified = any(f.get('verifies') for f in found)
        if verdict == 'depth_off' and verified:  # names the verifying check (平面立体 as before, 现场对照, ...)
            by = '、'.join(dict.fromkeys(f.get('stage') or '独立检查' for f in found if f.get('verifies')))
            level = 'medium'; missing.append(f'形状校验与{by}结论不一致：需复核')
        elif verdict == 'depth_off':
            level = 'low'
        elif strong and len(photos) >= 2 and not weak:
            level = 'high'
        elif verdict == 'ok' or (verdict == 'ambiguous' and len(photos) >= 2) or (verified and len(photos) >= 2):
            level = 'medium'
        elif verdict == 'ambiguous' or verified:
            level = 'low'
        else:
            level = 'unverified'
        for f in found:
            if f.get('reason'):
                reasons.append(f['reason'])
            if f.get('missing'):
                missing.append(f['missing'])
            if f.get('cap'):
                level = cap(level, f['cap'])
            if f.get('contradicts'):  # an independent check shows the model is wrong: that is evidence, not a lack of it
                level = 'low'
        kind = {'generated_mesh': '生成网格', 'primitive': '基本体', 'observed_surface': '实测表面'}.get(rep.get('kind'), rep.get('kind', '—'))
        mark = lambda i: '✓' if i in masked else '＋' if i in transferred else '·' if i in seen else '—'
        stages = [f"照片掩码：" + ' · '.join(f"{name}{mark(i)}" for i, name in enumerate(photo_names)) + ('（＋ 投影补出的掩码，· 看得到但无掩码）' if set(seen) - set(masked) else ''),
                  f"模型：{kind}" + ('（度量层校正：照片立体实测平面）' if e['id'] in corrected else ''),
                  ('摆放：轮廓 IoU ' + ' / '.join(f'{v:.2f}' for v in ious.values())) if ious else '摆放：无组装记录',
                  f"形状校验：{ {'ok': '一致', 'depth_off': '深度偏差', 'ambiguous': '含糊', 'inconclusive': '无法判断', None: '无法判断'}.get(verdict, verdict) }"]
        stages += [f"{f['stage']}：{f['text']}" for f in found if f.get('stage')]
        out[e['id']] = {'confidence': {'level': level, 'label': LEVEL[level], 'missing': missing, 'reasons': reasons},
                        'facts': [{'label': '置信度', 'kind': 'confidence', 'text': f"{LEVEL[level]}：" + '；'.join(reasons) + ('。缺：' + '；'.join(missing) if missing else '。')},
                                  {'label': '流程断点', 'kind': 'pipeline', 'text': '｜'.join(stages)}]}
    return out


def comparisons_from(result_dir):
    out = {}
    for f in glob.glob(os.path.join(result_dir, '*-comparison.json')):
        d = json.load(open(f)); out[d['object_id']] = d
    return out


def write(layer, assessed):  # edits the layer in memory (the build writes it once, after its guard)
    layer['confidence'] = {k: v['confidence'] for k, v in assessed.items()}
    for k, v in assessed.items():
        facts = [f for f in layer.setdefault('facts', {}).get(k, []) if f.get('kind') not in ('confidence', 'pipeline')]
        layer['facts'][k] = v['facts'] + facts
    counts = {}
    for v in assessed.values(): counts[v['confidence']['label']] = counts.get(v['confidence']['label'], 0) + 1
    return counts
# --- end of confidence.py ----------------------------------------------------------------------------------------------------

GUARDED, UNGUARDED = ('scale', 'reference_object', 'estop_recheck'), ('confidence', 'pipeline', 'check')
FIELD_AGREE_CM, FIELD_AGREE_K, FIELD_OFF_CM, FIELD_OFF_K, MISSING_BOTTOM_CM = 2., 2., 3., 3., 5.


def guarded(layer, hosts=()):
    """The e-stop scale and every fact of the e-stop's host entities (any entity with a scale / reference-object / e-stop
    recheck fact, and `hosts`: the config's button host), except their confidence, pipeline and check facts. By entity, not
    by text: a host's measured or specification fact may not change, whatever its wording."""
    facts = layer.get('facts', {}); hosts = set(hosts) | {k for k, v in facts.items() if any(f.get('kind') in GUARDED for f in v)}
    kept = {k: [f for f in facts.get(k, []) if f.get('kind') not in UNGUARDED] for k in sorted(hosts)}
    return json.dumps([layer['scale']['nativeToMeters'], kept], ensure_ascii=False, sort_keys=True)


def shape_rows(paths, scale):
    for path in paths:
        if os.path.exists(path):
            d = json.load(open(path)); k = scale / d['nativeToMeters']  # an older run at a superseded scale: same geometry, metres rescale
            for r in d['rows']:
                r['depthChangeM'] *= k
            return d['rows'], path
    raise FileNotFoundError(paths)


def extras(cfg, label_of, lower=None):
    """Findings of the extra checks as confidence inputs: {entityId: [finding]}; a check not in the config is skipped (a
    configured one whose file is missing has already failed the build). lower: the lower_edge results, for the field values."""
    out, inputs, keys, corrected = {}, cfg['inputs'], cfg['checkKeys'], cfg.get('corrected', {})
    add = lambda eid, **f: out.setdefault(eid, []).append(f)
    if lower:  # a photo-measured lower edge against the user's field value: independent of the shape check. Model path only
        rows = {(t['entityId'], t.get('part')): t for t in lower['targets']}  # (a two-view value does not use the model)
        for f in cfg.get('fieldValues', []):
            r = rows.get((f['entityId'], f.get('part')))
            if f.get('fieldCm') is None or not r or r.get('valueCm') is None or r.get('path') != 'model':
                continue
            dd, sg = r['valueCm'] - f['fieldCm'], r['sigmaCm']
            what = f"{f['labelZh']} {r['valueCm']:.1f} ±{sg:.1f} cm，现场 {f['fieldCm']:g} cm，差 {signed(dd)} cm"
            if abs(dd) <= max(FIELD_AGREE_CM, FIELD_AGREE_K * sg):
                add(f['entityId'], stage='现场对照', text=what + '：吻合（≤ max(2 cm, 2σ)）', verifies=True, reason=f'与现场值吻合（{what}）')
            elif abs(dd) > max(FIELD_OFF_CM, FIELD_OFF_K * sg):
                add(f['entityId'], stage='现场对照', text=what + '：不符（> max(3 cm, 3σ)）', cap='low',
                    missing=f'与现场值不符（{what}）：模型深度、摆放或部位需复核')
            else:
                add(f['entityId'], stage='现场对照', text=what + '：介于吻合与不符之间，不计入置信度')
    floor = inputs.get('floor')
    if floor:
        from workcell_checks.floor import status
        for eid, v in json.load(open(floor))['results']['floor']['objects'].items():
            v['status'] = status(v['bottomM']); b = v['bottomM'] * 100  # thresholds of the current module
            field = next((f['fieldCm'] for f in cfg.get('fieldValues', []) if eid == f['entityId'] and f.get('fieldCm') is not None), None)
            if field is not None and b - field > MISSING_BOTTOM_CM:  # the model ends above where the real object still is: its lower part is missing
                add(eid, missing=f'模型最低点 {b:.0f} cm，比现场底部 {field:.0f} cm 高 {b - field:.0f} cm：模型下部缺失', cap='medium')
            if v['status'] == 'sinks':
                add(eid, stage='地面', text=f'模型底部低于地面 {b:+.1f} cm', missing=f'模型底部低于地面 {-b:.1f} cm：摆放需复核', cap='medium')
            elif v['status'] in ('on_floor', 'hovers'):
                add(eid, stage='地面', text=f"模型底部离地 {b:+.1f} cm" + ('（贴地）' if v['status'] == 'on_floor' else '（若应落地则偏高）'))
    plane = inputs.get('planeStereo')
    if plane:  # strong only: peak >= .5, clear of the second peak by >= .15, reference view not grazing
        for eid, o in json.load(open(plane))['reports'][keys['planeStereo']]['objects'].items():
            if eid in corrected:
                continue  # measured on the imported model, which the layer replaces
            parts = [p for p in o.get('parts') or [] if p.get('verdict') in ('ok', 'offset')]
            best = max(parts, key=lambda p: p.get('pixels', 0), default=None)
            if not best:
                continue
            strong = best['nccAtBest'] >= .5 and best['nccAtBest'] - (best.get('secondPeakNcc') or -1) >= .15 and best.get('cosRef', 0) >= .25
            text = f"主平面沿视线 {best['rayChangeCm']:+.0f} cm、倾角差 {best.get('tiltDeg', 0):.0f}°（一致度 {best['nccAtModel']:.2f}→{best['nccAtBest']:.2f}）"
            if not strong:
                add(eid, stage='平面立体', text=text + '，证据弱，仅供参考'); continue
            if best['verdict'] == 'ok':
                add(eid, stage='平面立体', text=text + '：与模型一致', verifies=True, reason=f"双视角平面立体与模型一致（{best['rayChangeCm']:+.0f} cm）")
            else:
                add(eid, stage='平面立体', text=text + '：与模型不符', cap='low', contradicts=True,
                    missing=f"平面立体显示模型主平面偏 {best['rayChangeCm']:+.0f} cm、倾角差 {best.get('tiltDeg', 0):.0f}°：模型形状或位置需修正（条纹可能混叠，量级待第三视角确认）")
    transfer = inputs.get('maskTransfer')
    if transfer:  # the SAM box comes from the model itself, so a transferred mask counts only at held-out IoU >= .6
        for o in json.load(open(transfer))['objects']:
            if o['cell'] != keys['maskTransfer']:
                continue
            inside = max(o['reportMaskIoU'].values(), default=0)
            for t in o['sam']:
                k = t['photo'] - 1; iou = t['heldOutIoU']
                if iou >= .6:
                    add(o['entityId'], stage='投影补掩码', text=f"照片{t['photo']} 投影与分割吻合（IoU {iou:.2f}，留出验证）", photos=[k])
                elif iou < .3 and inside >= .6:
                    add(o['entityId'], stage='投影补掩码', text=f"照片{t['photo']} 投影处分割不到它（IoU {iou:.2f}，本身照片 {inside:.2f}）",
                        missing=f"照片{t['photo']} 里模型投影处没有对应物体：模型范围或身份需复核")
                else:
                    add(o['entityId'], stage='投影补掩码', text=f"照片{t['photo']} 投影吻合度 {iou:.2f}（不计入）")
    identity = inputs.get('identity')
    if identity:
        d = json.load(open(identity)); cell_key = keys['identity']
        ids = {p['objectId']: p['entityId'] for p in d['pairs'] if p['cell'] == cell_key}
        names = {}
        for p in d['pairs']:
            if p['cell'] != cell_key or p['verdict'] == 'single-view':
                continue
            names.setdefault(p['entityId'], []).append(p)
        for eid, pairs in names.items():
            worst = min(min(p['forward']['reprojection'], p['reverse']['reprojection']) for p in pairs)
            verdicts = {p['verdict'] for p in pairs}
            add(eid, stage='跨视角身份', text=('一致' if verdicts == {'same'} else '存疑') + f"（{len(pairs)} 对照片，最低重投影 {worst:.2f}）",
                missing=None if verdicts == {'same'} else '两张照片的掩码可能不是同一物体：需复核', cap=None if verdicts == {'same'} else 'medium')
            for p in pairs:
                for flag in p.get('flags') or []:
                    if 'mask in' in flag:
                        add(eid, missing='另一张照片的掩码漏掉了它可见表面的一部分（被别的物体掩码占用）：分割需复核', cap='medium')
        for o in d['sameViewOverlaps']:
            if o['cell'] == cell_key and o['ofSmallerMask'] > .5:
                a, b = o['objects']
                for me, other in ((a, b), (b, a)):
                    if me in ids:
                        add(ids[me], missing=f"照片{o['photoIndex0'] + 1} 中掩码与「{label_of(ids.get(other), other)}」重叠 {o['ofSmallerMask'] * 100:.0f}%：分割需复核", cap='medium')
    for eid, findings in corrected.items():
        out.setdefault(eid, []).extend(findings)
    return out


REASON_ZH = {'no_object': '报告里没有这个物体', 'no_part_mask': '没有这个部位的掩码（点选的部位在各照片都不可见或分割被拒）',
             'too_few_lines': '可用的下边界线不足 {minLines} 条：被别的物体挡住、出画，或部位不在物体最下沿',
             'too_few_direct_lines': '擦边线（模型只在上方 3–15 px 处）不计入后，直接落到模型上的下边界线不足 {minLines} 条',
             'single_view_see_through': '模型深度与 Pi3X 不符或模型缺失，只有 1 张照片可用，不能三角化',
             'two_view_rejected': '模型深度与 Pi3X 不符或模型缺失，两视角直线三角化不成立：{pairs}',
             'photos_disagree': '照片间不一致：{photos}', 'sigma_too_large': '不确定度 ±{sigmaCm:.1f} cm 超过 {maxSigmaCm:g} cm：估计 {estimateCm:.1f} cm（{photos}）'}
PAIR_ZH = {'tilt': '三角化出的线倾斜 {lineTiltDeg:.0f}°（不是横边）', 'overlap': '两张照片看到的线段只重叠 {overlap:.0%}',
           'straight': '下边界不是一条直线'}
GRADE_ZH = {'trusted': '可信', 'large': '偏大'}


def signed(x):
    return '0.0' if round(x, 1) == 0 else f'{x:+.1f}'.replace('-', '−')


def pm(x):
    return '—' if x is None else f'±{x:.1f}'


def used_photos(r):
    return [p for p in r.get('photos') or [] if p['photo'] in (r.get('usedPhotos') or [])]


def per_photo(r):
    """'照片 1 / 3：23.9 ±3.4 / 23.1 ±3.6' (model path) or the accepted pairs' '照片 1 + 3：21.8 ±1.3' (two-view)."""
    if r['path'] == 'model':
        used = used_photos(r)
        return '照片 ' + ' / '.join(str(p['photo']) for p in used) + '：' + ' / '.join(f"{p['medianCm']:.1f} {pm(p.get('sigmaCm'))}" for p in used)
    return '；'.join(f"照片 {p['photos'][0]} + {p['photos'][1]}：{p['medianCm']:.1f} {pm(p.get('sigmaCm'))}" for p in r.get('pairs') or [] if p['photos'] in (r.get('usedPairs') or []))


def why_sigma(r):
    """What dominates sigma: the Pi3X-vs-model range gap (or its 5 % bound without Pi3X), the segmentation edge or the spread
    (model path); the line fit or the segmentation edge (two-view). '' when the result does not say."""
    if r.get('path') == 'model':
        p = max(used_photos(r), key=lambda p: p.get('sigmaCm') or 0, default=None); t = (p or {}).get('sigmaTerms') or {}
        if not t:
            return ''
        top = max(('iqrCm', 'rangeCm', 'segCm'), key=lambda x: t.get(x) or 0)
        if top == 'rangeCm' and t.get('rangeSource') == 'pi3x':
            known = p.get('pi3xModelGapCm') is not None and p.get('sinDepression') is not None
            return '主要来自 Pi3X 与模型的距离差' + (f" {signed(p['pi3xModelGapCm'])} cm（边界上方 10–150 px 的中位数）× sin 俯角 {p['sinDepression']:.2f}" if known else '')
        return {'rangeCm': '主要来自距离项（这张照片没有 Pi3X：按距离的 5 % × sin 俯角保守计）', 'segCm': '主要来自分割边与照片亮度台阶的偏差',
                'iqrCm': '主要来自沿边高度的离散（IQR）'}[top]
    p = max((p for p in r.get('pairs') or [] if p['photos'] in (r.get('usedPairs') or [])), key=lambda p: p.get('sigmaCm') or 0, default=None)
    if not p or p.get('cmPerPxHypot') is None or not p.get('lineRmsPx'):
        return ''
    if ((p.get('sigmaTerms') or {}).get('segCm') or 0) > ((p.get('sigmaTerms') or {}).get('lineFitCm') or 0):
        return f"主要来自分割边与照片亮度台阶的偏差（1 px ≈ {p['cmPerPxHypot']:.2f} cm）"
    return f"1 px ≈ {p['cmPerPxHypot']:.2f} cm × 直线拟合残差 {max(p['lineRmsPx']):.1f} px"


def notes(r):
    """The flags a reader needs beside the number: grazed lines excluded, the edge on the model's own bottom, photos without
    Pi3X (5 % bound), segmentation-edge corrections."""
    used = used_photos(r); seg = [p for p in used if (p.get('segEdge') or {}).get('mode') == 'corrected' and abs(p['segEdge']['correctionCm']) >= .05]
    nopi = [str(p['photo']) for p in used if (p.get('sigmaTerms') or {}).get('rangeSource') == 'no_pi3x_5pct_of_range']
    return ''.join([f"；擦边 {r['grazedHits']} 条不计入" if r.get('grazedHits') else '',
                    '；下沿与模型轮廓底边重合（≤ 2 px）：读到的是模型自身的底边' if r.get('edgeIsModelBottom') else '',
                    f"；照片 {' / '.join(nopi)} 没有 Pi3X：距离项按距离的 5 % 计" if nopi else '',
                    ('；分割边按照片亮度台阶校正 ' + ' / '.join(f"照片 {p['photo']} {signed(p['segEdge']['correctionCm'])}" for p in seg) + ' cm') if seg else ''])


def edge_clause(r, label, params, field=None):
    """'罩壳下沿 24.0 ±0.6 cm（可信；照片 1 / 3：…；另计：尺度 ±0.4 cm，两视角扫描地面 −0.5 cm），差 0.0 cm' or
    '罩壳下沿本次无可靠值（reason）'. Grade: 可信 at sigma <= 1 cm, 偏大 (with what dominates) up to the publish gate."""
    if r is None:
        return f'{label}本次无可靠值（没有点选这个部位，未测）'
    extra = '' if r.get('estimateCm') is None else '；另计：尺度 ' + pm(r.get('scaleTermCm')) + ' cm' + (
        f"，两视角扫描地面 {signed(r['floorTermCm'])} cm" if r.get('floorTermCm') is not None else '')
    if r.get('valueCm') is None:
        pairs = '；'.join(f"照片 {p['photos'][0]} + {p['photos'][1]}：" + '，'.join(PAIR_ZH[x].format(**p) for x in (p['reason'] or '').split(',') if x)
                         + f"（{pm(p.get('sigmaCm'))} cm）" for p in r.get('pairs') or [] if not p['accepted'])
        photos = '；'.join(x for x in (per_photo(r), why_sigma(r) if r['reason'] == 'sigma_too_large' else '') if x) if r['reason'] in ('sigma_too_large', 'photos_disagree') else ''
        text = REASON_ZH[r['reason']].format(part=r.get('part'), pairs=pairs, photos=photos, sigmaCm=r.get('sigmaCm') or 0, estimateCm=r.get('estimateCm') or 0, **params)
        return f"{label}本次无可靠值（{text}{notes(r)}{extra}）"
    grade = GRADE_ZH[r['grade']] + ('' if r['grade'] == 'trusted' else '：σ 超过 1 cm' + ('，' + why_sigma(r) if why_sigma(r) else ''))
    how = per_photo(r) + ('，射线落到模型表面' if r['path'] == 'model' else '，两视角直线三角化')
    return f"{label} {r['valueCm']:.1f} {pm(r['sigmaCm'])} cm（{grade}；{how}{notes(r)}{extra}）" + (f"，差 {signed(r['valueCm'] - field)} cm" if field is not None else '')


def field_facts(cfg, results, key_of):
    """{entityId: 现场对照 text} from the lower_edge results and the config's fieldValues (exact entity ids). Like for like only:
    a part's measured lower edge against the field value; the model's lowest point (which may include bases or posts) is never
    differenced with it, only reported as 'model lower part missing' where it ends > 5 cm above the field edge (the same rule
    as the confidence cap). A fieldValues part label that none of its entity's lower_edge targets has is a typo: it raises."""
    rows = {(t['entityId'], t.get('part')): t for t in results['targets']}; params = results['params']; by, parts = {}, {}
    for t in results['targets']:
        parts.setdefault(t['entityId'], set()).add(t.get('part'))
    for f in cfg.get('fieldValues', []):
        if f['entityId'] in parts and f.get('part') not in parts[f['entityId']]:
            raise SystemExit(f"{cfg.get('report')}: fieldValues part {f.get('part')!r} of {f['entityId']} is not a lower_edge target part "
                             f"({sorted(map(str, parts[f['entityId']]))}): label typo?")
        by.setdefault(f['entityId'], []).append(f)
    out = {}
    for eid, entries in by.items():
        entries = sorted(entries, key=lambda f: f.get('fieldCm') is None)
        head = entries[0]; field = head.get('fieldCm'); row = lambda f: rows.get((f['entityId'], f.get('part')))
        clauses = [edge_clause(row(f), f['labelZh'], params, f.get('fieldCm')) for f in entries]
        mb = next((r['modelBottomCm'] for r in map(row, entries) if r and r.get('modelBottomCm') is not None), None)
        text = (f"现场 {field:g} cm（{head['labelZh']}）｜照片实测：" if field is not None else '照片实测：') + '；'.join(clauses)
        if mb is not None and field is not None and mb - field > MISSING_BOTTOM_CM:
            text += f"｜模型下部缺失：模型最低点 {mb:.1f} cm，比现场下沿高 {mb - field:.1f} cm"
        out[key_of(eid)] = text + '。'
    return out


def load_config(path, roots):
    """The per-report config with every ${NAME} in its strings replaced by --root NAME=DIR."""
    def sub(x):
        if isinstance(x, str):
            try:
                return string.Template(x).substitute(roots)
            except KeyError as missing:
                raise SystemExit(f'{path}: give --root {missing.args[0]}=DIR')
        return [sub(v) for v in x] if isinstance(x, list) else {k: sub(v) for k, v in x.items()} if isinstance(x, dict) else x
    cfg = json.load(open(path))
    cfg['inputs'] = sub(cfg['inputs'])
    for key in ('scaleInterval', 'modelPatch'):
        if key in cfg:
            cfg[key] = sub(cfg[key])
    return cfg


def build(cfg, layer_in, out):
    """Every input is read first; the layer is built and guarded in memory; only then are OUT.tmp and the asset copies (.tmp)
    written and renamed into place. Any failure removes the .tmp files and leaves OUT and its assets as they were."""
    layer = json.load(open(layer_in)); inputs = cfg['inputs']
    missing = [x for k, x in inputs.items() if k != 'shape' and x and not os.path.exists(x)]  # (shape: the first that exists)
    if missing:  # a configured check whose results are missing fails loudly, never silently drops its facts
        raise FileNotFoundError(f"{cfg.get('report')}: configured inputs missing: {missing}")
    layer.setdefault('scale', {'nativeToMeters': cfg['scale']}); layer.setdefault('facts', {})  # a fresh layer: the config's e-stop scale
    hosts = [cfg['button']['host']]; before = guarded(layer, hosts)
    assert abs(layer['scale']['nativeToMeters'] - cfg['scale']) < 1e-12, (cfg.get('report'), layer['scale'])
    view = json.load(open(inputs['view']))['publication']['snapshot']['revision']['document']
    rows, used = shape_rows(inputs['shape'], cfg['scale'])
    comparisons = comparisons_from(os.path.join(inputs['run'], 'result'))
    lower = inputs.get('lowerEdge'); lower = json.load(open(lower))['results']['lower_edge'] if lower else None
    estop = json.load(open(cfg['scaleInterval']['estopScale'])) if 'scaleInterval' in cfg else None
    mp = cfg.get('modelPatch'); patch = json.load(open(mp['patch'])) if mp else None
    copies = [(os.path.join(os.path.dirname(mp['patch']), os.path.basename(a['url'])), os.path.join(os.path.dirname(os.path.abspath(out)), os.path.basename(a['url'])))
              for a in (patch or {}).get('assets', [])]
    for src, _ in copies:
        if not os.path.exists(src):
            raise FileNotFoundError(src)
    label_of = lambda eid, fallback=None: (layer.get('labels') or {}).get(eid) or next((e['label'] for e in view['entities'] if e['id'] == eid), fallback)
    extra = extras(cfg, label_of, lower)  # reads the extra checks' result files

    def key_of(eid):  # exact entity ids only (configs give full ids)
        if eid in layer['facts'] or any(e['id'] == eid for e in view['entities']):
            return eid
        raise SystemExit(f"{cfg.get('report')}: no entity {eid!r} in the layer or the view (give full entity ids)")
    for eid in cfg.get('valueMFromText', []):  # valueM carried a superseded scale while the text was current: take the text's cm
        for f in layer['facts'].get(key_of(eid), []):
            if f.get('kind') == 'measured' and 'valueM' in f:
                f['valueM'] = round(float(f['text'].split(' cm')[0]) / 100, 4)
    if estop is not None:  # half the spread of red-only and yellow-only scales over the joint scale
        s = estop.get(cfg['scaleInterval']['key'], estop)
        if 'perFeature' in s:  # scripts/workcell_estop_scale.py output
            red, yellow = s['perFeature']['red lip'], s['perFeature']['yellow body']
        else:  # research-note format: widths, depth, fx of one photo
            native = lambda px: px * s['depthNative'] / s['fx']
            red, yellow = .04 / native(s['widthsPx']['red lip']), .08 / native(s['widthsPx']['yellow body'])
        layer['scale'].update(redOnly=round(red, 5), yellowOnly=round(yellow, 5), uncertaintyRelative=round(abs(red - yellow) / 2 / cfg['scale'], 4))
    if patch:  # a model corrected in the layer (e.g. by two-photo plane measurement), with its asset files
        layer.setdefault('models', {}).update(patch['models'])
        ids = {a['id'] for a in patch['assets']}
        layer['assets'] = [a for a in layer.get('assets', []) if a['id'] not in ids] + patch['assets']
        layer['facts'][mp['entity']] = [f for f in layer['facts'].get(mp['entity'], []) if f.get('kind') not in mp['dropFactKinds']] + mp['facts']
    if lower:  # photo lower edge vs the field value (scripts/workcell_checks/lower_edge.py), generated
        for key, text in field_facts(cfg, lower, key_of).items():
            layer['facts'][key] = [x for x in layer['facts'].get(key, []) if x.get('label') != '现场对照'] + [{'label': '现场对照', 'kind': 'check', 'text': text}]
    key = key_of(cfg['button']['host'])
    layer['facts'][key] = [f for f in layer['facts'].get(key, []) if f.get('label') != '按钮尺寸'] + [{'label': '按钮尺寸', 'kind': 'check', 'text': cfg['button']['text']}]
    for eid, rule in cfg.get('appendToMeasured', {}).items():
        for f in [f for f in layer['facts'].get(key_of(eid), []) if f.get('kind') == 'measured']:
            if rule['unless'] not in f['text']:
                f['text'] += rule['text']
    override = cfg.get('shapeOverride', {})
    rows = [r for r in rows if r['entityId'] not in override] + list(override.values())
    a = assess(view, rows, comparisons, cfg['photos'], corrected=set(layer.get('models', {})), extra=extra)
    counts = write(layer, a)
    if inputs.get('moge'):  # independent metric scale as a cross-check fact on the e-stop's host; never applied
        key = key_of(cfg['moge']['host'])
        layer['facts'][key] = [f for f in layer['facts'].get(key, []) if f.get('label') != '独立尺度对照（MoGe-3）'] + [{'label': '独立尺度对照（MoGe-3）', 'kind': 'check', 'text': cfg['moge']['text']}]
    if guarded(layer, hosts) != before:
        raise AssertionError(f"{cfg.get('report')}: e-stop scale or facts changed; {out} left as it was")
    staged = []  # ponytail: assets are renamed before the layer; a failing rename midway can leave earlier assets replaced (same names, same content)
    try:
        for src, dst in copies:
            staged.append((dst + '.tmp', dst)); shutil.copyfile(src, dst + '.tmp')
        staged.append((out + '.tmp', out))
        with open(out + '.tmp', 'w') as f:
            json.dump(layer, f, ensure_ascii=False, indent=1)
        for tmp, final in staged:
            os.replace(tmp, final)
    except BaseException:
        for tmp, _ in staged:
            if os.path.exists(tmp):
                os.remove(tmp)
        raise
    labels = layer.get('labels', {})
    print(cfg.get('report'), counts, 'shape:', os.path.relpath(used, os.path.dirname(os.path.dirname(used))))
    for k, v in a.items():
        print('  ', v['confidence']['label'], (labels.get(k) or next(e['label'] for e in view['entities'] if e['id'] == k))[:22], '|', '；'.join(v['confidence']['missing'])[:110])
    return counts


def _check():
    cams = [{'imageId': 'a'}, {'imageId': 'b'}]
    doc = {'cameras': cams, 'observations': [{'id': 'o1', 'imageId': 'a'}, {'id': 'o2', 'imageId': 'b'}],
           'entities': [{'id': x, 'activeModelRepresentationId': 'r', 'representations': [{'id': 'r', 'kind': 'generated_mesh'}],
                         'observationRefs': refs} for x, refs in [('far', ['o1', 'o2']), ('near', ['o1', 'o2']), ('one', ['o1'])]]}
    row = lambda x, off, ncc, pairs=[[0, 1, 500, .8]]: {'entityId': x, 'model': 'displayed', 'verdict': 'ok', 'depthChangeM': off, 'bestScale': 1 + off / 4, 'nccAtModel': ncc, 'pairsAtModel': pairs}
    a = assess(doc, [row('far', .2, .8), row('near', .01, .8), row('one', .01, .8)], {}, ['1', '2'])
    assert a['far']['confidence']['level'] == 'medium', a['far']   # +20 cm best depth is not high
    assert a['near']['confidence']['level'] == 'high'
    assert a['one']['confidence']['level'] == 'medium' and '可补掩码' in ''.join(a['one']['confidence']['missing'])
    b = assess(doc, [row('one', .01, .8)], {}, ['1', '2'], extra={'one': [{'stage': '投影补掩码', 'text': '照片2', 'photos': [1]}]})
    assert b['one']['confidence']['level'] == 'high', b['one']
    c = assess(doc, [row('near', .01, .8)], {}, ['1', '2'], extra={'near': [{'missing': '底部低于地面', 'cap': 'medium'}]})
    assert c['near']['confidence']['level'] == 'medium'
    d = assess(doc, [], {}, ['1', '2'], extra={'one': [{'missing': '平面不符', 'cap': 'low', 'contradicts': True}]})
    assert d['one']['confidence']['level'] == 'low'
    lower = {'targets': [dict(entityId='near', part='p', path='model', valueCm=23.0, sigmaCm=.6),  # |-1.0| <= 2 cm: verifies
                         dict(entityId='far', part='p', path='model', valueCm=28.5, sigmaCm=1.0),  # 4.5 > max(3, 3 x 1.0): caps at low
                         dict(entityId='one', part='p', path='twoView', valueCm=24.1, sigmaCm=.5)]}  # model-free: says nothing about the model
    ex = extras({'inputs': {}, 'checkKeys': {}, 'fieldValues': [dict(entityId=x, part='p', fieldCm=24, labelZh='下沿') for x in ('near', 'far', 'one')]}, None, lower)
    assert ex['near'][0]['verifies'] and ex['near'][0]['reason'].startswith('与现场值吻合') and ex['far'][0]['cap'] == 'low' and 'one' not in ex, ex
    e = assess(doc, [row('far', .01, .8)], {}, ['1', '2'], extra=ex)
    assert e['near']['confidence']['level'] == 'medium' and e['far']['confidence']['level'] == 'low', e  # (far: high without it)
    f = assess(doc, [dict(row('near', .01, .8), verdict='depth_off')], {}, ['1', '2'], extra=ex)
    assert f['near']['confidence']['level'] == 'medium' and '形状校验与现场对照结论不一致：需复核' in f['near']['confidence']['missing'], f['near']
    layer = {'scale': {'nativeToMeters': 1.2}, 'facts': {'e': [{'kind': 'scale', 'text': 'x'}, {'kind': 'check', 'text': '按钮'}, {'kind': 'measured', 'text': '可见约 8.5 cm'}],
                                                         'o': [{'kind': 'measured', 'text': '急停旁 3 cm'}], 'h': [{'kind': 'specification', 'text': '10 cm'}]}}
    g = guarded(layer, ['h']); layer['facts']['e'][1]['text'] += '（新）'; layer['facts']['o'][0]['text'] += '（新）'
    assert guarded(layer, ['h']) == g  # a host's check fact may change, and so may another entity's fact whatever its wording
    for k, i in (('e', 2), ('h', 0)):  # a host's (scale-fact host or button host) measured / specification fact may not
        x = json.loads(json.dumps(layer)); x['facts'][k][i]['text'] = '8.2 cm'; assert guarded(x, ['h']) != g, (k, i)
    layer['scale']['nativeToMeters'] = 1.21
    assert guarded(layer, ['h']) != g  # nor the e-stop scale
    _check_build()


def _check_build():
    """Generated 现场对照 texts (+-sigma, grade, terms, flags, reasons, like for like, the 5 cm rule, a part typo raises); a
    fresh layer builds and its field agreement verifies; exact ids; a guard failure (by entity), a missing input (a results file
    included) or a missing asset leaves OUT and its assets untouched and no temp file."""
    import tempfile
    params = dict(minLines=10, maxSigmaCm=2.5)
    ph = lambda k, m, sg, gap=None: dict(photo=k, medianCm=m, sigmaCm=sg, sigmaTerms=dict(iqrCm=.2, rangeCm=sg, rangeSource='pi3x'), pi3xModelGapCm=gap, sinDepression=.45)
    model = dict(entityId='eeee1111-x', part='recessed piece', path='model', valueCm=24.0, sigmaCm=.6, estimateCm=24.0, usedPhotos=[1, 3], modelBottomCm=14.9,
                 photos=[ph(1, 24.2, .5), ph(3, 23.7, .7)], scaleTermCm=.4, floorTermCm=-.47, grazedHits=0, grade='trusted')
    large = dict(entityId='iiii5555-x', part='housing', path='model', valueCm=23.4, sigmaCm=1.8, grade='large', usedPhotos=[1], modelBottomCm=23.0,
                 photos=[ph(1, 23.4, 1.8, 3.9)], grazedHits=12, edgeIsModelBottom=True)
    plate = dict(entityId='eeee1111-x', part='front plate', path='model', valueCm=None, reason='sigma_too_large', sigmaCm=3.5, estimateCm=21.6,
                 usedPhotos=[1], modelBottomCm=14.9, photos=[ph(1, 21.6, 3.5, 7.5)], scaleTermCm=.36, floorTermCm=None, grazedHits=3)
    rail = dict(entityId='gggg3333-x', part='bottom rail', path='twoView', valueCm=None, reason='two_view_rejected', modelBottomCm=22.0, usedPairs=[],
                pairs=[dict(photos=[1, 3], accepted=False, reason='overlap', sigmaCm=1.28, lineTiltDeg=9., overlap=0.)])
    cfg = {'report': 't', 'scale': 1.2, 'photos': ['照片1', '照片2'], 'checkKeys': {}, 'button': {'host': 'ffff2222-x', 'text': '规格'},
           'fieldValues': [{'entityId': 'eeee1111-x', 'part': 'front plate', 'fieldCm': None, 'labelZh': '前护板下沿'},
                           {'entityId': 'eeee1111-x', 'part': 'recessed piece', 'fieldCm': 24, 'labelZh': '罩壳下沿'},
                           {'entityId': 'gggg3333-x', 'part': 'bottom rail', 'fieldCm': 20, 'labelZh': '底横梁下沿'},
                           {'entityId': 'hhhh4444-x', 'part': 'housing', 'fieldCm': 24, 'labelZh': '罩壳下沿'},
                           {'entityId': 'iiii5555-x', 'part': 'housing', 'fieldCm': 24, 'labelZh': '罩壳下沿'}]}
    high = dict(entityId='hhhh4444-x', part='housing', path='model', valueCm=None, reason='photos_disagree', usedPhotos=[1, 2], modelBottomCm=44.0,
                photos=[ph(1, 23.0, .4), ph(2, 27.0, .5)], grazedHits=0)
    targets = [model, plate, rail, high, large]
    texts = field_facts(cfg, dict(params=params, targets=targets), lambda eid: eid)
    assert texts['eeee1111-x'] == ('现场 24 cm（罩壳下沿）｜照片实测：罩壳下沿 24.0 ±0.6 cm（可信；照片 1 / 3：24.2 ±0.5 / 23.7 ±0.7，射线落到模型表面；'
                                   '另计：尺度 ±0.4 cm，两视角扫描地面 −0.5 cm），差 0.0 cm；前护板下沿本次无可靠值（不确定度 ±3.5 cm 超过 2.5 cm：估计 21.6 cm'
                                   '（照片 1：21.6 ±3.5；主要来自 Pi3X 与模型的距离差 +7.5 cm（边界上方 10–150 px 的中位数）× sin 俯角 0.45）；擦边 3 条不计入；'
                                   '另计：尺度 ±0.4 cm）。'), texts
    assert texts['iiii5555-x'] == ('现场 24 cm（罩壳下沿）｜照片实测：罩壳下沿 23.4 ±1.8 cm（偏大：σ 超过 1 cm，主要来自 Pi3X 与模型的距离差 +3.9 cm'
                                   '（边界上方 10–150 px 的中位数）× sin 俯角 0.45；照片 1：23.4 ±1.8，射线落到模型表面；擦边 12 条不计入；'
                                   '下沿与模型轮廓底边重合（≤ 2 px）：读到的是模型自身的底边），差 −0.6 cm。'), texts  # 23.0 is not > 5 cm above 24
    assert '底横梁下沿本次无可靠值' in texts['gggg3333-x'] and '只重叠 0%（±1.3 cm）' in texts['gggg3333-x'] and '模型最低点' not in texts['gggg3333-x'], texts  # 22 - 20 <= 5
    assert '模型最低点' not in texts['eeee1111-x']  # 14.9 cm (with its base) is below the edge: no comparison
    assert '照片间不一致：照片 1 / 2：23.0 ±0.4 / 27.0 ±0.5' in texts['hhhh4444-x'] and texts['hhhh4444-x'].endswith('模型下部缺失：模型最低点 44.0 cm，比现场下沿高 20.0 cm。'), texts
    try:
        field_facts({**cfg, 'fieldValues': cfg['fieldValues'] + [{'entityId': 'hhhh4444-x', 'part': 'Housing', 'fieldCm': 24, 'labelZh': 'x'}]},
                    dict(params=params, targets=targets), lambda eid: eid); raise RuntimeError('a part typo passed')
    except SystemExit as error:
        assert 'label typo' in str(error), error
    assert why_sigma(dict(path='model', photos=[dict(photo=1, sigmaCm=None, sigmaTerms=None)], usedPhotos=[1])) == '' == why_sigma(dict(path='twoView', usedPairs=[[1, 2]]))
    entity = lambda i, label: {'id': i, 'label': label, 'activeModelRepresentationId': 'r', 'representations': [{'id': 'r', 'kind': 'generated_mesh'}], 'observationRefs': ['o1', 'o2']}
    doc = {'cameras': [{'imageId': 'a'}, {'imageId': 'b'}], 'observations': [{'id': 'o1', 'imageId': 'a'}, {'id': 'o2', 'imageId': 'b'}],
           'entities': [entity(i, i) for i in ('eeee1111-x', 'ffff2222-x', 'gggg3333-x', 'hhhh4444-x', 'iiii5555-x')]}
    with tempfile.TemporaryDirectory() as d:
        j = lambda name, x: (json.dump(x, open(os.path.join(d, name), 'w'), ensure_ascii=False), os.path.join(d, name))[1]
        os.mkdir(os.path.join(d, 'run')); os.mkdir(os.path.join(d, 'out'))
        cfg['inputs'] = {'view': j('view.json', {'publication': {'snapshot': {'revision': {'document': doc}}}}), 'run': os.path.join(d, 'run'),
                         'shape': [j('shape.json', {'nativeToMeters': 1.2, 'rows': []})],
                         'lowerEdge': j('lower.json', {'results': {'lower_edge': dict(params=params, targets=targets)}})}
        out = os.path.join(d, 'out', 'out.json'); asset = os.path.join(d, 'out', 'm.bin')
        build(cfg, j('fresh.json', {'publicationId': 'p', 'revisionId': 'r'}), out)
        built = json.load(open(out))
        assert built['scale']['nativeToMeters'] == 1.2 and built['facts']['eeee1111-x'][-1]['text'] == texts['eeee1111-x'], built['facts']
        conf = built['confidence']['eeee1111-x']  # no shape row; 24.0 +- 0.6 against the field 24: verified, two photos: medium
        assert conf['level'] == 'medium' and any(x.startswith('与现场值吻合') for x in conf['reasons']), conf
        try:
            build({**cfg, 'button': {'host': 'ffff2222', 'text': '规格'}}, out, out); raise RuntimeError('a prefix id was accepted')
        except SystemExit as error:
            assert 'give full entity ids' in str(error), error
        j('out/out.json', {'kept': True}); open(os.path.join(d, 'm.bin'), 'wb').write(b'new')
        patch = {'modelPatch': {'entity': 'eeee1111-x', 'patch': j('patch.json', {'models': {}, 'assets': [{'id': 'm', 'url': 'm.bin'}]}), 'dropFactKinds': [], 'facts': []}}
        bad = j('bad.json', {'scale': {'nativeToMeters': 1.2}, 'facts': {'ffff2222-x': [{'kind': 'measured', 'text': '可见约 8.5 cm'}]}})
        try:  # the button host's measured fact (no '急停' in it) is guarded by entity
            build({**cfg, **patch, 'appendToMeasured': {'ffff2222-x': {'unless': '复核', 'text': '（改）'}}}, bad, out); raise RuntimeError('guard did not fire')
        except AssertionError as error:
            assert 'e-stop scale or facts changed' in str(error), error
        for broken in ({**cfg, 'inputs': {**cfg['inputs'], 'shape': [os.path.join(d, 'missing.json')]}},
                       {**cfg, 'inputs': {**cfg['inputs'], 'lowerEdge': os.path.join(d, 'gone.json')}},
                       {**cfg, 'modelPatch': {**patch['modelPatch'], 'patch': j('patch2.json', {'models': {}, 'assets': [{'id': 'x', 'url': 'gone.bin'}]})}}):
            try:
                build(broken, bad, out); raise RuntimeError('a missing input was not caught')
            except FileNotFoundError:
                pass
        assert json.load(open(out)) == {'kept': True} and sorted(os.listdir(os.path.join(d, 'out'))) == ['out.json'], os.listdir(os.path.join(d, 'out'))
        build({**cfg, **patch}, bad.replace('bad', 'fresh'), out)
        assert open(asset, 'rb').read() == b'new' and sorted(os.listdir(os.path.join(d, 'out'))) == ['m.bin', 'out.json']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config'); p.add_argument('--layer', help='input layer JSON (may equal --out)'); p.add_argument('--out')
    p.add_argument('--root', action='append', default=[], help='NAME=DIR for ${NAME} in the config')
    p.add_argument('--data', help='the report inputs directory: ${DATA} in the config')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args(argv)
    if a.self_test:
        _check(); print('workcell_layer_build self-test passed'); return
    if not (a.config and a.layer and a.out):
        raise SystemExit('need --config, --layer and --out')
    build(load_config(a.config, {**dict(r.split('=', 1) for r in a.root), **({'DATA': a.data} if a.data else {})}), a.layer, a.out)


if __name__ == '__main__':
    main()
