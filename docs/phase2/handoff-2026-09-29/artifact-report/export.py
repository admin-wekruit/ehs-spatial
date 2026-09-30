"""Export a Fast report (round-5 merged run) into a small, self-contained bundle for a claude.ai artifact viewer:
room mesh per shot (decimated GLB), observed-surface GLB per shot (decimated), RecGen models per shot (decimated,
merged, one node per object id), slim cards + primitives + camera keyframes (data.json), and the video."""
import glob, json, os, shutil, sys
import numpy as np, open3d as o3d, trimesh

RUNS = '/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs'
VIDEOS = {  # warm calls of the round-5 merged runs
    'me340': ('r5b-int-me340-002', 'mvp-me340-e84efffd-1790741879', 'ME340 车间'),
    'samsclub': ('r5b-int-samsclub-a2-002', 'mvp-samsclub-a2-d5e0c855-1790741753', "Sam's Club 超市"),
    'walmart': ('r5b-int-walmart-002', 'mvp-walmart-c0761a2a-1790742209', 'Walmart 超市'),
}
ROOM_TRIS, SURF_BUDGET_MB, MODEL_BUDGET_MB = 160_000, 7.0, 6.0


def latest_patches(report_dir):
    out = {}
    for f in sorted(glob.glob(f'{report_dir}/patches/*.json')):
        out[os.path.basename(f).split('-', 1)[1][:-5]] = json.load(open(f))
    return out


def blob_list(patch):
    b = patch['blobs']
    return list(b.items()) if isinstance(b, dict) else [(None, x) for x in b]


def o3d_mesh(v, c, f):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(v, np.float64)),
                                  o3d.utility.Vector3iVector(np.asarray(f, np.int32)))
    if c is not None and len(c) == len(v):
        m.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(c, np.float64), 0, 1))
    m.remove_duplicated_vertices()  # glTF meshes come unwelded; quadric decimation needs shared edges
    return m


def decimate(v, c, f, target):
    m = o3d_mesh(v, c, f)
    if len(f) > target > 0:
        m = m.simplify_quadric_decimation(target_number_of_triangles=int(target))
        ext = float(np.max(np.asarray(m.get_max_bound()) - np.asarray(m.get_min_bound()))) or 1.0
        vox = ext / max(8.0, np.sqrt(target))
        for _ in range(6):  # quadric decimation stalls on open, non-manifold scans: fall back to vertex clustering
            if len(m.triangles) <= 1.5 * target:
                break
            m = m.simplify_vertex_clustering(voxel_size=vox, contraction=o3d.geometry.SimplificationContraction.Average)
            vox *= 1.6
    m.remove_unreferenced_vertices()
    col = np.asarray(m.vertex_colors) if m.has_vertex_colors() else None
    return np.asarray(m.vertices, np.float32), col, np.asarray(m.triangles, np.int32)


def tri(v, c, f):
    kw = {}
    if c is not None and len(c) == len(v):
        rgba = np.concatenate([np.clip(c, 0, 1) * 255, np.full((len(c), 1), 255)], 1).astype(np.uint8)
        kw['vertex_colors'] = rgba
    return trimesh.Trimesh(vertices=v, faces=f, process=False, **kw)


def geom_colors(g):
    try:
        vc = g.visual.to_color().vertex_colors if g.visual.kind == 'texture' else g.visual.vertex_colors
        return np.asarray(vc)[:, :3] / 255.0 if vc is not None and len(vc) == len(g.vertices) else None
    except Exception:
        return None


def quat_matrix(q):  # [x, y, z, w]
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


_CT = {5126: (np.float32, 4), 5121: (np.uint8, 1), 5123: (np.uint16, 2), 5125: (np.uint32, 4), 5120: (np.int8, 1), 5122: (np.int16, 2)}
_NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def read_glb(path):
    """Minimal glTF-binary reader for the pipeline's own GLBs (POSITION, COLOR_0, indices; node matrix or TRS).
    Returns [(node_name, V (n,3) in the node's parent frame after its own transform, C (n,3) 0-1 or None, F (m,3))]."""
    import struct
    raw = open(path, 'rb').read()
    jlen = struct.unpack('<I', raw[12:16])[0]
    js = json.loads(raw[20:20 + jlen])
    boff = 20 + jlen + 8
    bin_ = raw[boff:]

    def acc(i):
        a = js['accessors'][i]; bv = js['bufferViews'][a['bufferView']]
        dt, sz = _CT[a['componentType']]; nc = _NC[a['type']]
        start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        stride = bv.get('byteStride')
        if stride and stride != sz * nc:
            rows = np.frombuffer(bin_, np.uint8, count=a['count'] * stride, offset=start).reshape(a['count'], stride)
            arr = rows[:, :sz * nc].copy().view(dt).reshape(a['count'], nc)
        else:
            arr = np.frombuffer(bin_, dt, count=a['count'] * nc, offset=start).reshape(a['count'], nc)
        if a.get('normalized') and dt in (np.uint8, np.uint16):
            arr = arr.astype(np.float32) / (255.0 if dt == np.uint8 else 65535.0)
        return arr

    def node_T(n):
        if 'matrix' in n:
            return np.asarray(n['matrix'], np.float64).reshape(4, 4).T
        T = np.eye(4)
        if 'rotation' in n: T[:3, :3] = quat_matrix(n['rotation'])
        if 'scale' in n: T[:3, :3] = T[:3, :3] * np.asarray(n['scale'])
        if 'translation' in n: T[:3, 3] = n['translation']
        return T

    out = []
    def walk(i, P):
        n = js['nodes'][i]; T = P @ node_T(n)
        if 'mesh' in n:
            for pr in js['meshes'][n['mesh']]['primitives']:
                at = pr['attributes']
                V = acc(at['POSITION']).astype(np.float64)
                V = (T[:3, :3] @ V.T).T + T[:3, 3]
                C = acc(at['COLOR_0'])[:, :3].astype(np.float64) if 'COLOR_0' in at else None
                if C is not None and C.max() > 1.0: C = C / 255.0
                F = acc(pr['indices']).reshape(-1, 3).astype(np.int64) if 'indices' in pr else np.arange(len(V)).reshape(-1, 3)
                out.append((n.get('name', f'node{i}'), V, C, F))
        for ch in n.get('children', []):
            walk(ch, T)
    scene = js['scenes'][js.get('scene', 0)]
    for r in scene['nodes']:
        walk(r, np.eye(4))
    return out


def num(x, nd=3):
    return None if x is None else (round(float(x), nd) if isinstance(x, (int, float)) else x)


def slim_field(v):
    if not isinstance(v, dict):
        return None
    keep = {k: v.get(k) for k in ('value', 'u', 'status', 'unit', 'level', 'reason', 'bound') if v.get(k) is not None}
    for k in ('value', 'u'):
        if k in keep:
            keep[k] = [num(x) for x in keep[k]] if isinstance(keep[k], list) else num(keep[k])
    if isinstance(keep.get('reason'), str):
        keep['reason'] = keep['reason'][:160]
    return keep


def export(key):
    run, rep, title = VIDEOS[key]
    M = f'{RUNS}/{run}/mirror'
    P = latest_patches(f'{M}/reports/{rep}')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), key)
    os.makedirs(out, exist_ok=True)
    bp = lambda sha: f'{M}/blobs/sha256/{sha}'

    # cards
    cards_blob = dict(blob_list(P['object_cards']))['cards']
    cards = json.load(open(bp(cards_blob['sha256'])))
    by_id = {c['id']: c for c in cards}
    shots = sorted({int(c.get('shot', 0)) for c in cards})

    # room meshes
    for name, b in blob_list(P['room']):
        if b.get('format') != 'panoptes-mesh-v1':
            continue
        lay = b['byteLayout']; raw = open(bp(b['sha256']), 'rb').read(); n = lay['vertexCount']
        rows = np.frombuffer(raw, np.float32, count=n * 9, offset=lay['byteOffset']).reshape(n, 9)
        idx = np.frombuffer(raw, np.uint32, count=lay['indexCount'], offset=lay['indexByteOffset']).reshape(-1, 3)
        v, c, f = decimate(rows[:, :3], rows[:, 6:9], idx, ROOM_TRIS)
        shot = int(b['frame'].split('-')[1])
        tri(v, c, f).export(f'{out}/room-{shot}.glb')

    # observed surfaces: one GLB a shot, one node per card id
    for name, b in blob_list(P['surfaces']):
        src = bp(b['sha256']); size_mb = os.path.getsize(src) / 1e6
        parts = read_glb(src)
        ratio = min(1.0, SURF_BUDGET_MB / size_mb)
        new = trimesh.Scene()
        for node, v, c, f in parts:
            if len(f) == 0:
                continue
            if ratio < 1.0 and len(f) > 60:
                v, c, f = decimate(v, c, f, max(40, int(len(f) * ratio)))
            new.add_geometry(tri(v, c, f), node_name=node, geom_name=f'g-{node}')
        shot = None
        for s in P['surfaces']['data']['surfaces']:
            if s.get('object') in new.graph.nodes:
                shot = int(s.get('shot', 0)); break
        new.export(f'{out}/surfaces-{shot}.glb')

    # RecGen models: dedupe blobs, decimate, place with the entry's transform, one node per object id
    ents = P['models']['data']['models']
    blobs = dict(blob_list(P['models']))
    uniq = {}
    for e in ents:
        b = blobs.get(f"model-{e['object']}")
        if b:
            uniq.setdefault(b['sha256'], []).append(e)
    total_faces_budget = int(MODEL_BUDGET_MB * 1e6 / 30)  # ~30 bytes a face incl. vertices and colours
    per_mesh = max(300, min(3000, int(total_faces_budget * 0.35) // max(1, len(uniq))))
    scenes = {}
    for sha, es in uniq.items():
        parts = read_glb(bp(sha))
        V = np.concatenate([p[1] for p in parts]); F = np.concatenate([p[3] + sum(len(q[1]) for q in parts[:k]) for k, p in enumerate(parts)])
        C = np.concatenate([p[2] if p[2] is not None else np.full((len(p[1]), 3), .7) for p in parts])
        v, c, f = decimate(V, C, F, per_mesh)
        mesh = tri(v, c, f)
        for e in es:
            t = e['transform']; T = np.eye(4)
            T[:3, :3] = quat_matrix(t.get('quaternion', [0, 0, 0, 1])) * np.asarray(t.get('scale', [1, 1, 1]))
            T[:3, 3] = t.get('position', [0, 0, 0])
            shot = int(by_id.get(e['object'], {}).get('shot', 0))
            scenes.setdefault(shot, trimesh.Scene()).add_geometry(mesh, node_name=e['object'], geom_name=f'm-{sha[:12]}', transform=T)
    for f_ in glob.glob(f'{out}/models-*.glb'):
        os.remove(f_)
    model_files = {}
    for shot, sc in scenes.items():  # chunks of <= ~11 MB (artifact files must stay under 15 MB)
        nodes = [n for n in sc.graph.nodes_geometry]
        k, chunk, size = 0, trimesh.Scene(), 0
        for n in nodes:
            T, gname = sc.graph[n]; g = sc.geometry[gname]
            chunk.add_geometry(g, node_name=n, geom_name=gname, transform=T)
            size += len(g.faces) * 12 + len(g.vertices) * 16
            if size > 10.5e6:
                chunk.export(f'{out}/models-{shot}-{k}.glb'); model_files.setdefault(shot, []).append(f'models-{shot}-{k}.glb')
                k, chunk, size = k + 1, trimesh.Scene(), 0
        if len(chunk.graph.nodes_geometry):
            chunk.export(f'{out}/models-{shot}-{k}.glb'); model_files.setdefault(shot, []).append(f'models-{shot}-{k}.glb')
    generated = {e['object']: e for e in ents}

    # slim cards + primitives + where each object is
    fields = ('position_xy', 'top_above_floor', 'base_above_floor', 'height', 'width', 'depth', 'visible_length',
              'principal_axis_tilt_deg', 'planar_slope_deg')
    surf_by = {s['object']: s for s in P['surfaces']['data']['surfaces']}
    slim = []
    for c in cards:
        ident = c.get('identity') or {}; model = c.get('model') or {}; phys = c.get('physical') or {}; tm = c.get('time') or {}
        g = generated.get(c['id'])
        entry = {'id': c['id'], 'kind': c.get('kind'), 'shot': int(c.get('shot', 0)), 'name': ident.get('name') or ident.get('proposed'),
                 'decided_by': ident.get('decided_by'), 'category': (c.get('class') or {}).get('category'),
                 'physical': {k: slim_field(phys.get(k)) for k in fields if phys.get(k) is not None},
                 'scale': (phys.get('top_above_floor') or {}).get('scale'),
                 'time': {'first': tm.get('first_seen_s'), 'last': tm.get('last_seen_s'), 'state': tm.get('state')},
                 'views': (c.get('views') or {}).get('n')}
        if g:
            gate = g.get('gate', {})
            entry['model'] = {'kind': 'RecGen', 'iou': num(gate.get('fit_iou') or gate.get('silhouette_iou')), 'display': (gate.get('display') or '')[:120]}
            if g.get('bounds'):
                entry['box'] = [g['bounds']['min'], g['bounds']['max']]
        elif model.get('kind') in ('box', 'cylinder', 'plane', 'open frame'):
            entry['model'] = {'kind': model['kind'], 'position': model.get('position'), 'quaternion': model.get('quaternion'),
                              'size_m': model.get('size_m'), 'radius_m': model.get('radius_m'), 'length_m': model.get('length_m'),
                              'parts': model.get('parts'), 'chosen_by': (model.get('chosen_by') or '')[:120]}
            if model.get('position'):
                p = np.asarray(model['position']); h = np.asarray(model.get('size_m') or [model.get('radius_m', .1)] * 3) / 2
                entry['box'] = [(p - h).tolist(), (p + h).tolist()]
        elif model.get('kind') == 'observed surface':
            entry['model'] = {'kind': 'observed surface'}
        else:
            entry['model'] = {'kind': model.get('kind') or 'none'}
        s = surf_by.get(c['id'])
        if 'box' not in entry and s and s.get('min'):
            entry['box'] = [s['min'], s['max']]
        if 'box' in entry:
            entry['box'] = [[num(x) for x in entry['box'][0]], [num(x) for x in entry['box'][1]]]
        slim.append(entry)

    cams = []
    for s in P['cameras']['data']['shots']:
        cams.append({'shot': int(s['index']), 'times': [num(t) for t in s['times']],
                     'c2w': [[[num(x, 4) for x in row] for row in m[:3]] for m in s['c2w']],
                     'fov_y': num(s.get('fov_y_deg') or s.get('vfov_deg') or s.get('fov_deg')),
                     'keys': s.get('keys')})
    vid = blob_list(P['video'])[0][1]
    shutil.copy(bp(vid['sha256']), f'{out}/video.mp4')
    vd = P['video']['data']
    json.dump({'key': key, 'title': title, 'report': rep, 'video': {'fps': vd['fps'], 'w': vd['width'], 'h': vd['height']},
               'cameras': cams, 'cards': slim, 'model_files': {str(k): v for k, v in model_files.items()}, 'shots': sorted({e['shot'] for e in slim}),
               'counts': {'cards': len(slim), 'recgen': len(generated), 'primitives': sum(1 for e in slim if e['model']['kind'] in ('box', 'cylinder', 'plane', 'open frame')),
                          'surfaces': sum(1 for e in slim if e['model']['kind'] == 'observed surface')},
               'camera_keys': list(P['cameras']['data']['shots'][0].keys())},
              open(f'{out}/data.json', 'w'), ensure_ascii=False, separators=(',', ':'))
    sizes = {f: round(os.path.getsize(f'{out}/{f}') / 1e6, 2) for f in sorted(os.listdir(out))}
    print(key, sizes, 'total MB', round(sum(sizes.values()), 1))


if __name__ == '__main__':
    for k in (sys.argv[1:] or VIDEOS):
        export(k)
