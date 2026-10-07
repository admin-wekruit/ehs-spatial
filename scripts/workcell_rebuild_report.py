"""Rebuild a published four-view report on another geometry (e.g. licence-clean MVS instead of Pi3X), with the current
capture pipeline's contracts at every stage, so the new report is a separate publication next to the old one.

Config: configs/workcell-rebuild/<cell>.json (the old report's displayed entities and how each one is rebuilt).

Procedure (cell 090, research-notes/workcell-clean-report-090-2026-10-06): evidence -> surfaces -> SAM 3D from every photo
(research-notes/completion-licence-ab-2026-10-05/completion_ab.py, AB_RUN / AB_OUT / AB_FRAME=all) -> assembly (the same,
--stage assemble) -> compose with the candidates as box priors -> workcell_rebuild_publish.py trial -> box_faces on the trial view
(modal_apps/workcell_layer_trial.py --served-dir) -> boxes -> compare.py with those boxes (CMP_BOXES, CMP_COVERAGE_WEIGHT,
CMP_EXTENT_GATE_CM, CMP_EXPLODE_GATE) -> compose again with the measured-box fallbacks, until the choices repeat -> publish ->
box_faces, lower_edge, obvious_errors, workcell_fix_obvious, workcell_smooth_models on the live report -> layer.

  evidence  --config C --geometry GEOM_RUN --view OLD_VIEW.json --out NEW_RUN
            A run directory in the lucida-replica-01 / prepare_capture_evidence layout: the geometry run's frozen frames and
            point maps, its floor (plane and fit statistics kept), and object views from the report's own photo masks: the
            source run's object masks for run objects, the old report's observation polygons (union per photo, original
            pixels; as workcell_shape_check.load_report reads them) for every other entity. Views go through the serving
            pipeline's object_view (pinned files); a view with no geometry point inside its mask is kept with zero points.
            Also geometry/point_cloud.glb (the importer's native cloud: valid points of every frame, in frame order).
  surfaces  --config C --run NEW_RUN --scale SCALE.json
            The entities that are not completed by a generator, regenerated from the new point maps only:
            plane       RANSAC plane (seed 7, 300 iterations, 2 % of the support diagonal, SVD refit) on the entity's
                        point-map samples inside its masks (planeFrom parent: its parent's, for a see-through sheet); the largest mask's original pixels (grid step for <= 60k
                        vertices) cast as camera rays onto that plane, photo colours; < 50 samples -> no model (reason kept)
            floor_plane the same rays onto the report floor (+ lift), e.g. painted floor marking
            estop       the specification cylinder (diameter, height) at the triangulated e-stop, axis = floor normal
            floor       the report floor itself (floor_region: floor masks + pixels outside every object mask whose point-map
                        sample is within 3 cm of the floor + holes surrounded by such floor) cast onto the floor plane (later
                        photos only where earlier ones left 2 cm cells empty), the drawn reference surface of the floor entity
            -> NEW_RUN/surfaces/<id>.npz (world vertices, faces, colours) and surfaces/record.json
  compose   --config C --run NEW_RUN --completion OUT --choices RESULTS.json --scale SCALE.json --label TITLE [--boxes B.json]
            The assembled scene (result/, assemble_lucida_scene's format) and its public scene (build_capture_report, unchanged):
            sam3d objects = the chosen candidate (compare.py results.json: variant and decision) with compare.floor_fix, in its
            own object frame and assembled pose, decimated to 80k faces (build_layer.decimate); a decision 'measured box' with
            a boxes file = the box (as the viewer draws it) when its fit is not capped (box_faces R3: a capped box is the model's
            own); else the best candidate stays with its gates listed, or, when it fails the photo fit (IoU < 0.5), no model is
            shown (unavailable, with the reason); parts = the parent's faces whose centroids fall in the part's own
            masks in every photo that has one (unassigned faces within 10 cm follow their nearest assigned face), the parent
            keeps the rest; surfaces as written; 'observed_floor' exactly as assemble writes it (the importer binds the floor
            evidence to it); 'floor_surface' (an observed record) carries the floor's reference-surface mesh, which the
            publish step gives to a floor entity; context entities hidden; entities without a model -> unavailable_objects.
  boxes     --run-results BOX_FACES_RESULTS.json --manifest IMPORT_MANIFEST_OR_RESULT.json --out BOXES.json
            The box_faces output as the layer's boxes file (workcell-boxes-2026-10-06/build.py format) with finalize.py's rule
            (an unverified dimension shows the displayed model's size, source 'model') and objectIds {run object: entity id}.
  layer     --config C --run NEW_RUN --view VIEW.json --choices RESULTS.json --scale SCALE.json --out LAYER.json
            [--boxes B.json] [--lower-edge LE.json] [--field FAIR_AB.json] [--obvious OE.json] [--patches DIR,...] [--pages DIR]
            The new publication's own measurement layer (web/src/measurement-layer.ts): the e-stop scale and the report floor
            (as the publication's frame, restated), per-object confidence from the generic numbers of each method, a model
            note, the boxes (box_faces), 现场对照 facts from lower_edge (click part masks) and from the fair evaluator's
            near-face edge on the new point maps, per-object pipelines (every stage; the compare.py sheet for SAM 3D objects,
            copied to --pages as pipeline-<id8>.jpg), and layer model patches (--patches: smoothing / obvious-error fixes,
            {models, assets} files; their .bin files copied to --pages).
  selftest                       synthetic checks of fit_plane, box_mesh and partition
  coverage  --run NEW_RUN       per object view: mask pixels and point-map samples inside the mask
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SERVING = Path('/Users/adam/Desktop/panoptes-public/panoptes-serving')
PLATFORM = Path('/Users/adam/Desktop/Tesla/panoptes-platform')
RUNS = SERVING / 'outputs/candidate-evaluation'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def platform_module(rel):
    """A platform module by path (its package name 'scripts' collides with the serving and worktree ones)."""
    import importlib.util
    if str(PLATFORM) not in sys.path:
        sys.path.append(str(PLATFORM))
    spec = importlib.util.spec_from_file_location(Path(rel).stem, PLATFORM / rel)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def old_masks(view, entity_prefix):
    """{camera index: original-resolution bool mask}: union of the entity's observation polygons, even-odd per observation."""
    sys.path.insert(0, str(HERE))
    from workcell_shape_check import polygon_mask
    doc = view['publication']['snapshot']['revision']['document']
    entity = next(e for e in doc['entities'] if e['id'].startswith(entity_prefix))
    index = {c['imageId']: k for k, c in enumerate(doc['cameras'])}
    obs = {o['id']: o for o in doc['observations']}
    out = {}
    for oid in entity['observationRefs']:
        o = obs[oid]
        if o['imageId'] in index and o.get('originalPixelPolygons'):
            k = index[o['imageId']]; cam = doc['cameras'][k]
            m = polygon_mask(o['originalPixelPolygons'], (cam['height'], cam['width']))
            out[k] = out[k] | m if k in out else m
    return entity, out


def view_record(root, oid, frame_id, mask, provenance):
    """prepare_lucida_evidence.object_view, keeping views whose mask holds no geometry point (sparse point maps)."""
    sys.path.insert(0, str(SERVING / 'scripts/research'))
    from prepare_lucida_evidence import object_view
    try:
        return object_view(root, oid, frame_id, mask, provenance)
    except AssertionError:  # len(cloud) > 0: every file is written by then; the view has no point-map sample
        out = root / 'evidence/objects' / oid / frame_id
        geom = root / 'geometry/frames' / frame_id
        frame = next(f for f in json.loads((root / 'manifest.json').read_text())['frames'] if f['frame_id'] == frame_id)
        rel = lambda p: str(p.relative_to(root))  # noqa: E731
        y, x = np.nonzero(mask); w, h = Image.open(root / frame['input']).size
        pad = max(8, round(.025 * max(x.max() - x.min(), y.max() - y.min())))
        box = [max(0, int(x.min()) - pad), max(0, int(y.min()) - pad), min(w, int(x.max()) + pad + 1), min(h, int(y.max()) + pad + 1)]
        return {'frame_id': frame_id, 'rgb_path': frame['input'], 'mask_path': rel(out / 'mask.png'), 'rgba_path': rel(out / 'rgba.png'),
                'rgba_crop_xyxy_in_input': box, 'rgba_pixel_to_input': [[1, 0, box[0]], [0, 1, box[1]], [0, 0, 1]],
                'canonical_mask_path': rel(out / 'canonical_mask.npy'), 'points_path': rel(out / 'points.npy'), 'colors_path': rel(out / 'colors.npy'),
                'K_path': rel(geom / 'intrinsics.npy'), 'c2w_path': rel(geom / 'camera_to_world.npy'), 'canonical_rgb_path': rel(geom / 'canonical.png'),
                'pointmap_path': rel(geom / 'pts3d.npy'), 'valid_path': rel(geom / 'valid_mask.npy'),
                'content_valid_path': rel(geom / 'content_valid_mask.npy'), 'conf_path': rel(geom / 'conf.npy'),
                'K_pixel_grid': 'canonical518x518; source affine in manifest.json', 'mask_pixels': int(mask.sum()),
                'partial_point_count': 0, 'centroid_native': None, 'observed_only': True, 'provenance': provenance,
                'sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file()}}


def evidence(config, geometry, view_path, out):
    """geometry: a run with geometry/frames, evidence/canonical, evidence/floor.json and manifest.json (the MVS export)."""
    cfg = json.loads(Path(config).read_text()); geometry = Path(geometry).resolve(); out = Path(out).resolve()
    if out.exists():
        raise ValueError('choose a fresh run directory')
    source = RUNS / cfg['sourceRun']
    view = json.loads(Path(view_path).read_text())
    assert view['publication']['id'] == cfg['publicationId'], 'view is not the configured publication'
    sys.path.insert(0, str(SERVING / 'scripts/research'))
    import prepare_capture_evidence as pce
    manifest = json.loads((geometry / 'manifest.json').read_text())
    (out / 'input').mkdir(parents=True)
    for frame in manifest['frames']:
        shutil.copyfile(source / frame['input'], out / frame['input'])
        assert digest(out / frame['input']) == frame['sha256']
    shutil.copytree(geometry / 'evidence/canonical', out / 'evidence/canonical')
    shutil.copytree(geometry / 'geometry', out / 'geometry')
    manifest.update(experiment=cfg['experiment'], status='inputs_frozen', evidence={})
    geom_meta = manifest['geometry']
    save(out / 'manifest.json', manifest)
    pce.geometry(out)  # content masks by the importer's rule, the manifest's geometry record
    manifest = json.loads((out / 'manifest.json').read_text())
    for key in ('model', 'coordinate_system', 'candidate'):  # pce.geometry writes the Pi3X labels; keep the geometry run's own
        if key in geom_meta:
            manifest['geometry'][key] = geom_meta[key]
    manifest['geometry']['source_geometry_run'] = geometry.name  # a name, not a local path (the manifest is published)
    save(out / 'manifest.json', manifest)
    # the importer's native cloud: every frame's valid points, frame order, canonical colours (import_geometry_evidence)
    write_cloud = platform_module('scripts/import_geometry_evidence.py').write_cloud
    P, C = [], []
    for frame in manifest['frames']:
        g = out / 'geometry/frames' / frame['frame_id']
        valid = np.load(g / 'valid_mask.npy')
        P.append(np.load(g / 'pts3d.npy')[valid]); C.append(np.asarray(Image.open(g / 'canonical.png').convert('RGB'))[valid])
    rgb = np.concatenate(C)
    (out / 'geometry/point_cloud.glb').write_bytes(write_cloud(np.concatenate(P), np.column_stack([rgb, np.full(len(rgb), 255, np.uint8)])))
    # objects
    run_objects = {o['object_id']: o for o in json.loads((source / 'evidence/objects.json').read_text())['objects']}
    frames = [f['frame_id'] for f in manifest['frames']]
    result = []
    for ent in cfg['entities']:
        entity, masks = old_masks(view, ent['old'])
        views = []
        if ent.get('run'):
            src = run_objects[ent['id']]
            for v in src['views']:
                mask = np.asarray(Image.open(source / v['mask_path'])) > 0
                views.append(view_record(out, ent['id'], v['frame_id'], mask, v['provenance']))
            reference = src['reference_frame']
        else:
            for k, mask in sorted(masks.items()):
                prov = {'type': 'report observation polygons (union per photo)', 'publicationId': cfg['publicationId'],
                        'entityId': entity['id'], 'observationIds': [o for o in entity['observationRefs']]}
                views.append(view_record(out, ent['id'], frames[k], mask, prov))
            reference = max(views, key=lambda v: v['mask_pixels'])['frame_id'] if views else None
        if not views:
            raise ValueError(f"{ent['id']}: no photo mask")
        cents = [np.array(v['centroid_native']) for v in views if v['centroid_native'] is not None]
        result.append({'object_id': ent['id'], 'label': entity['label'], 'reference_frame': reference, 'source_inventory_indices': [],
                       'old_entity_id': entity['id'], 'views': views,
                       'physical_identity': {'status': 'the old report entity (one entity across its photos)', 'evidence': f"publication {cfg['publicationId']} entity {entity['id']}",
                                             'pairwise_visible_centroid_distances_native': [float(np.linalg.norm(a - b)) for i, a in enumerate(cents) for b in cents[i + 1:]]}})
    path = out / 'evidence/objects.json'
    save(path, {'version': 1, 'coordinate_system': manifest['geometry'].get('coordinate_system'), 'metric_scale_known': False,
                'status': 'complete for the listed objects', 'objects': result})
    manifest['evidence'].update(objects='evidence/objects.json', objects_sha256=digest(path), object_count=len(result),
                                object_views=sum(len(o['views']) for o in result), script=str(Path(__file__).name), current_script_sha256=digest(__file__))
    # floor: the geometry run's plane and fit statistics, views (pinned) from the source run's floor masks
    floor = json.loads((geometry / 'evidence/floor.json').read_text())
    src_floor = json.loads((source / 'evidence/floor.json').read_text())
    fviews = [view_record(out, 'observed_floor', v['frame_id'], np.asarray(Image.open(source / v['mask_path'])) > 0, v['provenance']) for v in src_floor['views']]
    pts = [np.load(out / v['points_path']) for v in fviews]; cols = [np.load(out / v['colors_path']) for v in fviews]
    np.save(out / 'evidence/floor_points.npy', np.concatenate(pts)); np.save(out / 'evidence/floor_colors.npy', np.concatenate(cols))
    floor.update(views=fviews, points_path='evidence/floor_points.npy', colors_path='evidence/floor_colors.npy',
                 total_observed_points=int(sum(len(p) for p in pts)), evidence_type='observed partial geometry only; not a generated asset')
    cams = [np.load(out / f'geometry/frames/{f}/camera_to_world.npy') for f in frames]
    n, d = np.asarray(floor['plane_native'][:3]), floor['plane_native'][3]
    assert all(c[:3, 3] @ n + d > 0 for c in cams), 'cameras below the floor'
    save(out / 'evidence/floor.json', floor)
    manifest['evidence'].update(floor='evidence/floor.json', floor_sha256=digest(out / 'evidence/floor.json'))
    manifest['status'] = 'input_geometry_and_object_evidence_complete'
    save(out / 'manifest.json', manifest)
    coverage(out)


def original_camera(run, frame_id):
    """(K, c2w, photo RGB, mask loader) of one frame at the photo's own resolution."""
    manifest = json.loads((run / 'manifest.json').read_text())
    frame = next(f for f in manifest['frames'] if f['frame_id'] == frame_id)
    g = run / 'geometry/frames' / frame_id
    K = np.linalg.inv(np.asarray(frame['input_to_canonical_pixel_centres'], float)) @ np.load(g / 'intrinsics.npy').astype(float)
    return K, np.load(g / 'camera_to_world.npy').astype(float), np.asarray(Image.open(run / frame['input']).convert('RGB'))


def fit_plane(P, seed=7, iterations=300, relative=.02):
    """RANSAC plane (n, d with n.X + d = 0), then SVD on the inliers; None below 50 samples."""
    if len(P) < 50:
        return None
    rng = np.random.default_rng(seed)
    thr = relative * float(np.linalg.norm(np.percentile(P, 95, 0) - np.percentile(P, 5, 0)))
    best = None
    for _ in range(iterations):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n); inl = np.abs((P - a) @ n) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    c = P[best].mean(0); n = np.linalg.svd(P[best] - c, full_matrices=False)[2][-1]
    r = np.abs((P - c) @ n)
    return dict(normal=n.tolist(), offset=float(-n @ c), thresholdNative=thr, samples=len(P), inlierFraction=float((r < thr).mean()),
                residualP95Native=float(np.percentile(r[r < thr], 95)))


def ray_plane_mesh(run, view, n, d, max_vertices=60000, keep=None, mask=None):
    """The view's original-resolution mask pixels (on a grid) cast onto the plane n.X + d = 0: vertices, faces, colours.
    keep(points) -> bool per vertex, to leave out what another view already covers; mask: instead of the view's mask file."""
    K, c2w, rgb = original_camera(run, view['frame_id'])
    mask = np.asarray(Image.open(run / view['mask_path'])) > 0 if mask is None else mask
    step = max(1, int(np.ceil(np.sqrt(mask.sum() / max_vertices))))
    ys, xs = np.mgrid[step // 2:mask.shape[0]:step, step // 2:mask.shape[1]:step]
    pix = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 1).astype(float)  # 2-D matmul (batched 3-D matmul segfaults here)
    rays = (pix @ (c2w[:3, :3] @ np.linalg.inv(K)).T).reshape(*xs.shape, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    o, n = c2w[:3, 3], np.asarray(n, float)
    cos = rays @ n
    t = -(o @ n + d) / np.where(np.abs(cos) > 1e-9, cos, np.nan)
    X = o + rays * t[..., None]
    ok = mask[ys, xs] & (np.abs(cos) >= .1) & (t > 0)
    if keep is not None:
        ok[ok] = keep(X[ok])
    idx = -np.ones(ok.shape, int); idx[ok] = np.arange(ok.sum())
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([b, e, c], -1).reshape(-1, 3)])
    F = F[(F >= 0).all(1)]
    return X[ok], F, rgb[ys[ok], xs[ok]], step


def floor_region(run, fid, n, d, S, objects, floor, band_m=.03, window=41, min_frac=.6, dilate=3):
    """The floor one photo sees, on the canonical grid: its floor masks, plus content pixels outside every object mask (dilated)
    whose point-map sample lies within band_m of the floor, plus unsampled pixels whose window-px neighbourhood is >= min_frac
    such floor (a hole in the sparse point map among floor) and whose ray meets the floor no farther than the farthest floor
    sample of that photo (never beyond the floor the photo shows). A sample above the band (an obstacle) is never floor."""
    import cv2
    g = run / 'geometry/frames' / fid
    P = np.load(g / 'pts3d.npy'); known = np.load(g / 'content_valid_mask.npy').astype(bool)
    alpha = np.zeros(known.shape, bool); alpha[:, 63:455] = True
    obj = np.zeros(known.shape, np.uint8)
    for o in objects.values():
        for v in o['views']:
            if v['frame_id'] == fid:
                obj |= np.load(run / v['canonical_mask_path']).astype(np.uint8)
    obj = cv2.dilate(obj, np.ones((2 * dilate + 1,) * 2, np.uint8)) > 0
    hgt = (P @ n + d) * S
    level = known & (np.abs(hgt) < band_m); above = known & (hgt >= band_m)
    free = alpha & ~obj
    k = np.ones((window, window), np.float32)
    nk = cv2.filter2D((known & free).astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    nf = cv2.filter2D((level & free).astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    K = np.load(g / 'intrinsics.npy').astype(float); c2w = np.load(g / 'camera_to_world.npy').astype(float)
    ys, xs = np.indices(known.shape); pix = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 1).astype(float)
    ray = pix @ (c2w[:3, :3] @ np.linalg.inv(K)).T; cos = ray @ n
    t = np.where(cos < -1e-9, -(c2w[:3, 3] @ n + d) / np.where(cos < -1e-9, cos, -1), np.inf).reshape(known.shape)  # depth along the ray to the floor
    reach = np.percentile(np.linalg.norm(P[level & free] - c2w[:3, 3], axis=-1) / np.linalg.norm(ray.reshape(*known.shape, 3)[level & free], axis=-1), 99) if (level & free).any() else 0
    hole = ~known & free & (nk >= 20) & (nf >= min_frac * np.maximum(nk, 1)) & (t <= reach)
    fm = np.zeros(known.shape, bool)
    for v in floor['views']:
        if v['frame_id'] == fid:
            fm |= np.load(run / v['canonical_mask_path']).astype(bool)
    region = (fm & ~above) | (free & level) | hole
    return region, {'floorMaskPx': int(fm.sum()), 'levelPx': int((free & level).sum()), 'holePx': int(hole.sum()), 'regionPx': int(region.sum())}


def floor_fill(run, objects, n, d, S, e1, e2, covered, cell, parts, fill_m=.04, low_m=.5):
    """The floor plane where no photo shows floor (under and behind objects): 4 cm cells inside the convex hull of the observed
    floor and of the low objects' own point-map samples (height < low_m), not already covered, in the observed floor's median colour.
    A flat floor runs under the whole cell; the report must not show a missing floor where the camera cannot see it."""
    import cv2
    pts = [np.c_[p[0] @ e1, p[0] @ e2] for p in parts]
    for o in objects.values():
        for v in o['views']:
            Q = np.load(run / v['points_path'])
            if len(Q):
                Q = Q[(Q @ n + d) * S < low_m]; pts.append(np.c_[Q @ e1, Q @ e2])
    xy = np.concatenate(pts)
    if len(xy) < 3:
        return None
    f = fill_m / S; lo = xy.min(0); idx = np.floor((xy - lo) / f).astype(np.int32)
    hull = cv2.convexHull(idx.reshape(-1, 1, 2))
    grid = np.zeros((idx[:, 1].max() + 2, idx[:, 0].max() + 2), np.uint8); cv2.fillConvexPoly(grid, hull, 1)
    j, i = np.nonzero(grid)
    cells = [[(int(np.floor((lo[0] + (ii + a) * f) / cell)), int(np.floor((lo[1] + (jj + b) * f) / cell))) for a in (.25, .75) for b in (.25, .75)]
             for ii, jj in zip(i, j)]
    keep = np.array([not any(c in covered for c in cs) for cs in cells], bool)  # a cell the photos show no floor in at all
    i, j = i[keep], j[keep]
    if not len(i):
        return None
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    uv = (np.stack([i, j], 1)[:, None, :] + corners[None]) * f + lo  # (cells, 4, 2)
    origin = -d * n  # the plane point nearest the origin (|n| = 1)
    V = (origin + uv[..., 0:1] * e1 + uv[..., 1:2] * e2).reshape(-1, 3)
    base = np.arange(len(i))[:, None] * 4
    F = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]])
    colour = np.median(np.concatenate([p[2] for p in parts]), axis=0).astype(np.uint8)
    return V, F, np.tile(colour, (len(V), 1)), 'fill', fill_m


def surfaces(config, run, scale_path):
    cfg = json.loads(Path(config).read_text()); run = Path(run).resolve(); scale = json.loads(Path(scale_path).read_text())
    S = scale['nativeToMeters']
    objects = {o['object_id']: o for o in json.loads((run / 'evidence/objects.json').read_text())['objects']}
    floor = json.loads((run / 'evidence/floor.json').read_text())
    fn, fd = np.asarray(floor['plane_native'][:3], float), float(floor['plane_native'][3])
    out = run / 'surfaces'; out.mkdir(exist_ok=True); record = {}

    def write(oid, V, F, C, **info):
        np.savez_compressed(out / f'{oid}.npz', vertices=np.asarray(V, np.float64), faces=np.asarray(F, np.int64), colors=np.asarray(C, np.uint8))
        record[oid] = dict(info, vertices=int(len(V)), faces=int(len(F)))

    for ent in cfg['entities']:
        oid, method = ent['id'], ent['method']
        obj = objects.get(oid)
        if method == 'plane':
            src = objects[ent['parent']] if ent.get('planeFrom') == 'parent' else obj  # a see-through sheet: its frame's plane
            P = np.concatenate([np.load(run / v['points_path']) for v in src['views']])
            plane = fit_plane(P) if len(P) else None
            if plane is None:
                record[oid] = dict(method=method, model=None, reason=f'{len(P)} point-map samples inside the masks (< 50): no plane, no model')
                continue
            view = max(obj['views'], key=lambda v: v['mask_pixels'])
            V, F, C, step = ray_plane_mesh(run, view, plane['normal'], plane['offset'])
            write(oid, V, F, C, method=method, plane=plane, planeFrom=src['object_id'], rayView=view['frame_id'], gridStepPx=step,
                  meaning='mask rays of one photo on the plane fitted to the new point maps inside the masks; visible region only, no thickness')
        elif method == 'floor_plane':
            lift = ent.get('lift', 0.) / S
            view = max(obj['views'], key=lambda v: v['mask_pixels'])
            V, F, C, step = ray_plane_mesh(run, view, fn, fd - lift)
            write(oid, V, F, C, method=method, liftM=ent.get('lift', 0.), rayView=view['frame_id'], gridStepPx=step,
                  meaning='mask rays of one photo on the report floor (+ lift): a painted marking on the floor')
        elif method == 'estop':
            import trimesh
            spec = cfg['estop']['specification']
            r, h = spec['yellowBodyMaxDiameterM'] / 2 / S, spec['heightM'] / S
            m = trimesh.creation.cylinder(radius=r, height=h, sections=64)
            z = fn / np.linalg.norm(fn); x = np.cross(z, [1., 0, 0]); x /= np.linalg.norm(x); R = np.column_stack([x, np.cross(z, x), z])
            V = np.asarray(m.vertices) @ R.T + np.asarray(scale['P3'])
            write(oid, V, m.faces, np.tile([235, 190, 30], (len(V), 1)), method=method, centerNative=scale['P3'], axisNative=z.tolist(),
                  radiusNative=r, heightNative=h, rotation=R.tolist(), meaning='specification cylinder at the triangulated e-stop, axis = floor normal')
    # the floor's drawn reference surface: every photo's floor region, later photos only on 2 cm cells the earlier ones left empty
    e1 = np.cross(fn, [1., 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(fn, e1); cell = .02 / S
    covered = set(); parts = []; regions = {}
    manifest = json.loads((run / 'manifest.json').read_text())
    for frame in manifest['frames']:
        fid = frame['frame_id']; region, stats = floor_region(run, fid, fn, fd, S, objects, floor)
        if not region.any():
            continue
        w, h = frame['width'], frame['height']
        mask = np.asarray(Image.fromarray(region[:, 63:455]).resize((w, h), Image.Resampling.NEAREST)).astype(bool)
        key = lambda X: [tuple(k) for k in np.floor(np.c_[X @ e1, X @ e2] / cell).astype(int)]  # noqa: E731
        V, F, C, step = ray_plane_mesh(run, {'frame_id': fid}, fn, fd, max_vertices=150000, mask=mask,
                                       keep=lambda X: np.array([k not in covered for k in key(X)]))
        covered.update(key(V)); parts.append((V, F, C, fid, step)); regions[fid] = stats
    fill = floor_fill(run, objects, fn, fd, S, e1, e2, covered, cell, parts)
    if fill is not None:
        parts.append(fill)
    off = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
    write('floor_surface', np.concatenate([p[0] for p in parts]), np.concatenate([p[1] + o for p, o in zip(parts, off)]),
          np.concatenate([p[2] for p in parts]), method='floor', plane=floor['plane_native'], views=[[p[3], p[4], int(len(p[1]))] for p in parts],
          fillFaces=int(len(fill[1])) if fill is not None else 0, regions=regions, meaning='each photo\'s floor region (floor masks, plus pixels outside every object mask whose point-map sample is within '
                                   '3 cm of the floor, plus holes whose neighbourhood is such floor) cast onto the report floor (the max-inlier plane); '
                                   'later photos only where earlier ones left 2 cm cells empty; the rest of the cell floor (the convex hull of that floor '
                                   'and of the low objects\' own points) as the same plane in the observed floor\'s median colour')
    save(out / 'record.json', record)
    for k, v in record.items():
        print(k.ljust(20), {x: v[x] for x in v if x in ('method', 'vertices', 'faces', 'reason', 'gridStepPx')}, (v.get('plane') or {}).get('inlierFraction') if isinstance(v.get('plane'), dict) else '')


NOTES = Path('/Users/adam/Desktop/panoptes-public/research-notes')


def research_module(rel, env):
    """A research-notes module (compare.py / build_layer.py), configured by its environment variables before import."""
    import importlib
    import os
    os.environ.update(env)
    sys.path.insert(0, str(NOTES / Path(rel).parent))
    return importlib.import_module(Path(rel).stem)


def box_mesh(box, n2m, floor):
    """The 12-triangle box exactly as the viewer draws a layer box (measurement-layer.ts boxGeometry): bottomM..topM above
    the floor along u, +-half L / W along l / w about the centre's floor point."""
    import trimesh
    n, d = floor; l, w, u = (np.asarray(a, float) for a in box['axes'])
    c = np.asarray(box['centerNative'], float); base = c - n * (n @ c + d)
    half = np.asarray(box['sizeM'][:2]) / n2m / 2; z = np.array([box['bottomM'], box['topM']]) / n2m
    V = np.array([base + sl * half[0] * l + sw * half[1] * w + z[k] * u for k in (0, 1) for sw in (-1, 1) for sl in (-1, 1)])
    hull = trimesh.convex.convex_hull(V)
    return np.asarray(hull.vertices), np.asarray(hull.faces)


def partition(run, parts, V, F, n2m, radius_m=.1):
    """Face labels (index into parts, -1 = the parent's): centroid inside the part's (2 px dilated) canonical mask in every photo
    where it projects into the image and has a mask of that part, at least one; then unassigned faces within radius_m of an
    assigned one take its label. V: world vertices."""
    import cv2
    from scipy.spatial import cKDTree
    X = V[F].mean(1); label = -np.ones(len(F), int)
    for j, part in enumerate(parts):
        ok, seen = np.ones(len(X), bool), np.zeros(len(X), bool)
        for view in part['views']:
            g = run / 'geometry/frames' / view['frame_id']
            K, c2w = np.load(g / 'intrinsics.npy').astype(float), np.load(g / 'camera_to_world.npy').astype(float)
            m = cv2.dilate(np.load(run / view['canonical_mask_path']).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            cam = (X - c2w[:3, 3]) @ c2w[:3, :3]; z = cam[:, 2]
            uv = np.round(cam[:, :2] / np.where(z[:, None] > 1e-9, z[:, None], np.nan) @ K[:2, :2].T + K[:2, 2])
            inside = (z > 1e-9) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < m.shape[1]) & (uv[:, 1] < m.shape[0])
            hit = np.zeros(len(X), bool); ui = uv[inside].astype(int); hit[np.nonzero(inside)[0]] = m[ui[:, 1], ui[:, 0]]
            ok &= ~inside | hit; seen |= hit
        label[(label == -1) & ok & seen] = j
    done = label >= 0
    if done.any() and (~done).any():
        dist, idx = cKDTree(X[done]).query(X[~done])
        near = dist <= radius_m / n2m
        label[np.nonzero(~done)[0][near]] = label[done][idx[near]]
    return label


def sub_mesh(V, F, C, keep):
    used = np.unique(F[keep]); remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    return V[used], remap[F[keep]], C[used]


def compose(config, run, completion, choices_path, scale_path, label, boxes_path=None):
    import trimesh
    cfg = json.loads(Path(config).read_text()); run = Path(run).resolve(); completion = Path(completion).resolve()
    scale = json.loads(Path(scale_path).read_text()); n2m = scale['nativeToMeters']
    env = {'AB_RUN': str(run), 'AB_OUT': str(completion), 'CMP_NOTES': str(Path(choices_path).parent), 'CMP_REFERENCE': 'none',
           'CMP_BOXES': str(boxes_path) if boxes_path else 'none', 'CMP_SCALE': str(scale_path)}
    C = research_module('completion-ab-090-2026-10-06/compare.py', env)
    BL = research_module('completion-ab-090-2026-10-06/build_layer.py', env)
    sys.path.insert(0, str(SERVING / 'scripts/research'))
    import assemble_lucida_scene as als
    from scipy.spatial import cKDTree
    choices = json.loads(Path(choices_path).read_text())
    objects = {o['object_id']: o for o in json.loads((run / 'evidence/objects.json').read_text())['objects']}
    surf = json.loads((run / 'surfaces/record.json').read_text())
    floor = json.loads((run / 'evidence/floor.json').read_text()); plane = np.asarray(floor['plane_native'], float)
    boxes = json.loads(Path(boxes_path).read_text()) if boxes_path else None
    out = run / 'result'
    if out.exists():
        shutil.rmtree(out)
    (out / 'images').mkdir(parents=True)
    meshes, entries, unavailable, built = {}, [], [], {}
    for ent in cfg['entities']:  # parents before their parts (config order)
        oid, method = ent['id'], ent['method']
        info = {'method': method}
        if method == 'sam3d':
            ch = choices.get(oid)
            if ch is None:
                unavailable.append({'id': oid, 'reason': 'no SAM 3D candidate assembled (too few point-map samples in its masks)'}); continue
            variant = ch['variant']; c = C.comparison(variant, oid)
            T = np.asarray(c['final_object_to_world'], float)
            d = np.load(completion / variant / f'{oid}.npz')
            world = C.world_mesh(variant, oid, c)
            fixed, notes = C.floor_fix(world, oid)
            _, near = cKDTree(world.vertices).query(fixed.vertices)
            Vw, Fw, Cw = BL.decimate(np.asarray(fixed.vertices), np.asarray(fixed.faces), d['colors'][near])
            Vw = (Vw - T[:3, 3]) @ np.linalg.inv(T[:3, :3]).T  # back to the model's own frame: the assembled pose stays its transform
            info.update(variant=variant, decision=ch['decision'], floorFix=notes, generationPhoto=ch.get('generationPhoto'))
            eid = (boxes or {}).get('objectIds', {}).get(oid)
            measured = bool(eid and eid in boxes['boxes'] and not boxes['research'][eid].get('capped'))  # a capped box is the model's own
            if ch['decision'].startswith('用测量盒') and measured:
                V, F = box_mesh(boxes['boxes'][eid], n2m, (plane[:3] / np.linalg.norm(plane[:3]), plane[3] / np.linalg.norm(plane[:3])))
                meshes[oid] = (V, F, np.tile([158, 158, 153], (len(V), 1)), np.eye(4)); info['shownAs'] = 'measured box'
            elif ch['decision'].startswith('用测量盒') and boxes and any(g.startswith('轮廓不符') for g in ch['sam3dFixed']['gates']):
                unavailable.append({'id': oid, 'reason': '没有可信模型：SAM 3D 候选都与照片轮廓不符（' + '；'.join(ch['sam3dFixed']['gates'])
                                    + '），盒子拟合被限制（只是模型自己的盒子），不能代替'}); continue
            else:  # a capped box cannot stand in: the best candidate stays, its remaining gates listed
                meshes[oid] = (Vw, Fw, Cw, T); info['shownAs'] = 'SAM 3D'
                if ch['decision'].startswith('用测量盒'):
                    info['flagged'] = ch['sam3dFixed']['gates']
            comp = dict(c); comp['refinement'] = {k: v for k, v in c.get('refinement', {}).items() if k != 'trajectory'}
            als.write_json(out / f'{oid}-comparison.json', comp)
            info['comparison'] = f'{oid}-comparison.json'
        elif method == 'part':
            parent = ent['parent']; siblings = [e for e in cfg['entities'] if e.get('parent') == parent and e['method'] == 'part']
            if parent not in meshes or built.get(parent, {}).get('shownAs') != 'SAM 3D':
                unavailable.append({'id': oid, 'reason': f'its parent {parent} has no completed model to split'}); continue
            if 'labels' not in built[parent]:
                V, F, Cc, T = meshes[parent]
                Vworld = V @ T[:3, :3].T + T[:3, 3]
                built[parent]['labels'] = partition(run, [objects[e['id']] for e in siblings], Vworld, F, n2m)
            labels = built[parent]['labels']; j = [e['id'] for e in siblings].index(oid)
            V, F, Cc, T = meshes[parent]
            if not (labels == j).any():
                unavailable.append({'id': oid, 'reason': 'no face of the parent model falls in this part\'s masks'}); continue
            meshes[oid] = (*sub_mesh(V, F, Cc, labels == j), T); info.update(parent=parent, faces=int((labels == j).sum()))
        else:
            rec = surf.get(oid)
            if not rec or not rec.get('faces'):
                unavailable.append({'id': oid, 'reason': (rec or {}).get('reason', 'no surface')}); continue
            z = np.load(run / 'surfaces' / f'{oid}.npz'); meshes[oid] = (z['vertices'], z['faces'], z['colors'], np.eye(4)); info.update(rec)
        built[oid] = info
    for parent in {e['parent'] for e in cfg['entities'] if e['method'] == 'part'}:  # the parent keeps the faces no part took
        if 'labels' in built.get(parent, {}):
            V, F, Cc, T = meshes[parent]; keep = built[parent]['labels'] < 0
            meshes[parent] = (*sub_mesh(V, F, Cc, keep), T); built[parent]['remainderFaces'] = int(keep.sum())
            built[parent].pop('labels')
    # scene objects in config order, then the floor's reference surface and the observed floor (as assemble)
    chunks, offset, world = [], 0, []
    def add(oid, V, F, Cc, T, entry):
        nonlocal offset
        m = trimesh.Trimesh(V, F, process=False)
        chunk = np.column_stack([V, m.vertex_normals, np.asarray(Cc)[:, :3] / 255.]).astype('<f4')
        idx = np.asarray(F, '<u4').ravel()
        chunks.extend([chunk.tobytes(), idx.tobytes()])
        entry['mesh'] = {'byte_offset': offset, 'vertex_count': len(chunk), 'stride': 9, 'index_byte_offset': offset + chunk.nbytes,
                         'index_count': len(idx), 'index_type': 'uint32'}
        offset += chunk.nbytes + idx.nbytes; world.append(V @ T[:3, :3].T + T[:3, 3]); entries.append(entry)
    for ent in cfg['entities']:
        oid = ent['id']
        if oid not in meshes:
            continue
        V, F, Cc, T = meshes[oid]; info = built[oid]; obj = objects[oid]
        model = {'sam3d': 'facebook/sam-3d-objects' if info.get('shownAs') == 'SAM 3D' else 'measured box (box_faces)',
                 'part': 'facebook/sam-3d-objects (part of ' + ent.get('parent', '') + ')', 'plane': 'plane fitted to the new point maps',
                 'floor_plane': 'report floor plane', 'estop': 'e-stop specification cylinder'}[ent['method']]
        entry = {'id': oid, 'label': obj['label'], 'source': 'generated', 'model': model, 'transform': als.decompose(T),
                 'frame_ids': [v['frame_id'] for v in obj['views']], 'source_inventory_indices': [], 'rebuild': {k: v for k, v in info.items() if k not in ('plane',)}}
        if ent.get('context'):
            entry.update(role='context', visible=False)
        if info.get('comparison'):
            comp = json.loads((out / info['comparison']).read_text())
            entry['metrics'] = {'comparison': info['comparison'], 'views': comp['views'], 'watertight': comp.get('watertight'), 'faces': int(len(F))}
        add(oid, V, F, Cc, T, entry)
    z = np.load(run / 'surfaces/floor_surface.npz')
    fv = objects  # noqa: F841
    add('floor_surface', z['vertices'], z['faces'], z['colors'], np.eye(4),  # the publish step turns it into the floor's reference surface
        {'id': 'floor_surface', 'label': '工位地面（拟合平面，参考面）', 'source': 'observed', 'model': 'report floor plane (max-inlier fit of the new point maps)',
         'transform': als.decompose(np.eye(4)), 'frame_ids': [v['frame_id'] for v in floor['views']], 'source_inventory_indices': [],
         'rebuild': surf['floor_surface']})
    floor_view = als.load_view(run, floor['views'][-1], 288)
    fm = als.observed_mesh(floor_view)
    add('observed_floor', np.asarray(fm.vertices), np.asarray(fm.faces), np.asarray(fm.visual.vertex_colors)[:, :3], np.eye(4),
        {'id': 'observed_floor', 'label': '观测到的工位地面', 'source': 'observed', 'model': 'source geometry', 'transform': als.decompose(np.eye(4)),
         'frame_ids': [floor_view['frame_id']], 'metrics': {'plane_residual_p95_native': floor['all_point_residual_p95_native']}})
    (out / 'scene.bin').write_bytes(b''.join(chunks))
    manifest = json.loads((run / 'manifest.json').read_text()); cameras = []
    for frame in manifest['frames']:
        fid = frame['frame_id']; g = run / 'geometry/frames' / fid
        shutil.copy2(g / 'canonical.png', out / 'images' / f'{fid}.png')
        w, h = Image.open(out / 'images' / f'{fid}.png').size
        cameras.append({'id': fid, 'label': f'照片 {int(fid.split("_")[-1])}', 'width': w, 'height': h, 'K': np.load(g / 'intrinsics.npy').tolist(),
                        'camera_to_world': np.load(g / 'camera_to_world.npy').tolist(), 'image': f'images/{fid}.png'})
    P = np.concatenate(world)
    scene = {'version': 1, 'run_id': manifest['experiment'], 'units': '未标定尺度（非米）', 'up': floor['up_native'],
             'bounds': {'min': P.min(0).tolist(), 'max': P.max(0).tolist()}, 'binary': 'scene.bin', 'glb': None, 'metrics_report': None,
             'cameras': cameras, 'objects': entries,
             'limitations': ['Licence-clean rebuild: MVS geometry (DA3-BASE start, RoMa matches + numpy Levenberg-Marquardt (Schur) bundle adjustment, no GPL; two-view triangulation), SAM 3D Objects completion; no Pi3X, no RecGen',
                             'Generated hidden surfaces and textures have no photographic ground truth',
                             'All supplied views contribute to geometry and placement; scores measure input consistency, not held-out accuracy',
                             'Static editable meshes; no robot joints, collision validation or physical calibration'],
             'unavailable_objects': unavailable, 'object_evidence': None, 'floor_plane': floor['plane_native']}
    als.write_json(out / 'scene.json', scene)
    als.write_json(out / 'comparisons.json', {'run_id': manifest['experiment'], 'objects': [json.loads((out / e['metrics']['comparison']).read_text())
                                                                                         for e in entries if (e.get('metrics') or {}).get('comparison')]})
    save(run / 'compose-record.json', {'entities': built, 'unavailable': unavailable, 'choices': str(choices_path), 'boxes': str(boxes_path)})
    if (run / 'public').exists():
        shutil.rmtree(run / 'public')
    sys.path.insert(0, str(SERVING / 'scripts/research'))
    import build_capture_report
    build_capture_report.build(run, label)
    public = json.loads((run / 'public/scene.json').read_text())
    for o in public['objects']:  # keep what build_capture_report copies from result/ only partly: role, visibility, provenance
        e = next(x for x in entries if x['id'] == o['id'])
        for k in ('role', 'visible', 'provenance'):
            if k in e:
                o[k] = e[k]
    (run / 'public/scene.json').write_text(json.dumps(public, ensure_ascii=False, indent=2) + '\n')
    for e in entries:
        print(e['id'].ljust(22), e['model'][:40].ljust(40), e['mesh']['index_count'] // 3, 'faces', (e.get('rebuild') or {}).get('decision', ''))
    print('unavailable', unavailable)


def boxes_file(results_path, manifest_path, out):
    r = json.loads(Path(results_path).read_text()); bf = (r.get('result') or r['results']['box_faces'])
    ids = json.loads(Path(manifest_path).read_text())['entityIds']
    for eid, b in bf['boxes'].items():  # finalize.py: unverified dimensions show the model's size
        x = bf['research'][eid]; prior = list(x['priorSizeM']) + [x['priorBottomM']]
        for i, d in enumerate(('L', 'W', 'H', 'bottom')):
            if b['dims'][d]['confidence'] == 'unverified' and b['dims'][d].get('source') != 'model':
                b['dims'][d].update(valueM=prior[i], sigmaCm=None, source='model')
        b['sizeM'] = [b['dims'][d]['valueM'] for d in ('L', 'W', 'H')]
        b['bottomM'] = b['dims']['bottom']['valueM']; b['topM'] = round(b['bottomM'] + b['sizeM'][2], 4)
    doc = dict(schema='panoptes-mvp-boxes-v2', revisionId=r.get('layerRevision'), nativeToMeters=bf['nativeToMeters'], floor=bf['floor'],
               boxes=bf['boxes'], objectIds={k: v for k, v in ids.items() if v in bf['boxes']}, research=bf['research'])
    save(out, doc)
    print(len(doc['boxes']), 'boxes;', sum(b['highlight'] for b in doc['boxes'].values()), 'highlighted;', sum(bool(x.get('capped')) for x in bf['research'].values()), 'capped')


LEVEL_ZH = {'high': '高', 'medium': '中', 'low': '低', 'unverified': '未验证'}


def confidence(oid, info, ch, surf, built):
    """One level per object from the numbers of the method that made its model (generic; no object-type rule)."""
    if info is None:
        return 'unverified', [], ['没有模型']
    m = info['method']
    if m == 'sam3d' and info['shownAs'] == 'SAM 3D':
        f = ch['sam3dFixed']; reasons = ['SAM 3D 摆放轮廓 IoU ' + ' / '.join(f"{p['iou']:.2f}" for p in f['perPhoto'].values()), f"深度残差 p50 {100 * f['meanDepthP50']:.1f}%"]
        level = 'high' if f['meanIou'] >= .85 and f['meanDepthP50'] <= .02 else 'medium'
        missing = []
        if f.get('coverage') is not None:
            reasons.append(f"生成照片点图覆盖 {100 * f['coverage']:.0f}%")
            if f['coverage'] < .4:
                level = 'medium' if level == 'high' else level; missing.append(f"生成照片的点图只覆盖掩码 {100 * f['coverage']:.0f}%（模型形状靠补全）")
        if info.get('flagged'):
            level = 'low'; missing += ['仍有明显错误门：' + '；'.join(info['flagged']) + '（盒子拟合被限制，不能替代）']
        return level, reasons, missing
    if m == 'sam3d':  # measured box
        return 'low', ['显示为测量盒（照片拟合）'], ['SAM 3D 候选都有明显错误：' + '；'.join(ch['sam3dFixed']['gates'])]
    if m == 'part':
        parent = built.get(info['parent']) or {}
        lv = 'medium' if parent.get('shownAs') == 'SAM 3D' else 'low'
        return lv, [f"父物体 SAM 3D 模型在本部件掩码内的部分（{info['faces']} 面）"], []
    if m == 'plane':
        pl = info['plane']; lv = 'medium' if pl['inlierFraction'] >= .8 else 'low'
        src = '（取父物体的点：透明板看到的是后面的东西）' if info.get('planeFrom') != oid else ''
        return lv, [f"平面拟合：{pl['samples']} 个点，内点 {100 * pl['inlierFraction']:.0f}%{src}"], ([] if lv == 'medium' else ['平面内点不足 80%：位置只是近似'])
    if m == 'floor_plane':
        return 'medium', ['地面上的标线：照片掩码投到报告地面'], []
    if m == 'estop':
        return 'medium', ['急停规格圆柱（直径 8 cm、高 10 cm），位置为三台相机三角化'], []
    return 'unverified', [], []


def field_facts(eid_of, le, fair, n2m):
    """现场对照 per entity: lower_edge (click part masks, the MVP definition) and the fair evaluator's near-face edge (both on the new
    geometry; the field value only compared, never an input)."""
    out = {}
    names = {'housing': '罩壳下沿', 'front plate': '带接地螺栓的黄色前护板下沿', 'bottom rail': '底横梁下沿'}
    field = {'housing': 24, 'bottom rail': 20}
    for t in (le or {}).get('targets', []):
        oid = t['objectId']; part = t.get('part')
        if t.get('valueCm') is not None:
            txt = f"{names.get(part, part)} {t['valueCm']:.1f} cm ±{t['sigmaCm']:.1f}（{t.get('grade') or ''}，{t.get('path') or ''}）"
        else:
            est = f"；估计 {t['estimateCm']:.1f} ±{t['sigmaCm']:.1f} cm" if t.get('estimateCm') is not None and t.get('sigmaCm') is not None else ''
            txt = f"{names.get(part, part)}：无可靠值（{t.get('reason') or '—'}{est}）"
        if part in field:
            txt = f"现场 {field[part]} cm｜" + txt + (f"，差 {t['valueCm'] - field[part]:+.1f} cm" if t.get('valueCm') is not None else '')
        out.setdefault(oid, []).append(txt)
    for oid, (label, cm, fcm) in (fair or {}).items():
        out.setdefault(oid, []).append(f"近面下沿（公平 A/B 评判器，新点图）{label} {cm:.1f} cm" + (f"，现场 {fcm} cm，差 {cm - fcm:+.1f} cm" if fcm else '（A/B 不计分，只列出）'))
    return {eid_of[o]: [{'label': '现场对照', 'kind': 'check', 'text': '｜'.join(v)}] for o, v in out.items() if o in eid_of}


def build_layer(config, run, view_path, choices_path, scale_path, out, boxes_path=None, le_path=None, fair_path=None, oe_path=None,
                patches='', pages=None, notes=None, completion=None):
    cfg = json.loads(Path(config).read_text()); run = Path(run).resolve()
    view = json.loads(Path(view_path).read_text()); doc = view['publication']['snapshot']['revision']['document']
    scale = json.loads(Path(scale_path).read_text()); n2m = scale['nativeToMeters']
    choices = json.loads(Path(choices_path).read_text())
    rec = json.loads((run / 'compose-record.json').read_text()); built = rec['entities']
    surf = json.loads((run / 'surfaces/record.json').read_text())
    ids = {}
    for e in doc['entities']:
        src = next((l.get('sourceRecordId') for l in e.get('lineage') or [] if isinstance(l, dict) and l.get('operation') == 'offline_import'), None)
        if src:
            ids[src] = e['id']
    frame = doc['coordinateFrames'][0]; g = frame['ground']
    layer = {'schemaVersion': 1, 'publicationId': view['publication']['id'], 'revisionId': view['publication']['sceneRevisionId'],
             'coordinateFrameId': frame['id'],
             'scale': {'nativeToMeters': n2m, 'status': 'operator_anchored', 'source': '急停红钮 4 cm + 黄色本体 8 cm 联合 · 三台相机 · 许可干净 MVS 几何（DA3-BASE 起始 + RoMa + numpy LM BA，无 GPL）',
                       'uncertaintyRelative': round(scale['uncertaintyRelative'], 4), 'redOnly': round(scale['redOnly'], 5), 'yellowOnly': round(scale['yellowOnly'], 5)},
             'ground': {'normal': g['normal'], 'offset': g['offset'], 'plane': g['plane'], 'source': 'max-inlier floor of the MVS point maps (geometry-backbone-ab-2026-10-06), the publication frame floor'},
             'facts': {}, 'confidence': {}, 'assets': [], 'models': {}, 'pipelines': {}}
    boxes = json.loads(Path(boxes_path).read_text()) if boxes_path else None
    if boxes:
        assert abs(boxes['nativeToMeters'] / n2m - 1) < 1e-9, 'boxes use another scale'
        # boxes of another import of the same run (another scene hash, other entity ids): keyed again by run object
        layer['boxes'] = {ids[o]: boxes['boxes'][e] for o, e in boxes['objectIds'].items() if o in ids and e in boxes['boxes']}
    unavailable = {u['id']: u['reason'] for u in rec['unavailable']}
    for ent in cfg['entities'] + [{'id': 'floor_surface', 'method': 'floor'}]:
        oid = ent['id']; eid = ids.get(oid)
        if not eid:
            continue
        info = built.get(oid); ch = choices.get(oid)
        if oid == 'floor_surface':
            continue
        level, reasons, missing = confidence(oid, info, ch, surf, built)
        if oid in unavailable:
            level, reasons, missing = 'unverified', [], [unavailable[oid]]
        if not ent.get('context'):
            layer['confidence'][eid] = {'level': level, 'label': LEVEL_ZH[level], 'reasons': reasons, 'missing': missing}
        if info is None:
            note = '不显示模型：' + unavailable.get(oid, '—')
        elif info['method'] == 'sam3d' and info['shownAs'] == 'SAM 3D':
            f = ch['sam3dFixed']
            note = (f"SAM 3D（照片 {ch['generationPhoto']} 生成，SAM License；输入 = 照片 + 掩码 + 新 MVS 点图，点图覆盖 {100 * (f.get('coverage') or 0):.0f}%）｜摆放轮廓 IoU "
                    + ' / '.join(f"{p['iou']:.2f}" for p in f['perPhoto'].values()) + f"｜深度残差 p50 {100 * f['meanDepthP50']:.1f}%｜最低点 {f['lowestCm']:.1f} cm｜尺寸 "
                    + '×'.join(f'{100 * x:.0f}' for x in f['extentM']) + ' cm' + (f"｜{'，'.join(f['fix'])}" if f['fix'] else '')
                    + (f"｜仍有：{'；'.join(info['flagged'])}" if info.get('flagged') else ''))
        elif info['method'] == 'sam3d':
            note = '测量盒代替模型：SAM 3D 候选都有明显错误（' + '；'.join(ch['sam3dFixed']['gates']) + '）'
        elif info['method'] == 'part':
            note = f"「{doc_label(doc, ids.get(info['parent']))}」的 SAM 3D 模型落在本部件照片掩码里的部分（{info['faces']} 面；掩码外 10 cm 内的面随最近的面）"
        elif info['method'] == 'plane':
            pl = info['plane']
            note = (f"平面面片（由新点图重算）：照片 {info['rayView'][-1]} 的掩码像素射线落到 RANSAC 平面上（{pl['samples']} 点，内点 {100 * pl['inlierFraction']:.0f}%，"
                    f"残差 p95 {100 * pl['residualP95Native'] * n2m:.1f} cm" + ('；透明板取父物体围栏的点' if info.get('planeFrom') != oid else '') + '）；只有看得到的一面，没有厚度')
        elif info['method'] == 'floor_plane':
            note = f"地面标线：照片 {info['rayView'][-1]} 的掩码像素射线落到报告地面（抬高 {100 * info['liftM']:.1f} cm 以免与地面重叠）"
        else:
            note = '急停规格圆柱（直径 8 cm、高 10 cm），位置为三台相机三角化的急停（新几何），轴沿地面法向；尺度由它定'
        layer['facts'][eid] = [{'label': '模型（许可干净版）', 'kind': 'note', 'text': note}]
    # the floor's own entity: its reference surface
    floor_eid = next((e['id'] for e in doc['entities'] if e.get('geometryRole') == 'floor' and any(r.get('sourceKind') == 'observed_reference_surface' for r in e['representations'])), None)
    if floor_eid:
        fs = surf['floor_surface']
        layer['facts'][floor_eid] = [{'label': '地面（参考面）', 'kind': 'note', 'text': f"新 MVS 点图的最大内点地面（残差 p95 {json.loads((run / 'evidence/floor.json').read_text())['all_point_residual_p95_native'] * n2m * 100:.2f} cm）；"
                                      f"照片 {'、'.join(v[0][-1] for v in fs['views'])} 的地面掩码像素射线落到该平面（后面的照片只补前面没盖到的 2 cm 格）"}]
    estop = ids.get(next(e['id'] for e in cfg['entities'] if e['method'] == 'estop'))
    if estop:
        feats = scale['features']
        by = lambda key: ' / '.join(f"{f['measuredM'] * 100:.2f}" for f in feats if key in f['feature'])  # noqa: E731
        layer['facts'][estop] += [
            {'label': '规格（参照物）', 'kind': 'specification', 'text': '红钮直径 4 cm · 黄色本体最大直径 8 cm · 高 10 cm'},
            {'label': '照片实测 · 红钮直径', 'kind': 'measured', 'text': f"{by('red')} cm（照片 1 / 2 / 3，规格 4）"},
            {'label': '照片实测 · 黄色本体最大直径', 'kind': 'measured', 'text': f"{by('yellow')} cm（照片 1 / 2 / 3，规格 8）"},
            {'label': '照片里 黄 ÷ 红', 'kind': 'measured', 'text': ' / '.join(f"{v['yellowOverRed']:.2f}" for v in scale['perPhoto'].values()) + '（规格 2.00）'},
            {'label': '尺度', 'kind': 'anchor', 'text': f"两项联合：1 原生单位 = {n2m * 100:.1f} cm（只用红钮 {scale['redOnly'] * 100:.1f}，只用黄色本体 {scale['yellowOnly'] * 100:.1f}），"
                                                       f"六个特征最大偏差 {scale['maxDeviation'] * 100:.2f}%（门限 4%）"}]
    for eid, facts in field_facts(ids, json.loads(Path(le_path).read_text()) if le_path else None,
                                  json.loads(Path(fair_path).read_text()) if fair_path else None, n2m).items():
        layer['facts'].setdefault(eid, []).extend(facts)
    for path in [x for x in patches.split(',') if x]:
        patch = json.loads(Path(path).read_text())
        layer['models'].update(patch['models'])
        for eid, m in patch['models'].items():  # a generic fix (workcell_fix_obvious.py): the display and its confidence follow it
            label = '显示平滑（高 / 中置信度、不需复核的物体）' if m['note'].startswith('显示模型平滑') else '显示修正（明显错误）'
            layer['facts'].setdefault(eid, []).insert(0, {'label': label, 'kind': 'note', 'text': m['note']})
            if m['note'].startswith('模型不可信') and eid in layer['confidence']:
                layer['confidence'][eid] = {'level': 'low', 'label': LEVEL_ZH['low'], 'reasons': ['显示为测量盒（照片拟合）'], 'missing': [m['note']]}
        ids_new = {a['id'] for a in patch['assets']}
        layer['assets'] = [a for a in layer['assets'] if a['id'] not in ids_new] + patch['assets']
        if pages:
            for a in patch['assets']:
                shutil.copyfile(Path(path).parent / Path(a['url']).name, Path(pages) / Path(a['url']).name)
    if boxes:  # box highlight: the layer's own confidence (box_faces R4) where it is low / unverified
        for eid, b in layer['boxes'].items():
            c = layer['confidence'].get(eid)
            extra = [f'图层：{x}' for x in (c or {}).get('missing', [])]
            b['highlightReasons'] = [x for x in b.get('highlightReasons') or [] if not x.startswith('图层：')] + extra
            b['highlight'] = bool(b['highlightReasons']) or (c or {}).get('level') in ('low', 'unverified')
    oe = json.loads(Path(oe_path).read_text()) if oe_path else None
    layer['pipelines'] = pipelines(cfg, run, doc, ids, built, choices, surf, layer, oe, unavailable, pages, notes, completion)
    save(out, layer)
    print(json.dumps({'entities': len(layer['confidence']), 'levels': {k: sum(1 for c in layer['confidence'].values() if c['level'] == k) for k in LEVEL_ZH},
                      'boxes': len(layer.get('boxes') or {}), 'highlighted': sum(1 for b in (layer.get('boxes') or {}).values() if b['highlight']),
                      'models': len(layer['models']), 'pipelines': len(layer['pipelines'])}, ensure_ascii=False))


def doc_label(doc, eid):
    return next((e['label'] for e in doc['entities'] if e['id'] == eid), eid)


def gen_photo(completion, variant, oid):
    return variant[-1] if variant.startswith('sam3d-') else json.loads((Path(completion) / 'sam3d/record.json').read_text())['objects'][oid]['frame_id'][-1]


def pipelines(cfg, run, doc, ids, built, choices, surf, layer, oe, unavailable, pages, notes, completion):
    """Every stage of every object, photos to measurements; the compare.py sheet for SAM 3D objects."""
    objects = {o['object_id']: o for o in json.loads((run / 'evidence/objects.json').read_text())['objects']}
    manifest = json.loads((run / 'manifest.json').read_text())
    n2m = layer['scale']['nativeToMeters']
    ba = '（' + cfg['geometry']['export'] + '）' if cfg.get('geometry') else ''
    geometry = (f"许可干净 MVS 几何{ba}（DA3-BASE 起始 → RoMa 匹配 + numpy LM（Schur）BA，无 GPL；两视图三角化，RoMa 置信度 0.05；MIT / Apache-2.0）：三张照片的相机与点图（有洞），"
                f"点图在本物体掩码里 " + '{}' + f" 个点；尺度由急停定（1 原生单位 = {n2m * 100:.1f} cm）；全部物体共用一个最大内点地面")
    oe = (oe.get('result') or oe.get('results', {}).get('obvious_errors') or oe) if oe else None
    oe_obj = (oe or {}).get('objects') or {}
    out = {}
    for ent in cfg['entities']:
        oid = ent['id']; eid = ids.get(oid)
        if not eid:
            continue
        obj = objects[oid]; info = built.get(oid); ch = choices.get(oid)
        st = [{'label': '照片与掩码', 'text': 'SAM 3 分割（SAM License；原报告同一套照片掩码）：' + '，'.join(f"照片 {v['frame_id'][-1]} {v['mask_pixels'] / 1e3:.0f}k 像素" for v in obj['views'])},
              {'label': '相机与深度', 'text': geometry.format(' / '.join(f"照片 {v['frame_id'][-1]} {v['partial_point_count']}" for v in obj['views']))}]
        url = None
        if ent['method'] == 'sam3d' and ch:
            st.append({'label': '补全候选', 'text': 'SAM 3D Objects（SAM License）每张有掩码的照片各生成一次，输入 = 照片 + 掩码 + 新 MVS 点图（无点处为空）：' + '；'.join(
                f"照片 {gen_photo(completion, v, oid)}："
                f"IoU {t['meanIou']:.2f}，深度 {100 * t['meanDepthP50']:.1f}%，点图覆盖 {100 * (t.get('coverage') or 0):.0f}%，" + (f"{len(t['gates'])} 个明显错误（{'；'.join(t['gates'])}）" if t['gates'] else '无明显错误')
                for v, t in ch['tried'].items())})
            f = ch['sam3dFixed']
            st.append({'label': '组装', 'text': f"同一组装（assemble_lucida_scene，9 自由度，全部照片的轮廓 + 深度）：照片 {ch['generationPhoto']} 的候选轮廓 IoU "
                       + ' / '.join(f"{p['iou']:.2f}" for p in f['perPhoto'].values()) + f"，深度残差 p50 {100 * f['meanDepthP50']:.1f}%"})
            st.append({'label': '选择与明显错误门', 'text': '下沉 / 悬空 / 底部与高度（对测量盒）/ 尺寸 G7 / 占了看得见的空地 / 深度 p50 > 6% / 轮廓 IoU < 0.5 / 比照片里的物体点低或高 15 cm（下部 / 上部缺失）；'
                       f"按「明显错误最少，再 IoU − 2×深度 + 0.5×点图覆盖」选：{('；'.join(f['gates']) or '全部通过')}" + ('；地面修正：' + '，'.join(f['fix']) if f['fix'] else '')})
            st.append({'label': '采用', 'text': (ch['decision'] if info and info.get('shownAs') != 'SAM 3D' or not ch['decision'].startswith('用测量盒') else
                                                 f"{ch['decision']}；但盒子拟合被限制（只是模型自己的盒子），不能代替：保留 SAM 3D，错误如实列出")
                       if info else '不显示模型：' + unavailable.get(oid, '')})
            sheet = Path(notes or '') / f'{oid}.jpg'
            if pages and sheet.exists():
                shutil.copyfile(sheet, Path(pages) / f'pipeline-{eid[:8]}.jpg'); url = f'measurement-layer/pipeline-{eid[:8]}.jpg'
        else:
            text = next((x['text'] for x in layer['facts'].get(eid, []) if x['label'] == '模型（许可干净版）'), '')
            st.append({'label': '模型', 'text': text})
        o = oe_obj.get(eid)
        if o is not None:  # the report's own obvious-error run (scripts/workcell_checks/obvious_errors.py) on what it shows
            fails = o.get('failed', [])
            st.append({'label': '明显错误检查 G1–G8（显示的模型）', 'text': ('明显：' + '；'.join(f"{f['gate']} {f.get('zh') or f.get('text')}" for f in fails if f.get('severity') == 'obvious') + '。'
                                                                  if any(f.get('severity') == 'obvious' for f in fails) else '无明显错误。')
                       + (('警告：' + '；'.join(f"{f['gate']} {f.get('zh') or f.get('text')}" for f in fails if f.get('severity') == 'warning')) if any(f.get('severity') == 'warning' for f in fails) else '')})
        fix = (layer.get('models') or {}).get(eid)
        if fix:
            st.append({'label': '显示平滑', 'text': fix['note'] + '（workcell_smooth_models.py）'} if fix['note'].startswith('显示模型平滑') else
                      {'label': '显示修正', 'text': fix['note'] + '（workcell_fix_obvious.py，通用规则）'})
        box = (layer.get('boxes') or {}).get(eid)
        if box:
            d = box['dims']
            st.append({'label': '测量（统一地面）', 'text': '长×宽×高 ' + '×'.join(f"{100 * d[k]['valueM']:.1f}" for k in 'LWH')
                       + f" cm，离地 {100 * d['bottom']['valueM']:.1f} cm；置信度 " + ' / '.join(f"{n} {LEVEL_ZH[d[k]['confidence']]}" for k, n in zip('LWH', '长宽高'))
                       + f" / 离地 {LEVEL_ZH[d['bottom']['confidence']]}" + ('；需复核：' + '；'.join(box['highlightReasons']) if box.get('highlightReasons') else '')})
        check = next((x['text'] for x in layer['facts'].get(eid, []) if x.get('kind') == 'check'), None)
        if check:
            st.append({'label': '现场对照', 'text': check})
        out[eid] = {'url': url, 'caption': '白 = 掩码，橙 = SAM 3D（新 MVS 几何）；第二行 = 每张照片生成的候选（点开看大图）' if url else None, 'stages': st}
    return out


def selftest():
    """Synthetic checks of the geometry helpers: plane fit, the viewer's box, the part split."""
    import tempfile
    rng = np.random.default_rng(0)
    P = np.c_[rng.uniform(-1, 1, (500, 2)), np.zeros(500)] + [0, 0, 2.]; P[:50, 2] += rng.uniform(.5, 1, 50)  # 10 % outliers
    pl = fit_plane(P); assert abs(abs(pl['normal'][2]) - 1) < 1e-6 and abs(abs(pl['offset']) - 2) < 1e-6 and pl['inlierFraction'] >= .89, pl
    assert fit_plane(P[:40]) is None
    n = np.array([0, 0, 1.]); box = {'axes': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'centerNative': [1., 2, .7], 'sizeM': [.2, .4, .5], 'bottomM': .1, 'topM': .6}
    V, F = box_mesh(box, 1., (n, 0.)); assert len(F) == 12 and np.allclose(V.min(0), [.9, 1.8, .1]) and np.allclose(V.max(0), [1.1, 2.2, .6])
    with tempfile.TemporaryDirectory() as tmp:  # a camera at the origin looking down +z; the part mask is the left half
        run = Path(tmp); g = run / 'geometry/frames/frame_0001'; g.mkdir(parents=True)
        K = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]]); np.save(g / 'intrinsics.npy', K); np.save(g / 'camera_to_world.npy', np.eye(4))
        m = np.zeros((100, 100), bool); m[:, :50] = True; np.save(run / 'm.npy', m)
        X = np.array([[-.5, 0, 2], [.5, 0, 2], [-.52, .1, 2]]); V = np.repeat(X, 3, 0) + np.tile([[0, 0, 0], [.001, 0, 0], [0, .001, 0]], (3, 1)); F = np.arange(9).reshape(3, 3)
        lab = partition(run, [{'views': [{'frame_id': 'frame_0001', 'canonical_mask_path': 'm.npy'}]}], V, F, 1.)
        assert lab.tolist() == [0, -1, 0], lab
    print('workcell_rebuild_report self-test passed')


def coverage(run):
    run = Path(run)
    for o in json.loads((run / 'evidence/objects.json').read_text())['objects']:
        print(o['object_id'].ljust(20), ' '.join(f"{v['frame_id'][-1]}:{v['mask_pixels'] // 1000}k/{v['partial_point_count']}" for v in o['views']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['evidence', 'surfaces', 'compose', 'boxes', 'layer', 'coverage', 'selftest'])
    p.add_argument('--config'); p.add_argument('--geometry'); p.add_argument('--view'); p.add_argument('--out'); p.add_argument('--run'); p.add_argument('--scale'); p.add_argument('--completion'); p.add_argument('--choices'); p.add_argument('--label'); p.add_argument('--boxes'); p.add_argument('--run-results'); p.add_argument('--manifest'); p.add_argument('--lower-edge'); p.add_argument('--field'); p.add_argument('--obvious'); p.add_argument('--patches', default=''); p.add_argument('--pages'); p.add_argument('--notes')
    a = p.parse_args()
    if a.stage == 'evidence':
        evidence(a.config, a.geometry, a.view, a.out)
    elif a.stage == 'surfaces':
        surfaces(a.config, a.run, a.scale)
    elif a.stage == 'layer':
        build_layer(a.config, a.run, a.view, a.choices, a.scale, a.out, a.boxes, a.lower_edge, a.field, a.obvious, a.patches, a.pages, a.notes, a.completion)
    elif a.stage == 'boxes':
        boxes_file(a.run_results, a.manifest, a.out)
    elif a.stage == 'compose':
        compose(a.config, a.run, a.completion, a.choices, a.scale, a.label, a.boxes)
    elif a.stage == 'selftest':
        selftest()
    else:
        coverage(a.run)
