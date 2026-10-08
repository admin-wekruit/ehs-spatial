"""Masks by projection: where a photo shows an object's model but the report has no mask for that object in that photo.

For each object and each photo without its mask, the model's visible silhouette is the set of pixels whose ray hits this model
first among all models (so other models and the model itself occlude), inside the frame. Its area, the fraction of the
model's whole projected silhouette that is visible (the rest is occluded or out of frame), a padded box and the silhouette
polygons (even-odd, original pixel coordinates) are returned. Candidates (area >= MIN_AREA px, fraction >= MIN_FRACTION) go to
a box-prompted segmenter on the photo (modal_apps/workcell_mask_transfer.py); the placement was fitted on the masked photos
only, so the segmenter's agreement with the silhouette is a held-out check of it."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import argus.checks.shape_core as wsc

MIN_AREA, MIN_FRACTION, PAD, STRIDE = 1500, .3, .08, 2


def scene_of(meshes):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    for V, F in meshes:  # geometry ids follow this order (every model has faces)
        scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    return scene


def cast(cam, scene, x0, y0, nx, ny, stride, count):
    """First model hit (id, -1 none) on an ny x nx grid of stride-px blocks from (x0, y0), rays through block centres."""
    import open3d as o3d
    ys, xs = np.mgrid[0:ny, 0:nx]
    pix = np.vstack([(x0 + stride * xs + stride / 2).ravel(), (y0 + stride * ys + stride / 2).ravel(), np.ones(xs.size)])
    d = (cam['R'].T @ np.linalg.inv(cam['K']) @ pix).T
    rays = np.hstack([np.broadcast_to(cam['C'], d.shape), d]).astype(np.float32)
    ids = scene.cast_rays(o3d.core.Tensor(rays))['geometry_ids'].numpy().astype(np.int64)
    ids[(ids < 0) | (ids >= count)] = -1
    return ids.reshape(ny, nx)


def full_area(cam, V, F, stride):
    """Pixel area of the model's whole projected silhouette, unoccluded and not clipped by the frame (3x frame at most)."""
    uv, z = wsc.project(cam, V); uv = uv[z > 0]
    if not len(uv):
        return 0
    lo = np.maximum(uv.min(0), [-cam['w'], -cam['h']]); hi = np.minimum(uv.max(0), [2 * cam['w'], 2 * cam['h']])
    if (hi <= lo).any():
        return 0
    s = max(stride, math.ceil(math.sqrt(np.prod(hi - lo) / 2e6)))  # ponytail: <= ~2M rays per model and photo
    n = np.ceil((hi - lo) / s).astype(int) + 1
    return int((cast(cam, scene_of([(V, F)]), lo[0], lo[1], n[0], n[1], s, 1) >= 0).sum()) * s * s


def polygons_of(mask, stride=1, eps=1.0):
    """Even-odd polygons (outer and hole contours) of a boolean mask, in original pixel-centre coordinates."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append(np.round(c * stride + (stride - 1) / 2, 1).tolist())
    return out


def assess(cams, objects, stride=STRIDE, min_area=MIN_AREA, min_fraction=MIN_FRACTION, pad=PAD):
    scene = scene_of([o['mesh'] for o in objects])
    out, candidates = {}, []
    for k, cam in enumerate(cams):
        todo = [i for i, o in enumerate(objects) if k not in o['masks']]
        label = cast(cam, scene, 0, 0, -(-cam['w'] // stride), -(-cam['h'] // stride), stride, len(objects))
        for i, o in enumerate(objects):  # in-sample reference: the same silhouette against the report's own mask
            if k in o['masks']:
                m = o['masks'][k][stride // 2::stride, stride // 2::stride][:label.shape[0], :label.shape[1]]
                m = np.pad(m, [(0, label.shape[0] - m.shape[0]), (0, label.shape[1] - m.shape[1])]); v = label == i
                out.setdefault(o['id'], dict(label=o['label'], maskPhotos=sorted(j + 1 for j in o['masks']), photos={}))
                out[o['id']].setdefault('reportMaskIoU', {})[k] = float((v & m).sum() / max(1, (v | m).sum()))
        for i in todo:
            o = objects[i]; vis = label == i; area = int(vis.sum()) * stride * stride
            row = dict(areaPx=area, fullAreaPx=0, visibleFraction=0.0, candidate=False)
            if area:
                row['fullAreaPx'] = full = full_area(cam, *o['mesh'], stride)
                row['visibleFraction'] = frac = min(1.0, area / full) if full else 0.0
                ys, xs = np.nonzero(vis)
                x0, x1, y0, y1 = xs.min() * stride, (xs.max() + 1) * stride, ys.min() * stride, (ys.max() + 1) * stride
                px, py = pad * (x1 - x0), pad * (y1 - y0)
                row['bbox'] = [max(0, round(x0 - px)), max(0, round(y0 - py)), min(cam['w'], round(x1 + px)), min(cam['h'], round(y1 + py))]
                row['candidate'] = area >= min_area and frac >= min_fraction
                if row['candidate']:
                    candidates.append(dict(entityId=o['id'], label=o['label'], photoIndex0=k, imageId=cam['imageId'], areaPx=area,
                                           visibleFraction=frac, bbox=row['bbox'], polygons=polygons_of(vis, stride)))
            out.setdefault(o['id'], dict(label=o['label'], maskPhotos=sorted(j + 1 for j in o['masks']), photos={}))['photos'][k] = row
    return out, candidates


def run(ctx, opts):
    objects, candidates = assess(ctx['cams'], ctx['objects'], opts.get('stride', STRIDE), opts.get('minAreaPx', MIN_AREA),
                                 opts.get('minVisibleFraction', MIN_FRACTION), opts.get('pad', PAD))
    return dict(status='ok', stride=opts.get('stride', STRIDE), minAreaPx=opts.get('minAreaPx', MIN_AREA),
                minVisibleFraction=opts.get('minVisibleFraction', MIN_FRACTION), pad=opts.get('pad', PAD),
                objects=objects, candidates=candidates)


def _check():
    """Plate A (2 x 2 at z 5) behind bar B (x 0.5..1.5 at z 3): from a camera whose frame starts at A's centre, A is visible
    on u/f in [0, .5/2.95] of its [-1/4.95, 1/4.95]: fraction .42; the polygons fill back to the visible mask."""
    box = lambda c, d: (wsc.primitive_mesh(dict(kind='box', dimensions=d))[0] + c, wsc.primitive_mesh(dict(kind='box', dimensions=d))[1])
    A, B = box(np.array([0, 0, 5.]), [2, 2, .1]), box(np.array([1, 0, 3.]), [1, 4, .1])
    cam = wsc.camera(dict(cameraToWorld=np.eye(4).tolist(), K=[[500, 0, 0], [0, 500, 240], [0, 0, 1]], width=640, height=480, imageId='c'))
    objs = [dict(id='A', label='A', mesh=A, masks={}), dict(id='B', label='B', mesh=B, masks={})]
    rows, cands = assess([cam], objs)
    a = rows['A']['photos'][0]
    want = (.5 / 2.95) / (2 / 4.95)
    assert abs(a['visibleFraction'] - want) < .03 and abs(a['fullAreaPx'] - (1000 / 4.95) ** 2) / (1000 / 4.95) ** 2 < .03, a
    assert [c['entityId'] for c in cands] == ['A', 'B'][:len(cands)] and cands[0]['entityId'] == 'A'
    vis = cast(cam, scene_of([A, B]), 0, 0, 320, 240, STRIDE, 2) == 0
    vis = cv2.resize(vis.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)
    back = wsc.polygon_mask(cands[0]['polygons'], (480, 640))
    assert (vis & back).sum() / (vis | back).sum() > .95
    objs[0]['masks'] = {0: back}
    again = assess([cam], objs)[0]['A']  # a photo with the report's mask is not a transfer target, only the in-sample reference
    assert not again['photos'] and again['reportMaskIoU'][0] > .95, again


if __name__ == '__main__':
    _check()
    print('transfer self-test passed')
