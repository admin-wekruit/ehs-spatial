"""Photo appearance for existing planar meshes; never change their geometry.

Frames require canonical K, camera-to-world pose, raw-to-canonical pixel-center
transform A, and the EXIF-oriented original RGB array textureRgb. Masks are in
canonical pixels. Visibility remains conditional on those masks and supplied
occluders; this is not a geometry or camera calibration stage.
"""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
import trimesh

from workcell_photo_objects import _project


def source_texture_frames(root, frames, sources):
    """Attach exact original rasters/transforms to existing camera records."""
    from workcell_photo_oneshot import _frame
    if not frames or max(frames) > len(sources):
        raise ValueError('Every textured photo needs its original source image')
    result = {}
    for photo, frame in frames.items():
        raw = _frame(root, photo)
        with Image.open(sources[photo - 1]) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
        if list(rgb.shape[:2]) != [raw['original_image']['height'], raw['original_image']['width']]:
            raise ValueError('Original texture dimensions disagree with recorded pixel transform')
        result[photo] = {**frame, 'textureRgb': rgb,
            'A': np.asarray(raw['input_mask_transform']['input_to_canonical_pixel_centres'], float)}
    return result


def _sample_image(image, pixels):
    # OpenCV limits each remap dimension to <32767; a 512px atlas exceeds that
    # when its supported texels are flattened into one column.
    chunks = [cv2.remap(image, p[:, 0].astype(np.float32).reshape(-1, 1),
                       p[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT).reshape((-1,) + image.shape[2:])
              for start in range(0, len(pixels), 16000) for p in [pixels[start:start + 16000]]]
    return np.concatenate(chunks) if chunks else np.empty((0,) + image.shape[2:], image.dtype)


def _occluded(points, center, polygons):
    blocked = np.zeros(len(points), bool)
    rays = points - center
    for polygon in polygons:
        polygon = np.asarray(polygon, float)
        for j in range(1, len(polygon) - 1):
            a, b, c = polygon[[0, j, j + 1]]
            ab, ac = b - a, c - a
            normal = np.cross(ab, ac)
            denominator = rays @ normal
            t = np.divide((a - center) @ normal, denominator,
                          out=np.full(len(points), np.inf), where=abs(denominator) > 1e-12)
            ids = np.flatnonzero((t > 1e-7) & (t < 1 - 1e-7) & ~blocked)
            q = center + t[ids, None] * rays[ids] - a
            determinant = (ab @ ab) * (ac @ ac) - (ab @ ac) ** 2
            if determinant <= 1e-20:
                continue
            u = ((ac @ ac) * (q @ ab) - (ab @ ac) * (q @ ac)) / determinant
            v = ((ab @ ab) * (q @ ac) - (ab @ ac) * (q @ ab)) / determinant
            blocked[ids[(u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)]] = True
    return blocked


def texture_planar_mesh(mesh, frames, masks, *, occluder_polygons=(), max_size=512):
    """Return a mesh copy with a baked RGB texture and source-coverage diagnostics.

    ponytail: one affine UV chart per planar sheet; curved/folded meshes require
    their existing separate surface charts, not flattening or remeshing here.
    The original vertex colors are baked into unsupported texels. One photo,
    selected in view-quality order with valid support, fills visible texels.
    """
    if type(max_size) is not int or not 16 <= max_size <= 2048:
        raise ValueError('Texture size must be an integer from 16 to 2048')
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if not len(faces) or not np.isfinite(vertices).all():
        raise ValueError('Texture needs finite nonempty mesh geometry')
    origin = vertices.mean(0)
    _, _, basis = np.linalg.svd(vertices - origin, full_matrices=False)
    extent = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    if extent <= 1e-10 or np.max(abs((vertices - origin) @ basis[2])) > extent * 1e-6:
        raise ValueError('Texture chart requires a planar mesh')
    local = (vertices - origin) @ basis[:2].T
    span = np.ptp(local, axis=0)
    if span.min() < extent * 1e-7:
        raise ValueError('Texture chart requires a nondegenerate planar mesh')
    if mesh.visual.kind == 'vertex':
        old_colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(float)
    elif (mesh.visual.kind == 'texture' and mesh.visual.material.baseColorTexture is None
          and 'color' in mesh.visual.vertex_attributes):
        old_colors = np.asarray(mesh.visual.vertex_attributes['color'])[:, :3].astype(float)
    else:
        raise ValueError('Planar photo bake requires the existing vertex colors without an existing image texture')
    material = getattr(mesh.visual, 'material', None)
    if material is not None and getattr(material, 'baseColorFactor', None) is not None:
        old_colors *= np.asarray(material.baseColorFactor)[:3] / 255.
    scale = (max_size - 5) / span.max()
    xy = (local - local.min(0)) * scale + 2
    width, height = (np.ceil(span * scale).astype(int) + 5).tolist()
    rgb = np.full((height, width, 3), 115, np.uint8)
    inside = np.zeros((height, width), bool)
    for face in faces:
        triangle = xy[face]
        lo = np.maximum(np.floor(triangle.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(triangle.max(0)).astype(int), [width - 1, height - 1])
        yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        q = np.stack([xx, yy], axis=-1) - triangle[0]
        ab, ac = triangle[1] - triangle[0], triangle[2] - triangle[0]
        denominator = ab[0] * ac[1] - ab[1] * ac[0]
        if abs(denominator) < 1e-12:
            continue
        u = (q[..., 0] * ac[1] - q[..., 1] * ac[0]) / denominator
        v = (ab[0] * q[..., 1] - ab[1] * q[..., 0]) / denominator
        good = (u >= -1e-7) & (v >= -1e-7) & (u + v <= 1 + 1e-7)
        colors = ((1 - u - v)[..., None] * old_colors[face[0]] +
                  u[..., None] * old_colors[face[1]] + v[..., None] * old_colors[face[2]])
        linear = np.clip(colors[good] / 255., 0, 1)
        srgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055)
        rgb[yy[good], xx[good]] = np.rint(srgb * 255).astype(np.uint8)
        inside[yy[good], xx[good]] = True
    yy, xx = np.nonzero(inside)
    points = origin + (((np.c_[xx, yy] - 2) / scale + local.min(0)) @ basis[:2])
    selected = np.full(len(points), -1, int)
    views = []
    for photo in sorted(masks):
        frame = frames[photo]
        image = np.asarray(frame['textureRgb'])
        A, K, pose = (np.asarray(frame[k], float) for k in ('A', 'K', 'pose'))
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Texture source must be the original RGB uint8 image')
        if A.shape != (3, 3) or K.shape != (3, 3) or pose.shape != (4, 4) or not all(np.isfinite(v).all() for v in (A, K, pose)):
            raise ValueError('Invalid source camera or pixel transform')
        center = pose[:3, 3]
        side = float(basis[2] @ (center - origin))
        facing = abs(side) / max(np.linalg.norm(center - origin), 1e-12)
        if facing < .15:
            continue
        projected, depth = _project(vertices, frame)
        raw = np.c_[projected, np.ones(len(projected))] @ np.linalg.inv(A).T
        raw = raw[:, :2] / raw[:, 2:3]
        if np.any(depth <= 0) or not np.isfinite(raw).all():
            continue
        lo = np.maximum(np.floor(raw.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(raw.max(0)).astype(int) + 1, image.shape[1::-1])
        if np.any(hi <= lo):
            continue
        crop = image[lo[1]:hi[1], lo[0]:hi[0]]
        sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())
        area = sum(abs(np.linalg.det(np.stack([raw[f[1]] - raw[f[0]], raw[f[2]] - raw[f[0]]]))) / 2 for f in faces)
        views.append({'photo': photo, 'facing': facing, 'side': side, 'projectedAreaRawPx': area, 'sharpness': sharpness})
    typical_sharpness = np.median([v['sharpness'] for v in views]) if views else 1.
    for view in views:
        view['score'] = view['projectedAreaRawPx'] * view['facing'] * float(np.clip((view['sharpness'] + 1) / (typical_sharpness + 1), .5, 1.5))
        view['selectedTexels'] = 0
    views.sort(key=lambda v: (-v['score'], v['photo']))
    for view in views:
        photo = view['photo']; frame = frames[photo]; image = frame['textureRgb']
        canonical, depth = _project(points, frame)
        raw = np.c_[canonical, np.ones(len(points))] @ np.linalg.inv(frame['A']).T
        raw = raw[:, :2] / raw[:, 2:3]
        mask = cv2.erode(np.asarray(masks[photo], np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
        support = _sample_image(mask.astype(np.float32), canonical) > .999
        good = support & (depth > 0) & np.isfinite(raw).all(1)
        good &= (raw[:, 0] >= 1) & (raw[:, 0] < image.shape[1] - 2) & (raw[:, 1] >= 1) & (raw[:, 1] < image.shape[0] - 2)
        ids = np.flatnonzero(good)
        ids = ids[~_occluded(points[ids], frame['pose'][:3, 3], occluder_polygons)]
        if len(ids):
            colors = _sample_image(image, raw[ids])
            rgb[yy[ids], xx[ids]] = colors
            selected[ids] = photo
            view['selectedTexels'] = int(len(ids))
            # ponytail: one coherent source per sheet avoids stripe tears from
            # camera mismatch. Multi-view registration is required before blending
            # another photo; unsupported texels retain the original appearance.
            break
    result = mesh.copy()
    result.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.c_[(xy[:, 0] + .5) / width, 1 - (xy[:, 1] + .5) / height],
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(rgb),
            baseColorFactor=[255, 255, 255, 255], metallicFactor=0., roughnessFactor=.8, doubleSided=True))
    unchanged = np.array_equal(result.vertices, vertices) and np.array_equal(result.faces, faces)
    if not unchanged:
        raise AssertionError('Photo texture changed mesh geometry')
    return result, {'geometryUnchanged': unchanged, 'textureSize': [width, height],
        'photoTexelFraction': float(np.mean(selected >= 0)) if len(selected) else 0.,
        'sourcePhotos': [v['photo'] for v in views if v['selectedTexels']], 'views': views,
        'sourceSelection': 'one highest-ranked photo with nonzero visible support per planar sheet; no cross-view gap filling',
        'unsupportedTexels': 'original linear vertex colors interpolated and encoded to sRGB on unchanged triangles',
        'visibility': 'eroded instance masks and supplied occluder polygons; conditional on segmentation and camera accuracy',
        'reverseSide': 'same double-sided display material, not independently observed reverse-face appearance'}


def texture_existing_guards(root, sources, out, max_size=512):
    """Replay appearance on existing left/right GLBs; never refit A4 geometry."""
    from workcell_guard_joint import _inputs
    root, out = Path(root), Path(out)
    if root.resolve() == out.resolve() or (out.exists() and any(out.iterdir())):
        raise ValueError('Texture replay requires a new or empty separate output directory')
    started = time.monotonic()
    frames, boards, association = _inputs(root)
    frames = source_texture_frames(root, frames, sources)
    scenes, polygons = {}, []
    for side in ('left', 'right'):
        scene = trimesh.load(root / f'guard-{side}.glb', force='scene', process=False)
        nodes = list(scene.graph.nodes_geometry)
        if len({scene.graph[n][1] for n in nodes}) != len(nodes):
            raise ValueError('Per-node photo textures require distinct existing mesh primitives')
        scenes[side] = scene
        for node in nodes:
            matrix, name = scene.graph[node]
            world = trimesh.transform_points(scene.geometry[name].vertices, matrix)
            _, _, basis = np.linalg.svd(world - world.mean(0), full_matrices=False)
            local = (world - world.mean(0)) @ basis[:2].T
            hull = cv2.convexHull(local.astype(np.float32), returnPoints=False).ravel()
            polygons.append(world[hull])
    out.mkdir(parents=True, exist_ok=True)
    objects = []
    for side, source in scenes.items():
        target = source.copy(); panels = []
        for index, node in enumerate(source.graph.nodes_geometry):
            matrix, name = source.graph[node]
            original = source.geometry[name]
            world = original.copy()
            if original.visual.kind == 'texture' and 'color' in original.visual.vertex_attributes:
                world.visual.vertex_attributes['color'] = original.visual.vertex_attributes['color'].copy()
            world.apply_transform(matrix)
            textured, diagnostic = texture_planar_mesh(world, frames, boards[side]['masks'],
                                                     occluder_polygons=polygons, max_size=max_size)
            # UVs were computed in world coordinates; the actual mesh and graph
            # stay in their original local coordinates and retain their indices.
            target.geometry[name].visual = textured.visual
            png = f'guard-{side}-panel-{index + 1}-texture.png'
            textured.visual.material.baseColorTexture.save(out / png)
            panels.append({'node': node, 'texture': png, **diagnostic})
        filename = f'guard-{side}.glb'
        target.export(out / filename)
        reopened = trimesh.load(out / filename, force='scene', process=False)
        if set(source.graph.nodes) != set(reopened.graph.nodes):
            raise AssertionError('Texture replay changed scene nodes')
        for node in source.graph.nodes_geometry:
            old_matrix, old_name = source.graph[node]; new_matrix, new_name = reopened.graph[node]
            old, new = source.geometry[old_name], reopened.geometry[new_name]
            if not (np.array_equal(old_matrix, new_matrix) and np.array_equal(old.vertices, new.vertices)
                    and np.array_equal(old.faces, new.faces)):
                raise AssertionError(f'Texture replay changed geometry or placement: {node}')
        objects.append({'id': f'v-guard-{side}', 'mesh': filename, 'geometryUnchanged': True,
                        'sourceMeshSha256': hashlib.sha256((root / filename).read_bytes()).hexdigest(),
                        'texturedMeshSha256': hashlib.sha256((out / filename).read_bytes()).hexdigest(),
                        'panels': panels})
    result = {'schemaVersion': 1, 'route': 'existing-guard-photo-texture', 'geometryUnchanged': True,
              'newModelCalls': 0, 'geometryOptimizationCalls': 0, 'objects': objects,
              'association': association, 'scope': 'left/right planar sheets only; center model not modified',
              'sourcePhotos': [{'photo': p, 'name': Path(path).name,
                  'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()} for p, path in enumerate(sources, 1)],
              'limitations': ['Appearance experiment only; unchanged geometry retains its existing errors.',
                  'Visibility is conditional on instance masks, supplied planar occluders and existing cameras.',
                  'Unsupported texels preserve the original vertex-color appearance; reverse sides are not reconstructed.'],
              'wallSeconds': time.monotonic() - started}
    (out / 'texture-results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
