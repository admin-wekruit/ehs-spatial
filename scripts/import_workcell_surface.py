"""Import the observed BOR1 GLB and evidence-backed inventory face IDs; no inference.

  .venv/bin/python scripts/import_workcell_surface.py --source /path/to/workcell-reconstruction-01 --run runs/user-bor1-02
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import struct
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ehs_spatial.interactive_report import case_objects, mask_index_png


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def glb_arrays(data):
    """Read the fixed, single-primitive trimesh export without adding a 3D dependency."""
    if len(data) < 20 or struct.unpack_from('<III', data) != (0x46546C67, 2, len(data)):
        raise ValueError('Expected a complete GLB 2.0 file')
    chunks, offset = {}, 12
    while offset < len(data):
        length, kind = struct.unpack_from('<II', data, offset)
        offset += 8
        if kind in chunks or offset + length > len(data):
            raise ValueError('Invalid GLB chunk layout')
        chunks[kind] = data[offset:offset+length]
        offset += length
    document = json.loads(chunks[0x4E4F534A])
    binary = chunks[0x004E4942]
    if len(document['meshes']) != 1 or len(document['meshes'][0]['primitives']) != 1:
        raise ValueError('The face map requires one mesh and one primitive')
    if (len(document['nodes']) != 1 or document['nodes'][0].get('mesh') != 0
            or any(key in document['nodes'][0] for key in ('matrix', 'translation', 'rotation', 'scale'))):
        raise ValueError('Expected native-world vertices without a node transform')
    primitive = document['meshes'][0]['primitives'][0]
    if primitive.get('mode', 4) != 4:
        raise ValueError('Expected indexed triangles')

    def array(index, kind, component, dtype, columns):
        accessor = document['accessors'][index]
        view = document['bufferViews'][accessor['bufferView']]
        if (accessor['type'] != kind or accessor['componentType'] != component
                or 'sparse' in accessor or view.get('buffer', 0) != 0 or 'byteStride' in view):
            raise ValueError('Unexpected GLB accessor layout')
        start = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        count = accessor['count'] * columns
        if start + count * np.dtype(dtype).itemsize > view.get('byteOffset', 0) + view['byteLength']:
            raise ValueError('Accessor exceeds its buffer view')
        return np.frombuffer(binary, dtype=dtype, count=count, offset=start).reshape(-1, columns)

    vertices = array(primitive['attributes']['POSITION'], 'VEC3', 5126, '<f4', 3)
    uv = array(primitive['attributes']['TEXCOORD_0'], 'VEC2', 5126, '<f4', 2)
    indices = array(primitive['indices'], 'SCALAR', 5125, '<u4', 1).reshape(-1)
    if (len(indices) % 3 or len(uv) != len(vertices) or not np.isfinite(vertices).all()
            or not np.isfinite(uv).all() or not len(indices) or int(indices.max()) >= len(vertices)):
        raise ValueError('Invalid triangle/UV arrays')
    # ponytail: the frozen export uses face-local vertices; reject a different export contract.
    if not np.array_equal(indices, np.arange(len(vertices))):
        raise ValueError('Expected face-local sequential vertices')
    return vertices.astype(float), indices.reshape(-1, 3), uv.astype(float)


def source_pixels(triangle_uv, *, atlas_grid=(2, 2), content_rect=(63, 0, 455, 518)):
    """glTF UVs have a top-left origin; atlas tiles are in original frame order."""
    grid = np.asarray(atlas_grid)
    tile = np.floor(triangle_uv * grid).astype(int)
    if (not ((triangle_uv > 0) & (triangle_uv < 1)).all()
            or not (tile == tile[:, :1]).all()):
        raise ValueError('Each face must lie inside one source-photo atlas tile')
    frame = tile[:, 0, 1] * grid[0] + tile[:, 0, 0] + 1
    rect = np.asarray(content_rect)
    pixels = (triangle_uv * grid - tile) * (rect[2:] - rect[:2]) + rect[:2] - 0.5
    return frame, pixels


def assign_faces(samples, masks, trusted, shape=(518, 518)):
    """Only a unique object containing all three corners and the centroid owns a face."""
    xy = np.floor(samples + 0.5).astype(int)
    height, width = shape
    within = ((xy[..., 0] >= 0) & (xy[..., 0] < width)
              & (xy[..., 1] >= 0) & (xy[..., 1] < height)).all(1)
    safe = np.clip(xy, [0, 0], [width-1, height-1])
    candidates = np.zeros(len(samples), np.uint16)
    winner = np.zeros(len(samples), dtype='<u4')
    for inv, mask in masks.items():
        if mask.shape != (height, width):
            raise ValueError('Object mask disagrees with the surface canonical grid')
        matches = trusted & within & mask[safe[..., 1], safe[..., 0]].all(1)
        candidates[matches] += 1
        winner[matches] = inv + 1
    winner[candidates != 1] = 0
    return winner, candidates


def _mask_records(run, objects, frame_id, mapped):
    _, _, _, _, issues, mask_uris = mask_index_png(run, objects, frame_id)
    masks = {inv:np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1])))).astype(bool)
             for inv,uri in mask_uris.items()}
    by_inv = {obj['inv']:obj for obj in objects}
    records = []
    for inv, mask in masks.items():
        obj = by_inv[inv]
        source = (run/'refinements'/f"{obj['refine_slug']}.json" if obj.get('refine_slug') else
                  run/'inventory/sam'/f"{frame_id}__{re.sub(r'[^a-z0-9]+','_',obj['label']).strip('_')}.json")
        records.append({'inv':inv, 'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),
                        'source_mask_sha256':digest(source), 'mapped_faces':int((mapped==inv+1).sum())})
    return masks, records, issues


def _save_correspondence(output, manifest, frame_numbers, samples, trusted, *, atlas_grid=(2,2), content_rect=(63,0,455,518)):
    path = output/'source-correspondence.npz'
    np.savez_compressed(path, frame_numbers=frame_numbers.astype('<u4'), samples=samples,
                        trusted=trusted.astype(bool))
    manifest['correspondence'] = {'path':path.name, 'sha256':digest(path),
        'asset_sha256':manifest['asset_sha256'], 'sampling':'three vertices and world centroid; canonical pixel centres',
        'cameras_sha256':hashlib.sha256(json.dumps(manifest['cameras'],sort_keys=True).encode()).hexdigest(),
        'trust':'saved acceptance of the native valid/confidence/content/4-percent-camera-depth test',
        'atlas_grid':list(atlas_grid), 'content_rect':list(content_rect)}
    manifest['version'] = 2


def refresh_surface_associations(run):
    """Re-label the unchanged GLB from current source masks, entirely within a run."""
    run = Path(run).resolve()
    output = run/'surface'
    manifest = json.loads((output/'surface.json').read_text())
    if digest(output/'surface.glb') != manifest['asset_sha256'] or digest(output/'face-inv.bin') != manifest['face_map_sha256']:
        raise ValueError('Surface assets changed; source association cannot be certified')
    vertices, faces, uv = glb_arrays((output/'surface.glb').read_bytes())
    if len(faces) != manifest['face_count']:
        raise ValueError('Surface face count changed')
    contract = manifest.get('correspondence')
    frame_numbers, uv_pixels = source_pixels(uv[faces], **({'atlas_grid':contract['atlas_grid'], 'content_rect':contract['content_rect']} if contract else {}))
    if ({int(r['frame_id'][-4:]) for r in manifest['frames']} != set(int(x) for x in np.unique(frame_numbers))
            or {r['frame_id'] for r in manifest['frames']} != {c['id'] for c in manifest['cameras']['frames']}):
        raise ValueError('Surface source frames/cameras do not cover its atlas faces')
    input_images = sorted((run/'input').glob('image_*'))
    for record in manifest['frames']:
        frame_id = record['frame_id']
        index = int(frame_id[-4:])
        if (digest(input_images[index-1]) != record['original_image_sha256']
                or digest(run/'geometry/frames'/frame_id/'canonical.png') != record['canonical_image_sha256']):
            raise ValueError(f'{frame_id}: original/canonical photograph differs from the GLB source')
        if int((frame_numbers == index).sum()) != record['faces']:
            raise ValueError(f'{frame_id}: atlas face assignment changed')
    if manifest.get('correspondence'):
        contract = manifest['correspondence']
        path = output/contract['path']
        if (not path.resolve().is_relative_to(output) or digest(path) != contract['sha256'] or contract['asset_sha256'] != manifest['asset_sha256']
                or contract['cameras_sha256'] != hashlib.sha256(json.dumps(manifest['cameras'],sort_keys=True).encode()).hexdigest()):
            raise ValueError('Stored surface correspondence changed')
        with np.load(path, allow_pickle=False) as data:
            samples, trusted = data['samples'], data['trusted']
            if not np.array_equal(data['frame_numbers'], frame_numbers):
                raise ValueError('Stored correspondence has different atlas frame IDs')
    else:
        # This exact v1 exporter certified every face of the frozen BOR1 GLB.
        # If even one face failed its depth check, aggregate counts cannot
        # recover which face failed: require the original saved correspondence.
        expected = 'glTF TEXCOORD_0 top-left UV; col=floor(2u), row=floor(2v), frame=2*row+col+1; canonical=(2*uv-tile)*[392,518]+[63,0]-0.5'
        if manifest.get('version') != 1 or manifest.get('uv_contract') != expected or any(r['unsupported_depth_faces'] != 0 for r in manifest['frames']):
            raise ValueError('This surface lacks per-face source support needed for safe reassociation')
        samples = np.zeros((len(faces),4,2),float)
        trusted = np.ones(len(faces),bool)
        triangles = vertices[faces]
        for frame in manifest['cameras']['frames']:
            chosen = np.flatnonzero(frame_numbers == int(frame['id'][-4:]))
            points = np.concatenate([triangles[chosen],triangles[chosen].mean(1)[:,None]],axis=1)
            pose, K = np.asarray(frame['camera_to_world']), np.asarray(frame['K']).copy()
            K[0,2] += 63  # v1 camera images are the hash-identified canonical crop.
            camera = (points-pose[:3,3]) @ pose[:3,:3]
            projected = camera @ K.T
            pixels = projected[...,:2]/projected[...,2:]
            if not np.isfinite(pixels).all() or not (camera[...,2]>0).all() or np.linalg.norm(pixels[:,:3]-uv_pixels[chosen],axis=-1).max() > .002:
                raise ValueError(f"{frame['id']}: stored cameras do not reproject the GLB source UVs")
            samples[chosen] = pixels
        _save_correspondence(output, manifest, frame_numbers, samples, trusted)
        manifest['correspondence']['migration'] = 'v1 hash-identified GLB; every face previously passed native depth support; embedded camera/UV reprojection reverified'
    if samples.shape != (len(faces),4,2) or trusted.shape != (len(faces),) or trusted.dtype != bool or not np.isfinite(samples).all():
        raise ValueError('Invalid saved surface correspondence arrays')
    _, objects = case_objects(run)
    labels = np.zeros(len(faces),dtype='<u4')
    for record in manifest['frames']:
        frame_id = record['frame_id']
        chosen = np.flatnonzero(frame_numbers == int(frame_id[-4:]))
        with Image.open(run/'geometry/frames'/frame_id/'canonical.png') as image:
            shape = (image.height,image.width)
        masks, _, issues = _mask_records(run, objects, frame_id, labels[chosen])
        mapped, candidates = assign_faces(samples[chosen], masks, trusted[chosen], shape)
        labels[chosen] = mapped
        _, mask_records, _ = _mask_records(run, objects, frame_id, mapped)
        record.update(matched_faces=int((mapped>0).sum()), ambiguous_faces=int((candidates>1).sum()),
                      unsupported_depth_faces=int((~trusted[chosen]).sum()),
                      no_matching_mask_faces=int((trusted[chosen] & (candidates==0)).sum()),
                      masks=mask_records, mask_issues=issues, canonical_shape_hw=list(shape))
    labels.tofile(output/'face-inv.bin')
    supported = [int(value)-1 for value in np.unique(labels) if value]
    manifest.update(inventory_sha256=digest(run/'inventory/inventory.json'), face_map_sha256=digest(output/'face-inv.bin'),
                    supported_inv=supported, unsupported_inv=[o['inv'] for o in objects if o['inv'] not in supported],
                    mapped_faces=int((labels>0).sum()), unknown_faces=int((labels==0).sum()))
    (output/'surface.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def import_surface(source, run, *, atlas_grid=(2,2)):
    source, run = source.resolve(), run.resolve()
    delivery = json.loads((source/'delivery.json').read_text())
    asset = (source/delivery['displayed_asset']).resolve()
    if not asset.is_relative_to(source):
        raise ValueError('Asset path leaves the source directory')
    texture = json.loads((asset.parent/'texture-metrics.json').read_text())
    if digest(asset) != delivery['displayed_sha256'] or digest(asset) != texture['output']['sha256']:
        raise ValueError('Source GLB hash differs from its verified delivery')
    inventory_path = run/'inventory/inventory.json'
    inventory_hash = digest(inventory_path)
    _, objects = case_objects(run)
    vertices, faces, uv = glb_arrays(asset.read_bytes())
    triangles = vertices[faces]
    content_rect = texture['selection']['content_rect']
    frame_number, canonical_uv = source_pixels(uv[faces], atlas_grid=atlas_grid, content_rect=content_rect)
    frame_ids = texture['include_frames']
    if sorted(set(int(x) for x in frame_number)) != sorted(int(name[-4:]) for name in frame_ids):
        raise ValueError('UV source frames disagree with texture provenance')
    geometry = Path(texture['source_geometry'])
    labels = np.zeros(len(faces), dtype='<u4')
    source_samples = np.zeros((len(faces),4,2),float)
    source_trusted = np.zeros(len(faces),bool)
    frame_records = []
    for frame_id in frame_ids:
        frame_index = int(frame_id[-4:])
        metadata = next(item for item in texture['frames'] if item['frame_id'] == frame_id)
        folder = geometry/'frames'/frame_id
        paths = {'pts3d':folder/'pts3d.npy', 'valid_mask':folder/'valid_mask.npy',
                 'camera_to_world':folder/'camera_to_world.npy', 'intrinsics':folder/'intrinsics.npy',
                 'canonical.png':folder/'canonical.png'}
        if {name:digest(path) for name,path in paths.items()} != metadata['geometry_sha256']:
            raise ValueError(f'{frame_id}: source geometry hash changed')
        if digest(folder/'conf.npy') != metadata['confidence_sha256']:
            raise ValueError(f'{frame_id}: source confidence changed')
        if (digest(run/'geometry/frames'/frame_id/'canonical.png') != metadata['geometry_sha256']['canonical.png']
                or digest(run/'input'/f'image_{frame_index:02d}.jpg') != metadata['source_image_sha256']):
            raise ValueError(f'{frame_id}: product and reconstruction photographs differ')
        chosen = np.flatnonzero(frame_number == frame_index)
        points = np.concatenate([triangles[chosen], triangles[chosen].mean(1)[:, None]], axis=1)
        c2w = np.load(paths['camera_to_world'], allow_pickle=False).astype(float)
        K = np.load(paths['intrinsics'], allow_pickle=False).astype(float)
        camera = (points-c2w[:3, 3]) @ c2w[:3, :3]
        projected = camera @ K.T
        samples = projected[..., :2] / projected[..., 2:]
        residual = np.linalg.norm(samples[:, :3]-canonical_uv[chosen], axis=-1)
        if not np.isfinite(samples).all() or residual.max() > .002:
            raise ValueError(f'{frame_id}: GLB UVs do not match the native camera projection')
        native = np.load(paths['pts3d'], allow_pickle=False)
        depth = ((native-c2w[:3, 3]) @ c2w[:3, :3])[..., 2]
        valid = np.load(paths['valid_mask'], allow_pickle=False).astype(bool)
        confidence = np.load(folder/'conf.npy', allow_pickle=False)
        xy = np.floor(samples+0.5).astype(int)
        height, width = valid.shape
        inside = ((samples[..., 0] >= content_rect[0]) & (samples[..., 0] <= content_rect[2]-1)
                  & (samples[..., 1] >= content_rect[1]) & (samples[..., 1] <= content_rect[3]-1)
                  & (camera[..., 2] > 0)).all(1)
        safe = np.clip(xy, [0,0], [width-1,height-1])
        observed = depth[safe[..., 1], safe[..., 0]]
        trusted = inside & (valid[safe[..., 1], safe[..., 0]] & np.isfinite(observed)
                           & (confidence[safe[..., 1], safe[..., 0]] >= .1)
                           & (np.abs(camera[..., 2]-observed) <= .04*observed)).all(1)
        source_samples[chosen] = samples
        source_trusted[chosen] = trusted
        _, _, _, _, issues, mask_uris = mask_index_png(run, objects, frame_id)
        masks = {inv:np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(',',1)[1])))).astype(bool)
                 for inv,uri in mask_uris.items()}
        mapped, candidates = assign_faces(samples, masks, trusted, valid.shape)
        labels[chosen] = mapped
        frame_records.append({'frame_id':frame_id, 'faces':len(chosen), 'matched_faces':int((mapped>0).sum()),
            'ambiguous_faces':int((candidates>1).sum()), 'unsupported_depth_faces':int((~trusted).sum()),
            'no_matching_mask_faces':int((trusted & (candidates==0)).sum()),
            'uv_reprojection_max_px':float(residual.max()), 'uv_reprojection_p95_px':float(np.quantile(residual,.95)),
            'geometry_sha256':metadata['geometry_sha256'], 'confidence_sha256':metadata['confidence_sha256'],
            'canonical_image_sha256':metadata['geometry_sha256']['canonical.png'],
            'original_image_sha256':metadata['source_image_sha256'], 'mask_issues':issues,
            'canonical_shape_hw':list(valid.shape),
            'masks':_mask_records(run,objects,frame_id,mapped)[1]})
    if digest(inventory_path) != inventory_hash:
        raise ValueError('Inventory changed during import')
    supported = [int(value)-1 for value in np.unique(labels) if value]
    cameras = json.loads((source/'cameras.json').read_text())
    output = run/'surface'
    output.mkdir(exist_ok=True)
    if any((output/name).exists() for name in ('surface.glb','face-inv.bin','surface.json')):
        raise ValueError('Surface output already exists; use a new run directory for replay')
    shutil.copyfile(asset, output/'surface.glb')
    labels.tofile(output/'face-inv.bin')
    # Keep the camera document and its native cropped source images intact.
    for camera in cameras['frames']:
        image = Path(camera['image'])
        if image.is_absolute() or '..' in image.parts: raise ValueError('Invalid camera image path')
        (output/image).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/image, output/image)
    manifest = {'version':1, 'asset':'surface.glb', 'face_map':'face-inv.bin', 'face_count':len(faces),
        'face_map_encoding':'uint32 little-endian; 0=unknown; value=inventory index + 1',
        'asset_sha256':digest(output/'surface.glb'), 'face_map_sha256':digest(output/'face-inv.bin'),
        'inventory_sha256':inventory_hash, 'source_asset':str(asset), 'source_frames':frame_ids,
        'source_delivery_sha256':digest(source/'delivery.json'), 'source_texture_metrics_sha256':digest(asset.parent/'texture-metrics.json'),
        'source_selection_sha256':texture['selection_sha256'], 'source_cameras_sha256':digest(source/'cameras.json'),
        'cameras':cameras, 'supported_inv':supported, 'unsupported_inv':[o['inv'] for o in objects if o['inv'] not in supported],
        'mapped_faces':int((labels>0).sum()), 'unknown_faces':int((labels==0).sum()), 'frames':frame_records,
        'uv_contract':'glTF TEXCOORD_0 top-left UV; col=floor(2u), row=floor(2v), frame=2*row+col+1; canonical=(2*uv-tile)*[392,518]+[63,0]-0.5',
        'assignment_rule':'Use only the face texture source frame. Three vertices and world centroid must pass native valid/conf>=0.1, content and 4% z checks and belong to exactly one accepted same-frame object mask. All other faces remain unknown.',
        'metric_scale_known':False, 'world_coordinates_changed':False,
        'product_geometry_registration':'not established; association uses hash-identical images and native source cameras only',
        'limitations':['Only selected frame3/4 evidence can label this mesh; frame1/2 object IDs are not inferred.',
                       'Image-mask agreement is not physical accuracy or new safety evidence.']}
    _save_correspondence(output, manifest, frame_number, source_samples, source_trusted,
                         atlas_grid=atlas_grid, content_rect=content_rect)
    (output/'surface.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    if (digest(output/'surface.glb') != digest(asset)
            or not np.array_equal(np.fromfile(output/'face-inv.bin',dtype='<u4'),labels)):
        raise ValueError('Imported surface readback failed')
    print(json.dumps({'manifest':str(output/'surface.json'), 'supported_inv':supported,
                      'mapped_faces':manifest['mapped_faces'],'unknown_faces':manifest['unknown_faces']}))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    import_surface(args.source, args.run)
