"""Small checks for the frozen GLB face order and evidence-only mask mapping."""
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from import_workcell_surface import assign_faces, glb_arrays, source_pixels


def test_glb_face_order_and_source_photo_uv():
    pixels = np.array([[[100, 100], [102, 100], [100, 102]],
                       [[300, 300], [302, 300], [300, 302]]], dtype=float)
    tiles = np.array([[[0, 1]], [[1, 1]]])
    uv = (((pixels + .5 - [63, 0]) / [392, 518] + tiles) / 2).astype('<f4')
    positions = np.arange(18, dtype='<f4').reshape(6, 3)
    arrays = [np.arange(6, dtype='<u4'), positions, uv.reshape(6, 2)]
    binary = b''.join(array.tobytes() for array in arrays)
    offsets = np.cumsum([0] + [array.nbytes for array in arrays])
    document = {'nodes':[{'mesh':0}], 'meshes':[{'primitives':[{
        'attributes':{'POSITION':1, 'TEXCOORD_0':2}, 'indices':0, 'mode':4}]}],
        'accessors':[{'bufferView':i, 'componentType':component, 'count':6, 'type':kind}
                     for i,component,kind in [(0,5125,'SCALAR'),(1,5126,'VEC3'),(2,5126,'VEC2')]],
        'bufferViews':[{'buffer':0, 'byteOffset':int(offsets[i]), 'byteLength':a.nbytes}
                       for i,a in enumerate(arrays)]}
    payload = json.dumps(document).encode()
    payload += b' ' * (-len(payload) % 4)
    chunks = (struct.pack('<II',len(payload),0x4E4F534A)+payload
              +struct.pack('<II',len(binary),0x004E4942)+binary)
    data = struct.pack('<III',0x46546C67,2,12+len(chunks))+chunks
    actual_vertices, faces, actual_uv = glb_arrays(data)
    np.testing.assert_array_equal(actual_vertices, positions)
    np.testing.assert_array_equal(faces, [[0,1,2],[3,4,5]])
    frames, decoded = source_pixels(actual_uv[faces])
    np.testing.assert_array_equal(frames, [3,4])
    np.testing.assert_allclose(decoded, pixels, atol=.0001)
    mixed = actual_uv[faces].copy()
    mixed[0,0,0] = .6
    with pytest.raises(ValueError, match='one source-photo'):
        source_pixels(mixed)
    with pytest.raises(ValueError, match='complete GLB'):
        glb_arrays(data[:-1])


def test_face_requires_unique_whole_sample_support():
    samples = np.array([[[100,100],[101,100],[100,101],[100.3,100.3]]]*5)
    samples[1,2] = [105,105]  # A boundary-spanning face cannot inherit the object.
    samples[2] += 100        # Both masks contain this face: remain unknown.
    samples[4] -= 200        # Out of bounds must never wrap to opposite mask edge.
    mask = np.zeros((518,518),bool)
    mask[100:103,100:103] = True
    mask[200:203,200:203] = True
    overlap = np.zeros_like(mask)
    overlap[200:203,200:203] = True
    labels, candidates = assign_faces(samples, {13:mask,18:overlap}, np.array([1,1,1,0,1],bool))
    np.testing.assert_array_equal(labels, [14,0,0,0,0])
    np.testing.assert_array_equal(candidates, [1,0,2,0,0])
    assert labels.dtype == np.dtype('<u4')
    assert labels.tobytes()[:4] == b'\x0e\0\0\0'


def test_saved_source_correspondence_relabels_new_inventory_without_old_geometry(tmp_path, monkeypatch):
    """The seed's UV/camera certificate is sufficient; missing support is an error."""
    from PIL import Image
    import import_workcell_surface as surface
    run = tmp_path/'run'
    out = run/'surface'; out.mkdir(parents=True)
    (run/'input').mkdir(); (run/'inventory').mkdir()
    pixels = np.array([[[100,100],[102,100],[100,102]],
                       [[300,300],[302,300],[300,302]]],float)
    uv = ((pixels+.5-[63,0])/[392,518]+np.array([[[0,1]],[[1,1]]]))/2
    vertices = np.concatenate([pixels,np.ones((2,3,1))],axis=-1).reshape(-1,3)
    faces = np.arange(6).reshape(2,3)
    monkeypatch.setattr(surface,'glb_arrays',lambda _: (vertices,faces,uv.reshape(-1,2)))
    (out/'surface.glb').write_bytes(b'fixed-seed-mesh')
    np.array([1,0],dtype='<u4').tofile(out/'face-inv.bin')
    inventory = run/'inventory/inventory.json'; inventory.write_text('{"objects":[{"inv":0}]}')
    cameras=[]; records=[]
    for index in range(1,5):
        source=run/f'input/image_{index:02}.png'; Image.new('RGB',(20,30),index).save(source)
        if index<3: continue
        frame_id=f'frame_{index:04}'
        geom=run/'geometry/frames'/frame_id; geom.mkdir(parents=True)
        Image.new('RGB',(518,518),index).save(geom/'canonical.png')
        records.append({'frame_id':frame_id,'faces':1,'original_image_sha256':surface.digest(source),
            'canonical_image_sha256':surface.digest(geom/'canonical.png'),'unsupported_depth_faces':0})
        cameras.append({'id':frame_id,'camera_to_world':np.eye(4).tolist(),
                        'K':[[1,0,-63],[0,1,0],[0,0,1]]})
    manifest={'version':1,'asset_sha256':surface.digest(out/'surface.glb'),
        'face_map_sha256':surface.digest(out/'face-inv.bin'),'face_count':2,'frames':records,
        'cameras':{'frames':cameras},'uv_contract':'glTF TEXCOORD_0 top-left UV; col=floor(2u), row=floor(2v), frame=2*row+col+1; canonical=(2*uv-tile)*[392,518]+[63,0]-0.5'}
    (out/'surface.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(surface,'case_objects',lambda _: ({},json.loads(inventory.read_text())['objects']))
    def masks(run, objects, frame_id, mapped):
        active={}
        for obj in objects:
            if (obj['inv']==0 and frame_id.endswith('3')) or (obj['inv']==1 and frame_id.endswith('4')):
                mask=np.zeros((518,518),bool); start=100 if obj['inv']==0 else 300
                mask[start:start+4,start:start+4]=True; active[obj['inv']]=mask
        return active,[],[]
    monkeypatch.setattr(surface,'_mask_records',masks)
    first=surface.refresh_surface_associations(run)
    assert first['version']==2 and first['supported_inv']==[0]
    assert np.array_equal(np.fromfile(out/'face-inv.bin',dtype='<u4'),[1,0])
    assert (out/'source-correspondence.npz').is_file()
    # A retained 2D-only observation creates no fake 3D face assignment.
    (run/'refinements.json').write_text('[{"label":"49-pixel test patch","geometry_status":"unmeasured"}]')
    inventory.write_text('{"objects":[{"inv":0},{"inv":1}]}')
    updated=surface.refresh_surface_associations(run)
    assert updated['supported_inv']==[0,1]
    assert updated['inventory_sha256']==surface.digest(inventory)
    assert np.array_equal(np.fromfile(out/'face-inv.bin',dtype='<u4'),[1,2])
    assert (out/'surface.glb').read_bytes()==b'fixed-seed-mesh'
    assert json.loads((run/'refinements.json').read_text())[0]['geometry_status']=='unmeasured'
    # Tampering with the native camera invalidates the saved sampling contract.
    updated['cameras']['frames'][0]['K'][0][2]+=1
    (out/'surface.json').write_text(json.dumps(updated))
    with pytest.raises(ValueError,match='correspondence changed'):
        surface.refresh_surface_associations(run)
