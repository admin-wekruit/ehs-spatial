"""Adapter: a published measurement layer (web/src/measurement-layer.ts shape) -> Scene (scene.py).

This is the only file that knows the layer's native units and box encoding. The engine never sees it.
  python adapter_measurement_layer.py <layer.json> <scene-id> <out scene.json>
"""
import json
import sys

from scene import Obj, Scene

CLASS_BY_WORD = (('light curtain', 'light_curtain'), ('fence', 'fence'), ('guard', 'guard'), ('bollard', 'bollard'), ('post', 'bollard'),
                 ('robot', 'robot'), ('cart', 'cart'), ('e-stop', 'estop'), ('emergency', 'estop'))


def classify(label):
    low = label.lower()
    return next((cls for word, cls in CLASS_BY_WORD if word in low), 'other')


def convert(layer, scene_id):
    ntm = float(layer['scale']['nativeToMeters'])
    objects = []
    for oid, box in layer.get('boxes', {}).items():
        label = (layer.get('labelsEn') or {}).get(oid) or (layer.get('labels') or {}).get(oid) or box['label']
        sigma = {k: (None if box['dims'][k]['sigmaCm'] is None else box['dims'][k]['sigmaCm'] / 100.0) for k in ('L', 'W', 'H', 'bottom')}
        views = sorted({p for face in box.get('faces', {}).values() for p in face.get('photos', [])})
        objects.append(Obj(id=oid, cls=classify(label), label=label, center_m=[c * ntm for c in box['centerNative']], axes=box['axes'],
                           size_m=list(box['sizeM']), bottom_m=float(box['bottomM']), top_m=float(box['topM']), sigma_m=sigma,
                           confidence=box['confidence'], floor_contact=bool(box['floorContact']), views=views))
    ground = layer.get('ground') or {}
    return Scene(scene_id=scene_id, objects=objects, ground_normal=list(ground.get('normal', [0, 0, 1])),
                 scale_rel_unc=layer['scale'].get('uncertaintyRelative'))


if __name__ == '__main__':
    layer_path, scene_id, out = sys.argv[1:4]
    scene = convert(json.load(open(layer_path)), scene_id)
    scene.dump(out)
    print(f'{scene_id}: {len(scene.objects)} objects ->', out, {o.cls: sum(1 for p in scene.objects if p.cls == o.cls) for o in scene.objects})
