"""The verdict layer's ONLY input: a metric scene in one world frame (metres, floor up = ground_normal).

Boundary rule (2026-10-07): nothing in this directory reads meshes, run directories, photos, pipeline manifests or
reconstruction code. An adapter (adapter_measurement_layer.py is one) produces a Scene from whatever the reconstruction
publishes; the engine (engine.py) consumes a Scene and emits verdicts. Changing the reconstruction never changes the
engine; changing a rule never changes the reconstruction.
"""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

CLASSES = ('fence', 'guard', 'bollard', 'light_curtain', 'robot', 'cart', 'estop', 'other')
LEVELS = ('high', 'medium', 'low', 'unverified')


@dataclass
class Obj:
    id: str
    cls: str                      # one of CLASSES (semantic input: the class the reconstruction / a reviewer assigned)
    label: str                    # display name
    center_m: list                # oriented box centre, world metres
    axes: list                    # [l, w, up] unit vectors (up = floor normal)
    size_m: list                  # [L, W, H] metres
    bottom_m: float               # lowest point above the floor (negative = below the fitted floor)
    top_m: float                  # highest point above the floor
    sigma_m: dict                 # {'L','W','H','bottom'} 1-sigma metres, or None when unmeasured
    confidence: str               # one of LEVELS
    floor_contact: bool
    views: list = field(default_factory=list)   # photo ids that saw the object (evidence refs; opaque to the engine)


@dataclass
class Scene:
    scene_id: str
    objects: list
    ground_normal: list           # unit vector, world frame
    scale_rel_unc: float | None   # relative uncertainty of the metric scale (half spread of the reference features), or None

    def dump(self, path):
        Path(path).write_text(json.dumps({'schema': 'panoptes.verdict.scene/0', **asdict(self)}, indent=1))

    @staticmethod
    def load(path):
        d = json.loads(Path(path).read_text())
        assert d.get('schema') == 'panoptes.verdict.scene/0', d.get('schema')
        objs = [Obj(**o) for o in d['objects']]
        for o in objs:
            assert o.cls in CLASSES and o.confidence in LEVELS, (o.id, o.cls, o.confidence)
        return Scene(d['scene_id'], objs, d['ground_normal'], d['scale_rel_unc'])
