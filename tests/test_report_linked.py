"""The report's 2×2 linked-selection block: photo / 3D / CAD / plan share one
selection key (`inv` = index in inventory.json["objects"]), the CAD cell is
driven by the renderer's floor_plan_map.json sidecar, and missing or stale CAD assets rebuild from the same inventory."""

import base64
import io
import json
import re
import sys
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.interactive_report import build_interactive_run_report, case_objects, mask_index_png, _TAIL_JS
from ehs_spatial.providers.sam3 import encode_coco_rle

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from scene_inventory import _render_plan, _refine_mask, _save_refinement_mask  # noqa: E402

BOR1 = REPO / "runs" / "user-bor1-02"
FLOORS = {"floor", "factory floor", "ceiling", "concrete floor", "ceiling structure"}


def _block(html: str) -> str:
    assert html.count('class="linked" data-section-group="linked"') == 1
    return html.split('class="linked" data-section-group="linked"')[1].split("cell 矩形约束")[0]


@pytest.mark.skipif(
    not (BOR1 / "inventory" / "floor_plan_map.json").exists(),
    reason="BOR1 run (with regenerated inventory) not present",
)
def test_bor1_block_links_four_panels_by_inv(monkeypatch):
    monkeypatch.chdir(REPO)
    html = build_interactive_run_report("user-bor1-02").read_text(encoding="utf-8")
    # still exactly the 14 section markers, three of them now cells of the block
    assert html.count('data-section="') == 14
    block = _block(html)
    assert re.findall(r'data-(?:sub)?section="([a-z]+)"', block) == ["photo", "viewer", "cad", "plan"]
    # the bridge to the 3D viewer
    assert 'id="v3d" class="v3d"' in block
    js = html.split("<script>")[-1]
    assert "panoptes:select" in js and "panoptes:selected" in js and "ehs-select" not in js
    # CAD overlay: one hit-area per on-plan object, keyed by inv, sized to the PNG
    fmap = json.loads((BOR1 / "inventory" / "floor_plan_map.json").read_text())
    cad = block.split('class="icad"')[1].split("</svg>")[0]
    cad_inv = [int(v) for v in re.findall(r'<polygon data-inv="(\d+)"', cad)]
    assert len(cad_inv) == len(fmap["objects"]) > 0
    assert f'viewBox="0 0 {fmap["width"]} {fmap["height"]}"' in cad
    # plan polygons carry the same key set; every key points at a non-floor inventory object
    inv = json.loads((BOR1 / "inventory" / "inventory.json").read_text())["objects"]
    plan_inv = [int(v) for v in re.findall(r'<polygon class="ip" data-inv="(\d+)"', block)]
    assert sorted(plan_inv) == sorted(cad_inv)
    assert all(inv[i]["label"] not in FLOORS and not inv[i].get("off_plan_reason") for i in plan_inv)
    assert len(plan_inv) == sum(1 for o in inv if o["label"] not in FLOORS and not o.get("off_plan_reason"))
    # toolbar, and the legacy intermediate figure is gone from the gallery
    assert "全不选" in html and "已选 0 个" in html
    assert "平面视图" not in html and "点云透视" in html and "测量平面图" in html


def _synthetic_run(tmp_path, with_map):
    run = tmp_path / "runs" / "run-syn"
    (run / "inventory").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "run-syn", "providers": {}}))
    (run / "scene.json").write_text(json.dumps({"entities": [], "warnings": []}))
    (run / "policies.json").write_text(json.dumps({"specs": [], "results": []}))
    objs = [
        {"label": "floor", "instance": 0, "height_m": 0.0, "size_m": "9x9", "camera_dist_m": 0.0,
         "centroid_xy": [0, 4], "footprint": [[-4, 0], [4, 0], [4, 8], [-4, 8]], "footprint_area_m2": 64.0},
        {"label": "bollard", "instance": 0, "height_m": 1.0, "size_m": "0.3x0.3", "camera_dist_m": 3.0,
         "centroid_xy": [1, 3], "footprint": [[0.9, 2.9], [1.1, 2.9], [1.1, 3.1], [0.9, 3.1]], "footprint_area_m2": 0.04},
        {"label": "safety fence", "instance": 0, "height_m": 1.8, "size_m": "2x0.1", "camera_dist_m": 5.0,
         "centroid_xy": [-1, 5], "footprint": [[-2, 4.9], [0, 4.9], [0, 5.1], [-2, 5.1]], "footprint_area_m2": 0.2},
    ]
    (run / "inventory" / "inventory.json").write_text(json.dumps({"objects": objs, "walls": []}))
    Image.new("RGB", (160, 124), "white").save(run / "inventory" / "floor_plan.png")
    if with_map:
        (run / "inventory" / "floor_plan_map.json").write_text(json.dumps({
            "width": 160, "height": 124,
            "objects": [{"legend": 1, "inv": 2, "label": "safety fence", "polygon": [[10, 10], [50, 10], [50, 14], [10, 14]], "centroid": [30, 12]},
                        {"legend": 2, "inv": 1, "label": "bollard", "polygon": [[80, 80], [84, 80], [84, 84], [80, 84]], "centroid": [82, 82]}],
        }))
    return run


def test_block_rebuilds_without_floor_plan_map(tmp_path, monkeypatch):
    _synthetic_run(tmp_path, with_map=False)
    monkeypatch.chdir(tmp_path)
    html = build_interactive_run_report("run-syn").read_text(encoding="utf-8")
    assert html.count('data-section="') == 14
    block = _block(html)
    cad = block.split('class="icad"')[1].split("</div>")[0]
    assert "<img" in cad and "<svg" in cad
    assert 'viewBox="0 0 1600 1240"' in cad
    # photo (no geometry frames) and 3D (no viewer.html) degrade to hints; plan still keyed by inv, floor skipped
    assert "无（无可点选实体或缺少几何帧）" in block and "无（viewer.html 缺失）" in block
    assert sorted(re.findall(r'<polygon class="ip" data-inv="(\d+)"', block)) == ["1", "2"]


def test_block_overlay_from_map(tmp_path, monkeypatch):
    run = _synthetic_run(tmp_path, with_map=True)
    (run / 'viewer.html').write_text('<p>current full viewer</p>')
    (run / 'viewer_small.html').write_text('<p>stale small viewer</p>')
    monkeypatch.chdir(tmp_path)
    html = build_interactive_run_report("run-syn").read_text(encoding="utf-8")
    assert 'current full viewer' in html and 'stale small viewer' not in html
    cad = _block(html).split('class="icad"')[1].split("</svg>")[0]
    assert 'viewBox="0 0 1600 1240"' in cad
    assert re.findall(r'<polygon data-inv="(\d+)"', cad) == ["2", "1"]
    assert "<title>1. safety fence</title>" in cad


def test_render_plan_writes_map_sidecar(tmp_path):
    objs = [{"label": "bollard", "inv": 7, "height_m": 1.0, "size_m": "0.3x0.3", "camera_dist_m": 3.0,
             "centroid_xy": [1.0, 3.0], "footprint": [[0.9, 2.9], [1.1, 2.9], [1.1, 3.1], [0.9, 3.1]],
             "rect_snapped": [[0.9, 2.9], [1.1, 2.9], [1.1, 3.1], [0.9, 3.1]]}]
    _render_plan(tmp_path / "floor_plan.png", [], objs, "t")
    assert (tmp_path / "floor_plan.png").exists()
    m = json.loads((tmp_path / "floor_plan_map.json").read_text())
    assert (m["width"], m["height"]) == (1600, 1240)
    (o,) = m["objects"]
    assert o["inv"] == 7 and o["legend"] == 1 and o["label"] == "bollard"
    assert len(o["polygon"]) == 4 and all(0 <= x <= 1600 and 0 <= y <= 1240 for x, y in o["polygon"])
    assert all(0 <= v for v in o["centroid"])


def test_stale_cad_map_rebuilds_when_inventory_changes(tmp_path, monkeypatch):
    run = _synthetic_run(tmp_path, with_map=True)
    monkeypatch.chdir(tmp_path)
    build_interactive_run_report('run-syn')
    path = run/'inventory/floor_plan_map.json'
    first = json.loads(path.read_text())
    inventory_path = run/'inventory/inventory.json'
    inventory = json.loads(inventory_path.read_text())
    inventory['objects'][1]['label'] = 'new bollard'
    inventory_path.write_text(json.dumps(inventory))
    build_interactive_run_report('run-syn')
    second = json.loads(path.read_text())
    assert first['inventory_sha256'] != second['inventory_sha256']
    assert next(o for o in second['objects'] if o['inv'] == 1)['label'] == 'new bollard'


def test_photo_masks_use_source_frame_and_canonical_grid(tmp_path, monkeypatch):
    run = _synthetic_run(tmp_path, with_map=False)
    inventory_path = run/'inventory/inventory.json'
    inventory = json.loads(inventory_path.read_text())
    inventory['objects'][1].update(label='bollard', frame='frame_0001')
    inventory['objects'][2].update(label='bollard', instance=0, frame='frame_0002')
    refinement = dict(inventory['objects'][1], label='red marker', refine_slug='marker', instance=None)
    inventory['objects'].append(refinement)
    inventory_path.write_text(json.dumps(inventory))
    # An un-ingested refinement cannot invent an ID outside inventory.
    (run/'refinements.json').write_text(json.dumps([{'label':'ghost', 'box':[0,0,2,2], 'footprint_xy':[[0,0],[1,0],[0,1]]}]))
    (run/'input').mkdir(); (run/'refinements').mkdir(); (run/'inventory/sam').mkdir()
    for number in (1, 2):
        frame = f'frame_{number:04d}'
        folder = run/'geometry/frames'/frame; folder.mkdir(parents=True)
        source = Image.new('RGB', (40,40), (30,60,90)); source.save(run/'input'/f'image_{number:02d}.png')
        canonical = Image.new('RGB', (8,4), 'white'); canonical.paste(source.resize((4,4)), (2,0)); canonical.save(folder/'canonical.png')
        mask = np.zeros((4,8), bool); mask[1:3, 2:4 if number==1 else 6] = True
        (run/'inventory/sam'/f'{frame}__bollard.json').write_text(json.dumps({'rle':[encode_coco_rle(mask)]}))
    raw = np.zeros((40,40), bool); raw[0:10,30:40] = True
    (run/'refinements/marker.json').write_text(json.dumps({'rle':[encode_coco_rle(raw)]}))
    alpha = np.zeros((4,8), np.float32); alpha[:,2:6]=1
    provider = {'image':{'shape':[4,8,3]}, 'alpha_mask':{'shape':list(alpha.shape),'dtype':str(alpha.dtype),'data':base64.b64encode(alpha.tobytes()).decode()}, 'original_image':{'width':40,'height':40}}
    (run/'geometry/provider').mkdir(); (run/'geometry/provider/frame_0001.json').write_text(json.dumps(provider))
    monkeypatch.chdir(tmp_path)
    _, objects = case_objects('run-syn')
    assert [o['inv'] for o in objects] == [1,2,3]
    for frame, expected in [('frame_0001',{1,3}),('frame_0002',{2})]:
        uri, width, height, visible, issues, masks = mask_index_png('run-syn', objects, frame)
        rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(',')[1])))).astype(np.int32)
        index = rgb[...,0]+(rgb[...,1]<<8)+(rgb[...,2]<<16)-1
        assert (width,height)==(8,4) and set(visible)==expected and set(masks)==expected and not issues
        assert np.all(index[:,:2]==-1) and np.all(index[:,6:]==-1)
        if frame=='frame_0001': assert index[0,5]==3 and index[0,4]==-1
    report = build_interactive_run_report('run-syn').read_text()
    assert 'class="photo-frame"' in report and report.count('class="iphoto"')==2
    assert 'data-frame="frame_0002"' in report
    assert report.index('data-section-group="linked"') < report.index('data-section="verdicts"')


def test_refinement_derived_mask_is_bound_to_source_and_accepted_entry(tmp_path):
    run = tmp_path/'run'; (run/'input').mkdir(parents=True); (run/'refinements').mkdir(); (run/'inventory').mkdir()
    Image.new('RGB', (12,8)).save(run/'input/image_01.png')
    source = run/'refinements/part.json'
    source.write_text(json.dumps({'rle':[encode_coco_rle(np.ones((8,12), bool))]}))
    source_bytes = source.read_bytes()
    canonical = np.zeros((4,8), bool); canonical[1:3,2:6]=True
    sha = _save_refinement_mask(run, 'frame_0001', 'part', canonical)
    assert source.read_bytes() == source_bytes
    assert np.array_equal(_refine_mask(run,'part',4,8,expected_sha256=sha), canonical)
    with pytest.raises(ValueError, match='does not match'):
        _refine_mask(run,'part',4,8,expected_sha256='wrong inventory snapshot')
    source.write_text(json.dumps({'rle':[encode_coco_rle(np.zeros((8,12), bool))]}))
    with pytest.raises(ValueError, match='does not match'):
        _refine_mask(run,'part',4,8,expected_sha256=sha)
    # A historically stretched RLE cannot regain provenance from its dimensions.
    source.write_text(json.dumps({'rle':[encode_coco_rle(canonical)]}))
    with pytest.raises(ValueError, match='Legacy resized refinement'):
        _refine_mask(run,'part',4,8,use_derived=False)


@pytest.mark.skipif(shutil.which('node') is None, reason='Node is needed to execute the inline selection bus')
def test_selection_bus_keeps_all_ids_and_replays_on_viewer_ready():
    bus_js = _TAIL_JS.split('  var bus=',1)[1].split('  L.bus=bus;',1)[0]
    bridge_js = _TAIL_JS.split('  // A ready/load handshake',1)[1].split('\n',1)[1].split('  bus.emit();',1)[0]
    script = r"""
const assert=require('node:assert/strict');
const pos={0:0,2:1,5:2,7:3,10:4,11:5},sent=[],handlers={},frameHandlers={};
const child={postMessage:(payload,origin)=>sent.push({payload,origin})};
const embedded={contentWindow:child,addEventListener:(name,cb)=>frameHandlers[name]=cb};
const L={querySelector:()=>embedded};
const window={origin:'http://localhost:8791',location:{origin:'null'}};
function addEventListener(name,cb){handlers[name]=cb;}
var bus=BUS;
BRIDGE
bus.set([0,2,5,7,10,11]);
assert.equal(bus.sel.size,6);
sent.length=0;
handlers.message({source:child,origin:window.origin,data:{type:'panoptes:ready',supported_inv:[0,2,5]}});
assert.deepEqual(sent[0],{payload:{type:'panoptes:select',inv:[0,2,5,7,10,11],exclusive:true},origin:window.origin});
handlers.message({source:{},origin:window.origin,data:{type:'panoptes:selected',inv:[2]}});
handlers.message({source:child,origin:'https://wrong.example',data:{type:'panoptes:selected',inv:[2]}});
assert.equal(bus.sel.size,6);
sent.length=0;
handlers.message({source:child,origin:window.origin,data:{type:'panoptes:selected',inv:[2,5,999]}});
assert.deepEqual([...bus.sel],[2,5]);assert.equal(sent.length,0);
frameHandlers.load();assert.deepEqual(sent[0].payload.inv,[2,5]);
bus.select(2);assert.deepEqual([...bus.sel],[5]);bus.set([]);assert.equal(bus.sel.size,0);
""".replace('BUS;',bus_js).replace('BRIDGE',bridge_js)
    subprocess.run([shutil.which('node'),'-e',script],check=True,capture_output=True,text=True)
