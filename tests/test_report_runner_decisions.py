"""Part C of the M2 runner: every decision rule reproduces the delivered number from cached runs (M2 design section 5), the ICP
per-iteration caps, the intrinsics_uncertain scale, the import's scale sentence, and the profiles' refusals.

CPU only, no GPU or Modal call: the cached runs are read, never written (new files go to pytest's tmp dirs). The tests that
read research-notes skip when $ART (default: this Mac's research-notes/phase2) is absent. The dense-gate tests replay the ICP
(about 40 s each) and the person filter reads every ME340 mask (about 2 min).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "modal_apps"), str(REPO)]
from report_runner import decide, profiles  # noqa: E402

ART = Path(os.environ.get("ART", "/Users/adam/Desktop/panoptes-public/research-notes/phase2"))
RUNS, CLIPS = ART / "runs", ART / "data/clips"
needs_art = pytest.mark.skipif(not (RUNS / "droid-me340-165-171").exists(), reason=f"no cached runs at {ART}")
SITES = {"me340": ("me340-165", "droid-me340-165-171"), "samsclub": ("samsclub-337", "droid-samsclub-337-157"), "walmart": ("walmart-190", "droid-walmart-190-168")}
METRIC = {"me340": "da3-posed-me340-223-shotc", "samsclub": "da3-posed-samsclub-a2-281", "walmart": "da3-posed-walmart-251-shot383"}


def run(name, out, **paths):
    return decide.decide(name, out, {k: str(v) for k, v in paths.items()})


@pytest.fixture(scope="module")
def shots(tmp_path_factory):
    """D2 on each site, written where the rules that read a shots decision find it."""
    out = {}
    for site, (clip, census) in SITES.items():
        folder = tmp_path_factory.mktemp(site)
        out[site] = (folder, run("shots", folder, segments=RUNS / "m0-integrate-cuts" / clip, census=RUNS / census))
    return out


def test_self_checks():
    decide.self_check()
    assert set(decide.RULES) == set(decide.VERSIONS) == set(profiles.M2_RULES)


@needs_art
def test_shots_pick_the_shot_with_most_census_keyframes(shots):
    expect = {"me340": ([226, 899], [1, 1, 18]), "samsclub": ([0, 420], [24, 8]), "walmart": ([383, 750], [28, 39])}
    for site, (primary, counts) in expect.items():
        result = shots[site][1]
        assert result["value"]["primary"] == primary and result["value"]["mapped"] and result["evidence"]["censusKeyframes"] == counts, (site, result)
    assert shots["me340"][1]["value"]["others"] == [[0, 14], [14, 226]]


@needs_art
def test_sam2_frames_equal_the_delivered_lists(shots):
    for site, delivered, count in (("me340", "sam2-me340-everything-184", 312), ("samsclub", "sam2-samsclub-everything-160", 267),
                                   ("walmart", "sam2-walmart-everything-181", 298)):
        clip, census = SITES[site]
        frames = run("sam2_frames", shots[site][0], segments=RUNS / "m0-integrate-cuts" / clip, census=RUNS / census)["value"]
        assert len(frames) == count and frames == sorted(json.loads((RUNS / delivered / "segment.json").read_text())["frames"]), site


@needs_art
def test_lens_frames_values_and_floor_plane_gate(shots):
    walmart = run("lens", shots["walmart"][0], shots=shots["walmart"][0], clip=CLIPS / "walmart-190",
                  moge=RUNS / "droid-walmart-190-shot383-250/fov-check.json", metric=RUNS / METRIC["walmart"])["value"]
    assert walmart["frames"] == json.loads((RUNS / "droid-walmart-190-shot383-250/fov-check.json").read_text())["frames"]
    assert walmart["fov_deg"] == 55.96 and round(walmart["clip_fov_deg"], 2) == 54.06 and walmart["keep"] and walmart["fx"] is None
    assert round(walmart["plane_inlier_fraction"], 3) == .884 and walmart["scale_status"] == "intrinsics_uncertain"
    sams = run("lens", shots["samsclub"][0], shots=shots["samsclub"][0], clip=CLIPS / "samsclub-337",
               moge=CLIPS / "samsclub-337-a2/shot-fov.json", metric=RUNS / METRIC["samsclub"])["value"]
    assert sams["frames"] == [f["frame"] for f in json.loads((CLIPS / "samsclub-337-a2/shot-fov.json").read_text())["frames"]]
    assert not sams["keep"] and round(sams["clip_fov_deg"], 2) == 60.08
    assert sams["fx"] == json.loads((CLIPS / "samsclub-337-a2/clip.json").read_text())["K"][0] == 610.5529403686523
    assert round(sams["plane_inlier_fraction"], 3) == .965 and sams["scale_status"] == "assumed_camera_height"
    me340 = run("lens", shots["me340"][0], shots=shots["me340"][0], clip=CLIPS / "me340-165", metric=RUNS / METRIC["me340"])["value"]
    assert round(me340["plane_inlier_fraction"], 3) == .959 and me340["scale_status"] == "assumed_camera_height" and me340["keep"] is None
    with pytest.raises(ValueError, match="refuse"):  # a MoGe record on other frames is never mixed in
        run("lens", shots["me340"][0], shots=shots["me340"][0], clip=CLIPS / "me340-165", moge=RUNS / "droid-walmart-190-shot383-250/fov-check.json")


@needs_art
def test_voxel(tmp_path):
    assert [run("voxel", tmp_path, metric=RUNS / METRIC[s])["value"] for s in ("me340", "samsclub", "walmart")] == [.01399, .00567, .011858]


@needs_art
def test_track_windows(shots):
    delivered = lambda *names: [json.loads((RUNS / n / "tracks.json").read_text())["frames"] for n in names]
    assert decide.windows(0, 899) == delivered("me340-sam31-tracks-178", "me340-sam31-tracks-179", "me340-sam31-tracks-180")  # the whole clip, as delivered
    sams = run("track_windows", shots["samsclub"][0], shots=shots["samsclub"][0])["value"]
    walmart = run("track_windows", shots["walmart"][0], shots=shots["walmart"][0])["value"]
    assert sams["shot"] == delivered("samsclub-a-sam31-tracks-263", "samsclub-a-sam31-tracks-265") and sams["stitch"]
    assert walmart["shot"] == delivered("walmart-sam31-tracks-253") and not walmart["stitch"]
    assert run("track_windows", shots["me340"][0], shots=shots["me340"][0])["value"]["others"] == [[[0, 14]], [[14, 226]]], "never across a cut"


@needs_art
def test_lingbot_stride_and_confidence(shots, tmp_path):
    assert [run("lingbot_stride", shots[s][0], shots=shots[s][0])["value"] for s in ("me340", "samsclub", "walmart")] == [2, 1, 1]
    confs = [run("lingbot_conf", tmp_path, diagnose=RUNS / r)["value"] for r in ("me340-lingbot-map-222", "samsclub-a-lingbot-map-264", "walmart-lingbot-map-258")]
    assert confs == [1.06, 1.74, 1.01]  # Walmart copied ME340's 1.06 (deviation D4)


@needs_art
def test_splat_pick_and_inferred_floor(tmp_path):
    for clean in ("me340-splat-217/skip-2.5m-60k-elong4", "samsclub-a2-splat-287", "walmart-splat-260"):
        assert run("splat_pick", tmp_path, clean=RUNS / clean)["value"] == "negligible_1", clean
    kept = [run("inferred_floor", tmp_path, floor=RUNS / f)["value"] for f in ("me340-inferred-floor-244", "samsclub-a2-inferred-floor-296-icp", "walmart-inferred-floor-265")]
    assert kept == [True, False, False]


@needs_art
def test_overlay_band(tmp_path):
    me340 = run("overlay", tmp_path / "me340", clip=CLIPS / "me340-165")
    (y0, y1), = me340["value"]["bands"]
    assert 640 <= y0 <= 661 and 689 <= y1 <= 712 and me340["evidence"]["stillMovedMatches"] == 11865
    assert me340["value"]["texture_rows"] == f"{y0}:{y1}" and me340["value"]["lingbot_rows"] == f"{y0 - 2}:{y1 + 2}"
    assert me340["value"]["splat_captions"] == [y0, y1, 160, 1120] and me340["value"]["objects_no_captions"] is False
    for clip, matches in (("samsclub-337", 186), ("walmart-190", 74)):
        value = run("overlay", tmp_path / clip, clip=CLIPS / clip)
        assert value["value"]["bands"] == [] and value["value"]["lingbot_rows"] == "0:0" and value["value"]["splat_captions"] == [0, 0, 0, 0]
        assert value["evidence"]["stillMovedMatches"] == matches


def test_floor_frames_use_the_operator_list_only_on_its_clip(tmp_path):
    """D18 / O1: the operator's frames when the list names this clip (source video sha256 and window), else the rule: a
    new window or video of the same site never gets another clip's hand list."""
    (tmp_path / "shots.json").write_text(json.dumps({"value": {"primary": [383, 750], "others": [[0, 383]], "mapped": True}}))
    source = {"video_sha256": "a" * 64, "start_s": 190., "end_s": 220.}
    (tmp_path / "operator.json").write_text(json.dumps({"floorFrames": [0, 66, 747], "clip": source}))
    for window, expected in ((source, [0, 66, 747]), ({**source, "end_s": 219.}, decide.spaced_floor_frames([383, 750])),
                             ({**source, "video_sha256": "b" * 64}, decide.spaced_floor_frames([383, 750]))):
        (tmp_path / "clip").mkdir(exist_ok=True)
        (tmp_path / "clip/clip.json").write_text(json.dumps({"source": window}))
        got = run("floor_frames", tmp_path / "out", shots=tmp_path / "shots.json", operator=tmp_path / "operator.json", clip=tmp_path / "clip")
        assert got["value"] == expected and got["evidence"]["source"] == ("operator" if window == source else "rule"), window
    assert run("floor_frames", tmp_path / "out", shots=tmp_path / "shots.json")["value"] == decide.spaced_floor_frames([383, 750])


@needs_art
def test_other_shot_gate(tmp_path):
    value = run("other_shot", tmp_path, registration=RUNS / "me340-cutaway-register-304")["value"]
    assert value["accepted"] and value["shot"] == [14, 226] and round(value["centreResidualShare"] * 100, 2) == .92 and round(value["rotationResidualDeg"], 2) == 2.38


@needs_art
def test_generator_plan_keeps_review_exclusions_for_the_box(tmp_path):
    value = run("generator_plan", tmp_path, sam3d=RUNS / "walmart-object-models-267-sam3d", recgen=RUNS / "walmart-object-models-264",
                review=REPO / "docs/phase2/box-review-303/walmart.json")["value"]
    assert "object-156" in value["box_excludes"] and set(value["excluded_all"]) == {"object-080", "object-156", "object-158"}
    assert "object-156" in json.loads((REPO / "docs/phase2/box-review-303/walmart.json").read_text())["boxesApproved"], "the delivered box this fixes"
    assert not set(value["recgen_entities"]) & set(value["excluded_all"])


@needs_art
@pytest.mark.parametrize("site,raw,delivered,offset_raw,offset_icp,max_deg,max_move", [
    ("walmart", "walmart-lingbot-map-258", "walmart-lingbot-map-268-icp", -.087, .009, .84, .041),
    ("samsclub", "samsclub-a2-lingbot-map-286", "samsclub-a2-lingbot-map-295-icp", None, 0., .97, .081)])
def test_dense_gate_replays_the_delivered_icp(tmp_path, site, raw, delivered, offset_raw, offset_icp, max_deg, max_move):
    import hashlib
    value = run("dense_gate", tmp_path, map=RUNS / raw, fused=RUNS / METRIC[site])["value"]
    assert value["use"] == "icp" and value["dir"] == "icp" and not value["steps"]["refused"]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ("dense-points.glb", "point-attributes.npz", "remote.json", "plan.json", "diagnose.json", "lingbot-run.json"):
        assert digest(tmp_path / "icp" / name) == digest(RUNS / delivered / name), name  # below the caps the script's output is unchanged
    ours, theirs = (json.loads(p.read_text()) for p in (tmp_path / "icp/points.json", RUNS / delivered / "points.json"))
    assert {k: v for k, v in theirs.items() if k not in ("metrics", "comparison_images")} == ours
    assert round(value["steps"]["max_step_deg"], 2) == max_deg and round(value["steps"]["max_step_move_m"], 3) == max_move
    raw_m, icp_m = value["metrics"]["raw"], value["metrics"]["icp"]
    assert not decide.dense_passes(raw_m) and decide.dense_passes(icp_m)
    assert abs(icp_m["floor_offset_m"] - offset_icp) <= .01, icp_m
    if offset_raw is not None:
        assert abs(raw_m["floor_offset_m"] - offset_raw) <= .01 and raw_m["within_25cm_all_points"] >= .45, raw_m  # Walmart fails on its floor alone
    else:
        assert raw_m["within_25cm_all_points"] == .4036 and icp_m["within_25cm_all_points"] == .6155


def test_icp_refuses_a_step_over_the_cap(tmp_path):
    import trimesh
    from lingbot_dense_map import write_points_glb
    fused, source, out = tmp_path / "fused", tmp_path / "map", tmp_path / "out"
    fused.mkdir(), source.mkdir()
    slab = trimesh.creation.box((10., 10., .02))  # a floor; the map sits 0.5 m above it, so the first step moves every point 0.5 m
    slab.export(fused / "mono-anchored-mesh.ply")
    (fused / "metric-scale.json").write_text(json.dumps({"metres_per_native_unit": 1.}))
    points = trimesh.sample.sample_surface(slab, 20_000, seed=1)[0] + [0, 0, .5]
    write_points_glb(source / "dense-points.glb", points.astype(np.float32), np.zeros((len(points), 3), np.uint8))
    np.savez_compressed(source / "point-attributes.npz", fill=np.zeros(len(points), bool), spacing=np.full(len(points), .01, np.float16))
    (source / "points.json").write_text(json.dumps({"cell_native": .01, "note": "synthetic"}))
    (source / "remote.json").write_text(json.dumps({"cell_native_final": .01}))
    done = subprocess.run([sys.executable, str(REPO / "scripts/lingbot_icp_refine.py"), str(source), str(out), str(fused)], capture_output=True, text=True)
    assert done.returncode == 2, done.stderr[-2000:]
    refused = json.loads((out / "refused.json").read_text())
    assert refused["step_deg"] > 2 or refused["step_move_m"] > .3
    assert not (out / "dense-points.glb").exists() and json.loads((out / "steps.json").read_text())


@needs_art
def test_static_filter_clears_the_me340_floor_patch_named_man(tmp_path):
    result = run("static_filter", tmp_path, object_map=RUNS / "me340-entity-names-200", masks=RUNS / "me340-masks-194",
                 dynamic_masks=RUNS / "me340-dynamic-masks-188/masks", droid=RUNS / "droid-me340-165-171")
    assert "object-031" in result["value"]["cleared"] and result["evidence"]["medianShare"]["object-031"] < .02
    filtered = {e["entityId"]: e for e in json.loads((tmp_path / "object-map.json").read_text())["entities"]}
    assert filtered["object-031"]["label"] == decide.UNNAMED and filtered["object-031"]["clearedLabel"] == "man"
    assert not any(e in filtered for e in result["value"]["moved"])


def test_intrinsics_uncertain_scale_claims_no_metres():
    from ehs_spatial.video import contract_scale
    from ehs_spatial.platform.contracts import empty_document, validate_document
    record = contract_scale({"scale_status": "intrinsics_uncertain", "metres_per_native_unit": 3.37, "plane_inlier_fraction": .884})
    assert record["status"] == "uncalibrated" and record.get("nativeToMeters") is None and record["intrinsicsUncertain"]
    document = empty_document()
    document["coordinateFrames"] = [{"id": "f", "convention": "opencv", "scale": record, "ground": None}]
    validate_document(document)  # the platform takes it as it is: no metres, so no rule decides
    assert contract_scale({"scale_status": "uncalibrated"}) == {"status": "uncalibrated"}
    assert contract_scale({"scale_status": "device_metric", "metres_per_native_unit": 1.})["status"] == "operator_anchored"
    stated = contract_scale({"scale_status": "assumed_camera_height", "metres_per_native_unit": 1.25, "camera_height_native_median": 1.28})
    assert stated["status"] == "model_estimated" and stated["nativeToMeters"] == 1.25 and stated["anchor"]["metres"] == 1.6
    with pytest.raises(ValueError):
        contract_scale({"scale_status": "something_new"})


@needs_art
def test_import_scale_sentence_and_lens(shots):
    import import_video_scene as importer
    for site in METRIC:
        scale = json.loads((RUNS / METRIC[site] / "metric-scale.json").read_text())
        assert scale["model_estimated_metres_per_native_unit"] is None
        sentence = importer.scale_limitation(scale, device=False)
        assert "20%" not in sentence and "disagrees" not in sentence and "1.6 m carry height, not measured" in sentence, sentence  # X13
    with_estimate = {**scale, "model_estimated_metres_per_native_unit": 2.7, "height_anchor_vs_model_estimate": -.2}
    assert "disagrees with the model scale estimate by about 20%" in importer.scale_limitation(with_estimate, device=False)
    source = (REPO / "scripts/import_video_scene.py").read_text()
    assert "about 20% on this clip" not in source and "scale_limitation(scale, device)" in source
    lens = run("lens", shots["walmart"][0], shots=shots["walmart"][0], clip=CLIPS / "walmart-190", metric=RUNS / METRIC["walmart"])
    walmart = importer.lens_scale(json.loads((RUNS / METRIC["walmart"] / "metric-scale.json").read_text()), shots["walmart"][0] / "lens.json")
    assert lens["value"]["scale_status"] == walmart["scale_status"] == "intrinsics_uncertain"
    assert importer.contract_scale(walmart)["status"] == "uncalibrated" and "88.4%" in importer.scale_limitation(walmart, device=False)
    run("lens", shots["samsclub"][0], shots=shots["samsclub"][0], clip=CLIPS / "samsclub-337", metric=RUNS / METRIC["samsclub"])
    sams = json.loads((RUNS / METRIC["samsclub"] / "metric-scale.json").read_text())
    assert importer.lens_scale(sams, shots["samsclub"][0] / "lens.json") == sams, "a lens that passed changes nothing"
    with pytest.raises(AssertionError, match="another metric-scale"):
        importer.lens_scale(sams, shots["walmart"][0] / "lens.json")


def test_profiles_refuse_every_unverified_licence():
    commercial, research, delivered = (profiles.PROFILES[n] for n in ("commercial", "research", "delivered"))
    for hf_id, (revision, _, licence, allowed) in profiles.MODELS.items():
        spec = {"models": [("any", hf_id, revision, None)]}
        assert (profiles.refuse(commercial, spec) is None) == (allowed is True and revision != profiles.UNPINNED), hf_id
        assert profiles.refuse(research, spec) is None and profiles.refuse(delivered, spec) is None
    for refused in ("depth-anything/DA3-GIANT-1.1", "TRI-ML/RecGen", "facebook/map-anything", "robbyant/lingbot-map", "Ruicheng/moge-3-vitl",
                    "princeton-vl/DROID-SLAM", "depth-anything/DA3-BASE#camera"):
        assert profiles.refuse("commercial", {"models": [("r", refused, profiles.MODELS[refused][0], None)]}), refused
    assert "no pinned revision" in profiles.refuse("commercial", {"models": [("depth", "depth-anything/DA3-BASE", profiles.UNPINNED, None)]})
    assert profiles.refuse("commercial", {"models": [("x", "someone/new-model", "abc", None)]}), "an unknown model is refused"
    assert delivered.cache_only and delivered.rules == "adopted" and not research.cache_only and not commercial.cache_only
    assert commercial.namer == "qwen3vl" and not commercial.dense_map and "recgen" not in commercial.generators
    assert all(profiles.refuse(commercial, {"models": [(role, *row[:3])]}) for role, row in commercial.models.items() if role in ("camera", "register", "lens")), \
        "commercial stays blocked on the licences still to verify (DROID, MoGe-3 weights, DA3-BASE camera head)"
    assert profiles.QWEN3VL_NAMING_GATE["status"] == "not-run"


def test_profile_pins_equal_the_modal_apps():
    import mono_room, moge3_app, sam2_everything, sam3_app, sam3_video, lingbot_room, droid_room, sam3d_research
    pinned = {"depth-anything/DA3-GIANT-1.1": mono_room.DA3_REVISIONS["depth-anything/DA3-GIANT-1.1"], "depth-anything/DA3-BASE": mono_room.DA3_REVISIONS["depth-anything/DA3-BASE"],
              "Ruicheng/moge-3-vitl": moge3_app.REVISION, "facebook/sam2.1-hiera-large": sam2_everything.REVISION, "facebook/sam3": sam3_app.REVISION,
              "facebook/sam3.1": sam3_video.MODEL_REVISION, "robbyant/lingbot-map": lingbot_room.WEIGHTS_REV, "princeton-vl/DROID-SLAM": droid_room.REV,
              "facebook/sam-3d-objects": sam3d_research.MODEL_REVISION}
    assert {k: profiles.MODELS[k][0] for k in pinned} == pinned
    assert sam3_app.REVISION == sam3_video.SAM3_REVISION and profiles.MODELS["robbyant/lingbot-map"][1] == lingbot_room.WEIGHTS_SHA


@needs_art
def test_qwen3vl_namer_leaves_names_blank_until_its_gate_passes(tmp_path):
    out = tmp_path / "named"
    done = subprocess.run([sys.executable, str(REPO / "scripts/name_video_entities.py"), "--object-map", str(RUNS / "me340-object-map-195"),
                           "--masks", str(RUNS / "me340-masks-194"), "--output", str(out), "--namer", "qwen3vl"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    named = json.loads((out / "object-map.json").read_text())
    assert named["namer"] == {"namer": "qwen3vl", "gate": profiles.QWEN3VL_NAMING_GATE}
    assert not list(out.glob("request-*")) and json.loads((out / "names.json").read_text()) == {}, "no request was made"
    agnostic = [e for e in named["entities"] if e.get("labelSource") == "class-agnostic segment, not named"]
    assert agnostic and all(e["label"] == "unnamed surface" for e in agnostic)
