"""r5 (models): the model bench's hand-off, dumped by a report run (options {"fixture_dump": "<name>"}) after every layer is written
(not analysis time). X7's fixture (fixture.npz + fixture.json: per shot keys, depth, cameras, K, person; per first-pass
object its SAM 3 logits on every keyframe it was seen on; RGB) plus what the cards were built from (every object's lifted
points, the shots' floor inputs) and the final cards, so the bench rebuilds each card's own points (cards.prepare).
RGB: every keyframe of every shot (BGR as decoded).

    python -m fast_report.fixture --self-check
"""
import json
from pathlib import Path

import numpy as np

SHOT_KEYS = ("normal", "point_m", "mpu", "u_floor_m", "scale_status", "plumb_deg", "plumb_walls")


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def dump(folder, report_id, objs, geo, frames, fps, cards, shots_in, objects, points):
    """objs: sam3d_inputs() rows (masks_lr); geo: the core's shots; frames: every decoded frame (BGR); cards: the final cards;
    shots_in: cards.build's shots; objects / points: the objects rows and their lifted points (points_v3 order)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    arrays, rows = {}, []
    by_id = {o["id"]: o for o in objects}
    for i, o in enumerate(objs):
        fr = sorted(int(f) for f in o["masks_lr"])
        arrays[f"o{i}_frames"] = np.array(fr, np.int32)
        arrays[f"o{i}_logits"] = np.stack([_np(o["masks_lr"][f]).astype(np.float16) for f in fr]) if fr else np.zeros((0, 288, 288), np.float16)
        rows.append({k: o[k] for k in ("id", "shot", "word", "box_min_m", "box_max_m")} | {"centroid_m": by_id.get(o["id"], {}).get("centroid_m")})
    for s in geo:
        i = s["index"]
        arrays.update({f"s{i}_keys": np.array(s["keys"], np.int32), f"s{i}_depth": _np(s["depth_m"]).astype(np.float16),
                       f"s{i}_c2w": _np(s["c2w_m"]).astype(np.float64), f"s{i}_K": _np(s["K"]).astype(np.float32),
                       f"s{i}_person": np.packbits(_np(s["person"]).astype(bool), axis=-1)})
    for j, p in enumerate(points):
        arrays[f"p{j}_world"] = np.asarray(p["world"], np.float32)
        arrays[f"p{j}_frame"] = np.asarray(p["frame"], np.int16)
    used = sorted({int(k) for s in geo for k in s["keys"]})
    arrays["frame_ids"] = np.array(used, np.int32)
    arrays["frames"] = np.stack([frames[f] for f in used])
    np.savez(folder / "fixture.npz", **arrays)
    shots = [{"index": s["index"], **{k: s.get(k) for k in SHOT_KEYS}, "sharp": _np(s["sharp"]).tolist() if s.get("sharp") is not None else None}
             for s in shots_in or []]
    meta = {"report": report_id, "n_frames": len(frames), "fps": fps, "wh": list(frames[0].shape[1::-1]), "objects": rows,
            "shots": [{"index": s["index"], "person_width": int(_np(s["person"]).shape[-1])} for s in geo], "cards_shots": shots,
            "points_ids": [o["id"] for o in objects[:len(points)]], "cards": cards,
            "note": "RGB of every keyframe, BGR as decoded; person masks bit-packed on the last axis"}
    (folder / "fixture.json").write_text(json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return {"folder": str(folder), "objects": len(rows), "cards": len(cards), "keyframes": len(used), "bytes": (folder / "fixture.npz").stat().st_size}


def load(folder, light=False):
    """-> {meta, z (the npz, lazily), shots (sam3d.stage's), objs (masks_lr)}; frames stay in z['frames']. light: meta and z only
    (a worker reading cards' points: every object's logits would take ~1 GB a process)."""
    folder = Path(folder)
    meta = json.loads((folder / "fixture.json").read_text())
    z = np.load(folder / "fixture.npz")
    if light:
        return {"meta": meta, "z": z}
    shots = []
    for s in meta["shots"]:
        i = s["index"]
        shots.append({"index": i, "keys": z[f"s{i}_keys"].tolist(), "depth_m": z[f"s{i}_depth"].astype(np.float32), "c2w_m": z[f"s{i}_c2w"],
                      "K": z[f"s{i}_K"], "person": np.unpackbits(z[f"s{i}_person"], axis=-1)[..., :s["person_width"]].astype(bool)})
    objs = [{**o, "masks_lr": {int(f): lg for f, lg in zip(z[f"o{i}_frames"], z[f"o{i}_logits"])}} for i, o in enumerate(meta["objects"])]
    return {"meta": meta, "z": z, "shots": shots, "objs": objs}


def card_points(fx, card):
    """A card's joined lifted points (world, local keyframe): its own object's and every absorbed one's (cards_chunk's join)."""
    at = {oid: j for j, oid in enumerate(fx["meta"]["points_ids"])}
    js = [at[i] for i in [card["id"], *card["physical"].get("merged_from", [])] if i in at]
    if not js:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int16)
    return (np.concatenate([fx["z"][f"p{j}_world"] for j in js]), np.concatenate([fx["z"][f"p{j}_frame"] for j in js]))


def self_check():
    import tempfile
    rng = np.random.default_rng(0)
    geo = [{"index": 0, "keys": [3, 9], "depth_m": rng.random((2, 4, 6)), "c2w_m": np.repeat(np.eye(4)[None], 2, 0), "K": np.repeat(np.eye(3)[None], 2, 0),
            "person": rng.random((2, 4, 6)) > .5}]
    objs = [{"id": "obj-0-1", "shot": 0, "word": "box", "box_min_m": [0, 0, 0], "box_max_m": [1, 1, 1], "masks_lr": {9: rng.random((288, 288)), 3: rng.random((288, 288))}}]
    objects = [{"id": "obj-0-1", "centroid_m": [.5, .5, .5]}, {"id": "obj-0-2"}]
    points = [{"world": rng.random((5, 3)), "frame": np.array([0, 1, 1, 0, 1])}, {"world": rng.random((2, 3)), "frame": np.array([1, 1])}]
    frames = [np.full((4, 6, 3), f, np.uint8) for f in range(12)]
    card = {"id": "obj-0-1", "physical": {"merged_from": ["obj-0-2"]}}
    with tempfile.TemporaryDirectory() as d:
        info = dump(d, "rep", objs, geo, frames, 5., [card], [{"index": 0, "normal": [0, 0, 1], "mpu": 1.}], objects, points)
        fx = load(d)
        assert info["keyframes"] == 2 and fx["z"]["frames"][1][0, 0, 0] == 9
        assert np.array_equal(fx["shots"][0]["person"], geo[0]["person"]) and fx["shots"][0]["keys"] == [3, 9]
        assert sorted(fx["objs"][0]["masks_lr"]) == [3, 9] and fx["meta"]["objects"][0]["centroid_m"] == [.5, .5, .5]
        w, f = card_points(load(d, light=True), card)
        assert len(w) == 7 and list(f) == [0, 1, 1, 0, 1, 1, 1]
    print("fixture self-check ok: dump / load round trip, packed person masks, a card's merged points")


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
