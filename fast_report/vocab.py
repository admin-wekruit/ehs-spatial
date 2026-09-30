"""r5b/vocab: SAM 3's wave-2 words without a VLM (the user, 2026-09-29: the VLM comes last and rarely; round 4's review: one
Qwen request set the words that found 77-90 % of the objects). A source scores words per frame on FRAMES evenly spaced frames;
rank() keeps the words with evidence across the video (at most PER_CLASS a canonical class, at most N_WORDS), never a STOP
word. Sources compared in r5b (scripts/r5b_vocab.py; modal_apps/r5b_vocab.py measures them off the pipeline):
  'taxonomy'  the fixed canonical classes (cards.TAXONOMY) + the frozen bank's free labels of the site's family (no model)
  'pe'        PE-Core-L zero-shot (the naming cascade's encoder, already resident) over WORDS: LVIS + Objects365 names + the
              taxonomy's classes and aliases + WORKSHOP + RETAIL, on square tiles of each frame (Apache-2.0 weights)
  'yoloe-pf'  YOLOE-26L prompt-free (its built-in 4585-tag vocabulary; AGPL-3.0: internal profile only), offline
  'ram'       RAM++ (Recognize Anything Plus, Apache-2.0 code and weights) image tags on the frames, kept when they are object
              nouns of the 'pe' word list; in the pipeline one RAM++ process in its own venv (/opt/ram, the package's pins) on GPU 1
  'qwen'      the round-4 Qwen scene vocabulary (vlm.vocab): opt-in comparison only
r5b's comparison (runs/r5b-vocab-results/compare.md) chose 'ram' by the rule written before its results were read (selection-rule.md):
the most held-out items typed right over the three videos; 'pe' found ME340's workshop words but missed Walmart's shoes and boxes.
"""
import numpy as np

from fast_report import cards

FRAMES, N_WORDS, PER_CLASS, MIN_SCORE, TILE_TOP = 16, 40, 2, .5, 5
DEFAULT = "qwen+ram"  # r5b integrate: the user accepted ONE Qwen scene-vocabulary request a video: RAM++'s words, then Qwen's (the
# union of the two best sources, runs/r5b-results/summary.md's comparison); without the VLM, RAM++'s alone. r5b/vocab's pick was 'ram'
N_UNION = 60  # the union's words at most (SAM 3's wave 2 is warmed at 58 words)
RAM_PY, RAM_HF = "/opt/ram/bin/python", "/v/models/hf-r5b"  # the RAM++ venv; its weights and BERT tokenizer (modal_apps/r5b_vocab.py cached them)
RAM_REPO, RAM_CKPT = "xinyu1205/recognize-anything-plus-model", "ram_plus_swin_large_14m.pth"
TEXT = "/v/layers/label-bank/r5b/pe-text-173c4bb37cb3.npz"  # modal_apps/r5b_vocab.py: the word list's PE-Core text (x13's templates)
RULE = (f"PE-Core-L zero-shot over the word list on {FRAMES} evenly spaced frames x 29 square tiles (3 scales); a word's score is the sum over "
        f"frames of its best tile's softmax p (a tile's top {TILE_TOP}); kept: score >= {MIN_SCORE}, <= {PER_CLASS} words a canonical "
        f"class, <= {N_WORDS} words, never a STOP or floor word")
FLOOR_CLASSES = ("spill", "floor marking")  # floor states and paint: wave 1's 'spill' covers the first; a shiny floor read as 'wet floor
# machine-shop / shop-floor things the taxonomy's names miss (r5b, written before any r5b run; round 2's held-out labels named
# tool holders, angle plates, locker doors, a pedestal fan, a vise, machine covers)
WORKSHOP = ("milling machine", "lathe", "drill press", "band saw", "surface grinder", "cnc machining center", "machine vise", "angle plate",
            "clamping kit", "step block", "t slot bolt", "machine tool holder", "end mill holder", "collet chuck", "drill chuck", "end mill",
            "drill bit", "tap wrench", "hex key set", "tool cabinet", "roll cabinet", "tool chest", "pegboard", "tool rack", "workbench",
            "locker", "cabinet door", "drawer", "air hose", "extension cord", "cable reel", "power strip", "shop vacuum", "chip brush",
            "coolant tank", "chip pan", "dial indicator", "height gauge", "caliper", "micrometer", "pedestal fan", "ceiling light fixture",
            "fluorescent light", "roll up door", "door track", "emergency stop button", "fire extinguisher", "first aid kit", "trash can",
            "parts bin", "storage bin", "shop towel", "safety cabinet", "welding table", "welder", "grinder", "belt sander", "air compressor",
            "hydraulic press", "arbor press", "metal bar stock", "sheet metal", "metal block", "clamp", "bench vise", "machine guard",
            "control panel", "digital readout", "electrical panel", "conduit", "pipe", "duct", "cable tray", "column", "i beam", "ladder",
            "step stool", "cart", "tool box", "tool tray", "wooden board", "pallet", "cardboard box", "plastic bin", "bucket", "spray bottle",
            "oil can", "safety glasses", "glove", "hard hat", "whiteboard", "clipboard", "computer monitor", "keyboard", "chair", "stool")
RETAIL = ("store shelf", "gondola shelf", "display rack", "endcap display", "price tag", "shelf label", "shopping cart", "shopping basket",
          "pallet of goods", "shrink wrapped pallet", "stacked boxes", "cardboard box", "shoe box", "case of water bottles", "water bottle",
          "paper towels", "toilet paper", "tissue box", "detergent bottle", "cereal box", "snack bag", "canned food", "egg carton",
          "plastic crate", "pallet jack", "hand truck", "stocking cart", "shoe", "sneaker", "boot", "sandal", "slipper", "handbag",
          "backpack", "clothing rack", "hanger", "folded clothes", "mannequin", "checkout counter", "cash register", "freezer case",
          "refrigerated display case", "banner", "aisle sign", "hanging sign", "stroller", "car seat", "toy", "stuffed animal", "pillow")
STOP = frozenset((
    # people and bodies: the people layer's, never an object word
    "person", "people", "man", "woman", "men", "women", "boy", "girl", "child", "kid", "baby", "adult", "customer", "shopper", "worker",
    "employee", "staff", "clerk", "cashier", "student", "crowd", "family", "couple", "hand", "arm", "face", "head", "leg", "foot", "finger",
    "hair", "body", "pedestrian", "shirt", "polo shirt", "t shirt", "sweatshirt", "hoodie", "jeans", "trousers", "pants", "uniform",
    # places and scenes: not a thing
    "store", "shop", "supermarket", "grocery", "grocery store", "market", "warehouse", "factory", "workshop", "garage", "plant", "room",
    "aisle", "hallway", "corridor", "interior", "indoor", "indoors", "building", "office", "mall", "shopping mall", "retail", "industry",
    "industrial", "machine shop", "laboratory", "lab", "classroom", "department store", "retail store", "shopping", "showroom", "scene",
    "interior design", "home improvement store", "convenience store", "discount store", "hardware store", "shoe store", "stockroom",
    # surfaces (the floor and walls are the room's, never objects)
    "floor", "flooring", "wall", "ceiling", "ground", "tile", "tile floor", "roof", "concrete", "pavement", "road", "hardwood", "wood floor",
    "laminate", "carpet", "rug",
    # attributes, abstractions, the frame itself
    "sale", "white", "black", "red", "blue", "green", "yellow", "orange", "gray", "grey", "brown", "pink", "purple", "silver", "gold",
    "metal", "steel", "wood", "wooden", "plastic", "glass", "aluminum", "iron", "color", "colour", "shadow", "reflection", "text", "logo",
    "number", "letter", "image", "photo", "picture", "view", "stuff", "thing", "object", "item", "variety", "assortment", "row", "line",
    "stack", "pile", "group", "lot", "many", "full", "empty", "open", "stand"))


def clean(name):
    """An LVIS / Objects365 / tagger name -> plain lower-case words ('bow_(weapon)' -> 'bow', 'Cabinet/shelf' -> ['cabinet',
    'shelf'] via split('/') by the caller)."""
    n = str(name).lower().replace("_", " ").split("(")[0]
    return " ".join(n.replace("-", " ").split())


def usable(word):
    """A word SAM 3 may run: not a STOP word, not a non-object class, 1-4 words."""
    w = clean(word)
    return bool(w) and w not in STOP and cards.norm(w) not in STOP and cards.canonical(w) not in (cards.NOT_OBJECT, *FLOOR_CLASSES) \
        and len(w.split()) <= 4


def candidates(*lists):
    """The 'pe' word list: the taxonomy's classes and aliases, WORKSHOP, RETAIL, then the given lists (LVIS, Objects365 ...),
    cleaned, usable only, deduplicated (by the singular form) in that order."""
    seen, out = set(), []
    tax = [w for fam in cards.TAXONOMY.values() for c, ws in fam.items() for w in (c, *ws)]
    for w in [*tax, *WORKSHOP, *RETAIL, *[x for lst in lists for y in lst for x in str(y).split("/")]]:
        w = clean(w)
        k = cards.norm(w)
        if k not in seen and usable(w):
            seen.add(k)
            out.append(w)
    return out


def rank(per_frame, n=N_WORDS, min_score=MIN_SCORE, per_class=PER_CLASS, skip=()):
    """[{word: score in [0, 1]} per frame] -> [(word, video score)]: the score summed over frames (evidence across the video), words
    with at least min_score, best first, at most per_class words of one canonical class (a free word is its own class), at most n;
    STOP words and `skip` (wave 1's words) never."""
    tot = {}
    for d in per_frame:
        for w, s in d.items():
            w = clean(w)
            if usable(w) and w not in skip:
                tot[w] = tot.get(w, 0.) + float(s)
    out, per = [], {}
    for w, s in sorted(tot.items(), key=lambda x: (-x[1], x[0])):
        if s < min_score or len(out) >= n:
            break
        c = cards.canonical(w) or w
        if per.get(c, 0) >= per_class:
            continue
        per[c] = per.get(c, 0) + 1
        out.append((w, round(s, 3)))
    return out


def taxonomy_words(bank_meta=(), family=None, exclude=(), min_rows=3):
    """Option 'taxonomy': every canonical class, then the frozen bank's free labels ('~head', no class) with >= min_rows rows of
    this family from sites not in `exclude` (the bank's own names, most rows first)."""
    words = [c for fam in cards.TAXONOMY.values() for c in fam if usable(c)]
    count = {}
    for m in bank_meta:
        if m.get("family") == family and m.get("site") not in exclude and str(m.get("label", "")).startswith("~"):
            count[m["label"][1:]] = count.get(m["label"][1:], 0) + 1
    return words + [w for w, k in sorted(count.items(), key=lambda x: (-x[1], x[0])) if k >= min_rows and usable(w) and w not in words]


def load(path, dev):
    """The 'pe' source's text (modal_apps/r5b_vocab.py) -> {words, text (torch fp16 on dev), scale, file, sha256}."""
    import hashlib
    import io
    import torch
    from pathlib import Path
    raw = Path(path).read_bytes()
    z = np.load(io.BytesIO(raw))
    return {"words": [str(w) for w in z["words"]], "text": torch.from_numpy(z["word_text"].astype(np.float16)).to(dev), "scale": float(z["scale"]),
            "file": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def union(first, then, n=N_WORDS):
    """Two ranked word lists [(word, score)] -> one: `first`'s words, then `then`'s not already in (by the singular form), <= n."""
    seen, out = set(), []
    for w, s in [*first, *then]:
        if cards.norm(w) not in seen and len(out) < n:
            seen.add(cards.norm(w))
            out.append((w, s))
    return out


def object_tags(per_frame, words):
    """Tagger output [{tag: p}] -> only the tags that are object nouns (the 'pe' word list: the taxonomy, LVIS, Objects365,
    WORKSHOP, RETAIL): a tagger's scene and activity tags ('store', 'job', 'fill', 'courtyard') are never SAM 3 words."""
    objs = {cards.norm(clean(w)) for w in words}
    return [{t: p for t, p in d.items() if cards.norm(clean(t)) in objs} for d in per_frame]


def ram_worker():
    """The RAM++ process (its own venv, RAM_PY): boot loads RAM++ swin-L on its GPU (weights and tokenizer from RAM_HF, offline) and
    tags a noise frame; then each {frames: [(h, w, 3) uint8 BGR]} -> {per_frame: [{tag: sigmoid p}] (the tags over RAM++'s own
    per-class thresholds), s}. modal_apps/r5b_vocab.ram measured the same code off the pipeline."""
    from fast_report import sam3d
    st = {}

    def boot():
        import time as _t
        import torch
        from huggingface_hub import hf_hub_download
        from ram import get_transform
        from ram.models import ram_plus
        t = _t.perf_counter()
        model = ram_plus(pretrained=hf_hub_download(RAM_REPO, RAM_CKPT), image_size=384, vit="swin_l").eval().to("cuda")
        got = []
        model.fc.register_forward_hook(lambda m, i, o: got.append(o.detach()))
        st.update(model=model, got=got, tf=get_transform(image_size=384), torch=torch, thr=model.class_threshold.cpu().numpy(),
                  drop=set(np.asarray(getattr(model, "delete_tag_index", []), int).tolist()), tags=[str(x) for x in model.tag_list])
        handle({"frames": [np.zeros((720, 1280, 3), np.uint8)] * 2}, None)
        return {"load_s": round(_t.perf_counter() - t, 2), "tags": len(st["tags"]), "gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)}

    def handle(msg, _):
        import time as _t
        from PIL import Image
        torch, t = st["torch"], _t.perf_counter()
        x = torch.stack([st["tf"](Image.fromarray(np.ascontiguousarray(f[..., ::-1]))) for f in msg["frames"]]).to("cuda")
        st["got"].clear()
        with torch.inference_mode():
            st["model"].generate_tag(x)
        p = torch.sigmoid(st["got"][-1].squeeze(-1).float()).cpu().numpy()
        torch.cuda.empty_cache()  # GPU 1's other tenants (vLLM, SAM 3, the splat) get the activations back
        per = [{st["tags"][j]: round(float(q[j]), 4) for j in np.flatnonzero(q > st["thr"]) if j not in st["drop"]} for q in p]
        return {"per_frame": per, "s": round(_t.perf_counter() - t, 3)}
    sam3d.serve(handle, boot)


class Ram:
    """The pipeline's RAM++ process on one GPU (sam3d.Pool, one worker); tags(frames) -> {per_frame, s}."""

    def __init__(self, gpu):
        from fast_report import sam3d
        env = sam3d.worker_env(RAM_PY, CUDA_VISIBLE_DEVICES=gpu, HF_HOME=RAM_HF, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)
        self.pool = sam3d.Pool([RAM_PY, "-c", "from fast_report.vocab import ram_worker; ram_worker()"], 1, env, "ram")

    def ready(self, timeout=None):
        return self.pool.ready(timeout)[0]

    def tags(self, frames, timeout=60):
        return self.pool.submit({"frames": list(frames)}).result(timeout)

    def close(self):
        self.pool.close()


def with_ram(image):
    """/opt/ram: recognize-anything's pins (the versions modal_apps/r5b_vocab.ram_image ran), uv's Python 3.10."""
    py = RAM_PY
    return image.run_commands("python -m pip install uv==0.8.22", "uv venv --python 3.10 /opt/ram",
                              f"uv pip install --python {py} torch==2.0.1 torchvision==0.15.2 timm==0.4.12 transformers==4.25.1 fairscale==0.4.4 "
                              "pillow 'numpy<2' huggingface_hub==0.16.4 scipy opencv-python-headless",
                              f"uv pip install --python {py} --no-deps git+https://github.com/xinyu1205/recognize-anything.git")


def pick_frames(n_total, k=FRAMES):
    return sorted({int((i + .5) * n_total / k) for i in range(k)})


def tiles(h, w):
    """Square crops (x0, y0, side) of an h x w frame at three scales, so a small thing fills a tile at one of them: full-height
    squares across it (3), half-height (4 across x 2 down) and third-height (6 x 3): 29 crops for 1280 x 720."""
    out = []
    for side, rows in ((h, 1), (h // 2, 2), (h // 3, 3)):
        across = max(1, int(np.ceil(w / side)))
        xs = np.linspace(0, w - side, across + (1 if side == h else 0)).round().astype(int) if w > side else [0]
        for r in range(rows):
            out += [(int(x), r * side, side) for x in xs]
    return out


def pe_scores(enc, frames_bgr, text, words, scale, top=TILE_TOP):
    """PE-Core-L zero-shot on each frame's tiles -> [{word: max over the frame's tiles of its softmax p}] (a tile's top `top` words
    only). frames_bgr: [(h, w, 3) uint8]; text: (W, D) unit torch tensor on enc.dev; scale: the model's logit scale."""
    import torch
    import torch.nn.functional as F
    from fast_report.cascade import PE_SIDE
    crops, owner = [], []
    for i, f in enumerate(frames_bgr):
        t = torch.from_numpy(np.ascontiguousarray(f[..., ::-1])).to(enc.dev)
        for x, y, s in tiles(*f.shape[:2]):
            crops.append(F.interpolate(t[y:y + s, x:x + s].permute(2, 0, 1)[None].float() / 255, size=(PE_SIDE, PE_SIDE), mode="bilinear",
                                       antialias=True, align_corners=False)[0])
            owner.append(i)
    out = [{} for _ in frames_bgr]
    for b in range(0, len(crops), 32):  # 32 a batch: its activations stay in GPU 1's cache (64: +4 GiB there, run r5b-vocab-final-001)
        e = enc.pe_embed(torch.stack(crops[b:b + 32]))
        p = (scale * e @ text.float().T).softmax(-1)
        v, j = p.topk(top, dim=-1)
        for k, (vv, jj) in enumerate(zip(v.cpu().numpy(), j.cpu().numpy())):
            d = out[owner[b + k]]
            for s, wi in zip(vv, jj):
                d[words[wi]] = max(d.get(words[wi], 0.), float(s))
    return out


def self_check():
    assert clean("bow_(weapon)") == "bow" and clean("Cabinet") == "cabinet" and not usable("floor") and not usable("person") and usable("vise")
    assert not usable("Man") and not usable("supermarket") and usable("fire extinguisher") and not usable("a very long name of a thing")
    assert not usable("wet floor") and not usable("floor line") and usable("pallet jack"), "floor states and paint are never words"
    c = candidates(["Cabinet/shelf", "Person", "zebra_(animal)", "vise"])
    assert "cabinet" in c and "shelf" in c and "zebra" in c and "person" not in c and len(c) == len({cards.norm(w) for w in c})
    assert c.index("shelf") < c.index("angle plate") < c.index("zebra"), "taxonomy first, then WORKSHOP/RETAIL, then the lists"
    frames = [{"cardboard box": .9, "carton": .8, "box": .7, "floor": .9, "man": .8}, {"cardboard box": .6, "shelf": .4, "vise": .05},
              {"shelf": .3, "fan": .2}]
    got = rank(frames, n=3)
    assert [w for w, _ in got] == ["cardboard box", "carton", "shelf"], got  # box: 2 words of one class at most; STOP words never
    assert rank(frames, n=10, min_score=.5, skip=("shelf",)) == [("cardboard box", 1.5), ("carton", .8)]
    tw = taxonomy_words([{"family": "retail", "site": "a", "label": "~stroller"}] * 3 + [{"family": "retail", "site": "b", "label": "~toy"}] * 3,
                        "retail", ["b"])
    assert "vise" in tw and tw[-1] == "stroller" and "toy" not in tw, tw[-3:]  # the scored site's own rows never
    t = tiles(720, 1280)
    assert len(t) == 29 and all(x + s <= 1280 and y + s <= 720 for x, y, s in t) and t[0] == (0, 0, 720) and t[2][0] == 560, t
    assert len(tiles(480, 640)) == 3 + 3 * 2 + 4 * 3, "a 4:3 frame: 480, 240 and 160 px squares"
    assert pick_frames(900, 4) == [112, 337, 562, 787]
    assert object_tags([{"store": .9, "shoe": .8, "Job": .7, "shopping cart": .6}], ["shoe", "shopping cart", "box"]) == [{"shoe": .8, "shopping cart": .6}]
    assert union([("shoe", 3.), ("box", 2.)], [("shoes", 5.), ("vise", 1.)], n=3) == [("shoe", 3.), ("box", 2.), ("vise", 1.)]
    print("vocab self-check ok: STOP words, candidate list order, rank (evidence, per-class cap), taxonomy + bank labels, tiles")


if __name__ == "__main__":
    self_check()
