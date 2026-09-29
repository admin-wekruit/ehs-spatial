"""r4/coverage: extra object instances where SAM 3's words found nothing. A detector's boxes on every 5 fps keyframe ->
SAM 3's tracker (the box-prompt head of the checkpoint already resident) on the SAM 3 pass's own backbone features (HF
Sam3VideoModel's sharing: backbone last_hidden_state -> the tracker's neck; no second image-encoder pass) -> masks that
join densify's pool: a mask mostly on pixels a SAM 3 mask already holds is dropped (its word stays a vote), the rest join an
existing object by the lift's voxel rule or are lifted into new objects (>= 2 keyframes). On the pick maps they only fill
pixels no entity holds: every SAM 3 pick stays as it was.

No VLM anywhere here: a box object's name is the detector's word ('detected word, unverified', like SAM 3's), or
'unidentified object' when the detector is not sure; the physical values come from the lift like every card's.

Box sources compared (r4 probe; licences checked on the model pages 2026-09-29):
  yoloe    YOLOE-26L-seg text prompts = the canonical taxonomy's classes      ultralytics 8.4: AGPL-3.0 (code and weights)
  yoloepf  YOLOE-26L-seg prompt-free (its own 4585-word list)                 AGPL-3.0
  yolo11   YOLO11x, COCO's 80 classes -> taxonomy                             AGPL-3.0
  yolo26   YOLO26x, COCO's 80 classes -> taxonomy                             AGPL-3.0
  owlv2    OWLv2 large ensemble: objectness + taxonomy words (X10's route)    Apache-2.0
  SAM 3 tracker (box prompts)                                                 SAM License (commercial use with conditions)

    python -m fast_report.coverage --self-check
"""
import numpy as np

from fast_report import cards

LICENCES = {"yoloe": "AGPL-3.0 (ultralytics YOLOE-26; the user accepts it for now)", "yoloepf": "AGPL-3.0 (ultralytics YOLOE-26 prompt-free)",
            "yolo11": "AGPL-3.0 (ultralytics YOLO11)", "yolo26": "AGPL-3.0 (ultralytics YOLO26)", "owlv2": "Apache-2.0 (google/owlv2-large-patch14-ensemble)",
            "sam3 tracker": "SAM License (Meta; commercial use allowed with conditions)"}
WEIGHTS = {"yoloe": "yoloe-26l-seg.pt", "yoloepf": "yoloe-26l-seg-pf.pt", "yolo11": "yolo11x.pt", "yolo26": "yolo26x.pt"}
OWL_MODEL = "google/owlv2-large-patch14-ensemble"
WDIR = "/v/models/ultralytics"
# the taxonomy's classes a box may carry: every canonical class but the flat marks and building surfaces (the floor rule and the
# background decide those) and people (tracked apart)
NOT_BOXED = {"floor marking", "wall panel", "spill"}
TAXO = sorted(c for fam in cards.TAXONOMY.values() for c in fam if c not in NOT_BOXED)
COCO_TO = {"bicycle": "bicycle", "car": "car", "motorcycle": "motorcycle", "truck": "truck", "bench": "bench", "backpack": "bag",
           "umbrella": "umbrella", "handbag": "bag", "suitcase": "bag", "sports ball": "ball", "skateboard": "skateboard", "bottle": "bottle",
           "wine glass": "cup", "cup": "cup", "bowl": "bowl", "chair": "chair", "couch": "couch", "potted plant": "potted plant", "bed": "bed",
           "dining table": "table", "toilet": "toilet", "tv": "monitor", "laptop": "computer", "mouse": "computer mouse", "remote": "remote",
           "keyboard": "keyboard", "cell phone": "phone", "microwave": "microwave", "oven": "oven", "toaster": "toaster", "sink": "sink",
           "refrigerator": "refrigerator", "book": "book", "clock": "clock", "vase": "vase", "scissors": "scissors", "teddy bear": "toy",
           "hair drier": "hair dryer", "toothbrush": "toothbrush", "fire hydrant": "fire hydrant", "stop sign": "sign", "parking meter": "meter",
           "traffic light": "traffic light", "kite": "toy", "frisbee": "toy", "banana": "food", "apple": "food", "orange": "food",
           "sandwich": "food", "broccoli": "food", "carrot": "food", "hot dog": "food", "pizza": "food", "donut": "food", "cake": "food",
           "knife": "knife", "fork": "cutlery", "spoon": "cutlery", "baseball bat": "bat", "baseball glove": "glove", "tie": "clothing",
           "surfboard": "board", "skis": "skis", "snowboard": "board", "tennis racket": "racket", "airplane": "toy", "bus": "vehicle",
           "train": "vehicle", "boat": "boat"}  # person and animals: not boxed (people are tracked apart)
MIN_BOX_PX, MAX_BOX_SHARE = 24 * 24, .6  # source px: smaller boxes are below the lift's area floor; a box over 60 % of the frame is a scene
MAX_MASK_SHARE = .4  # a mask over 40 % of the frame is a region (the probe's gridwall-with-slippers mask: 49 %; no dev click changed), not one thing
# no 'SAM 3 has it' gate (None): the dev misses lie on SAM 3 masks that never became objects. r4 dev run 001 (Walmart), a box mask
# dropped when > 50 % of it lay on kept SAM 3 masks of its keyframe: 8341 of 10149 dropped, 4 of 42 dev misses opened; run 002 at
# 90 %: 20 of 42 opened on Walmart, 5 of 40 on Sam's Club, where 22 of the 40 misses had a box mask dropped as a near-copy of
# SAM 3's (unlifted) masks. A copy of a mask that did become an object joins that object and paints nothing new (fill)
CLAIMED_MAX = None
NMS_IOU = .7  # OWLv2 has no NMS: neighbouring patches box the same thing (r4 dev run 001: 81 boxes a keyframe)
ON_FLOOR_MAX, ON_PEOPLE_MAX = .5, .3  # a box mask mostly on SAM 3's floor (not a flat class) is the floor; on a person, the person
FLAT = ("mat", "rug", "cable", "hose", "drain", "pallet", "board", "sheet", "tape", "cord")  # classes that lie on the floor and stay
NAME_MIN = {"yoloe": .3, "yolo11": .3, "yolo26": .3, "owlv2": .2, "yoloepf": .3}  # a word below this is no word: 'unidentified object'
# box floor per source (objectness for OWLv2): set on the dev clicks of rounds 2 and 3 (seeds 2 and 61, runs/r4-coverage-probe-002): at
# 0.1 OWLv2's masks lie under 52 of 96 missed clicks and 9 of 117 background clicks, at 0.15 under 45 and 4
SCORE = {"yoloe": .15, "yolo11": .25, "yolo26": .25, "owlv2": .1, "yoloepf": .15}
RULE = ("OWLv2 objectness >= 0.1 (dev clicks of rounds 2 and 3, seeds 2 and 61), boxes >= 24 x 24 px and <= 60 % of the frame; NMS at IoU 0.7, a SAM 3 "
        "tracker mask per box, kept when <= 40 % of the frame (on SAM 3's own masks too: those never lifted are the misses), <= 50 % on SAM 3's "
        "floor (flat classes excepted) and <= 30 % on people; it joins densify's pool (joins an object: pick pixels only, no points; "
        "else lifted, >= 2 keyframes); on the pick maps it fills only pixels no entity holds; its word only at class score >= 0.2")


def word_of(src, name):
    """A detector's class name -> the word it votes with: COCO's through COCO_TO (person / animals: None), the rest as given."""
    if src in ("yolo11", "yolo26"):
        return COCO_TO.get(name)
    return name


def keep_box(xyxy, hw, word, agnostic=False):
    """Size and class gate on one box (source px); a class-agnostic source (OWLv2's objectness) may carry no word."""
    w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
    return (agnostic or word is not None) and w * h >= MIN_BOX_PX and w * h <= MAX_BOX_SHARE * hw[0] * hw[1]


def is_flat(word):
    return bool(word) and cards.head_match(word, FLAT) is not None


def keep_mask(area, claimed, on_floor, on_people, word, frame_px=None):
    """The mask gates (pixel counts on the same grid) -> (kept, reason)."""
    if area <= 0:
        return False, "empty"
    if frame_px and area > MAX_MASK_SHARE * frame_px:
        return False, "a region"
    if CLAIMED_MAX is not None and claimed > CLAIMED_MAX * area:
        return False, "sam3 has it"
    if on_people > ON_PEOPLE_MAX * area:
        return False, "on a person"
    if on_floor > ON_FLOOR_MAX * area and not is_flat(word):
        return False, "floor"
    return True, "kept"


def fill(lab, masks, ids):
    """Gap-fill painting: lab (h, w) int map (0 = nothing), masks (n, h, w) bool, ids (n,) the entity code of each (0 = none).
    Masks paint only where lab is 0, the smaller mask on top (segment.paint's rule). -> the new map (a copy)."""
    out = lab.copy()
    free = out == 0
    for i in np.argsort([-int(m.sum()) for m in masks], kind="stable"):  # larger first, the smaller overwrites
        if ids[i]:
            out[masks[i] & free] = ids[i]
    return out


class Boxer:
    """SAM 3's tracker (Sam3TrackerModel from the same checkpoint) with its backbone dropped: image embeddings from the SAM 3
    detector pass's backbone output (the tracker's own neck on it), boxes -> one mask each (multimask off)."""

    def __init__(self, dev, model_id, revision, cache_dir, keep_backbone=False):
        import torch
        from transformers import Sam3TrackerModel
        self.dev = dev
        self.model = Sam3TrackerModel.from_pretrained(model_id, revision=revision, cache_dir=cache_dir, torch_dtype=torch.bfloat16).eval()
        if not keep_backbone:  # the detector pass computes it: ~0.9 GB less on this GPU (dropped before the move)
            self.model.vision_encoder.backbone = None
        self.model.to(dev)
        self.pos = None

    def embed(self, hidden):
        """hidden: (B, 72 * 72, C) the SAM 3 detector's backbone last_hidden_state at 1008 -> the tracker's image embeddings
        (Sam3TrackerModel.get_image_features + forward's reshaping, with the backbone replaced by this input)."""
        import torch
        m = self.model
        with torch.cuda.device(self.dev), torch.inference_mode():
            b, n, c = hidden.shape
            s = int(round(n ** .5))
            fpn, _ = m.vision_encoder.neck(hidden.to(self.dev).view(b, s, s, c).permute(0, 3, 1, 2))
            fm = list(fpn)
            fm[0] = m.mask_decoder.conv_s0(fm[0])
            fm[1] = m.mask_decoder.conv_s1(fm[1])
            fm = [f.flatten(2).permute(2, 0, 1) for f in fm]
            fm[-1] = fm[-1] + m.no_memory_embedding
            return [f.permute(1, 2, 0).view(b, -1, *size) for f, size in zip(fm, m.backbone_feature_sizes)]

    def decode(self, emb, j, boxes, chunk=64):
        """emb: embed()'s list; j: the frame in it; boxes (n, 4) xyxy in the 1008 x 1008 squashed frame -> (low-res logits
        (n, h, w) float, predicted IoU (n,))."""
        import torch
        if not len(boxes):
            return torch.zeros((0, 288, 288), device=self.dev), torch.zeros(0, device=self.dev)
        one = [e[j:j + 1] for e in emb]
        lg, sc = [], []
        with torch.cuda.device(self.dev), torch.inference_mode():
            for i in range(0, len(boxes), chunk):
                bx = torch.as_tensor(np.asarray(boxes[i:i + chunk], np.float32), device=self.dev)[None]
                o = self.model(image_embeddings=one, input_boxes=bx, multimask_output=False)
                lg.append(o.pred_masks[0, :, 0].float())
                sc.append(o.iou_scores[0, :, 0].float())
        return torch.cat(lg), torch.cat(sc)


class Detector:
    """One box source on one GPU: detect(list of BGR uint8 frames) -> per frame (xyxy (n, 4) source px, score (n,), words)."""

    def __init__(self, src, dev, wdir=WDIR, imgsz=1280):
        self.src, self.dev, self.imgsz = src, dev, imgsz
        if src == "owlv2":
            self.owl = Owl(dev)
            self.q = self.owl.encode(TAXO)
            return
        from pathlib import Path
        from ultralytics import YOLO, YOLOE
        path = str(Path(wdir) / WEIGHTS[src])
        if src == "yoloe":
            self.model = YOLOE(path)
            pe_path = Path(wdir) / f"{WEIGHTS[src]}.taxo-{len(TAXO)}.pt"
            import torch
            if pe_path.exists():
                pe = torch.load(pe_path, map_location="cpu")
            else:  # the text encoder (MobileCLIP 2, downloaded once); cached so a run never loads it
                pe = self.model.get_text_pe(TAXO)
                torch.save(pe.cpu(), pe_path)
            self.model.set_classes(TAXO, pe.to(dev))
        else:
            self.model = (YOLOE if src == "yoloepf" else YOLO)(path)
        self.names = self.model.names

    def detect(self, frames, floor):
        if self.src == "owlv2":
            import torch
            out = []
            rgb = torch.from_numpy(np.ascontiguousarray(np.stack(frames)[..., ::-1])).to(self.dev)
            from torchvision.ops import nms
            for xyxy, obj, word, s in self.owl.detect(rgb, self.q):
                k = torch.zeros_like(obj, dtype=torch.bool)
                k[nms(xyxy, obj, NMS_IOU)] = True
                k &= obj >= floor
                out.append((xyxy[k].cpu().numpy(), obj[k].cpu().numpy(), [TAXO[i] if v >= NAME_MIN["owlv2"] else None
                                                                          for i, v in zip(word[k].tolist(), s[k].tolist())]))
            return out
        res = self.model.predict(list(frames), imgsz=self.imgsz, conf=floor, iou=.6, agnostic_nms=True, max_det=150, quantize=16,
                                 device=self.dev.index, verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), [word_of(self.src, self.names[int(c)]) for c in r.boxes.cls.tolist()])
                for r in res]


class Owl:
    """X10's OWLv2 large (ensemble): objectness per box, class logits against the encoded queries."""

    def __init__(self, dev, cache_dir="/v/models/hf"):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.dev = dev
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL, cache_dir=cache_dir, torch_dtype=torch.float16).to(dev).eval()
        self.proc = Owlv2Processor.from_pretrained(OWL_MODEL, cache_dir=cache_dir)
        ip = self.proc.image_processor
        self.mean = torch.tensor(ip.image_mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=dev).view(1, 3, 1, 1)
        self.side = self.model.config.vision_config.image_size

    def encode(self, words):
        import torch
        with torch.inference_mode():
            tok = self.proc.tokenizer(words, padding="max_length", max_length=16, truncation=True, return_tensors="pt").to(self.dev)
            e = self.model.owlv2.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        return getattr(e, "pooler_output", e)

    def detect(self, rgb, queries, top=100):
        """rgb (B, H, W, 3) uint8 on this GPU -> per image (boxes xyxy px (top by objectness), objectness, top word, its score)."""
        import torch
        import torch.nn.functional as F
        n, h, w = rgb.shape[:3]
        side = max(h, w)
        x = torch.full((n, 3, side, side), .5, device=self.dev)
        x[:, :, :h, :w] = rgb.permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(self.side, self.side), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - self.mean) / self.std).half()
        out = []
        with torch.inference_mode():
            m = self.model
            emb, _ = m.image_embedder(pixel_values=x)[:2]
            b, gh, gw, d = emb.shape
            feats = emb.reshape(b, gh * gw, d)
            boxes = m.box_predictor(feats, emb)
            obj = m.objectness_predictor(feats).float().sigmoid()
            logits = m.class_predictor(feats, queries[None].expand(b, -1, -1).to(feats.dtype), None)[0].float().sigmoid()
            for i in range(b):
                order = obj[i].argsort(descending=True)[:top]
                bx = boxes[i][order].float()
                xyxy = torch.stack([bx[:, 0] - bx[:, 2] / 2, bx[:, 1] - bx[:, 3] / 2, bx[:, 0] + bx[:, 2] / 2, bx[:, 1] + bx[:, 3] / 2], 1) * side
                xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
                xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
                s, word = logits[i][order].max(1)
                out.append((xyxy, obj[i][order], word, s))
        return out


def fill_maps(maps, box, ent, qs, dev=None):
    """maps: [(entry, label map, sx, sy, keyframe)], replaced in place; box: {pool index: (source, word, packed map-grid mask)};
    ent: object index + 1 per pool mask (0 = none); qs: keyframe per pool mask. Each box mask of an object fills its keyframe's
    free pixels with that object (fill; on `dev` with segment.paint, the same rule). -> pixels filled."""
    by_q = {}
    for g, (_, _, up) in box.items():
        if ent[g]:
            by_q.setdefault(int(qs[g]), []).append((up, int(ent[g])))
    n = 0
    for i, e in enumerate(maps):
        got = by_q.get(e[4])
        if not got:
            continue
        if dev is None:
            lab = fill(e[1], np.unpackbits(np.stack([u for u, _ in got]), axis=2).astype(bool), [k for _, k in got])
        else:
            import torch
            from fast_report import segment
            top = segment.paint(segment.unpack(np.stack([u for u, _ in got]), dev))  # 0 = none, k + 1 = the smallest mask there
            ids = torch.tensor([0] + [k for _, k in got], device=dev)[top]
            old = torch.from_numpy(e[1]).to(dev)
            lab = torch.where(old == 0, ids.to(old.dtype), old).cpu().numpy()
        n += int((lab != e[1]).sum())
        maps[i] = (e[0], lab, *e[2:])
    return n


def to_1008(xyxy, hw):
    """Source-px boxes -> the squashed 1008 x 1008 frame SAM 3 sees (segment.Sam3.vision resizes without keeping the aspect)."""
    s = np.array([1008 / hw[1], 1008 / hw[0]] * 2, np.float32)
    return np.asarray(xyxy, np.float32).reshape(-1, 4) * s


def self_check():
    assert word_of("yolo11", "person") is None and word_of("yolo26", "tv") == "monitor" and word_of("yoloe", "rack") == "rack"
    assert "rack" in TAXO and "shelf" in TAXO and "merchandise" in TAXO and "floor marking" not in TAXO
    hw = (720, 1280)
    assert keep_box([0, 0, 100, 100], hw, "box") and not keep_box([0, 0, 10, 10], hw, "box") and not keep_box([0, 0, 1280, 720], hw, "box")
    assert not keep_box([0, 0, 100, 100], hw, None)
    assert keep_mask(100, 40, 0, 0, "box") == (True, "kept") and keep_mask(100, 100, 0, 0, "box") == (True, "kept")  # no 'SAM 3 has it' gate
    assert keep_mask(100, 0, 80, 0, "box")[1] == "floor" and keep_mask(100, 0, 80, 0, "extension cord") == (True, "kept")
    assert keep_mask(100, 0, 0, 40, "bag")[1] == "on a person" and keep_mask(450, 0, 0, 0, "bag", frame_px=1000)[1] == "a region"
    lab = np.zeros((6, 8), np.int16)
    lab[:, :2] = 7  # an existing entity
    big, small = np.zeros((6, 8), bool), np.zeros((6, 8), bool)
    big[:, :6], small[2:4, 1:4] = True, True
    out = fill(lab, np.stack([small, big]), [5, 4])
    assert (out[:, :2] == 7).all(), "an existing pick is never overwritten"
    assert out[3, 2] == 5 and out[0, 4] == 4 and out[0, 7] == 0, out
    assert (fill(lab, np.stack([big]), [0]) == lab).all(), "a mask with no entity paints nothing"
    maps = [({"f": 3}, lab, 2., 2., 3), ({"f": 4}, lab, 2., 2., 4)]
    n = fill_maps(maps, {0: ("yoloe", "box", np.packbits(small, axis=1)), 1: ("yoloe", None, np.packbits(big, axis=1))}, np.array([5, 0]), np.array([3, 3]))
    assert n == 4 and maps[0][1][3, 2] == 5 and maps[0][1][0, 4] == 0 and maps[1][1] is lab, (n, maps[0][1])
    b = to_1008([[0, 0, 1280, 720]], hw)
    assert np.allclose(b, [[0, 0, 1008, 1008]])
    print("coverage self-check ok: word mapping, box and mask gates, gap-fill painting, box scaling")


if __name__ == "__main__":
    import sys
    if "--self-check" in sys.argv:
        self_check()
