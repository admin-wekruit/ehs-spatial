"""Probe: can Qwen3-VL-Embedding + Qwen3-VL-Reranker find and name our reconstructed objects without a cloud model?

Inputs are one tight crop per named entity (its largest observed view, box plus margin, mask outlined) and the names
Gemini gave them, used only as a rough reference, never as truth. Two uses, both as the Qwen docs pipeline them (embedding for
recall, reranker for precision; github.com/QwenLM/Qwen3-VL-Embedding at REPO_COMMIT):
  finding  a text query -> cosine over all crops -> top RECALL -> reranker score -> order
  naming   a crop -> cosine over a label vocabulary -> top NAME_CANDIDATES labels -> reranker picks one
Three Modal GPU calls (A100-80GB, else H100, else A100-40GB), retries 0, each saved before the next: embedding recall, reranking for finding, reranking
for naming on a sample of NAME_SAMPLE named crops (the reranker scores pairs one at a time: all 775 x 5 did not fit in 40 min).

  python modal_apps/qwen3vl_retrieval_probe.py --names RUN [RUN ...] --masks ROOT [ROOT ...] --output NEW_DIR
"""
import argparse
import io
import json
from pathlib import Path
import re
import time

import modal

REPO_COMMIT = "393e2978d27852b0d0230d6994f37f9c15bed73c"
EMBEDDING, RERANKER = "Qwen/Qwen3-VL-Embedding-8B", "Qwen/Qwen3-VL-Reranker-8B"
RECALL, NAME_CANDIDATES, NAME_SAMPLE = 30, 5, 150
QUERIES = ["pallet", "shopping cart", "ladder", "fire extinguisher", "exit sign", "computer monitor", "CNC machine control panel",
           "tool storage drawer cabinet", "cardboard box", "price tag", "ceiling light", "paper towels", "workbench", "trash can"]
EHS_TERMS = ["fire extinguisher", "exit sign", "ladder", "guard rail", "safety cone", "forklift", "hand truck", "electrical panel",
             "first aid kit", "eyewash station", "spill kit", "hose reel", "pallet jack", "warning sign", "machine guard"]
FIND_INSTRUCTION = "Given the name of an object, retrieve images that show that object."
NAME_INSTRUCTION = "Given an image of one outlined object, retrieve the name that describes it."

app = modal.App("panoptes-qwen3vl-retrieval-probe")
cache = modal.Volume.from_name("qwen3vl-retrieval-cache", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git")
         .pip_install("torch==2.8.0", "torchvision==0.23.0", "transformers==4.57.3", "qwen-vl-utils==0.0.14", "accelerate>=1.12.0", "scipy", "pillow", "numpy<2.3")
         .run_commands(f"git clone https://github.com/QwenLM/Qwen3-VL-Embedding /opt/qvl && cd /opt/qvl && git checkout {REPO_COMMIT}")
         .env({"PYTHONPATH": "/opt/qvl", "HF_HOME": "/cache/huggingface", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}))


def local_snapshot(repo):
    """Load from a downloaded snapshot: by hub id, the reranker's processor fails on its extra chat template
    (additional_chat_templates/reranker.jinja resolves to None under transformers 4.57.x), a local directory does not."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo)


@app.function(image=image, cpu=2, timeout=900, retries=0, volumes={"/cache": cache})
def check_processor():
    """CPU only: the reranker's processor loads from a local snapshot (no weights are read)."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(local_snapshot(RERANKER), trust_remote_code=True, padding_side="left")
    cache.commit()
    return type(processor).__name__


@app.function(image=image, gpu=["A100-80GB", "H100", "A100-40GB"], timeout=1800, retries=0, max_containers=1, volumes={"/cache": cache})
def recall(crops, queries, vocabulary):
    """Embedding stage: crops: {id: jpeg} -> top RECALL crops per query and top NAME_CANDIDATES labels per crop."""
    import numpy as np
    import torch
    from PIL import Image
    from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
    started = time.perf_counter()
    ids = sorted(crops)
    embedder = Qwen3VLEmbedder(model_name_or_path=local_snapshot(EMBEDDING), torch_dtype=torch.bfloat16, attn_implementation="sdpa")

    def embed(items, batch=8):
        return torch.cat([embedder.process(items[n:n + batch]).float().cpu() for n in range(0, len(items), batch)]).numpy()

    crop_vec = embed([{"image": Image.open(io.BytesIO(crops[i])).convert("RGB")} for i in ids])
    query_vec = embed([{"text": q, "instruction": FIND_INSTRUCTION} for q in queries])
    label_vec = embed([{"text": v, "instruction": FIND_INSTRUCTION} for v in vocabulary])
    find_sim, name_sim = query_vec @ crop_vec.T, crop_vec @ label_vec.T
    return {"finding": {q: [(ids[j], float(find_sim[n, j])) for j in np.argsort(-find_sim[n])[:RECALL]] for n, q in enumerate(queries)},
            "naming": {ids[n]: [(vocabulary[j], float(name_sim[n, j])) for j in np.argsort(-name_sim[n])[:NAME_CANDIDATES]] for n in range(len(ids))},
            "seconds": time.perf_counter() - started, "gpu": torch.cuda.get_device_name()}


@app.function(image=image, gpu=["A100-80GB", "H100", "A100-40GB"], timeout=1800, retries=0, max_containers=1, volumes={"/cache": cache})
def rerank(requests):
    """Reranker stage: [(key, instruction, query dict, [document dicts])] with images as jpeg bytes -> {key: scores}."""
    import torch
    from PIL import Image
    from src.models.qwen3_vl_reranker import Qwen3VLReranker
    started = time.perf_counter()
    reranker = Qwen3VLReranker(model_name_or_path=local_snapshot(RERANKER), torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    picture = lambda d: {**d, "image": Image.open(io.BytesIO(d["image"])).convert("RGB")} if "image" in d else d
    out = {key: reranker.process({"instruction": instruction, "query": picture(query), "documents": [picture(d) for d in documents]})
           for key, instruction, query, documents in requests}
    return {"scores": out, "seconds": time.perf_counter() - started, "pairs": sum(len(r[3]) for r in requests)}


def crop_of(entity, masks, pad=.15, least=64):
    """The entity's largest observed view, cut to its box with a margin, its mask outlined thinly: one object, not a scene."""
    import cv2
    import numpy as np
    key = max(entity["observationBoxes"], key=lambda k: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(entity["observationBoxes"][k]))
    part, frame, instance = key.split(":")
    folder = next(masks.glob(f"{part}-*/frame-{int(frame):05d}"))
    image = cv2.imread(str(folder / f"frame-{int(frame)}.png"))
    mask = cv2.imread(str(folder / f"instance-{instance}-mask.png"), 0) > 0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 255), 1)
    x0, y0, x1, y1 = entity["observationBoxes"][key]
    w, h = max(x1 - x0, least), max(y1 - y0, least)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = int(max(cx - w * (.5 + pad), 0)), int(min(cx + w * (.5 + pad), image.shape[1]))
    y0, y1 = int(max(cy - h * (.5 + pad), 0)), int(min(cy + h * (.5 + pad), image.shape[0]))
    return cv2.imencode(".jpg", image[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def norm(label):
    label = re.sub(r"[^a-z ]", " ", label.lower()).strip()
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 and not w.endswith("ss") else w for w in label.split())


def relevant(query, gemini):
    """Rough reference: the Gemini name shares the query's head noun (last word), singularised."""
    return bool(gemini) and norm(query).split()[-1] in norm(gemini).split()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    crops, gemini = {}, {}
    for run_dir, masks in zip(args.names, args.masks):
        for entity in json.loads((run_dir / "object-map.json").read_text())["entities"]:
            if entity.get("label") is None:
                continue
            ident = f"{run_dir.name.split('-')[0]}/{entity['entityId']}"
            crops[ident] = crop_of(entity, masks)
            gemini[ident] = entity["label"] if entity.get("labelStatus") in ("clear", "partial") else None
    vocabulary = sorted({norm(v) for v in gemini.values() if v and norm(v) not in ("unnamed surface", "")} | {norm(t) for t in EHS_TERMS})
    named_ids = sorted(i for i in crops if gemini[i])
    sample = named_ids[::max(1, len(named_ids) // NAME_SAMPLE)][:NAME_SAMPLE]  # every k-th named crop, all three videos
    with app.run():
        found = recall.remote(crops, QUERIES, vocabulary)
        (args.output / "recall.json").write_text(json.dumps(found, indent=1))
        find_scores = rerank.remote([(q, FIND_INSTRUCTION, {"text": q}, [{"image": crops[i]} for i, _ in hits]) for q, hits in found["finding"].items()])
        (args.output / "rerank-find.json").write_text(json.dumps(find_scores, indent=1))
        name_scores = rerank.remote([(i, NAME_INSTRUCTION, {"image": crops[i]}, [{"text": v} for v, _ in found["naming"][i]]) for i in sample])
        (args.output / "rerank-name.json").write_text(json.dumps(name_scores, indent=1))
    answer = {"finding": found["finding"], "naming": {i: found["naming"][i] for i in sample}, "rerank_find": find_scores["scores"], "rerank_name": name_scores["scores"],
              "report": {"recall_seconds": found["seconds"], "rerank_find_seconds": find_scores["seconds"], "rerank_name_seconds": name_scores["seconds"],
                         "gpu": found["gpu"], "crops": len(crops), "queries": len(QUERIES), "vocabulary": len(vocabulary), "naming_sample": len(sample)}}
    (args.output / "probe.json").write_text(json.dumps({**answer, "gemini": gemini, "vocabulary": vocabulary, "queries": QUERIES,
                                                          "models": [EMBEDDING, RERANKER], "repo_commit": REPO_COMMIT}, indent=1))
    rows, sheet = [], []
    for q in QUERIES:
        hits, scores = answer["finding"][q], answer["rerank_find"][q]
        embed_top = [i for i, _ in hits[:10]]
        rerank_top = [i for i, _ in sorted(zip([i for i, _ in hits], scores), key=lambda x: -x[1])[:10]]
        in_pool = sum(relevant(q, gemini[i]) for i, _ in hits)
        rows.append({"query": q, "embedding_p10": sum(relevant(q, gemini[i]) for i in embed_top) / 10,
                     "reranked_p10": sum(relevant(q, gemini[i]) for i in rerank_top) / 10, "reference_hits_in_top30": in_pool,
                     "reranker_top_score": round(max(scores), 3), "reranker_scores_over_0.5": sum(s > .5 for s in scores)})
        sheet.append((q, embed_top[:8], rerank_top[:8]))
    named = [(i, gemini[i], answer["naming"][i][0][0], max(zip(answer["rerank_name"][i], [v for v, _ in answer["naming"][i]]))[1])
             for i in answer["naming"] if gemini[i] and norm(gemini[i]) != "unnamed surface"]
    agree = lambda a, b: norm(a) == norm(b) or norm(a).split()[-1] == norm(b).split()[-1]
    summary = {"finding": rows,
               "naming": {"crops_with_gemini_name": len(named), "embedding_top1_agrees": sum(agree(g, e) for _, g, e, _ in named),
                          "reranked_top1_agrees": sum(agree(g, r) for _, g, _, r in named),
                          "disagreements_sample": [{"crop": i, "gemini": g, "reranked": r} for i, g, _, r in named if not agree(g, r)][:40]},
               "report": answer["report"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    contact_sheet(sheet, crops, args.output / "finding-sheet.jpg")
    print(json.dumps({k: v for k, v in summary.items() if k != "naming"} | {"naming": {k: v for k, v in summary["naming"].items() if k != "disagreements_sample"}}, indent=1))


def contact_sheet(rows, crops, path):
    import cv2
    import numpy as np
    tile = lambda i: cv2.resize(cv2.imdecode(np.frombuffer(crops[i], np.uint8), cv2.IMREAD_COLOR), (150, 150))
    blocks = []
    for q, embed_top, rerank_top in rows:
        for label, top in (("embedding", embed_top), ("reranked", rerank_top)):
            strip = np.hstack([tile(i) for i in top] + [np.zeros((150, 150, 3), np.uint8)] * (8 - len(top)))
            head = np.full((150, 170, 3), 255, np.uint8)
            cv2.putText(head, q[:18], (4, 60), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
            cv2.putText(head, label, (4, 90), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 180) if label == "reranked" else (90, 90, 90), 1)
            blocks.append(np.hstack([head, strip]))
    cv2.imwrite(str(path), np.vstack(blocks), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", type=Path, nargs="+", help="name_video_entities.py output directories (object-map.json with names)")
    parser.add_argument("--masks", type=Path, nargs="+", help="the mask root each object map was built from, same order")
    parser.add_argument("--output", type=Path)
    run(parser.parse_args())
