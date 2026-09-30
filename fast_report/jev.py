"""r5b (models): Jev-Omni as the display-model router's decider, on its OWN GPU (a separate Modal class: never on the report's two
A100s). X13's loader and X8's batched forward (route/jev measured it: 0.088 s a question on an A100 in batches of 8), moved here
from modal_apps/route_jev.py so the report's app and the route/jev bench share it.

  load()            the model from the x8 volume (HF_HOME=/v/x8/hf, offline): -> state
  decide(st, qs)    qs [(key, jpeg, state, question, options)] -> {"probs": {key: [p]}, "compute_s", "load_s", "peak_gib"}
"""
import io
import sys
import time

JEV = "akhilaaa3/Jev-Omni"
JEV_FILES = ["config.json", "generation_config.json", "model*.safetensors*", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
             "chat_template.jinja", "decision_config.json", "head.pt", "jev_omni.py", "sha256.json", "verification.json", "README.md"]
JEV_BATCH = 8


def load():
    import torch
    from huggingface_hub import snapshot_download
    t = time.perf_counter()
    path = snapshot_download(JEV, allow_patterns=JEV_FILES)
    sys.path.insert(0, path)
    import jev_omni
    st = {"mod": jev_omni, "torch": torch, "clf": jev_omni.load_jev_omni(), "full": {}}
    _, decoder = jev_omni._find_backbone(st["clf"].model)
    decoder.register_forward_hook(lambda _m, _a, out: st["full"].__setitem__("h", out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]))
    st["load_s"] = round(time.perf_counter() - t, 2)
    return st


def batch(st, reqs):
    """X8's batched forward (right padding, each row's last real token, the card's head). reqs: [(PIL, state, q, options)]."""
    torch, clf = st["torch"], st["clf"]
    convs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": st["mod"]._prompt(s_, q, opts)}]}]
             for im, s_, q, opts in reqs]
    proc = clf.processor
    proc.tokenizer.padding_side = "right"
    inputs = proc.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", padding=True,
                                      enable_thinking=False)
    inputs = {k: v.to("cuda", dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to("cuda") for k, v in inputs.items()}
    last = inputs["attention_mask"].sum(1) - 1
    clf._capture.clear()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        clf.model(**inputs, use_cache=False, **clf._extra)
        hidden = st["full"]["h"][torch.arange(len(reqs), device=last.device), last].float()
        logits = clf.head(hidden, torch.tensor([len(r[3]) for r in reqs], device="cuda"))
    return [logits[i, :len(r[3])].float().softmax(-1).cpu().tolist() for i, r in enumerate(reqs)]


def decide(st, qs):
    import subprocess
    from PIL import Image
    t = time.perf_counter()
    reqs = [(Image.open(io.BytesIO(j)).convert("RGB"), s_, q, o) for _, j, s_, q, o in qs]
    out = {}
    for s in range(0, len(reqs), JEV_BATCH):
        for (k, *_), p in zip(qs[s:s + JEV_BATCH], batch(st, reqs[s:s + JEV_BATCH])):
            out[k] = [round(v, 6) for v in p]
    st["torch"].cuda.synchronize()
    return {"probs": out, "compute_s": round(time.perf_counter() - t, 4), "load_s": st["load_s"],
            "peak_gib": round(st["torch"].cuda.max_memory_reserved() / 2 ** 30, 2),
            "gpu": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip()}
