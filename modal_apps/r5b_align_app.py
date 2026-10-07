"""r5b integrate: the alignment check (scripts/r5b_align.py) where the reports are: one ephemeral CPU container reads each report's
layers and blobs on the layers Volume, draws every object's shown model from its best keyframe's camera against its outline, and
sends back the numbers, the worst generated models' sheet and two overlay frames (small JPGs).

    modal run modal_apps/r5b_align_app.py --reports REPORT[,REPORT...] --out DIR
"""
import base64
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
app = modal.App("panoptes-r5b-align")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "opencv-python-headless", "scipy")
for _f in ("r5b_align.py", "r5b_models.py", "r5b_render.py", "r4_models.py"):
    image = image.add_local_file(HERE.parent / "scripts" / _f, f"/repo/scripts/{_f}")
for _f in ("__init__.py", "cards.py", "display_model.py", "instances.py", "surface.py"):  # r5b_models.r4b (the audit sheets' r4b column)
    image = image.add_local_file(HERE.parent / "fast_report" / _f, f"/repo/fast_report/{_f}")
LAYERS = modal.Volume.from_name("panoptes-fb-layers")


@app.function(image=image, volumes={"/v/layers": LAYERS}, cpu=8, memory=16384, timeout=1800, retries=0)
def align(report: str) -> dict:
    sys.path.insert(0, "/repo/scripts")
    import r5b_align
    res, files = r5b_align.run(Path("/v/layers"), report)
    return {"result": res, "files": {k: base64.b64encode(v).decode() for k, v in files.items()}}


@app.function(image=image, volumes={"/v/layers": LAYERS}, cpu=4, memory=16384, timeout=1800, retries=0)
def sheets(report: str, n: int, strat: int, seed: int, only: str = "") -> dict:
    """scripts/r5b_models.sheets on the Volume: n random + strat stratified cards' shown model vs outline (the audit by eye)."""
    import tempfile
    sys.path[:0] = ["/repo", "/repo/scripts"]
    import r5b_models
    d = Path(tempfile.mkdtemp())
    r5b_models.sheets(Path("/v/layers"), report, d, n=n, strat=strat, seed=seed, only=only or None)
    return {p.name: base64.b64encode(p.read_bytes()).decode() for p in d.iterdir() if p.is_file()}


@app.local_entrypoint()
def audit(reports: str, out: str, n: int = 0, strat: int = 20, seed: int = 29):
    for rep, files in zip(reports.split(","), sheets.map(reports.split(","), kwargs={"n": n, "strat": strat, "seed": seed})):
        d = Path(out) / rep
        d.mkdir(parents=True, exist_ok=True)
        for k, v in files.items():
            (d / k).write_bytes(base64.b64decode(v))
        print(rep, sorted(files))


@app.local_entrypoint()
def main(reports: str, out: str):
    for rep, x in zip(reports.split(","), align.map(reports.split(","))):
        d = Path(out) / rep
        d.mkdir(parents=True, exist_ok=True)
        (d / "alignment.json").write_text(json.dumps(x["result"], indent=1))
        for k, v in x["files"].items():
            (d / k).write_bytes(base64.b64decode(v))
        print(rep, json.dumps(x["result"]["summary"]))
