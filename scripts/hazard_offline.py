"""The hazard judge on round-1 runs, off the pipeline: every object with a check (judge rules on the final cards) -> its 2 x 2
evidence image (fast_report.hazard.evidence) -> Gemini by exec in the report-workspace container (hazard.exec_gemini) ->
answers. Also builds agent-audit contact sheets (sheets/) of items for labelling by looking.

  python scripts/hazard_offline.py --out RUNS/mvp2-judge-hazard-001 [--sites me340,samsclub,walmart] [--ask] [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import judge_offline as jo  # noqa: E402
from fast_report import hazard, judge  # noqa: E402


def items_of(site, out, version=2):
    d = jo.load(site)
    ctx = d["ctx"]
    outl = ctx.get("outlines") or []
    ctx["outlines_by_frame"] = {f["sourceFrame"]: f["objects"] for f in outl}
    rows = judge.evaluate(d["cards"], ctx)
    by = {c["id"]: c for c in d["cards"]}
    want = {}
    for r in rows:
        q = hazard.CHECK_Q.get(r["check"])
        if q and by.get(r["subject"], {}).get("kind") == "object":
            want.setdefault(r["subject"], {"checks": [], "questions": []})
            want[r["subject"]]["checks"].append(r["check"])
            if q not in want[r["subject"]]["questions"]:
                want[r["subject"]]["questions"].append(q)
    items = []
    (out / "images").mkdir(parents=True, exist_ok=True)
    for oid, w in want.items():
        jpg, keys = hazard.evidence(by[oid], ctx, judge.frame_at, judge.marks_on, judge.som, version)
        if jpg is None:
            continue
        name = f"{site}-{oid}.jpg"
        (out / "images" / name).write_bytes(jpg)
        items.append({"site": site, "id": oid, "name": judge.name_of(by[oid]), "questions": w["questions"], "checks": w["checks"], "keys": keys,
                      "image": f"images/{name}"})
    return items, rows


def ask(items, out, tag, version=2):
    for it in items:
        it["jpeg"] = (out / it["image"]).read_bytes()
    reqs = hazard.batches(items, version=version)
    t0, got, rec = time.time(), {}, {}
    for r in reqs:
        rec[r["key"]] = {"ids": r["ids"], "t_submit": 0.}

    def done(key, prov, err):
        rec[key].update(t_done=round(time.time() - t0, 2), error=err, usage=(prov or {}).get("usage"), status=(prov or {}).get("status"),
                        output_text=(prov or {}).get("output_text"))
        if prov and prov.get("status") == "completed":
            got.update({f"{i}|{q}": v for (i, q), v in hazard.parse(prov["output_text"], rec[key]["ids"]).items()})
    with open(out / f"events-{tag}.jsonl", "a") as log:
        code = hazard.exec_gemini(reqs, done, log)
    for it in items:
        it.pop("jpeg", None)
    return {"exit_code": code, "wall_s": round(time.time() - t0, 2), "requests": rec, "answers": got}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sites", default="me340,samsclub,walmart")
    ap.add_argument("--ask", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--version", type=int, default=2, help="1: hazard-001's evidence and wording; 2: criteria in the questions, a context tile")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for site in a.sites.split(","):
        items, rows = items_of(site, a.out, a.version)
        if a.limit:
            items = items[:a.limit]
        (a.out / f"items-{site}.json").write_text(json.dumps(items, indent=1))
        print(site, len(items), "items", sum(len(i["questions"]) for i in items), "questions", flush=True)
        if a.ask:
            res = ask(items, a.out, site, a.version)
            (a.out / f"answers-{site}.json").write_text(json.dumps(res, indent=1))
            print(site, {k: res[k] for k in ("exit_code", "wall_s")}, len(res["answers"]), "answers",
                  sorted((v.get("t_done"), v.get("error")) for v in res["requests"].values())[-3:], flush=True)
