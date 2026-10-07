"""mvp3 integrate: what the hazard judge sends to Gemini over one analysis, replayed off the GPU on a finished run: every cards
version in put order, the layers as they stood at that patch (judge_offline.load upto), one shared answer cache, a stand-in
Gemini that answers p 0.5 at once (the answers do not change what is asked). Per call: requests, evidence images (one 2 x 2
image per object per request), questions, and input tokens estimated at the run's own relayed tokens per image. Run it
under this branch's judge and under an earlier branch's (PYTHONPATH) for before / after.

  PYTHONPATH=.:scripts python scripts/hazard_asks_mvp3.py RUN [RUN ...] --out FILE.json
"""
import argparse
import glob
import json
import sys
from concurrent.futures import Future
from pathlib import Path

from fast_report import judge  # first: an earlier branch's judge on PYTHONPATH stays the one judge_offline uses
import judge_offline as jo  # noqa: E402


def stand_in(sent):
    def ask(reqs):
        sent.append({"requests": len(reqs), "images": sum(len({i for i, _ in r["ids"]}) for r in reqs), "questions": sum(len(r["ids"]) for r in reqs)})
        out = {}
        for r in reqs:
            f = Future()
            f.set_result({"status": "completed", "usage": {}, "output_text": json.dumps(
                {"answers": [{"id": i, "question": q, "p_yes": .5, "why": ""} for i, q in r["ids"]]})})
            out[r["key"]] = f
        return out
    return ask


def unanswered(jpegs, prompt, n, priority):
    f = Future()
    f.set_result({"probs": None, "mass": 0.})
    return f


def replay(run, report):
    """-> [{version seq, requests, images, questions}] over the call's cards versions."""
    seqs = [json.loads(Path(p).read_text())["seq"] for p in sorted(glob.glob(str(run / "mirror/reports" / report / "patches/*-object_cards.json")))]
    carried, sent, per = {}, [], []
    for n, seq in enumerate(seqs, 1):
        d = jo.load(run=run, report=report, upto=seq)
        k = len(sent)
        judge.run(d["cards"], {**d["ctx"], "put_order": n}, judge._Writer(), judge._Clock(), ask=unanswered, hazard_ask=stand_in(sent), carried=carried)
        per.append({"cards_seq": seq, **{x: sum(s[x] for s in sent[k:]) for x in ("requests", "images", "questions")}})
    return per


def measured(run):
    """The run's own relayed hazard requests (both calls): requests and counted input tokens."""
    f = run / "hazard-relay-events.jsonl"
    ev = [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []
    b = [e for e in ev if e.get("phase") == "budget_evidence"]
    return {"requests": len(b), "input_tokens": sum(e["data"].get("inputTokens") or 0 for e in b)}


def main(a):
    out = {"judge": judge.__file__}
    for r in a.runs:
        run = jo.RUNS / r if not Path(r).is_absolute() else Path(r)
        calls = sorted((json.loads(Path(f).read_text()) for f in glob.glob(str(run / "call-*.json"))), key=lambda c: c["call"])
        rep = {"measured_both_calls": measured(run), "calls": {}}
        for c in calls:
            per = replay(run, c["run"]["report"])
            rep["calls"][c["kind"]] = {"report": c["run"]["report"], "per_version": per,
                                      **{x: sum(v[x] for v in per) for x in ("requests", "images", "questions")}}
            print(run.name, c["kind"], {x: rep["calls"][c["kind"]][x] for x in ("requests", "images", "questions")}, flush=True)
        out[run.name] = rep
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    main(ap.parse_args())
    sys.exit(0)
