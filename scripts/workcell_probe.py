"""Workcell rules v2 probe: safety-device vocabulary + fence min-height.

Part 1 (paid, cached): run the four NEW perception classes (safety sensor,
emergency stop button, warning sign, safety light) through the production
SAM3Adapter with the production first-hit synonym fallback, against the
owner's real factory photo and two MTMC warehouse frames. A miss is a valid
result — these classes may not be visible, or SAM may not know the phrase;
nothing is tuned to force a hit.

Part 2 (free, offline): evaluate the reviewed docs/policies/compiled_v2
specs against the cached demo-real-factory SceneMap through the production
evaluate_policies path — the fence min-height verdict with the mono band.

House pattern: every paid response is disk-cached under
outputs/workcell_probe_v1 in the MAIN repo, spend is gated behind --live
with a printed call count (~$0.01/call), per-image failures skip.

  uv run --env-file .env python scripts/workcell_probe.py          # cached/offline
  uv run --env-file .env python scripts/workcell_probe.py --live   # authorize spend
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehs_spatial.app import load_policy_specs  # noqa: E402
from ehs_spatial.contracts import SceneMap  # noqa: E402
from ehs_spatial.policy import evaluate_policies  # noqa: E402
from ehs_spatial.providers.sam3 import LABEL_PROMPTS, SAM3Adapter  # noqa: E402

REPO = Path("/Users/adam/Desktop/Tesla/ehs-spatial")
WORK = REPO / "outputs" / "workcell_probe_v1"
COST_PER_CALL_USD = 0.01

NEW_CLASSES = (
    "safety sensor",
    "emergency stop button",
    "warning sign",
    "safety light",
)
MTMC_FRAMES = REPO / "outputs/datasets/mtmc/MTMC_Tracking_2025/val/Warehouse_016/frames_png"
IMAGES = {
    "factory": REPO / "runs/demo-real-factory/input/image_01.png",
    "mtmc-cam0": MTMC_FRAMES / "Camera_003600.png",
    "mtmc-cam3": MTMC_FRAMES / "Camera_03_004200.png",
}
FIXTURES = Path(__file__).resolve().parents[1] / "docs" / "policies" / "compiled_v2"
SCENE_JSON = REPO / "runs/demo-real-factory/scene.json"


def _cache_path(image_id: str, prompt: str) -> Path:
    return WORK / "cache" / f"{image_id}__{prompt.replace(' ', '_')}.json"


class CachingSubscriber:
    """Pay once, iterate free. Wraps fal subscribe with a per-(image, prompt)
    disk cache; refuses to spend unless --live authorized it."""

    def __init__(self, *, live: bool) -> None:
        self.live = live
        self.live_calls = 0
        self.cached_hits = 0
        self.current_key: str | None = None  # set per call by the probe loop

    def __call__(self, endpoint: str, *, arguments: dict) -> dict:
        cache = _cache_path(*json.loads(self.current_key))
        if cache.is_file():
            self.cached_hits += 1
            return json.loads(cache.read_text())
        if not self.live:
            raise SystemExit(
                f"uncached call needed for {self.current_key}; rerun with --live"
            )
        import fal_client

        self.live_calls += 1
        response = fal_client.subscribe(endpoint, arguments=arguments)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(response))
        return response


def probe(live: bool) -> dict:
    subscriber = CachingSubscriber(live=live)
    adapter = SAM3Adapter(subscriber=subscriber)
    rows = {}
    for image_id, image_path in IMAGES.items():
        for label in NEW_CLASSES:
            row = {"hit": False, "prompt": None, "masks": 0, "top_score": None}
            errors = 0
            # Production semantics: first prompt that returns masks wins.
            for prompt in LABEL_PROMPTS[label]:
                subscriber.current_key = json.dumps([image_id, prompt])
                try:
                    observations = adapter.segment(
                        image_path,
                        prompt=prompt,
                        label=label,
                        frame_id=image_id.replace("-", "_"),
                        output_dir=WORK / "masks" / image_id,
                    )
                except Exception as error:
                    # A provider failure is NOT an honest miss; the row is
                    # marked so a locked account never reads as absence.
                    print(f"  ! {image_id} / {prompt!r}: {error}")
                    errors += 1
                    observations = []
                if observations:
                    row = {
                        "hit": True,
                        "prompt": prompt,
                        "masks": len(observations),
                        "top_score": max(o.score for o in observations),
                    }
                    break
            row["provider_errors"] = errors
            rows[f"{image_id}/{label}"] = row
    print(
        f"\nprovider calls: {subscriber.live_calls} live "
        f"(~${subscriber.live_calls * COST_PER_CALL_USD:.2f}), "
        f"{subscriber.cached_hits} cached"
    )
    print(f"\n{'image/class':<42} {'hit':<5} {'prompt':<22} masks  top_score")
    for key, row in rows.items():
        status = "HIT" if row["hit"] else ("ERR" if row["provider_errors"] else "miss")
        print(
            f"{key:<42} {status:<5} "
            f"{row['prompt'] or '-':<22} {row['masks']:<6} "
            f"{row['top_score'] if row['top_score'] is not None else '-'}"
        )
    return rows


def evaluate_fixtures() -> list[dict]:
    specs = load_policy_specs(FIXTURES)
    scene = SceneMap.model_validate_json(SCENE_JSON.read_text())
    frame_count = len({f for e in scene.entities for f in e.evidence_frame_ids}) or 1
    results = evaluate_policies(specs, scene, capture_frame_count=frame_count)
    print(f"\noffline evaluation vs {SCENE_JSON} ({frame_count} frame(s), mono band):")
    for spec, result in zip(specs, results):
        print(f"  {result.status.value:<22} {spec.policy_id}")
        for violation in result.violations:
            print(
                f"      {violation.subject_id}: measured "
                f"{violation.measured}{violation.unit} "
                f"(limit {violation.threshold}{violation.unit})"
            )
        for warning in result.warnings:
            print(f"      {warning}")
        for fact in result.facts:
            print(f"      fact: {fact.subject_id} {fact.predicate} {fact.value}{fact.unit}")
    return [r.model_dump(mode="json") for r in results]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    rows = probe(args.live)
    results = evaluate_fixtures()
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "report.json").write_text(
        json.dumps({"probe": rows, "policy_results": results}, indent=2) + "\n"
    )
    print(f"\nwrote {WORK / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
