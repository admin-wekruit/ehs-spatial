"""The runner's three profiles, the model pin and licence table, and what each profile refuses (M2 design section 6).

  research    the M2 rules, every model the research reports used
  commercial  the M2 rules, only models whose licence is verified for commercial use and whose weights are pinned
  delivered   cache-only: the adopted values of the three delivered reports; any cache miss stops the run

commercial is True (verified), False (not allowed) or None (still to verify: refused like False). A pin here is what a new run
asks for; the modal apps carry the same constants (tests/test_report_runner_decisions.py keeps them equal).
"""
from dataclasses import dataclass, field

UNPINNED = "unpinned"

# hf_id (or the service's own id) -> (revision, weights sha256, licence, commercial)
MODELS = {
    "princeton-vl/DROID-SLAM": ("2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99", None, "code BSD-3-Clause; droid.pth weights: verify", None),
    "depth-anything/DA3-GIANT-1.1": ("72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19", None, "CC BY-NC 4.0", False),
    "depth-anything/DA3-BASE": ("f4a6c9b3c95e41c82048423d3493a81ec3fa810e", None, "Apache-2.0", True),
    "depth-anything/DA3-BASE#camera": ("f4a6c9b3c95e41c82048423d3493a81ec3fa810e", None, "Apache-2.0 card; unposed camera head output: verify", None),
    "Ruicheng/moge-3-vitl": ("184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be", None, "MIT card; weights: verify", None),
    "facebook/sam2.1-hiera-large": ("665f8e2ad61cf5f53d65644ff27c8ee525124610", None, "Apache-2.0", True),
    "facebook/sam3": ("3c879f39826c281e95690f02c7821c4de09afae7", None, "SAM License", True),
    "facebook/sam3.1": ("daa63191845a41281374e725f4c9e51c7a824460", None, "SAM License", True),
    "facebook/sam-3d-objects": ("2e73555018d2741ccd486e56c24fac41155a1dc6", None, "SAM License (no ITAR/military/nuclear uses)", True),
    "TRI-ML/RecGen": ("bc0df7de2e43314830039a35a720731d4c4fac65", None, "Toyota Research Institute Non-Commercial", False),
    "robbyant/lingbot-map": ("204754b72bb24f561f8d7e7e1e4e4cd9e809adf9", "ee665103348e07e6b826d529b8e61de8f413d5432a4f2e84970d6c8fd2e1cd72",
                             "no written licence from Robbyant", False),
    "facebook/map-anything": (UNPINNED, None, "CC BY-NC 4.0", False),
    "facebook/map-anything-apache": ("00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a", None, "Apache-2.0", True),
    "Qwen/Qwen3-VL-8B-Instruct": ("0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", None, "Apache-2.0", True),
    "Qwen/Qwen3-VL-Embedding-8B": (UNPINNED, None, "Apache-2.0", True),
    "Qwen/Qwen3-VL-Reranker-8B": (UNPINNED, None, "Apache-2.0", True),
    "gemini (panoptes-report-workspace)": (UNPINNED, None, "cloud API terms", None),
}

# The Qwen3-VL naming agreement gate (150 crops, named by Gemini as the reference). It needs a GPU, so this workflow did not run
# it; until it passes, the commercial profile leaves names blank and therefore every generated model too (the import still runs).
QWEN3VL_NAMING_GATE = {"status": "not-run", "set": "150 named crops over the three delivered object maps",
                       "reason": "needs GPU (Qwen3-VL-Embedding + Reranker on Modal); the M2 build workflow ran CPU only"}

M2_RULES = {name: f"{name}@1" for name in ("shots", "other_shot", "lens", "voxel", "overlay", "lingbot_stride", "lingbot_conf", "dense_gate",
                                             "inferred_floor", "track_windows", "splat_pick", "sam2_frames", "generator_plan", "floor_frames", "static_filter")}


def row(role, hf_id):
    return role, (hf_id, *MODELS[hf_id])  # "#camera": the same weights, a use with its own licence question


@dataclass(frozen=True)
class Profile:
    name: str
    models: dict  # role -> (hf_id, revision | 'unpinned', weights_sha256 | None, licence, commercial True | False | None)
    generators: tuple
    dense_map: bool
    namer: str
    cache_only: bool
    rules: object  # {decision: 'name@v'} or 'adopted'
    caps: dict = field(default_factory=dict)  # splat_minutes, generator_usd: per stage, never more than the run's budget
    omit: tuple = ()  # M2 stages this profile leaves out of the graph (delivered: the stages the delivered reports never had)


RESEARCH_MODELS = dict([row("camera", "princeton-vl/DROID-SLAM"), row("depth", "depth-anything/DA3-GIANT-1.1"), row("register", "depth-anything/DA3-GIANT-1.1"),
                        row("lens", "Ruicheng/moge-3-vitl"), row("segment", "facebook/sam2.1-hiera-large"), row("floor", "facebook/sam3"),
                        row("tracks", "facebook/sam3.1"), row("sam3d", "facebook/sam-3d-objects"), row("recgen", "TRI-ML/RecGen"),
                        row("dense", "robbyant/lingbot-map"), row("names", "gemini (panoptes-report-workspace)"), row("events", "Qwen/Qwen3-VL-8B-Instruct")])
COMMERCIAL_MODELS = {k: v for k, v in RESEARCH_MODELS.items() if k not in ("recgen", "dense")} | dict(
    [row("depth", "depth-anything/DA3-BASE"), row("register", "depth-anything/DA3-BASE#camera"), row("names", "Qwen/Qwen3-VL-Reranker-8B"),
     row("names-recall", "Qwen/Qwen3-VL-Embedding-8B")])
CAPS = {"splat_minutes": 58, "generator_usd": 10.}

PROFILES = {
    "research": Profile("research", RESEARCH_MODELS, ("sam3d", "recgen", "box"), True, "gemini", False, M2_RULES, CAPS),
    "commercial": Profile("commercial", COMMERCIAL_MODELS, ("sam3d", "box"), False, "qwen3vl", False, M2_RULES, CAPS),
    # the delivered reports ran all three generators and neither the static filter (D20) nor the lens gate on the import (D4)
    "delivered": Profile("delivered", {}, ("sam3d", "recgen", "box"), True, "gemini", True, "adopted", {}, ("static_filter", "lens_gate")),
}


def _models(spec):
    """(role, hf_id, revision, weights_sha256) rows of a stage: a StageSpec's .models, or a dict with 'models'."""
    rows = spec.get("models", ()) if isinstance(spec, dict) else getattr(spec, "models", ())
    return [tuple(r) + (None,) * (4 - len(tuple(r))) for r in rows]


def refuse(profile, spec):
    """Why this profile may not run this stage, or None. Only commercial refuses a stage for its models: a licence that is
    False or still to verify, an unpinned revision, the RecGen generator, the LingBot dense map. research refuses nothing;
    delivered refuses nothing here, it only ever serves hits (Profile.cache_only)."""
    profile = PROFILES[profile] if isinstance(profile, str) else profile
    if profile.name != "commercial":
        return None
    for role, hf_id, revision, _ in _models(spec):
        known = MODELS.get(hf_id)
        if known is None:
            return f"{role}: {hf_id} is not in the licence table"
        if known[3] is not True:
            return f"{role}: {hf_id} licence '{known[2]}' is {'not allowed' if known[3] is False else 'not verified'} for commercial use"
        if (revision or UNPINNED) == UNPINNED or known[0] == UNPINNED:
            return f"{role}: {hf_id} has no pinned revision"
        if revision != known[0]:
            return f"{role}: {hf_id} revision {revision} is not the pinned {known[0]}"
    return None


if __name__ == "__main__":
    assert refuse("research", {"models": [("depth", "depth-anything/DA3-GIANT-1.1", UNPINNED, None)]}) is None
    assert "not allowed" in refuse("commercial", {"models": [("depth", "depth-anything/DA3-GIANT-1.1", MODELS["depth-anything/DA3-GIANT-1.1"][0], None)]})
    assert refuse("commercial", {"models": [("depth", "depth-anything/DA3-BASE", MODELS["depth-anything/DA3-BASE"][0], None)]}) is None
    assert "pinned" in refuse("commercial", {"models": [("depth", "depth-anything/DA3-BASE", UNPINNED, None)]})
    assert PROFILES["delivered"].cache_only and not PROFILES["commercial"].dense_map and "recgen" not in PROFILES["commercial"].generators
    print("profiles self-check passed")
