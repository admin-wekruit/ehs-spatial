# EHS Spatial — first-principles requirements (eval + understanding)

Owner request 2026-07-20: "as PM and as eng lead, from first principles, what do
we need for EHS spatial eval + understanding." This document is the standing
answer; the validation ladder and build order derive from it.

## First principle

The product answers one question: **"is this industrial scene arranged in
compliance with written safety rules?"** Answering it requires exactly five
capabilities, no more:

| # | Capability | Question it answers |
|---|---|---|
| C1 | Semantic perception | what objects are here (per-site vocabulary) |
| C2 | Metric-spatial state | where / how big / what pose — position, extent, **orientation, tilt**, height, all in metres |
| C3 | Spatial relations | distances, containment, straddling, **overhang**, stacking, aisle occupancy |
| C4 | Normative evaluation | which written rules apply and what do they say about this state |
| C5 | Evidence & abstention | can every verdict be traced to pixels+points+numbers; when evidence is insufficient, say so |

Everything in the system is in service of one of these. Anything else is scope
creep.

## The binding layer (owner's "SAM3 的 map")

C1 outputs live in 2D pixels; C2 outputs live in 3D points. The **binding**
mask × pointmap → `Entity3D` is where semantics acquire metric state — it is a
first-class component, not glue:

- today: `_select_points` (mask pixels ∩ valid pointmap) → cluster → footprint
  hull + height; TO ADD: orientation (XY PCA yaw), tilt (3D principal axis vs
  gravity), layered footprints (overhang);
- binding error is its own eval axis (a perfect mask on a misaligned pointmap
  still yields a wrong entity);
- geometry slot is plural by design: multi-view (MapAnything today, DA3-Apache
  candidate) for full scenes; monocular depth (Depth Anything 3 / MoGe-2 MIT)
  as scale cross-check and single-photo fallback; scale anchors (camera height
  primary, EXIF intrinsics, known-dimension object) as configuration.

## PM view

**Users**: EHS manager (wants auditable verdicts + tolerance statements), site
operator (10-minute capture protocol, no training), future reviewer (HITL).

**Success metrics** (product-level, each must have a number before GA claims):
1. per-rule recall at fixed false-positive rate (false negatives are the
   dangerous mode; INSUFFICIENT_EVIDENCE rate reported alongside);
2. measurement tolerance statement ("clearance verdicts valid to ±X cm in
   0.3–3 m") backed by the tiered eval;
3. audit completeness: 100% of verdict sentences carry fact_ids (deterministic
   check, already enforced);
4. cost & latency per verdict; on-prem-capable proof (V4 swap).

**Data strategy — three sources, distinct jobs**:
- **Synthetic parametric** (owned, infinite, mathematically true): rule
  semantics + geometry-code correctness. Cannot validate models.
- **Public real** (Redwood PD, ETH3D NC-internal, EHS-class detection sets —
  search running): model quality tiers, component recall.
- **Owned captures** (design-partner site, protocol: 4 photos, measured camera
  height, tape distances, inclinometer for tilts): the only source that is
  simultaneously real, EHS-semantic, and commercially citable. Long-term moat.

## Eng lead view — eval architecture (each capability gets its own gauge)

| Eval | Measures | Ground truth | Status |
|---|---|---|---|
| E1 semantic | per-class recall/precision @ threshold sweep, prompt-ensemble sensitivity, unseen-vocab generalization | labeled EHS imagery | designed (V3); dataset hunt running |
| E2 metric | reconstruction distance error by input tier | laser scans / tape | **done, 3 tiers: 0.5 cm / 14.4 / 27.5–34 MAE** |
| E2.5 binding | mask→3D entity error (footprint IoU vs oracle, centroid error) | synthetic oracle + annotated real | partially implicit in offline eval; make explicit |
| E3 state | orientation° / tilt° / height accuracy | synthetic oracle; real: inclinometer + tape | **missing — next build** |
| E4 rule semantics | predicate correctness on adversarial scenario matrix (boundary 0.58/0.60/0.62, straddle, 45° yaw, leaning ladder, overhang) + domain-partner truth table | analytic truth + human sign-off | **missing — next build** |
| E5 end-to-end | verdict vs expert judgment on real scenes; per-rule recall@FPR | owned captures + partner review | blocked on owned data |
| E6 robustness | model-swap invariance (SAM3↔LLMDet+SAM-HQ, MapAnything↔DA3), viewpoint perturbation (answer must track geometry, not priors) | same packs re-run | V4 |

**Build order** (each step unblocks the next, none speculative):
1. `Entity3D` gains `orientation_deg`, `tilt_deg`, layered footprint (binding
   layer upgrade) + unit tests. [C2]
2. Synthetic generator: parametric scenario matrix (angle/tilt/straddle/
   boundary/overhang) + E3/E4 offline assertions. [C3/C4 truth]
3. Rule engine: predicate types generalized toward presence/state/zone/distance
   (state = orientation/tilt/height bands), rules as per-site config with
   clause refs. [C4]
4. Owner capture protocol extended: tape + inclinometer per scene → E5 seed.
5. V3 fence/EHS-class recall on found datasets → E1.
6. V4 swap matrix → E6 + on-prem proof.

**Standing invariants** (already validated, do not regress): deterministic core
(146+ tests, oracle 4/4), fact_id grounding 100%, asymmetric failure
(INSUFFICIENT over guess), pay-once-iterate-free eval design, every model
behind a swappable adapter, licenses tracked per artifact.
