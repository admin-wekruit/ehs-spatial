# Reconciliation: safety-compliance handoff vs. EHS Spatial MVP

Date: 2026-07-20. Repo state: `feature/ehs-spatial-mvp` @ `b94e6d9`.

The original safety-compliance handoff (fixed-camera inspection module) predates the
current carrier-agnostic photo MVP. Where they conflict, the owner's newer decisions
govern. This table records each conflict and its standing decision so deviations are
explicit, not silent.

| # | Handoff mandate | Current repo | Standing decision | Owner sign-off needed later? |
|---|---|---|---|---|
| 1 | Fully on-prem; no proprietary cloud APIs anywhere | All inference is cloud: Replicate MapAnything, fal SAM 3.1, Gemini 3.5 Flash | **Owner decision 2026-07-20: criterion is "on-prem capable", not on-prem-now.** Each model slot must have a self-hostable path: MapAnything (Apache checkpoint ✓), SAM 3.1 (weights downloadable under Meta SAM license ✓, legal review pending), Gemini 3.5 Flash (not self-hostable → swap target is an open VLM behind the existing adapter). Cloud endpoints are dev-time conveniences | Only SAM-license legal review + verifying each slot's self-host path before production |
| 2 | Swappable inference interface (backend swap, not rewrite) | Duck-typed constructor injection (`ehs_spatial/pipeline.py:14-29`); no Protocol/ABC; shared type is `ProviderError` only | Injection is sufficient for one implementation per role; formalize an interface only when a second backend exists | No (trigger: second backend) |
| 3 | TypeScript/Node orchestration + Firebase + Qdrant | Pure Python (Gradio, Open3D, Shapely, Pydantic) | **Owner decision 2026-07-20: Firebase dropped entirely; orchestration language open (TypeScript/Go/Rust/Python all acceptable).** Python stays for MVP | No |
| 4 | YAML rule engine: presence/state/zone/distance predicates + clause refs + severity | One hardcoded deterministic rule `fence_clearance` (`ehs_spatial/rules.py`), 0.6 m demo threshold | One rule proves the closed loop; rule engine deferred until rule #2 exists | Revisit at rule #2 |
| 5 | Apache/MIT models only; Grounding DINO for detection | fal SAM 3.1 (Meta custom SAM license — commercial allowed with trade-control/indemnity/CA-jurisdiction clauses, not Apache) | SAM 3.1 chosen for segmentation quality via hosted API | **YES** — license posture vs Apache-only preference before commercial packaging |
| 6 | Fixed-camera metrology ladder (image-space polygons → floor homography) | Carrier-agnostic 4-photo MapAnything multi-view metric geometry, camera-height scale | Newer product decision supersedes: carrier-agnostic is the bet; homography tiers not applicable | No |
| 7 | HITL review loop (CVAT/Label Studio → FiftyOne → LoRA later) | None | Out of MVP scope; MVP proves geometry+rule+grounding loop first | Deferred (Phase 2 class work) |
| 8 | Rules carry official regulation clause refs; auditable verdicts | 0.6 m is explicitly a demo threshold, "not an official EHS standard" (README, UI banner) | Frozen out of MVP by owner instruction | **YES** — before any customer-facing compliance claim |
| 9 | Asymmetric failure: below-confidence → human review, never auto-pass | `INSUFFICIENT_EVIDENCE` verdict path exists (fence/movable evidence gates in `rules.py`); no review queue | Verdict-level behavior already conservative; queue is HITL scope (#7) | Deferred with #7 |

## EHS eval datasets (V3/E1 recall) — status 2026-07-20

- **LOCO** (pallet/forklift/pallet_truck) — landed, 5,593 imgs, CC0 commercial-citable. `outputs/datasets/loco/`.
- **CylinDeRS** (gas_cylinder) — landed, 7,060 imgs / 25,278 boxes, CC BY 4.0 (m4d.iti.gr NC-ND discrepancy to verify). `outputs/datasets/cylinders/`.
- **SODA** (construction fence/scaffold) — ABANDONED: only surviving mirror is Baidu (slow, account-gated); SharePoint/aliyun mirrors dead; no HF/Zenodo/Kaggle copy of the *construction* SODA exists (Kaggle "SODA-A/D" is a different small-object dataset). Owner decision: use any anonymous fence source instead. Fence recall料 to come from a keyless HF/Zenodo/GitHub fence-detection set (fetch in progress).

Non-conflicts worth noting: Gemini never produces verdicts or distances (advisory,
fact_id-grounded only) — this matches the handoff's "compliance judgment stays in the
rule engine, never in model weights" principle. The deterministic evaluator over
geometry facts is exactly the handoff's rule-engine shape, just n=1 rules.
