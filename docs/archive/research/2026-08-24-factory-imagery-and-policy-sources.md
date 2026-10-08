# Factory imagery + policy-text sources — scouted, verified, parked

Date 2026-08-24. Three parallel scouts (factory-interior datasets /
multi-view industrial sources / written policy corpora) with an adversarial
verification pass that decoded actual frames. 58 candidates.

**Status: PARKED.** The MVP audit's ruling stands — stop adding benchmarks,
close the product loop. This file exists so the research is not re-done.

## Imagery — what survived frame-level verification

| Source | Domain | Multi-view | Metric GT | License | Verdict |
|---|---|---|---|---|---|
| **NVIDIA PhysicalAI-SmartSpaces MTMC_Tracking_2026** | real warehouse, 28 scenes | yes, calibrated | camera calibration | CC-BY-4.0, ungated | **ADOPT — best find** |
| **LOCO** (TUM) | real warehouse/logistics halls | weak (walkthrough frames) | none (2D boxes) | **CC0** | **ADOPT** as imagery |
| NVIDIA SDG-Warehouse (synthetic ops) | warehouse, EHS incident scenarios | 5–10 synced views | synthetic exact | OpenMDW-1.1, commercial OK | ADOPT as synthetic arm |
| ARKitScenes | domestic/office | yes | FARO laser | Apple custom | already in use |
| ETH3D | mixed indoor incl. plant room | yes | laser | CC-BY-NC-SA | already in use, internal only |

Verified as the frame check: the MTMC warehouse frame decodes to a genuine
room-scale aisle with pallet racking, a forklift and people — the domain we
claim to serve.

### Downgraded by the verify pass — the useful part of the exercise

- **IndEgo** (Fraunhofer): the first scout's top pick. Frame decode killed it
  on scene content — egocentric activity footage, not room-scale inspection
  views.
- **SH17**: "PPE in manufacturing" does not survive looking at the pixels —
  shallow-depth-of-field stock portraits, not halls.
- **Hilti SLAM**: indoor but rig/fisheye footage; rosbag plumbing, NC.
- **Places365 factory classes**: content is right, access + NC license wrong
  for anything but an internal smoke set.

## Policy text — OSHA is public domain and quotable verbatim

The single most useful finding for the policy compiler. US 29 CFR is a
government work: quote and ingest freely.

| Rule | What it gives the compiler |
|---|---|
| **1910.303(g)** Table S-1 | electrical working-space depths — clean object-to-wall distances, verbatim from the eCFR XML |
| **1910.29** | fall-protection heights — single-object height predicates, the *easiest* regime for a monocular harness |
| **1910.253(b)** | oxygen/fuel-gas cylinder separation — the best object-to-object distance rule available |
| **1910.219** | a height threshold that gates whether a guard is required at all — conditional geometry |
| **1910.36 / .37** | egress width and clearance — a positive and a negative case in one place |
| **eCFR Versioner API** | the ingestion mechanism: pull subparts D/E/N/O/S in one script |

**Deliberate refusal cases** — as valuable as the compilable ones, because
the compiler must decline them:

- **1910.176(a)** "sufficient safe clearances" — states no number.
- **1910.333(c)** Table S-5 — threshold keyed on *voltage*, not observable.
- **1910.157(d)** extinguisher travel distance — *path* distance, needs
  floor-plan topology, not single-photo depth.

ISO 13857 / 13854 / NFPA 70E: numbers are usable, **tables must never be
redistributed** (paywalled, restrictive). Cite, do not ingest. EU Machinery
Regulation 2023/1230 (applies 2027-01-20) is a structural case, not a numeric
source.

## If and when this is unparked

1. MTMC warehouse — real-domain harness test on genuine warehouse aisles.
2. OSHA subset into the policy compiler test set, refusal cases included.
   This is the cheapest way to prove the compiler declines what it should.
3. LOCO — entity/recall layer only; it has no metric ground truth.
