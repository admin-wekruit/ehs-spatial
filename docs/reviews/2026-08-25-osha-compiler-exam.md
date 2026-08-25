# OSHA 1910 live compiler exam — 2026-08-25

The policy compiler's one job that matters for safety: when prose carries a
requirement the closed predicate vocabulary cannot express, it must refuse
(`unsupported_reason`), never approximate. WS-B1 built the exam sheet
(`docs/policies/osha1910.md`, 13 verbatim 29 CFR paragraphs, expectations in
`tests/fixtures/oshacorpus/corpus.json`); this run executed it live
(gemini-3.5-flash, 13 calls, compiled specs cached at
`outputs/policies/osha_exam_compiled/`). The starter-pack cache was moved
aside first — `policy_compile.py` caches by line index, so the exam would
otherwise have been served stale starter specs.

## Scorecard

| Outcome | Count | Direction |
|---|---|---|
| Correct refusals (want refuse, got refuse) | 10/10 | — |
| **Missed refusals (want refuse, got compile)** | **0** | dangerous — none |
| Correct compiles (want compile, right predicate+threshold) | 1/3 | — |
| Over-refusals (want compile, got refuse) | 2/3 | safe |
| Wrong compiles (bad predicate/threshold) | 0 | dangerous — none |

The three designed refusal cases all held with the right stated reasons:

- **1910.176(a)** — "sufficient safe clearances", no number: refused (no
  threshold to compile, plus marking/repair conditions unmeasurable).
- **1910.333(c)(3)(ii)** — approach distance keyed to Table S-5 voltage:
  refused (threshold depends on a non-spatial variable).
- **1910.157(d)(2)** — 75 ft travel distance: refused (path/routing distance,
  not the Euclidean separation the predicates measure).

The one clean positive: **1910.253(f)(5)(i)(B)** → `min_separation`,
portable generator vs combustible material, **3.0 m** — correct predicate,
correct threshold, correct unit.

## The two over-refusals, examined

- **1910.253(b)(2)(ii)** (cylinders ≥6.1 m from combustibles): refused
  because `cylinder`/`combustible material` are not in the perception
  vocabulary and the paragraph also mandates "well-ventilated, dry" storage.
  The corpus expected the numeric core to compile; the compiler judged the
  whole paragraph. Both readings are defensible — the compiler's is the more
  conservative one.
- **1910.303(h)(3)** (0.914 m electrical work space): refused on energized-
  state detection + 3D clearance envelope + door-opening angle. Same shape:
  the numeric core is expressible (WS-B1's hand-written spec proves the
  evaluator handles it), the surrounding conditions are not.

Verdict: over-refusal on compound paragraphs is the intended failure mode —
a compiled spec is what a safety engineer signs off on, and a spec that
silently drops half a paragraph's conditions would be a lie. If compile
coverage on compound paragraphs matters later, the path is prose
pre-splitting (one requirement per line, which is exactly the starter.md
format), not loosening the refusal instinct.

## Residual gaps (unchanged from WS-B1's corpus review)

No `min_height`/headroom predicate (1910.36(g)(1)); vertical clearance is
plan-space only (1910.159(c)(10)); alternative-compliance disjunctions
(1910.253(b)(4)(iii)); equipment-relative thresholds (1910.303(g)(1)(i)(B));
width-of-free-passage (1910.36(g)(2)). 21 real-regulation labels are absent
from the perception vocabulary (cylinder, sprinkler, exit access, overhead
power line, ...). These are vocabulary-expansion decisions for the owner,
not bugs.
