<!-- Survey A (rule languages, representations, tooling), produced 2026-10-07 by a research agent for docs/research/verdict-layer-rules-2026-10-07.md. Verbatim; links as the agent reported them. -->

# Verdict-layer survey for Panoptes: representations, rule languages, tooling

Scope: static 3D workcell scene (≈10–100 objects, OBB + mesh + floor gap + confidence + views, metric scale from e-stop reference) → per-requirement **pass / fail / needs-measurement** verdict with evidence. Rule classes: TOPOLOGY, SHAPE/GEOMETRY (numeric), SEMANTIC; video/temporal later. Versions checked October 2026. "Unverified" marks claims I could not confirm from a primary source.

Throughout, the four running rules are: **(a)** light-curtain distance `S ≥ K·T + C`, K=1600 mm/s, T=0.2 s, C=8·(d−14) for d=14..40 mm (note ISO 13855 uses K=2000 for S≤500 mm and allows K=1600 only when the result stays ≥500 mm — encode that branch too: https://www.leuze.com/en-at/navigation/01993da1a26a7b1b978fd45fee026108); **(b)** fence floor gap ≤180 mm / ISO 13857 opening-vs-reach tables (https://evs.ee/en/iso-13857-2019); **(c)** every hazard zone topologically enclosed by guards or light curtains; **(d)** an e-stop reachable from each operator position.

A design premise for all options: **geometry is computed in Python, not in the logic**. Relations such as `adjacent`, `encloses`, `gap_to_floor`, `distance`, `between`, `line_of_sight` are derived once from OBBs/meshes (shapely 2.x for plan-view footprints — GEOS predicates are strictly 2D, Z is discarded: https://shapely.readthedocs.io/en/latest/faq.html; trimesh for 3D containment/`contains`/booleans: https://pypi.org/project/trimesh; Open3D for OBBs: https://www.open3d.org/docs/latest/python_api/open3d.geometry.OrientedBoundingBox.html) and emitted as facts. The rule language then only needs comparison, arithmetic, negation, recursion and aggregation.

---

## 1. RDF + OWL 2 + SWRL + SPARQL

**What/maturity.** OWL 2 (W3C Rec, profiles EL/QL/RL: https://www.w3.org/TR/owl2-profiles/) is a description logic for *inference under the open-world assumption (OWA)*. SWRL (W3C member submission, https://www.w3.org/submissions/SWRL/) adds Horn rules with math/comparison built-ins (`swrlb:multiply`, `swrlb:greaterThan`, https://www.daml.org/2004/04/swrl/builtins.html). SPARQL 1.1 is the query language. Reasoner health is poor: HermiT's repo is abandoned (last release 1.3.8/1.4.3 bundled with Protégé, https://github.com/phillord/hermit-reasoner), Pellet is continued as Openllet 2.6.x (Java, https://github.com/Galigator/openllet), ELK 0.6 is EL-only and fast (https://github.com/liveontologies/elk-reasoner); see "OWL reasoners still usable in 2023" (https://arxiv.org/abs/2309.06888).

**Python.** owlready2 0.51 (LGPL; bundles HermiT and Pellet JARs, requires Java; SWRL rules with built-ins run under `sync_reasoner_pellet`: https://owlready2.readthedocs.io/en/v0.51/reasoning.html, https://owlready2.readthedocs.io/en/v0.48/rule.html), rdflib 7.x (BSD), owlrl 6.0.2 (OWL-RL/RDFS forward chaining in pure Python, https://pypi.org/project/owlrl).

**Rules.**
(a) SWRL: `LightCurtain(?lc) ^ resolution_mm(?lc,?d) ^ stopTime_s(?lc,?t) ^ distToHazard_mm(?lc,?s) ^ swrlb:subtract(?dm,?d,14) ^ swrlb:multiply(?c,8,?dm) ^ swrlb:multiply(?kt,1600,?t) ^ swrlb:add(?smin,?kt,?c) ^ swrlb:greaterThanOrEqual(?s,?smin) -> DistanceCompliant(?lc)` — derives *pass* only; "fail" needs the negation OWL lacks, so you end up writing the complement as a SPARQL query anyway.
(b) OWL class: `Fence and (floorGap_mm some xsd:decimal[<= 180])` ⊑ `GapCompliant` — a fence with no measured gap is simply *unknown*, never a violation.
(c) Enclosure is "no uncovered boundary segment exists": a closed-world statement, inexpressible in OWL without asserting completeness (OWA kills it).
(d) `OperatorPosition and (canReach some EStop)` — again derives pass only; `canReach` must be precomputed.

**Verdict/explanation.** Entailment yes/no; justification extraction exists only in Java (OWL API), not in owlready2. **Verdict: avoid as the rule engine**; at most use RDF/OWL as an exchange vocabulary. Compare SHACL-vs-OWL semantics in https://arxiv.org/abs/2507.12286 and https://arxiv.org/abs/2108.06096.

## 2. SHACL (Core, SHACL-SPARQL, SHACL-AF/Rules), pySHACL, SHACL 1.2

**What/maturity.** SHACL 1.0 (W3C Rec 2017, https://www.w3.org/TR/shacl/) validates RDF under the *closed-world* assumption and returns a structured **validation report** (`sh:focusNode`, `sh:resultPath`, `sh:value`, `sh:sourceShape`, `sh:resultMessage` with `{$value}` templating, three severities `sh:Violation/Warning/Info`). SHACL-SPARQL adds arbitrary SPARQL constraints; SHACL-AF (https://w3c.github.io/data-shapes/shacl-af/) adds rules (TripleRule/SPARQLRule) and functions. **SHACL 1.2** is in progress in the W3C Data Shapes WG: Core WD (latest 2026-08-28, https://www.w3.org/TR/2026/WD-shacl12-core-20260828/), SPARQL Extensions WD (https://www.w3.org/TR/2025/WD-shacl12-sparql-20250828/), Rules WD with a new "SHACL Rules Language" text syntax (https://www.w3.org/TR/shacl-rules/), Node Expressions FPWD Jan 2026 (https://lists.w3.org/Archives/Public/public-review-announce/2026Jan/0000.html). All still Working Drafts — do not depend on 1.2-only features yet.

**Python.** pySHACL 0.40.1 (Apache-2.0, 2026-07-28, https://pypi.org/project/pyshacl/) implements Core, SHACL-SPARQL, SHACL-AF rules/functions and node expressions. TopBraid is the commercial reference (Java). Explanation tooling: xpSHACL builds justification trees + LLM text from reports (https://arxiv.org/abs/2507.08432).

**Rules.**
(a) SHACL-SPARQL:
```
pan:LCDistance a sh:NodeShape ; sh:targetClass pan:LightCurtain ;
 sh:sparql [ sh:message "S={?s} mm < required {?smin} mm (K=1600,T={?t},C=8*(d-14),d={?d})" ;
  sh:select """SELECT $this ?s ?smin ?t ?d WHERE {
   $this pan:resolution_mm ?d ; pan:stopTime_s ?t ; pan:distToHazard_mm ?s .
   BIND(1600*?t + 8*(?d-14) AS ?smin) FILTER(?s < ?smin) }""" ] .
```
(b) SHACL Core: `sh:property [ sh:path pan:floorGap_mm ; sh:maxInclusive 180 ; sh:minCount 1 ; sh:message "gap {$value} mm > 180 mm" ]` — put `sh:minCount 1` in a *second* shape with `sh:severity sh:Warning` so "no measurement" maps to **needs-measurement** rather than fail.
(c) Enclosure = "an outside cell reaches a hazard cell through free cells": `SELECT $this WHERE { ?o a pan:OutsideCell ; pan:adjFree+ ?c . ?c pan:inZone $this }` — works, but SPARQL property paths cannot filter per step, so `adjFree` edges must be materialised first (SHACL rule or Python).
(d) `sh:sparql SELECT $this WHERE { FILTER NOT EXISTS { $this pan:canReach ?e . ?e a pan:EStop } }` with `canReach` precomputed.

**Strengths.** CWA, native three-level verdict, standardised machine-readable evidence, arithmetic via SPARQL `BIND`, aggregation via `GROUP BY/COUNT`. **Limits.** RDF plumbing (IRIs, Turtle) for a 50-object scene; recursion only via property paths; no floats-with-units; pySHACL performance is fine at this scale. **Fit:** strongest standards-based option; best if customers want RDF deliverables (the BIM lineage in §7 uses it).

## 3. Spatial calculi, GeoSPARQL, QSR tools, 3D ontologies

RCC-8 and Egenhofer's 9-intersection/DE-9IM are the standard qualitative topologies (survey: https://arxiv.org/abs/1606.00133). **GeoSPARQL 1.1** (OGC 2024, https://docs.ogc.org/is/22-047r1/22-047r1.html) exposes Simple-Features, RCC-8 and DE-9IM predicates (`geof:rcc8ntpp`, `geof:sfContains`, `geof:relate`) plus 1.1 additions (SHACL shapes, more functions), but the spec contains no 3D semantics; triplestores mostly flatten or mishandle 3D (https://isprs-archives.copernicus.org/articles/XL-4-W7/11/2015/isprsarchives-XL-4-W7-11-2015.pdf, compliance benchmark https://arxiv.org/abs/2102.06139). Rule (c) in GeoSPARQL: `FILTER(!geof:rcc8ntpp(?zoneWKT, ?guardedWKT))` on plan-view footprints — fine for fences in plan, blind to floor gaps and heights.

QSR constraint solvers **SparQ** (Lisp, GPL-3, https://github.com/dwolter/SparQ) and **GQR** (C++, https://openscience.org/gqr/) do composition/path-consistency over *unknown* relations. Panoptes has metric geometry, so relations are *computed*, not inferred — skip them. **CLP(QS)** (https://sfbtr8.spatial-cognition.de/aigaion/index.php/publications/show/90.html) and **ASPMT(QS)** (https://arxiv.org/abs/1606.07860; polynomial constraints via SMT inside ASP) are research prototypes, but their architecture — logic for rules, numeric theory for geometry — is exactly the clingo + Python split recommended below.

Python geometry: shapely 2.x (2D, `relate` gives the DE-9IM string), trimesh (3D meshes, `contains`, OBB, boolean), Open3D, PyVista (visualisation). Ontologies: **BOT** (W3C LBD, zones/elements/adjacency: https://www.semantic-web-journal.net/content/bot-building-topology-ontology-w3c-linked-building-data-group-0), **ifcOWL** (https://standards.buildingsmart.org/documents/20170830_LDWG_ifcOWLontology.pdf), **IndoorGML 2.0** (OGC Aug 2025, cell-space + navigation graph: https://www.ogc.org/ogc-publishes-indoorgml-2-0-part-1-conceptual-model-standard/), **CityGML 3.0** space concept (https://docs.ogc.org/is/20-010/20-010.html), **SOSA/SSN** for sensors (https://www.w3.org/TR/vocab-ssn/). Borrow BOT's `containsZone/adjacentZone/hasElement` naming and IndoorGML's "cell + connectivity graph" idea; adopting any of them wholesale buys nothing for a workcell.

## 4. Logic programming and production-rule engines

**ASP / clingo.** clingo 5.8.0 (MIT, pip `clingo`, 2025-04: https://github.com/potassco/clingo/releases) gives negation-as-failure, recursion, integer arithmetic, aggregates (`#count/#sum/#min/#max`), and a stable Python API. Numeric extensions: clingo-dl (difference constraints, https://potassco.org/clingoDL/), clingcon (integer linear, https://anaconda.org/conda-forge/clingcon), **clingo-lpx** 1.3.0 (rational linear constraints via simplex, 2025-12, https://github.com/potassco/clingo-lpx; theory paper https://arxiv.org/abs/1707.04053). Use integer millimetres/milliseconds and you rarely need them. Explanations: xclingo (annotated derivation trees, https://arxiv.org/abs/2009.10242), xASP2 (why/why-not graphs incl. aggregates, https://par.nsf.gov//servlets/purl/10462534) — or simply make verdict atoms carry their evidence.

The four rules (facts come from Python):
```
% (a)
req_s(LC,S) :- light_curtain(LC), res_mm(LC,D), stop_ms(LC,T), D>=14, D<=40, S = 1600*T/1000 + 8*(D-14).
fail(lc_dist,LC,have(H),need(N)) :- req_s(LC,N), dist_hazard_mm(LC,H), H < N.
pass(lc_dist,LC) :- req_s(LC,N), dist_hazard_mm(LC,H), H >= N.
needs_meas(lc_dist,LC) :- light_curtain(LC), not dist_hazard_mm(LC,_).
% (b)  ISO 13857 table as facts: req_sr(Emin,Emax,Sr).
fail(fence_gap,F,gap(G)) :- fence(F), floor_gap_mm(F,G), G > 180.
needs_meas(fence_gap,F) :- fence(F), floor_gap_mm(F,_), conf(F,C), C < 70.
fail(reach,O,have(S),need(R)) :- opening(O,E), hazard_dist_mm(O,S), req_sr(Lo,Hi,R), E>Lo, E<=Hi, S<R.
% (c)  cells from a plan-view grid; blocked(C) if a guard/light curtain OBB covers C
reach(C) :- outside(C).  reach(C2) :- reach(C1), adj(C1,C2), not blocked(C2).
fail(enclosure,Z,via(C)) :- hazard_zone(Z), cell_of(Z,C), reach(C).
pass(enclosure,Z) :- hazard_zone(Z), not fail(enclosure,Z,_).
% (d)
reachable(P,E) :- operator_pos(P), estop(E), dist_mm(P,E,D), D<=600, height_mm(E,H), H>=600, H<=1700, not occluded(P,E).
fail(estop,P) :- operator_pos(P), not reachable(P,_).  pass(estop,P,E) :- reachable(P,E).
```
Verdict priority (`fail > needs_meas > pass`) is one more rule. The grounder complains on floats, so keep units integral.

**Datalog.** Soufflé (C++, UPL-1.0; aggregates, arithmetic; proof trees via `-t explain` and `explainnegation`: https://souffle-lang.github.io/provenance) is excellent but a separate binary with thin Python glue. **pyDatalog is unmaintained** (https://github.com/pcarbonn/pyDatalog). **Nemo** (Rust, existential rules + arithmetic + aggregates, Python API: https://github.com/knowsys/nemo) is the live alternative. LogicBlox lineage survives commercially only.

**Prolog.** SWI-Prolog + **janus-swi** 1.5.2 (official bidirectional bridge, pip; pyswip on PyPI is stale: https://www.swi-prolog.org/FAQ/Python.md, https://pypi.org/project/janus-swi/). Floats are native (`S is 1600*T + 8*(D-14)`), explanation requires a meta-interpreter. Equivalent power to clingo, less declarative about "all answers".

**Production/policy engines.** durable_rules is dead (last release ~2020, https://cloudsmith.com/navigator/pypi/durable-rules); Drools is Java/Rete (https://www.drools.org/). **OPA/Rego** has arithmetic, `sum/count/max` (https://www.openpolicyagent.org/docs/policy-reference/builtins/aggregates), `opa eval --explain` traces, but no recursion (rule (c) needs a precomputed closure) and Python only via subprocess/WASM. **CEL** (Google's cel-expr-python, open-sourced March 2026: https://opensource.googleblog.com/2026/03/announcing-cel-expr-python-the-common-expression-language-in-python-now-open-source.html) and **JSON-Logic** (https://pypi.org/project/json-logic) are expression languages: perfect for (a)/(b) thresholds stored as data, useless for (c)/(d). Verdict: production engines handle numeric rules and tracing; only logic programming handles topology natively.

## 5. Temporal/signal logics for the video path (STL = Signal Temporal Logic, not the mesh format)

LTL (discrete, qualitative), MTL (bounded intervals), **STL** (real-valued predicates over signals) with *robustness semantics* — a signed margin, e.g. `ρ = S_measured − S_required` — are the standard toolkit; RTAMT's paper is a good primer (https://arxiv.org/abs/2501.18608). Tools: **RTAMT** (Python, offline/online, discrete and dense time, ROS bridge: https://pypi.org/project/rtamt/), **MoonLight** (Java with Python/MATLAB interfaces; implements **STREL** — STL plus `reach/escape/somewhere/everywhere` over a *weighted graph*, i.e. a scene graph: https://github.com/MoonLightSuite/MoonLight, https://arxiv.org/abs/2104.14333), **Reelay** (C++ header-only, past-time LTL/MTL/STL and first-order extensions: https://arxiv.org/abs/2604.22384), **Breach** (MATLAB), **S-TaLiRo** (MATLAB, https://home.cs.colorado.edu/~srirams/papers/sTaliro-tacas11.pdf) and **PSY-TaLiRo** (Python falsification: https://arxiv.org/abs/2106.02200). Spatio-temporal logics: SpaTeL (https://calinbelta.com/wp-content/uploads/2023/12/SpaTel-HSCC2015.pdf), SSTL (https://arxiv.org/abs/1706.09334), STL-GO graph operators (https://arxiv.org/abs/2507.15147), Differentiable SpaTiaL for object-geometry manipulation specs (https://arxiv.org/abs/2604.02643).

Scene-graph + temporal logic for monitoring is an active AD line: PerceMon/TQTL (https://arxiv.org/abs/2108.08289) → **STPL** adds spatial operators and is polynomially monitorable offline (https://arxiv.org/abs/2206.14372); **Scene Flow Specifications** (FSE 2025) show first-order-over-scene-graphs + LTLf encodes 96 % of AD specs vs 76 % for prior logics (https://conf.researchr.org/details/fse-2025/fse-2025-research-papers/72/Scene-Flow-Specifications-Encoding-and-Monitoring-Rich-Temporal-Safety-Properties-of); **Abstract Scene Graphs** (FMAS 2025) formalise spatial properties for runtime monitoring (https://arxiv.org/abs/2511.14430); STADA generates tests from LTLf over scene flows (https://arxiv.org/abs/2603.10940). Takeaway for Panoptes: STL has no object quantifiers, so run the static verdict rules per frame and feed STL with the resulting *signals* (`gap_mm(fence3,t)`, `enclosed(zone1,t)`), e.g. RTAMT `always[0,T] (gap_fence3 <= 180)` and `always (door_open -> eventually[0,2] robot_stopped)`. Robustness then equals your evidence margin.

## 6. 3D scene graphs as representation

Armeni et al. 2019 define the hierarchy building→room→object→camera with attributes and relations (https://3dscenegraph.stanford.edu). **Hydra** (RSS 2022, https://arxiv.org/abs/2201.13360), Hydra-Multi (https://arxiv.org/abs/2304.13487) and **Clio** (task-driven open-set, https://arxiv.org/abs/2404.13696) all use **Spark-DSG**: a C++ layered graph (places, rooms, objects, buildings, agents) with typed inter/intra-layer edges, node attributes including bounding box, position and semantic label, JSON serialisation, pip-installable Python bindings, BSD-2 (https://github.com/MIT-SPARK/Spark-DSG). **ConceptGraphs** stores object nodes with CLIP features/captions and LLM-labelled edges; relation labels ≈90 % accurate vs ≈70 % node labels (https://arxiv.org/abs/2309.16650); **HOV-SG** adds floor/room hierarchy (https://arxiv.org/abs/2403.17846); **Open3DSG** predicts graphs from point clouds (https://arxiv.org/abs/2402.12259); SceneGraphFusion is the incremental GNN precursor (https://arxiv.org/abs/2103.14898). Temporal follow-ups: **Khronos** spatio-temporal SLAM (https://arxiv.org/abs/2402.13817), Lost & Found change tracking in dynamic scene graphs (https://arxiv.org/abs/2411.19162), Aion 4D scene graphs (https://www.alphaxiv.org/abs/2512.11903), relationship-aware hierarchical 3DSGs (https://arxiv.org/abs/2602.02456).

Querying: in practice NetworkX in-process or an LLM reading serialised JSON; the 2025 MIT study shows **Cypher over a graph DB** scales far better than dumping the graph into an LLM context (https://arxiv.org/abs/2510.16643), Neo4j demoed versioned robot scene memory (https://neo4j.com/?p=428928), and Beetz's group converts USD scenes into ontology knowledge graphs answering competency questions (https://arxiv.org/abs/2507.11770). Compliance use is rare but emerging: regulation-aligned PPE checking compares predicted scene graphs with rule-derived graphs (https://www.sciencedirect.com/science/article/abs/pii/S1474034626003046); SafeSceneReason pairs "safety scene graphs" with declarative rules and executable programs (https://arxiv.org/abs/2608.09230); scene-graph-guided hazard scenario synthesis (https://arxiv.org/abs/2511.13970). None do metric guard/light-curtain rules — Panoptes would be first here.

## 7. BIM/AEC code-compliance lineage (the solved analogue)

Eastman et al. 2009 set the four stages — rule interpretation, model preparation, rule execution, reporting (https://doi.org/10.1016/j.autcon.2009.07.002) — and this is exactly your pipeline. **RASE** marks regulatory text with Requirement/Applicability/Selection/Exception operators before formalisation (https://architektur-informatik.scix.net/paper/w78-2011-Paper-45); **LegalRuleML** (OASIS 2021) encodes deontic/defeasible legal rules in XML (https://docs.oasis-open.org/legalruleml/legalruleml-core-spec/v1.0/legalruleml-core-spec-v1.0.html) — heavyweight, use only if you must trace to statute clauses. **Solibri** ships ~50 parameterised rule templates, closed source (https://help.solibri.com/hc/en-us/articles/1500005009042-Understanding-Checking). The semantic-web branch encodes clauses as SHACL over ifcOWL/BOT (French smart building code: https://arxiv.org/abs/1910.00334; ifcOWL SHACL validation at LDAC: https://ceur-ws.org/Vol-2636/07paper.pdf). **ACCORD** (Horizon Europe 2022–2025, https://cordis.europa.eu/project/id/101056973) built the AEC3PO ontology, RASE-based manual plus NLP-assisted rule formalisation, and SHACL as the rule carrier (https://build-up.ec.europa.eu/en/news-and-events/events/accord-digital-building-permits-semantics-and-rule-formalisation-processes, https://ceur-ws.org/Vol-3633/paper13.pdf), followed by a Building Compliance Ontology with SHACL checks for structural codes (https://zenodo.org/records/15374643). **buildingSMART IDS 1.0** (final standard June 2024, https://www.buildingsmart.org/?p=30065, https://github.com/buildingSMART/IDS) is a facet-based constraint language (entity/attribute/property/classification/material/partOf, with enumerations, regex, bounds) — good for SEMANTIC attribute checks, no cross-object arithmetic or topology; IfcOpenShell's `ifctester` validates it in Python. LLM-based ACC: I-SNACC (NLP + logic, not LLM) reports 95.2 % precision / 100 % recall on IBC 2015 Ch. 10 (https://par.nsf.gov/biblio/10420120); 2025 work has GPT/Claude/Gemini/Llama generating Python check scripts against Revit models without reported accuracy figures in the abstract (https://arxiv.org/abs/2506.20551); a 2026 systematic review surveys interpretation methods (https://www.tandfonline.com/doi/full/10.1080/09613218.2026.2637965); an AiC 2025 framework targets transparency/validation of rule interpretation (https://www.sciencedirect.com/science/article/pii/S0926580525006387). The "97.9 % code classification" figure that circulates in secondary sources is **unverified**.

## 8. Neuro-symbolic pattern: LLM compiles, engine judges

The pattern is well established: **Logic-LM** (LLM → symbolic solver, +39.2 pts over standard prompting, self-refinement from solver errors: https://arxiv.org/abs/2305.12295); LLM→ASP (https://arxiv.org/abs/2307.07699), LLASP fine-tuning (https://proceedings.kr.org/2024/78/kr2024-0078-coppolillo-et-al.pdf), self-correcting LLM+ASP (https://aclanthology.org/2026.findings-acl.1151/); temporal: **nl2spec** (CAV 2023) maps sub-formulas back to NL fragments so a human can accept/edit each sub-translation (https://arxiv.org/abs/2303.04864), **NL2LTL** (IBM, pip, MIT: https://github.com/IBM/nl2ltl), **NL2TL** (28 K pairs, >95 % with domain fine-tuning: https://arxiv.org/abs/2305.07766), Lang2LTL (https://people.csail.mit.edu/ajshah/publication/corl-2022-lang2ltl); RDF: **NL2SHACL-Bench** (2026) finds LLMs produce syntactically valid SHACL reliably but fail on semantic equivalence for complex logical/path patterns (https://arxiv.org/abs/2608.07530), PolicyKG compiles institutional policy to SHACL at corpus scale (https://arxiv.org/abs/2608.09028), LLM→SPARQL with retrieval (https://arxiv.org/abs/2410.06062). Failure modes: wrong operator precedence, silent unit/constant errors, dropped exceptions (the RASE "E"), hallucinated predicates not in the schema. Validation that works: (1) **round-trip** formal→NL→formal with an equivalence check — diagnosis-guided repair raised formal equivalence from 45–61 % to 83–85 % on statutory text (https://arxiv.org/abs/2604.25031); (2) **LLM-generated unit test cases** for specs, then run them (https://arxiv.org/abs/2510.23350); (3) nl2spec-style sub-translation review; (4) differential testing of a compiled rule against hand-written reference rules on synthetic scenes. For Panoptes: constrain the LLM to a fixed predicate vocabulary, require one positive and one negative synthetic scene per rule, and keep the standard clause ID in the rule as metadata.

---

## Ranked recommendation for Panoptes

**Representation: a typed scene graph in Python, not flat JSON and not RDF-first.** Keep today's object list, add an explicit relation layer computed geometrically (`adjacent`, `contains`, `between`, `floor_gap`, `distance`, `line_of_sight`, plan-view occupancy cells) and store it as a NetworkX graph with a JSON(-LD-compatible) serialisation. Flat JSON forces every rule to recompute topology; RDF adds IRI/Turtle friction with no benefit until a customer asks for SHACL reports — at which point the same graph exports to rdflib in ~50 lines.

**Rule languages per class.**
1. **ASP/clingo for TOPOLOGY and SEMANTIC and as the single verdict engine** — closed-world, recursion for enclosure/reachability, aggregates, integer arithmetic in mm, three-valued verdicts fall out of negation-as-failure, evidence carried in the atoms, pip-installable, MIT.
2. **Numeric SHAPE/GEOMETRY rules**: write the formula in clingo (integer mm) when it feeds other rules; otherwise keep thresholds as data (CEL/JSON-Logic style dict → evaluated in Python). Reach for clingo-lpx only if rationals become unavoidable.
3. **SHACL (pySHACL) as an optional export/reporting layer** for standards-minded customers; it is the closest to BIM practice and its validation report is a ready evidence format.
4. **Temporal (video)**: run the static engine per frame, feed boolean/margin signals to RTAMT STL; look at MoonLight/STREL if you need graph-spatial temporal operators.

**MVP stack.** Python 3.11+, numpy, shapely 2.x, trimesh, Open3D (existing), NetworkX 3.x, `clingo` 5.8, `xclingo` (explanations), pydantic for the `Verdict{rule_id, clause_ref, status, measured, required, evidence_objects, views, confidence}` schema; later `rtamt`; optional `rdflib` + `pyshacl` for export. One module computes facts from the scene graph; one `.lp` file per rule family; one Python function maps answer-set atoms to Verdict records.

**Avoid and why.** OWL/SWRL (open world, no fail-by-absence, dead Java reasoners); QSR solvers SparQ/GQR (you have metric geometry); GeoSPARQL as a store (2D-only, 3D unsupported in practice); durable_rules/pyDatalog (unmaintained); IDS/LegalRuleML as the core (attribute-only / deontic XML, no geometry); letting the LLM judge — use it only to compile clauses into clingo rules, validated by round-trip plus synthetic pass/fail scenes.

| Option | Topology (c,d) | Numeric (a,b) | Semantic | Explanation/evidence | Python maturity | Fit for Panoptes |
|---|---|---|---|---|---|---|
| Flat JSON + ad-hoc Python | recomputed per rule | good (floats) | good | hand-built | native | baseline; does not scale in rules |
| Typed scene graph (NetworkX) + Python checks | good (graph algos) | good | good | hand-built | native | good; no declarative rules |
| RDF + OWL 2 + SWRL | poor (OWA, no NAF) | weak (built-ins, Pellet only) | strong classes | justifications Java-only | owlready2 0.51 (needs Java) | avoid |
| RDF + SHACL / SHACL-SPARQL (pySHACL 0.40) | medium (paths, materialise edges) | good via BIND | good | excellent (validation report, severities) | good, pure Python | strong for export/reporting |
| GeoSPARQL 1.1 | 2D RCC-8/DE-9IM only | n/a | n/a | query results | few stores, no 3D | internal shapely only |
| ASP / clingo 5.8 (+lpx) | excellent (recursion, NAF, aggregates) | integer; rationals via lpx | good | good (evidence atoms, xclingo/xASP) | excellent (pip, MIT) | **recommended core** |
| Datalog Soufflé / Nemo | excellent | good (floats in Soufflé) | good | proof trees (Soufflé explain) | external binary / young API | alternative to clingo |
| Prolog + janus-swi | excellent | excellent (floats) | good | meta-interpreter needed | good (official bridge) | fine if team knows Prolog |
| OPA/CEL/JSON-Logic | weak (no recursion) | good | good | OPA trace | subprocess / new cel-expr-python | threshold tables only |
| QSR tools SparQ/GQR | composition over unknowns | none | none | none | Lisp/C++ | avoid |
| STL (RTAMT/MoonLight) | STREL only | excellent (robustness margin) | n/a | robustness value | RTAMT pip; MoonLight Java | video path |
| 3DSG + Cypher/Neo4j | good | good | good | query results | Neo4j driver | later, multi-scene memory |
