<!-- Survey E (from natural-language specs to executable checks: alternatives to per-sentence template compilation; critique and recommended architecture), produced 2026-10-08 by a research agent for docs/research/verdict-spec-to-check-2026-10-08.md. Verbatim; "unverified" marks are the agent's own. -->

# Spec-to-check pipelines for Panoptes: critical survey and recommendation

Scope: literature/tooling survey only; no code touched; no `.platform` directory read. "unverified" marks claims I could not confirm beyond a search snippet or an abstract.

## 0. The thing being critiqued, in one line

Panoptes compiles safety text one sentence at a time into a fixed RASE JSON over a closed vocabulary (classes/zones/edges/attributes/declared inputs), validates to compiled / needs_input / vocabulary_gap / refused, renders to ASP, grounds against a thin 3D scene graph. The user's complaint ("many rules are natural language; we'd have to arrange them all") is the right complaint. The literature says: the back half (scene graph + metric relations + deterministic grounding + explicit refusal states) is the part the field is converging on; the front half (per-sentence arrangement into a fixed template, refusal on vocabulary gap, no document context, no retrieval, no verification of the arrangement) is the part the field has already moved past.

## 1. Function-library mapping

**LLM-FuncMapper** (Zheng et al., Tsinghua, arXiv Aug 2023; journal version Jan 2026): a database of **66 atomic functions** distilled from building codes; claim that "almost 100% of computer-processible clauses" are expressible; LLM retrieves relevant functions with **81.55% top-5** few-shot, ~19% better than fine-tuned BERT; case study composes functions into executable code. Caveat: "computer-processible" is defined by their own clause classifier (long-tail distribution acknowledged). https://arxiv.org/abs/2308.08728 ; interpretability classifier: https://arxiv.org/pdf/2309.14374

**Solibri**: >50 parametric rule templates, enumeration/numeric parameters, Ruleset Manager; anything beyond templates needs Java via the API. TU Wien: "numerous complex specifications and tasks cannot be checked with the standard checking rule templates". Zhang & El-Gohary: rules "with multiple conjunctions and/or disjunctions and restrictions and/or exceptions could take 30 minutes or more to input into the rule templates of the Solibri Model Checker". https://help.solibri.com/hc/en-us/articles/1500005009042-Understanding-Checking ; https://help.solibri.com/hc/en-us/articles/1500004623681-Rule-Parameters ; https://www.tuwien.at/en/cee/ibb/zdb/research/ongoing-research-projects/bim-checking-rule-development ; https://par.nsf.gov/servlets/purl/10347911

**Solihin & Eastman (2015)** rule classes: 1 explicit data, 2 simple derived attributes, 3 extended data structures / higher-level semantics, 4 "proof of solution". A template of "one edge + operator + threshold" covers classes 1-2 only. https://repository.gatech.edu/entities/publication/c798e843-9a07-4c8e-bf4a-0dc635d2069a (unverified page content)

**Fuchs, Hellin, Borrmann (TUM, EC3 2026)**: LLM agent in the CodeAct pattern generates *reusable* checking functions for class-3 regulations, with "iterative refinement with a strong verifier" against BIM data; functions reused across scenarios; "geometric reasoning requiring auxiliary constructions remains challenging". Code: github.com/stefan-1992/ACC-function-generation. No headline success rate in the abstract (unverified). https://ec-3.org/?p=15476

How libraries grow in practice: FuncMapper mined codes offline; TUM lets the agent write and the verifier keep functions; Solibri sells them. Nobody reports coverage statistics for machinery standards.

## 2. Program synthesis per clause

**Revit script generation (Electronics 14(11):2146, Jun 2025)**: 8 LLMs, **12 rules** (IRC/IMC), two buildings, Revit PythonShell. Scripts that ran without error: Grok 76.7%, DeepSeek 33.3%, Claude 3.5 Sonnet 23.7%, GPT-4 13.2%, Gemini/Perplexity <15%, Llama-405B and Copilot 0%. Correctness validated by hand; failures: IronPython quirks, version drift, model metadata quality, semantic errors. https://arxiv.org/abs/2506.20551

**ViperGPT / VisProg** (2023): LLM writes Python over a vision API. Re-analysis: gains over BLIP-2 "can be attributed to its selection of task-specific modules" and vanish with task-agnostic modules; 3x more runtime errors on A-OKVQA. https://arxiv.org/abs/2303.08128 ; https://arxiv.org/pdf/2311.06411

**PropTest (EMNLP Findings 2024)**: LLM writes property tests (type, syntax, semantic properties) for the generated program before execution: GQA 46.1% (+6.0) with Llama3-8B, RefCOCO+ 59.5% (+8.1). This is the cheapest published "auto-generated tests for generated checks" result. https://aclanthology.org/2024.findings-emnlp.483/

**CodeAct (ICML 2024)**: code as action space gives up to 20% higher success than JSON/text actions on M3ToolEval (82 tasks). Directly relevant: a fixed JSON template is the thing CodeAct beat. https://proceedings.mlr.press/v235/wang24h.html

**Compliance-to-Code (May 2025 / Jan 2026)**: 1,159 clauses from 361 Chinese financial regulations, each decomposed into subject / condition / constraint / contextual info + inter-clause relations, with deterministic Python mappings and explanations; FinCheck pipeline. Accuracy not in abstract (unverified). https://arxiv.org/abs/2505.19804

**Failure modes of LLM autoformalization** ("Know Your Limits", Jun 2026, ContractNLI → Z3): "scope laundering" (reporting solver-inconsistent answers without running the solver), implicit-constraint blindness, wrong Z3 code despite structured prompting. https://arxiv.org/abs/2606.16118

**Building code review case study (Oct 2025)**: MCP tool-agent pipelines beat RAG reasoning "in rigor and reliability"; small models unstable. https://arxiv.org/abs/2510.02634

## 3. Agentic tool use over the scene graph

**CompAgent (Oct 2025, CVPR 2026 GRAIL-V workshop)**: planning agent picks tools (object detectors, face analyzers, NSFW, captioning); verification agent fuses image + tool outputs + policy text; up to **76% F1 on UnsafeBench**, +10 over SOTA. Media/ad compliance; no uncertainty handling or cost reported. https://arxiv.org/abs/2511.00171

**RieMind (Mar 2026)**: LLM grounded in a persistent 3D scene graph; tools in four namespaces (memory summary; scene queries incl. nearest-neighbour; geometry: bbox dims, volume, surface area, distances for convex/concave; egocentric/allocentric transforms). VSI-Bench static (4,185 Qs): GPT-4.1 89.5%, GPT-4o 85.2% (base VLM 35.3%), Qwen2.5-VL-7B 64.1% (base 31.2%); prior fine-tuned SOTA 73.6%. 1-4 tool calls per question. Caveats: scene graphs from ground-truth annotations; no "cannot determine"; small models fail on 5-6-call chains. https://arxiv.org/abs/2603.15386

**MIT "Structured Interfaces" (Oct 2025, Ray/Arkin/Biggie/Fan/Carlone/Roy)**: QA with GPT-4.1, small/large graph: Cypher tool 85/77%, context dump 75/33%, Python tool 51/32%. PDDL grounding: Cypher 65/56, context 65/41, Python 38/51. Input tokens on the large graph: Cypher 2,395 vs Python 6,237 vs context 582,202. Claude Opus 4.1 top overall; Qwen3-32B works locally. https://arxiv.org/abs/2510.16643

**SGR-BIM (Jun 2026, HKUST)**: cross-modal KG aligning user intent, regulatory semantics and BIM geometry; **679 expert-verified fire-code queries, 84.3%**, +8.6 over an "enhanced-tool single-agent" baseline. https://arxiv.org/abs/2606.12065

**DriveReg (2024-25)**: see §5; 1-2 s per decision, explanations rated 4.93/5.

Cost per clause: not reported by any of these (unverified for all). Auditability: tool traces and Cypher queries are logged, but verdicts are still LLM-emitted; none of these are deterministic.

## 4. Semantic parsing to logic / ontologies

**I-SNACC (ITcon 2023, Wu, Xue, Zhang, Purdue)**: pattern-matching rules turn IBC 2015 Ch.10 sentences into Horn clauses in **B-Prolog**; IFC → "invariant signatures" → logic facts; **95.2% P / 100% R** on two real projects; 60 hand-made non-compliance cases; first iteration was 51.7% and reached 100% only after a *manual logic-rule refinement module*; rule development 24.23 → 13.22 min per non-compliance case. Note "manual adaptations" are expected for other chapters. https://itcon.org/papers/2023_01-ITcon-Wu.pdf

**Zhang & El-Gohary** pipeline (information extraction → information transformation → logic): IE P/R 0.969/0.944 on IBC 2009 quantitative requirements (chapter summary); 2015 JCCE transformation paper; 2023 transformer-based IFC-regulation concept alignment ~80% accuracy (unverified exact). https://ascelibrary.org/doi/10.1061/%28ASCE%29CP.1943-5487.0000427 ; https://experts.illinois.edu/en/publications/transformer-based-approach-for-automated-context-aware-ifc-regula/

**RASE / ACCORD**: RASE (Hjelseth & Nisbet 2011) marks Requirement/Applicability/Selection/Exception on text *and tables*, tested on a standard with tables (Dubai). ACCORD formalizes RASE into the AEC3PO ontology (RASEStatement, DefinitionStatement, Table/Cell/Row/Column modules) and BCRL/SHACL; D2.2 adopts a *hybrid* manual/automated route "because the accuracy level of automated translation methods is not yet fully proven". RASE process: separate normative/definitive/informative text, mark clauses, mark complex terms and tables, producing nested boxes. CODE-ACCORD corpus: **862 sentences, only self-contained ones** ("express complete rules without needing additional context"), 12 annotators, 4,297 entities, 4,329 relations. Al-Turki et al. (IEEE Access 2024): GPT-4o → YAML RASE with few-shot / fine-tune / active learning with expert feedback; no numbers in abstract (unverified). Nisbet & Çidik (EC3 2025) still searching for a theoretical basis for RASE's four tags; RASE is applied "down to an individual cell in a table". https://cumincades.scix.net/pdfs/w78-2011-Paper-45.pdf ; https://accordproject.eu/wp-content/uploads/2024/02/ACCORD_D2.2_BCO_Ontology_and_Rules_Format.pdf ; https://arxiv.org/abs/2403.02231 ; https://ec-3.org/wp-content/uploads/2025/10/EC32025_278.pdf ; https://github.com/Accord-Project/aec3po

**LLM → LegalRuleML** (Fuchs et al. 2024, GPT-3.5/4 few-shot with CoT + self-consistency; no accuracy in abstract) and intermediate representations for semantic parsing (reversible IR cuts training time ~4x, +6.6 F1). https://arxiv.org/abs/2407.21060 ; https://www.ucl.ac.uk/bartlett/construction/sites/bartlett_construction/files/improving_the_semantic_parsing_of_building_regulations_through_intermediate_representations.pdf

**LegalRuleML**: defeasible rules, override priorities, exceptions as explicit defeaters. https://arxiv.org/abs/1711.06128

**Han et al. (Buildings 2026)**: BERT extraction + CFG structural validation + confidence-based expert review; 95.8% translation accuracy, 98.3% executable; expert effort **168 h vs 1,620 h (−90%)**; regulatory-change processing −94%. https://www.mdpi.com/2075-5309/16/4/719

## 5. Retrieval and alignment

**DriveReg** (Traffic Regulation Retrieval agent): VLM writes a scene description + query; paragraph retrieval (ada-002, FAISS, τ=0.28) then sentence refinement (MiniLM); **562 rules** across Boston/LA/Singapore; retrieval accuracy 100% on 20 scenarios vs BM25 60% / LLaDA 55%; decision compliance 98-99% hypothesized, 91-95% real; retrieval raised GPT-4o compliance 91→93%. https://arxiv.org/abs/2410.04759

**Lawful AD (Apr 2026)**: anchors law retrieval in a scenario taxonomy; 5,897 scenarios; law-scenario matching +29.1%, mandatory-requirement accuracy +36.9%, prohibitive +38.2% vs generic LLM retrieval. https://arxiv.org/abs/2604.24562

**BifrostRAG (Jul 2025)**: dual graphs (entity network + document-structure navigator) over OSHA; multi-hop F1 **87.3** vs OpenAI vector RAG 75.0, Neo4j graph RAG 56.5. https://arxiv.org/abs/2507.13625

**The grounding gap is the vocabulary problem, measured**: VLTL-Bench (Dec 2025) finds NL→LTL *lifted* translation at 78-99% but *grounded* end-to-end accuracy collapses: NL2TL 54.4-60.1%, nl2spec 29.6-34.8%, NL2LTL 26.2-38.4%, Lang2LTL 37.9-72.1%. https://arxiv.org/abs/2507.00877 . **GinSign (Dec 2025)** treats grounding to a "system signature" (predefined atomic propositions + typed constants) as structured classification and reaches 95.5% grounded logical equivalence (1.4x SOTA); it assumes the signature exists. https://arxiv.org/abs/2512.16770 . GuardEn's tool-aware grounding (substitute keywords when detector confidence < 0.8, cached) is worth 16.5% F1 in ablation (§9).

## 6. Autoformalization with verification

**nl2spec (CAV 2023)**: 36 expert-written hard specs; Codex minimal prompt 44.4%, in-distribution prompt 58.3%, **interactive sub-translation refinement 86.1% after 1.4 loops on average**; UI shows (NL fragment → subformula) pairs the user edits. https://arxiv.org/abs/2303.04864

**NL2TL (EMNLP 2023)**: >95% with <10% training data, 28K-pair dataset; **NL2LTL** (IBM, AAAI 2023). https://arxiv.org/abs/2305.07766 ; https://ojs.aaai.org/index.php/AAAI/article/view/27068

**ARc / Automated Reasoning checks (AWS, Nov 2025 / Jul 2026)**: k redundant LLM translations to logic, compared by SMT equivalence to get a confidence; **>99% soundness, near-zero false positives**; auditable artifacts; GA in Bedrock Guardrails (Aug 2025, "up to 99%"); policies carry tests with outcomes VALID / INVALID / SATISFIABLE / IMPOSSIBLE / TRANSLATION_AMBIGUOUS; automatic refinement proposes rule edits shown with test-impact diffs and a fidelity report, human approval mandatory; Feb 2026: rules link to source statements. Needs a domain model (variables, types) first. https://arxiv.org/abs/2511.09008 ; https://aws.amazon.com/about-aws/whats-new/2025/08/automated-reasoning-checks-amazon-bedrock-guardrails ; https://aws.amazon.com/blogs/machine-learning/automated-reasoning-policy-refinement-in-amazon-bedrock ; https://aws.amazon.com/about-aws/whats-new/2026/02/automated-reasoning-policies-include-references

**P4IR (Jun 2026)**: SFT + GRPO to emit a code-skeleton IR for building rules; −23.8% tree-edit distance, −38.6% Levenshtein vs SFT; beats Claude Opus/Sonnet 4.5, GPT-5.2, Qwen-3-Max, GLM-4.7 zero-shot; reward is structural, not functional correctness. https://arxiv.org/abs/2606.22402

**Catala (ICFP 2021)**: law text and code in one literate document for lawyers and programmers; default/exception logic native; F*-verified compiler; found a bug in the official French family-benefits implementation. "Closing the Loop" (Jun 2026) uses a Catala-extended calculus as an RL verifier; demonstrations only, no numbers. https://arxiv.org/abs/2103.03198 ; https://arxiv.org/abs/2606.23913

**LLM + SMT for financial statutes (2025)**: 87 Taiwan FSC enforcement cases, **86.2%** correct SMT code. https://arxiv.org/abs/2601.06181

**LogiSafetyGen (Jan 2026)**: policies → LTLf restricted to two templates; a *signature validator* rejects formulas naming non-existent API predicates; 73.9% of generated formulas accepted after author review; fuzzer builds traces satisfying functional + compliance oracles. https://arxiv.org/html/2601.08196v1

What all of these need: a target language with semantics, a signature, and a test oracle. Panoptes has the first (ASP) and third (scenes) but has frozen the second.

## 7. Whole-document handling

Evidence that sentence-level is the wrong unit: CODE-ACCORD discarded every non-self-contained sentence; RASE/AEC3PO model definitions, tables (Table/Cell/Row/Column) and nested section boxes explicitly; BifrostRAG needed a document-navigator graph to answer multi-hop OSHA questions; the Frontiers 2025 ontology study lists "cross-reference currency tracking absent" and "nested conditions, exceptions, hierarchical relationships inadequately handled" as open gaps (https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2025.1575913/full). Policy KGs: Baldwin & Ghanavati (Apr 2026) show KG augmentation helps all five LLMs on 42 cross-policy QA tasks and that an LLM-discovered schema matches a formal ontology (https://arxiv.org/abs/2604.27713); EDC gives extract-define-canonicalize for schema-light KG building (https://arxiv.org/abs/2404.03868). Defeasibility: LegalRuleML. Editions: Akoma Ntoso point-in-time versioning and amendment links (OASIS 2018) (https://publications.cohubicol.com/typology/akoma-ntoso/). Standards bodies: IEC "SMART Standards" utility model (Level 3 machine-readable content, adopted by ISO/CEN/CENELEC), DIN/DKE first SMART products 2024/25 in ReqIF, VDE SIS GmbH from Jan 2026 (https://experts.cen.eu/key-initiatives/smart-standards/ ; https://www.vde.com/en/press/press-releases/smart-standards-simplify-ai-testing). ISO 13857:2019 Table 2 (reach over protective structures: hazard-zone height × structure height → horizontal distance) is not public; it must be extracted as a typed lookup function from the licensed PDF (https://www.iso.org/standard/69569.html).

## 8. Human review UX and cost

Published review surfaces: nl2spec's editable sub-translation table (1.4 loops to 86.1%); AWS's rule-diff + test-impact + fidelity-report screen with mandatory approval; Catala's literate law-plus-code; Solibri's parameter forms (≥30 min per complex rule); I-SNACC's rule-editing console (13.22 min per case); Han et al.'s confidence-triaged review (168 h total vs 1,620 h); MonitorVLM-v2's entropy-driven routing to human reviewers. Nobody reports review minutes per *clause* for an LLM-compiled rule (unverified gap); Han et al. is the closest whole-pipeline number.

## 9. Rule-free baselines

**ConstructionSite-10K** (~10K images): Molmo-7B single prompt P 85.7 / R 48.0 / F1 61.5; 10-prompt ensemble F1 67.2 (R 79.6); Qwen2-VL-2B F1 66.7 → 72.6 with ensemble (R 98.0, P 67.2). https://arxiv.org/abs/2511.15720 . **HomeSafeBench**: 1,000 human-validated tasks (v2; an earlier summary cites 12,900 data points — unverified); best VLM F1 34.7% vs human 98.0%; fine-tuned 4B 45.3%; "precision far exceeds recall" = under-reporting hazards. https://arxiv.org/abs/2509.23690 . Cambridge 2026 study: VLMs high recall / low precision on violation VQA, poor grounding (https://www.cambridge.org/core/services/aop-cambridge-core/content/view/4F9F8B39B34FD6F2B201C9947CDF42E8/S2632673626100446a.pdf/are-large-pre-trained-vision-language-models-effective-construction-safety-inspectors.pdf).

**MonitorVLM-v2 (Aug 2026)**: casts inspection as single-token prediction over a finite set of rule IDs; SymPO contrastive training; entropy triage to humans; 4-month mining deployment on 10 feeds found 2.78x the confirmed violations of manual inspection. https://arxiv.org/abs/2608.00975

**GuardEn (EMNLP 2026)** — the closest relative of Panoptes. Atomic propositions are `(object, attribute)` or `(object1, relation, object2)`; a decomposer with a verifier loop splits a rule into a proposition tree over ∧/∨/¬ which *is* the executable artifact; grounding builds a scene graph per image from SAM 3 + SigLIP 2 + Depth Anything 3 + OCR with predefined geometric/depth relation rules, then "tool-aware grounding" proposes substitute keywords when detector confidence < 0.8 and caches hits. SafetyVisionBench ~34K images (24K unsafe + 10K hard-safe) from FDA Food Code, NYC Building Code, Meta standards, UK CAP Code. Average F1 **78.7 vs 68.9** best baseline (+9.8); construction 86.4 vs CompAgent 80.6. Ablations: remove relation grounding −30.1%, verifier loop −27.4%, tool-aware grounding −16.5%, scene-grounded execution −12.8%. Limits: coverage bounded by rule completeness; deductive only. What it does better than Panoptes: (i) the compile step is *verified* by a loop, not single-shot; (ii) vocabulary gaps trigger *re-grounding with synonyms*, not refusal; (iii) relations are computed by predefined geometric rules, same as Panoptes, and that is where its F1 comes from. https://arxiv.org/abs/2609.18328

---

## (a) Fair critique of closed-vocabulary RASE-compile

Weaknesses, with evidence:

1. **Vocabulary gap → refuse is the wrong policy.** The measured failure point of NL→logic is grounding, not translation (VLTL-Bench: 78-99% lifted vs 26-72% grounded). Systems that *align* against a signature recover most of it (GinSign 95.5%; GuardEn +16.5 F1 from substitute-keyword grounding; Zhang & El-Gohary ~80% IFC alignment). Panoptes treats the signature as a gate and discards the clause.
2. **One sentence, no context.** The only corpus of RASE-style annotations (CODE-ACCORD) had to exclude non-self-contained sentences; RASE's own authors operate on nested sections and table cells; AEC3PO carries Definition and Table modules. ISO 13857 Table 2, ISO 13855's speed/formula clauses, IEC 60204-1's cross-referenced definitions cannot be arranged sentence by sentence.
3. **Fixed template = Solibri's ceiling re-implemented.** "One edge + operator + threshold" is Solihin-Eastman class 1-2; class 3-4 clauses need code, which is why Solibri exposes a Java API and why TUM/FuncMapper let the LLM write or compose functions. CodeAct showed code actions beat JSON actions by up to 20%.
4. **Unverified single-shot formalization.** The arrangement is an autoformalization with no redundancy, equivalence check or tests. ARc gets >99% soundness only by k-redundant translation + SMT equivalence; "Know Your Limits" documents scope laundering and missed constraints; P4IR shows even Claude Opus 4.5 / GPT-5.2 emit structurally wrong rule skeletons zero-shot.
5. **No retrieval.** Compile-everything-upfront is unnecessary; DriveReg and Lawful-AD show scene→clause retrieval at 100% / +29% matching, so only applicable clauses need compiling and reviewing.
6. **Scale and review are unmeasured.** No per-clause effort number exists for this design; the only pipeline-level effort data (Han et al. −90%; I-SNACC 13 min/case) come from extraction+review designs with confidence triage and literate review, which Panoptes lacks.

Strengths worth keeping: deterministic grounding and verdicts (clingo), explicit NEEDS_INPUT / NEEDS_MEASUREMENT / CANNOT_DETERMINE states (MONIR-style four-valued compliance reasoning is the current theory: https://arxiv.org/abs/2606.04619), guard bands, evidence trail, and the scene graph + computed metric relations — GuardEn's ablation says relation grounding is the single biggest contributor (−30.1% without it).

## (b) Comparison

| Approach | Manual effort / clause | Scales to 100s of pages | Tables / definitions / exceptions | Auditability | Reported accuracy | Maturity | Fit for metric 3D rules |
|---|---|---|---|---|---|---|---|
| Closed-vocab RASE JSON → ASP (current) | High (LLM draft + human arrangement, refusal on gap) | Poor (no retrieval, no context) | Weak (sentence unit) | High (deterministic) | None published | Prototype | Good back end, poor front end |
| Function-library mapping (FuncMapper, Solibri, TUM) | Low-med once library exists; library building is the cost | Medium (long-tail functions) | Medium (functions can wrap tables) | High if functions are reviewed | 81.55% top-5 retrieval; "~100%" expressible (self-defined) | Solibri commercial; LLM variants research | Good: edges become functions |
| Program synthesis per clause + tests | Low draft, review needed | Good with retrieval | Good (code can read tables/defs) | Medium (code review) | 13-77% runs-clean on 12 rules; PropTest +6-8 | Research | Good if API is typed and small |
| Agentic tool use (RieMind, MIT Cypher, SGR-BIM, CompAgent) | Very low | Good | Medium | Medium (traces, non-deterministic verdicts) | 85-89% VSI static; 84.3% on 679 queries; 76-80 F1 | Research | Good for evidence gathering, not for verdicts |
| Semantic parsing → logic (I-SNACC, ACCORD) | Med-high (pattern rules, markup) | Poor-medium | ACCORD models tables/defs; sentence-level parsers do not | High | 95.2 P / 100 R on one chapter | ACCORD EU pilot | Medium (built for BIM) |
| Retrieval + alignment (stage, not system) | Very low | Required for scale | Needs structured KG | Medium | 100% / +29.1% matching; F1 87.3 multi-hop | Research, easy | Necessary |
| Autoformalization w/ verification (ARc, nl2spec, P4IR) | Low draft, interactive fix (1.4 loops) | Good | Needs signature incl. tables | Very high (artifacts, tests) | >99% soundness; 86.1% interactive | ARc commercial | Good: ASP is a fine target |
| VLM direct (ConstructionSite-10K, HomeSafeBench) | Zero | Trivial | None | Low | F1 61-73; 34.7% home | Deployed (MonitorVLM-v2) | Poor for metric thresholds |
| GuardEn compile + scene-graph grounding | Low (verified decomposition) | Medium | Weak on tables | High (proposition tree) | F1 78.7 avg, 86.4 construction | Research, strong | Closest sibling; lacks metric tables/guard bands |

## (c) Recommended architecture

Goal: extraction instead of arrangement; deterministic, auditable verdicts preserved.

1. **Document → structured clause KG** (copies BifrostRAG's document-navigator graph, AEC3PO's Definition/Table modules, EDC canonicalization, Akoma Ntoso IDs/editions). Nodes: standard edition, clause, definition, table, figure, note, cross-reference. Tables become typed lookup functions (`iso13857_table2(hazard_h, structure_h) -> horiz_dist`), definitions become alias sets, exceptions become defeater edges (LegalRuleML). This is extraction with an LLM plus layout parsing, reviewed once per document, not per sentence.
2. **Concept alignment as a learned, cached aligner** (GinSign structured classification; GuardEn substitute-keyword loop; Zhang & El-Gohary transformer alignment). Output is an alignment table clause-term → scene class/zone/attribute with confidence; engineers review the *table* (dozens of rows), never individual clauses. "vocabulary_gap" becomes "alignment below threshold → review row", not refusal.
3. **Per-scene retrieval of applicable clauses** (DriveReg two-level retrieval; Lawful-AD taxonomy anchors keyed on the workcell taxonomy: robot, fixed guard, light curtain, opening, operator zone). Only retrieved clauses proceed; log retrieval set for audit.
4. **Check synthesis against a small typed scene API** (CodeAct/TUM pattern; FuncMapper composition; P4IR-style skeleton IR later for fine-tuning). The API is exactly the current relation library (3D distance, horizontal gap, floor gap, z-overlap, reach-over triple, line of sight, plan occupancy) plus table lookups and `declare_input(...)`. The LLM emits a short check (Python or ASP) with sub-translations (nl2spec) mapping each clause fragment to the API calls it used.
5. **Automatic verification before any human sees it**: k-redundant translations compared by *differential execution* on generated test scenes (ARc's idea with scenes instead of SMT); LLM-written property tests (PropTest); signature validator rejecting unknown predicates (LogiSafetyGen); metamorphic tests (scale the scene, verdict must not change; cross the threshold, verdict must flip). Disagreement → review queue with the diff.
6. **Literate review**: one screen per clause showing clause text with highlighted fragments, sub-translations, the check code, the test scenes and their verdicts, alignment rows used (nl2spec table + AWS refinement screen + Catala literate layout). Confidence triage (Han et al.; MonitorVLM-v2 entropy) so engineers review the uncertain 20%.
7. **Deterministic execution with evidence**: reviewed checks compile to ASP as today (or run as pure functions emitting facts into clingo), four-valued outcomes with guard bands, per-verdict evidence = clause ID, table row, measured values with uncertainty, alignment rows.

Build first (two weeks of work, not months): stage 1 for ISO 13857 + ISO 13855 only (tables and definitions as functions), the typed scene API with the existing relations, stages 4-5 for 20 retrieved clauses, and the review screen as a static HTML per clause. Measure against the current compiler on the same 20 clauses.

Metrics: **compile rate** (clauses executable without refusal, target >80% vs current); **alignment coverage** (clause terms resolved ≥ threshold); **review time per clause** (stopwatch; Han-style hours per document); **verdict agreement** with an engineer on a held-out scene set (κ and per-state confusion including CANNOT_DETERMINE); **false-refusal rate**; **retrieval recall** of applicable clauses against an engineer's checklist; **test-disagreement rate** between redundant translations (proxy for ambiguity, as ARc's TRANSLATION_AMBIGUOUS).

## (d) Drop / keep

Drop: the per-sentence fixed JSON template as the compilation target; refusal on vocabulary gap; "one edge + operator + threshold" as the only requirement shape; compile-everything-upfront; unverified single-shot arrangement.

Keep: the thin scene graph with oriented boxes, heights and uncertainties; the metric relation library (promote it to the typed API); clingo grounding and determinism; guard bands; the five verdict states; the evidence trail; the closed set of *classes/zones/edges* as a signature to align against, not a gate to refuse at.

Unverified items flagged above: exact Zhang & El-Gohary 2023 alignment accuracy; Al-Turki 2024 numbers; Fuchs EC3 2026 success rate; Compliance-to-Code and COLING 2025 accuracy; HomeSafeBench size discrepancy; Solihin & Eastman page content; any per-clause cost for agentic systems.
