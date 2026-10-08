"""L4 plugin llm-extract@0: safety-concept TEXT (markdown: numbered requirement paragraphs, each ending in clause citations such as
"(TS-0011963 Rev 10, 8.1.4–8.1.6)") -> ClauseGraph through Claude (ehs_spatial.verdict.llm.complete, Haiku by default), verified
deterministically, aligned + retrieved like clause-kg@0, and diffed against the hand-extracted reference. README.md here, section
llm-extract@0."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from ehs_spatial.verdict import llm
from ehs_spatial.verdict.contracts import Clause, ClauseGraph, Signature
from ehs_spatial.verdict.layers.l4_spec.alignment import align
from ehs_spatial.verdict.layers.l4_spec.clause_kg import SIGNATURE_PATH, SPEC_DIR
from ehs_spatial.verdict.layers.l4_spec.retrieve import retrieve
from ehs_spatial.verdict.plugins import register

TEXT, REFERENCE, EXAMPLE = "samples/manual-loading-station-concept-2026-10-08.md", "clauses-ts0011963-v0.json", "TS0011963:Rev10/8.1.4"
_HEADING, _ITEM, _BULLET = re.compile(r"^#{1,6}\s*(.*)"), re.compile(r"^\s*\d+\.\s+(.*)"), re.compile(r"^\s*[-*]\s+(.*)")
_PAREN = re.compile(r"（([^（）]*)）|\(([^()]*)\)")          # a full-width pair may hold ASCII parens: "6.3.2.2 a)", "Formula (13)"
_STD = re.compile(r"\b(TS-\d+) (Rev \d+)|\b((?:ISO|IEC|EN)(?:/TS)? \d+(?:-\d+)*):(\d{4})")
_CALL, _RANGE = re.compile(r"\b([a-z_][a-z0-9_]*)\("), re.compile(r"^(\d+(?:\.\d+)*)[–-](\d+(?:\.\d+)*)$")
DEVICE_NOTES = re.compile(r"measure|device|装置", re.I)       # heading of the device-note bullets (the page's "Measure" readings)


# ---------------------------------------------------------------- step A: deterministic split
def citations(text: str) -> list[str]:
    """The parenthesised groups of a text that name a standard: 'TS-0011963 Rev 10, 8.1.4–8.1.6'."""
    return [c for c in (a or b for a, b in _PAREN.findall(text)) if _STD.search(c)]


def split_units(text: str) -> list[dict]:
    """Units: every numbered item (the 'Configuration requirements' list) and every bullet under a device-note heading, each with at
    least one citation. Hazard / purpose bullets and the picture description are not requirements. Continuation lines join."""
    units, section, cur = [], "", None
    for line in text.splitlines():
        if (h := _HEADING.match(line)):
            section, cur = h.group(1), None
        elif (item := _ITEM.match(line)) or ((bullet := _BULLET.match(line)) and DEVICE_NOTES.search(section)):
            cur = {"kind": "requirement" if item else "device_note", "text": (item or bullet).group(1).strip()}
            units.append(cur)
        elif cur and line.strip():
            cur["text"] += " " + line.strip()
        else:
            cur = None
    units = [u for u in units if citations(u["text"])]
    return [{"id": f"u{i:02d}", **u, "citations": citations(u["text"])} for i, u in enumerate(units, 1)]


def std_id(name: str, edition: str) -> str:
    """'TS-0011963', 'Rev 10' -> 'TS0011963:Rev10'; 'ISO 13849-1', '2023' -> 'ISO13849-1:2023' (the clauses files' convention)."""
    return re.sub(r"^([A-Z]+)[ -]*", r"\1", name).replace(" ", "") + ":" + edition.replace(" ", "")


def standards(text: str) -> dict[str, dict[str, str]]:
    """{id: {name, edition, title}} for every standard named in a citation of the text, e.g. TS0011963:Rev10 -> TS-0011963 / Rev 10."""
    out = {}
    for c in citations(text):
        for ts, rev, iso, year in _STD.findall(c):
            name, edition = (ts, rev) if ts else (iso, year)
            out[std_id(name, edition)] = {"name": name, "edition": edition, "title": f"{name} {edition}" if ts else f"{name}:{edition}"}
    return dict(sorted(out.items()))


def cited_refs(cites: list[str]) -> dict[str, list[str]]:
    """{standard id: [clause refs without spaces]}: 'TS-0011963 Rev 10, 8.1.4–8.1.6; ISO 13855:2024, Formula (13), 8.3' ->
    {'TS0011963:Rev10': ['8.1.4–8.1.6'], 'ISO13855:2024': ['Formula13', '8.3']}. A ';' segment without a standard keeps the previous."""
    out: dict[str, list[str]] = {}
    for c in cites:
        sid = None
        for seg in c.split(";"):
            if (m := _STD.search(seg)):
                sid, seg = std_id(m.group(1) or m.group(3), m.group(2) or m.group(4)), seg[m.end():]
            if sid:
                out.setdefault(sid, []).extend(r for r in (re.sub(r"[^A-Za-z0-9.–-]", "", x) for x in re.split(r",|\band\b", seg)) if r)
    return out


def is_cited(clause_id: str, cites: list[str]) -> bool:
    """The numeric head of the clause number ('4.3.4-upper' -> 4.3.4, 'Table2', 'Formula13') is cited for that standard; a cited
    range such as 8.1.4–8.1.6 includes 8.1.5; 'Table 2 note c' cites Table2 but '4.1.1' does not cite 4.1."""
    sid, _, num = clause_id.partition("/")
    if not (m := re.match(r"[A-Za-z]*\d+(?:[.-]\d+)*", num)):
        return False
    head = m.group()
    for ref in cited_refs(cites).get(sid, []):
        if re.fullmatch(re.escape(head) + r"(?:[^\d.].*)?", ref):
            return True
        if (r := _RANGE.match(ref)) and re.fullmatch(r"\d+(?:\.\d+)*", head):
            lo, hi, h = (tuple(int(x) for x in s.split(".")) for s in (r.group(1), r.group(2), head))
            if lo <= h <= hi:
                return True
    return False


# ---------------------------------------------------------------- step B: the model's output schema and prompts
class Binding(BaseModel):
    var: str                   # selection variable, a capital letter
    values: list[str]          # Signature classes / zones it ranges over


class Requirement(BaseModel):
    predicate: str             # Signature predicate or attribute; "exists" for presence
    args: list[str]
    operator: Literal[">=", "<=", ">", "<", "==", "!=", "table", "exists"]
    threshold: float | None    # mm (or the predicate's unit); null when the clause uses a table or formula
    table: str | None          # a table id of the clauses files, e.g. ISO13857:2019/Table2
    formula: str | None
    unit: str
    inputs: list[str]          # declared inputs the check needs


class Candidate(BaseModel):
    clause_id: str             # "<standard id>/<clause>", e.g. TS0011963:Rev10/8.1.4, ISO13855:2024/9.3
    title: str
    paraphrase: str
    rule_class: Literal["geometry", "topology", "semantic", "procedural"]
    selection: list[Binding]
    applicability: list[str]   # predicate atoms "perimeter_of(F, Z)" or a bare attribute "operator_interaction"
    requirement: Requirement
    exceptions: list[str]      # Signature attributes that switch the clause off
    tags: list[str]            # classes / zones the clause is about
    photo_checkable: bool
    source_quote: str          # the exact words of the unit that carry the number
    vocabulary_gaps: list[str]  # terms the text needs that the vocabulary lacks


class Extraction(BaseModel):
    candidates: list[Candidate]


CONVENTIONS = """You extract machine-checkable safety requirements from ONE unit of a safety-concept text into clause candidates.
- One candidate per distinct requirement; a unit may yield several (a minimum height, a different minimum under an exception, a maximum
  opening ...). Return no candidates when the unit states no checkable requirement.
- clause_id = "<standard id>/<clause number>" with exactly the standard ids listed below and a clause number the unit cites (a cited
  range 8.1.4–8.1.6 covers 8.1.5). Never cite a clause the unit does not cite; a table is "<standard id>/TableN", a formula "<standard id>/FormulaN".
- rule_class: geometry (a measured height / distance / gap against a number, table or formula), topology (presence, reachability,
  enclosure between objects and zones), semantic (an attribute of an object: colour, resolution, interlock), procedural (control
  logic, documentation, training: not checkable from a scene, photo_checkable = false).
- selection: variables (capital letters) over Signature classes or zones. applicability: Signature predicate atoms that must hold,
  e.g. "perimeter_of(F, Z)", or a bare Signature attribute. requirement: a Signature predicate or attribute on the variables, an
  operator, and a threshold (number) or table (id) or formula (text); unit "mm" for lengths (metres become millimetres: 0.6 m -> 600),
  "bool" for presence / membership; inputs = the declared inputs the check needs. exceptions: attributes that switch the clause off.
  tags: the classes / zones the clause is about.
- Every number must come from the unit's text; source_quote = the exact words carrying it. Every class, zone, predicate, attribute
  and input must be a name from the vocabulary below; a term the text needs that the vocabulary lacks goes into vocabulary_gaps,
  never into the fields. Paraphrase in your own words, short. Nothing is verified against the standard's text."""


def vocabulary(sig: Signature) -> str:
    preds = "\n".join(f"  {n}({', '.join(p.args)})" + (f" -> {p.unit}" if p.unit else "") + (f"   # {p.note}" if p.note else "")
                      for n, p in sig.predicates.items())
    return (f"Vocabulary (Signature {sig.version})\nclasses: {' '.join(sig.classes)}\nzones: {' '.join(sig.zones)}\npredicates:\n{preds}\n"
            f"attributes: {' '.join(sig.attributes)}\ndeclared_inputs: {' '.join(sig.declared_inputs)}")


def _num(t: float | None):
    return int(t) if t is not None and t == int(t) else t


def candidate_of(c: Clause) -> Candidate:
    """A reference clause in the model's output shape (the worked example; tests reuse it as the fake model's answer)."""
    r = c.requirement
    req = Requirement(predicate=str(r.get("predicate") or r.get("attribute")), args=[str(a) for a in r.get("args", [])],
                      operator=str(r.get("operator")), threshold=r.get("threshold"), table=r.get("table"), formula=r.get("formula"),
                      unit=str(r.get("unit", "mm")), inputs=[str(i) for i in r.get("inputs", [])])
    return Candidate(clause_id=c.id, title=c.title, paraphrase=c.paraphrase, rule_class=c.rule_class, requirement=req,
                     selection=[Binding(var=k, values=v) for k, v in c.selection.items()], applicability=c.applicability,
                     exceptions=c.exceptions, tags=c.tags, photo_checkable=c.photo_checkable, source_quote="", vocabulary_gaps=[])


def worked_example(units: list[dict], reference: ClauseGraph, clause_id: str) -> str:
    """The unit citing `clause_id` and that clause as a candidate (source_quote = the unit's sentence carrying its threshold)."""
    c = next((c for c in reference.clauses if c.id == clause_id), None)
    u = next((u for u in units if c and is_cited(c.id, u["citations"])), None)
    if not (c and u):
        return ""
    cand, t = candidate_of(c), _num(c.requirement.get("threshold"))
    cand.source_quote = next((s.strip(" *") for s in re.split(r"[。；;]", u["text"]) if t is not None and str(t) in s), "")
    return (f"Worked example.\nUnit: {u['text']}\nOne of its candidates (this unit yields two more, one per requirement):\n"
            + cand.model_dump_json(indent=1))


# ---------------------------------------------------------------- step C: deterministic verification, output, diff
def verify(c: Candidate, unit: dict, sig: Signature, tables: set[str]) -> list[str]:
    """Flags, sorted: vocabulary_gap:<term>, ungrounded:<threshold> (digits not in the unit text; 0.6 m grounds 600 mm), uncited,
    unknown_table:<id>. No model, no network."""
    names = set(sig.classes) | set(sig.zones) | set(sig.predicates) | set(sig.attributes) | set(sig.declared_inputs)
    r = c.requirement
    terms = ({v for b in c.selection for v in b.values} | {p for a in c.applicability for p in (_CALL.findall(a) or [a.strip()])}
             | set(r.inputs) | set(c.exceptions) | set(c.tags) | ({r.predicate} - {"exists"}))
    flags = [f"vocabulary_gap:{t}" for t in terms - names]
    if r.threshold is not None and r.unit != "bool":
        body = unit["text"]
        for cite in unit["citations"]:
            body = body.replace(cite, "")
        nums = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", body)}
        if r.threshold not in nums and r.threshold / 1000 not in nums:
            flags.append(f"ungrounded:{_num(r.threshold)}")
    if not is_cited(c.clause_id, unit["citations"]):
        flags.append("uncited")
    if r.table and r.table not in tables:
        flags.append(f"unknown_table:{r.table}")
    return sorted(flags)


def to_clause(c: Candidate, stds: dict[str, dict[str, str]], sig: Signature, photo_checkable: bool) -> Clause:
    sid = c.clause_id.partition("/")[0]
    st = stds.get(sid) or {"name": sid.partition(":")[0], "edition": sid.partition(":")[2]}
    r = c.requirement
    req: dict[str, object] = {"attribute" if r.predicate in sig.attributes else "predicate": r.predicate, "args": r.args,
                              "operator": r.operator, "unit": r.unit, "inputs": r.inputs}
    req.update({k: v for k, v in (("threshold", _num(r.threshold)), ("table", r.table), ("formula", r.formula)) if v is not None})
    return Clause(id=c.clause_id, standard=st["name"], edition=st["edition"], title=c.title, paraphrase=c.paraphrase,
                  rule_class=c.rule_class, selection={b.var: b.values for b in c.selection}, applicability=c.applicability,
                  requirement=req, exceptions=c.exceptions, tags=c.tags, photo_checkable=photo_checkable)


def requirement_key(r: dict) -> dict:
    return {"predicate": r.get("predicate") or r.get("attribute"), "operator": r.get("operator"), "threshold": r.get("threshold"),
            "table": r.get("table"), "formula": r.get("formula")}


def diff(graph: ClauseGraph, reference: ClauseGraph) -> dict:
    a, b = {c.id: c for c in graph.clauses}, {c.id: c for c in reference.clauses}
    same, different = [], []
    for i in sorted(a.keys() & b.keys()):
        ka, kb = requirement_key(a[i].requirement), requirement_key(b[i].requirement)
        same.append(i) if ka == kb else different.append({"id": i, "llm": ka, "reference": kb})
    out = {"reference": reference.version, "ids_only_llm": sorted(a.keys() - b.keys()), "ids_only_reference": sorted(b.keys() - a.keys()),
           "same_id_same_requirement": same, "same_id_different_requirement": different}
    out["summary"] = (f"{len(same)} of {len(b)} reference clauses reproduced (same id, same requirement); {len(different)} same id, "
                      f"different requirement; {len(out['ids_only_llm'])} only in the LLM output; {len(out['ids_only_reference'])} only in the reference")
    return out


@register("L4", "llm-extract", "0")
class LLMExtract:
    """inputs: spec_dir (falls back to the package spec/ when the text is not there), signature, scene (Scene | None).
    cfg: spec_text, reference, example, model, effort, cache_dir. Output: clauses, alignment, retrieved, extraction_report, diff
    (when the reference exists), llm_calls (non-cached calls; the runner sums it)."""

    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]:
        spec_dir = Path(inputs.get("spec_dir") or SPEC_DIR)
        if not (spec_dir / cfg.get("spec_text", TEXT)).exists():
            spec_dir = SPEC_DIR
        text = (spec_dir / cfg.get("spec_text", TEXT)).read_text()
        ref_path = spec_dir / cfg.get("reference", REFERENCE)
        reference = ClauseGraph.load(ref_path) if ref_path.exists() else None
        signature = inputs.get("signature") or Signature.load(SIGNATURE_PATH)
        units, stds = split_units(text), standards(text)
        system = "\n\n".join([CONVENTIONS, "Standard ids of this document: " + "; ".join(f"{i} = {s['title']}" for i, s in stds.items()),
                              vocabulary(signature), worked_example(units, reference, cfg.get("example", EXAMPLE)) if reference else ""]).rstrip()
        cache_dir = cfg.get("cache_dir") or os.environ.get("PANOPTES_LLM_CACHE") or workdir.parents[1] / "llm-cache"
        tables = {t.id for p in sorted(spec_dir.glob("clauses-*.json")) for t in ClauseGraph.load(p).tables}
        model, calls, clauses, per_clause, dups, gaps = cfg.get("model") or llm.MODEL, 0, [], {}, [], set()
        for u in units:
            user = f"Unit {u['id']} ({u['kind']})\nCitations: {'; '.join(u['citations'])}\nText: {u['text']}"
            out, meta = llm.complete(system, user, Extraction, model=cfg.get("model"), effort=cfg.get("effort"), cache_dir=cache_dir)
            calls, model, u["candidates"] = calls + (not meta["cached"]), meta["model"], len(out.candidates)
            for cand in out.candidates:
                if cand.clause_id in per_clause:
                    dups.append({"clause_id": cand.clause_id, "unit": u["id"]})
                    continue
                flags = verify(cand, u, signature, tables)
                gaps.update(cand.vocabulary_gaps, (f.partition(":")[2] for f in flags if f.startswith("vocabulary_gap:")))
                clauses.append(to_clause(cand, stds, signature, cand.photo_checkable and not any(f.startswith("vocabulary_gap:") for f in flags)))
                per_clause[cand.clause_id] = {"unit": u["id"], "source_quote": cand.source_quote, "flags": flags, "declared_gaps": cand.vocabulary_gaps}
        graph = ClauseGraph(version=f"llm-extract-0+{model}", clauses=clauses,
                            standards=[{"id": i, "title": s["title"], "edition": s["edition"], "url": ""} for i, s in stds.items()],
                            tables=reference.tables if reference else [], definitions=reference.definitions if reference else {})
        flagged = sum(bool(v["flags"]) for v in per_clause.values())
        report = {"model": model, "llm_calls": calls, "units": units, "clauses": per_clause, "duplicates": dups, "vocabulary_gaps": sorted(gaps),
                  "summary": f"{len(units)} units -> {len(clauses)} clauses ({flagged} flagged, {len(dups)} duplicates), "
                             f"{len(gaps)} vocabulary gaps, {calls} llm calls ({model})"}
        result = {"clauses": graph, "alignment": align(graph, signature), "retrieved": retrieve(graph, inputs.get("scene")),
                  "extraction_report": report, "llm_calls": calls}
        if reference:
            result["diff"] = diff(graph, reference)
        return result
