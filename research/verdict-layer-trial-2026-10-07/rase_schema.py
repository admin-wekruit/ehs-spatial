"""The arranged form of one natural-language rule (RASE: Requirement / Applicability / Selection / Exception) over the closed
vocabulary (vocabulary.json), and the validator that decides what the arranged rule can do today:
  compiled              every referenced class / zone / edge / attribute is available and no declared input is needed
  compiled_needs_input  expressible, but one or more declared inputs (stop time, resolution, table lookup, ...) must be supplied
  vocabulary_gap        references items the vocabulary only plans (named gaps) -> representation work before it can run
  refused               the compiler set unsupported_reason: photos / this vocabulary can never express it
  invalid               references something the vocabulary does not even plan -> compiler hallucination, rejected
  python rase_schema.py compiled_v1_draft.json          prints the table and writes compiled_v1_draft.validated.json
"""
import json
from pathlib import Path
import sys

from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent
VOCAB = json.loads((HERE / 'vocabulary.json').read_text())


class Selection(BaseModel):
    variables: dict[str, list[str]]           # variable -> allowed classes (or zone types)


class Applicability(BaseModel):
    relations: list[str] = Field(default_factory=list)   # edge names that must hold between the variables, e.g. "perimeter_of(F, Z)"
    zones: list[str] = Field(default_factory=list)       # zone types that must exist
    note: str = ''


class Requirement(BaseModel):
    edge: str                                 # edge name measured (or 'exists' / 'enclosed')
    operator: str                             # one of VOCAB['operators']
    threshold: float | None = None
    unit: str | None = None
    formula: str | None = None                # when the threshold is derived, e.g. "1600*T + 8*(d-14)"
    inputs: list[str] = Field(default_factory=list)      # declared inputs the formula / lookup needs


class Exception_(BaseModel):
    attributes: list[str] = Field(default_factory=list)  # attributes that switch the rule off, e.g. "interlock"
    note: str = ''


class ArrangedRule(BaseModel):
    rule_id: str
    source_text: str
    clause: str = ''
    selection: Selection
    applicability: Applicability = Applicability()
    requirement: Requirement | None = None
    exception: Exception_ = Exception_()
    unsupported_reason: str | None = None
    compiled_by: str = ''


def _names(section, key):
    block = VOCAB[section]
    avail = block['available'] if isinstance(block['available'], list) else list(block['available'].keys())
    planned = block['planned'] if isinstance(block['planned'], list) else list(block['planned'].keys())
    return avail, planned


def validate(rule: ArrangedRule):
    """-> (status, details) following the docstring's five statuses."""
    if rule.unsupported_reason:
        return 'refused', {'reason': rule.unsupported_reason}
    classes_a, classes_p = _names('classes', None); zones_a, zones_p = _names('zones', None)
    edges_a, edges_p = _names('edges', None); attrs_a, attrs_p = _names('attributes', None)
    gaps, invalid = [], []
    for var, allowed in rule.selection.variables.items():
        for c in allowed:
            if c in classes_a or c in zones_a:
                continue
            (gaps if c in classes_p or c in zones_p else invalid).append(f'class/zone {c} (variable {var})')
    for rel in rule.applicability.relations:
        name = rel.split('(')[0].strip()
        if name in edges_a:
            continue
        (gaps if name in edges_p else invalid).append(f'relation {name}')
    for z in rule.applicability.zones:
        if z in zones_a:
            continue
        (gaps if z in zones_p else invalid).append(f'zone {z}')
    inputs = []
    if rule.requirement:
        name = rule.requirement.edge
        if name not in edges_a and name != 'exists':
            (gaps if name in edges_p else invalid).append(f'edge {name}')
        if rule.requirement.operator not in VOCAB['operators']:
            invalid.append(f'operator {rule.requirement.operator}')
        for inp in rule.requirement.inputs:
            (inputs if inp in VOCAB['declared_inputs'] else invalid).append(inp)
    for a in rule.exception.attributes:
        if a in attrs_a:
            continue
        (gaps if a in attrs_p else invalid).append(f'attribute {a}')
    if invalid:
        return 'invalid', {'invalid': invalid}
    if gaps:
        return 'vocabulary_gap', {'gaps': gaps, 'inputs': inputs}
    if inputs:
        return 'compiled_needs_input', {'inputs': inputs}
    return 'compiled', {}


def main(path):
    rules = [ArrangedRule.model_validate(r) for r in json.loads(Path(path).read_text())['rules']]
    rows, out = [], []
    for r in rules:
        status, details = validate(r)
        out.append({**r.model_dump(), 'status': status, 'details': details})
        req = (f"{r.requirement.edge} {r.requirement.operator} {r.requirement.threshold if r.requirement.formula is None else r.requirement.formula}"
               + (f" {r.requirement.unit}" if r.requirement.unit else '')) if r.requirement else '—'
        sel = '; '.join(f"{v} ∈ {{{', '.join(c)}}}" for v, c in r.selection.variables.items())
        rows.append(f"| {r.rule_id} | {r.source_text[:70]}{'…' if len(r.source_text) > 70 else ''} | {sel} | {', '.join(r.applicability.relations) or '—'} | {req} | **{status}** | "
                    f"{'; '.join(details.get('gaps', []) + details.get('inputs', []) + details.get('invalid', []) + ([details['reason']] if 'reason' in details else []))} |")
    print('| 规则 | 自然语言 | Selection | Applicability | Requirement | 状态 | 缺什么 / 原因 |'); print('|---|---|---|---|---|---|---|'); print('\n'.join(rows))
    counts = {}
    for o in out:
        counts[o['status']] = counts.get(o['status'], 0) + 1
    print('counts:', counts)
    Path(path).with_suffix('.validated.json').write_text(json.dumps({'vocabulary': VOCAB['schema'], 'rules': out}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main(sys.argv[1])
