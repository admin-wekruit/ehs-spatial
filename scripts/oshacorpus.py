"""Real OSHA 1910 text -> classified policy-compiler exam corpus.

The policy compiler must refuse what its closed predicate vocabulary cannot
express. Paraphrased starter rules cannot prove that; real regulation text
can, because real paragraphs mix expressible numeric clearances with
requirements the vocabulary has no words for (path distance, voltage-keyed
tables, minimum headroom, alternative-compliance disjunctions). This script
fetches the US-federal public-domain source text from the eCFR API, caches
it as fixtures, extracts per-paragraph citations, and classifies each
paragraph:

  compile - carries an explicit numeric spatial requirement the vocabulary
            expresses (predicate + threshold + unit recorded),
  refuse  - a spatial requirement the vocabulary cannot express; the
            compiler must set unsupported_reason, never approximate,
  skip    - not a spatial requirement at all.

Offline by default (tests must never touch the network); --fetch refreshes
the cached XML. curl is used for the fetch because this environment's
python lacks SSL certs (same workaround as scripts/nvwarehouse_eval.py).

Outputs:
  tests/fixtures/oshacorpus/corpus.json  full classified corpus
  docs/policies/osha1910.md              exam sheet: one curated rule per
                                         line, ready for the orchestrator to
                                         run scripts/policy_compile.py --live
"""

import argparse
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "oshacorpus"
CORPUS_JSON = FIXTURES / "corpus.json"
EXAM_MD = ROOT / "docs" / "policies" / "osha1910.md"

# Pinned so a --fetch refresh reproduces the cached fixtures byte-for-byte
# unless OSHA actually amends the text.
ECFR_DATE = "2026-08-01"
ECFR_URL = (
    "https://www.ecfr.gov/api/versioner/v1/full/{date}/title-29.xml"
    "?part=1910&section={section}"
)

SECTIONS = [
    "1910.36",   # exit routes: design and construction
    "1910.37",   # exit routes: maintenance and operation
    "1910.157",  # portable fire extinguishers
    "1910.159",  # automatic sprinkler systems
    "1910.176",  # materials handling: clearances
    "1910.212",  # machine guarding
    "1910.253",  # oxygen-fuel gas welding and cutting
    "1910.303",  # electrical: general / working space
    "1910.333",  # electrical: work practices / approach distances
]

# Curated exam expectations, keyed by exact citation. Everything not listed
# here defaults to "skip": the exam only claims classifications a human
# actually reviewed against the vocabulary.
EXPECTATIONS: dict[str, dict] = {
    # ---- compile: explicit numeric requirement the vocabulary expresses ----
    "1910.253(b)(2)(ii)": {
        "expected": "compile",
        "expected_predicate": "min_separation",
        "expected_threshold": 6.1,
        "expected_unit": "m",
        "note": "cylinders at least 20 feet (6.1 m) from highly combustible "
        "materials: plain Euclidean floor-plan separation",
    },
    "1910.253(f)(5)(i)(B)": {
        "expected": "compile",
        "expected_predicate": "min_separation",
        "expected_threshold": 3.0,
        "expected_unit": "m",
        "note": "portable generators not used within 10 feet (3 m) of "
        "combustible material",
    },
    "1910.303(h)(3)": {
        "expected": "compile",
        "expected_predicate": "keep_clear",
        "expected_threshold": 0.914,
        "expected_unit": "m",
        "note": "minimum clear work space 914 mm (3.0 ft) wide about "
        "over-600V equipment; the 6.5 ft vertical band rides along but the "
        "enforceable plan-space core is the 3 ft clearance",
    },
    # ---- refuse: spatial, but outside the vocabulary ----
    "1910.176(a)": {
        "expected": "refuse",
        "refuse_reason": "'sufficient safe clearances' carries no numeric "
        "threshold; nothing to compile without inventing a number",
    },
    "1910.333(c)(3)(ii)": {
        "expected": "refuse",
        "refuse_reason": "approach distance comes from Table S-5, keyed to "
        "circuit voltage: a non-spatial variable the scene cannot measure",
    },
    "1910.157(d)(2)": {
        "expected": "refuse",
        "refuse_reason": "75-foot limit is walking travel distance along a "
        "path, not the Euclidean separation the predicates measure",
    },
    "1910.157(d)(4)": {
        "expected": "refuse",
        "refuse_reason": "50-foot Class B limit is travel distance along a "
        "path, and conditional on fire class besides",
    },
    "1910.36(g)(1)": {
        "expected": "refuse",
        "refuse_reason": "minimum ceiling height 2.3 m: vocabulary has "
        "max_height only, no minimum-height/headroom predicate",
    },
    "1910.36(g)(2)": {
        "expected": "refuse",
        "refuse_reason": "28-inch minimum width of a passage is a property "
        "of free space between obstructions, not a labeled subject-object "
        "separation",
    },
    "1910.253(b)(4)(iii)": {
        "expected": "refuse",
        "refuse_reason": "20 ft separation OR a rated barrier: an "
        "alternative-compliance disjunction; compiling only the distance "
        "arm would raise false violations where a barrier exists",
    },
    "1910.303(g)(1)(i)(A)": {
        "expected": "refuse",
        "refuse_reason": "working-space depth comes from Table S-1, keyed "
        "to voltage class and opposing-surface condition",
    },
    "1910.303(g)(1)(i)(B)": {
        "expected": "refuse",
        "refuse_reason": "width is 762 mm or equipment width, whichever is "
        "greater: threshold conditional on per-entity dimensions",
    },
    "1910.159(c)(10)": {
        "expected": "refuse",
        "refuse_reason": "18-inch sprinkler clearance is vertical, between "
        "a head above and material below; separation predicates measure "
        "plan-space (XY footprint) gaps",
    },
}


# ---------------------------------------------------------------------------
# Citation parsing. CFR paragraph markers ladder down a fixed hierarchy:
# (a) -> (1) -> (i) -> (A) -> (1) -> (i). A marker either continues one of
# the currently open levels (it is that level's successor) or starts the
# next level down (it is that level's first token). Checking successors
# deepest-first resolves the classic (i)-after-(h) alpha/roman ambiguity.
# ---------------------------------------------------------------------------

_DEPTH_TYPES = ("alpha", "num", "roman", "ALPHA", "num", "roman")
# 1910.212 predates the modern ladder and uses italic (a)..(i) at depth 4.
_START = (("a",), ("1",), ("i",), ("A", "a"), ("1",), ("i",))
_ROMAN = re.compile(r"^[ivxlcdm]+$")
_ROMAN_PAIRS = ((10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"))


def _roman_to_int(tok: str) -> int:
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    for ch, nxt in zip(tok, tok[1:] + " "):
        v = values[ch]
        total += -v if nxt != " " and values.get(nxt, 0) > v else v
    return total


def _int_to_roman(n: int) -> str:
    out = ""
    for value, glyph in _ROMAN_PAIRS:
        while n >= value:
            out += glyph
            n -= value
    return out


def _successor(marker: str, level_type: str) -> str | None:
    if level_type == "num":
        return str(int(marker) + 1) if marker.isdigit() else None
    if level_type == "roman":
        if not _ROMAN.match(marker):
            return None
        return _int_to_roman(_roman_to_int(marker) + 1)
    # alpha / ALPHA: next letter, same case as what the level opened with.
    if len(marker) == 1 and marker.isalpha() and marker != "z" and marker != "Z":
        return chr(ord(marker) + 1)
    return None


def _advance(stack: list[str], token: str) -> bool:
    """Consume one marker token, mutating the open-level stack."""
    for depth in range(len(stack) - 1, -1, -1):
        if _successor(stack[depth], _DEPTH_TYPES[depth]) == token:
            del stack[depth + 1 :]
            stack[depth] = token
            return True
    if len(stack) < len(_START):
        starts = _START[len(stack)]
        if token in starts:
            stack.append(token)
            return True
        # Recovery: a start marker can hide inside a parenthesized aside
        # the heading matcher refuses (e.g. 1910.253(c)(2) "...(for use
        # with cylinders...)). (i) ..."). Seeing that level's SECOND
        # marker at paragraph start means the first was missed - open the
        # level there rather than mislabel every following sibling.
        level_type = _DEPTH_TYPES[len(stack)]
        if any(_successor(s, level_type) == token for s in starts):
            stack.append(token)
            return True
    return False


_LEAD_MARKER = re.compile(r"^\(([a-zA-Z0-9]{1,4})\)\s*")
# "(b) Exemptions. (1) Where ..." - a heading sentence, then the first
# marker of the next level down. The no-parens guard keeps mid-sentence
# cross-references like "Paragraph (d) of this section" from matching;
# the lookahead (instead of consuming a space) admits runs like "(i)(A)".
_HEAD_MARKER = re.compile(r"^([^()]{1,90}?[.—:])\s*\(([a-zA-Z0-9]{1,4})\)(?=[\s(])")


def parse_section(xml_text: str, section: str) -> list[dict]:
    """Per-paragraph text with citation ids like '1910.176(a)'.

    Only direct <P> children of the section DIV are regulation text;
    <NOTE> guidance, tables, and source credits are not requirements and
    are excluded on purpose.
    """
    root = ET.fromstring(xml_text)
    stack: list[str] = []
    rows = []
    for p in root.iterfind("P"):
        text = " ".join("".join(p.itertext()).split())
        rest = text
        while True:
            if (match := _LEAD_MARKER.match(rest)) and _advance(
                stack, match.group(1)
            ):
                rest = rest[match.end() :]
                continue
            # One paragraph may ladder several levels down through interior
            # headings ("(c) Manifolding-(1) Fuel-gas manifolds. (i) ...").
            match = _HEAD_MARKER.match(rest)
            if (
                match
                and len(stack) < len(_START)
                and match.group(2) in _START[len(stack)]
            ):
                stack.append(match.group(2))
                rest = rest[match.end() :]
                continue
            break
        citation = section + "".join(f"({m})" for m in stack)
        rows.append({"citation": citation, "text": text})
    return rows


def build_corpus() -> list[dict]:
    rows = []
    for section in SECTIONS:
        xml_path = FIXTURES / f"{section}.xml"
        for row in parse_section(xml_path.read_text(), section):
            expectation = EXPECTATIONS.get(row["citation"], {"expected": "skip"})
            rows.append({**row, **expectation})
    found = {row["citation"] for row in rows}
    missing = sorted(set(EXPECTATIONS) - found)
    if missing:  # a parser regression must fail loudly, not mislabel the exam
        raise RuntimeError(f"curated citations not found in parse: {missing}")
    return rows


def fetch(section: str) -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "curl",
            "-sSfL",
            ECFR_URL.format(date=ECFR_DATE, section=section),
            "-o",
            str(FIXTURES / f"{section}.xml"),
        ],
        check=True,
    )


def write_exam_sheet(rows: list[dict]) -> None:
    """One curated rule per line, the format policy_compile.py reads.

    The orchestrator can then run the live compiler over real OSHA prose:
      uv run --env-file .env python scripts/policy_compile.py \
        --policies docs/policies/osha1910.md --live
    and diff its output against the expected columns in corpus.json
    (exam line N corresponds to the Nth non-skip corpus row).
    """
    lines = [
        "# OSHA 1910 compiler exam sheet - generated by scripts/oshacorpus.py",
        "# Verbatim 29 CFR paragraphs (US federal public domain, eCFR "
        f"{ECFR_DATE}).",
        "# Expected outcomes live in tests/fixtures/oshacorpus/corpus.json;",
        "# rows marked 'refuse' there MUST compile with unsupported_reason set.",
        "",
    ]
    lines += [
        f"- [{row['citation']}] {row['text']}"
        for row in rows
        if row["expected"] != "skip"
    ]
    EXAM_MD.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="refresh cached eCFR XML (network; default replays the cache)",
    )
    args = parser.parse_args(argv)

    if args.fetch:
        for section in SECTIONS:
            fetch(section)
            print(f"fetched {section}")

    rows = build_corpus()
    CORPUS_JSON.write_text(json.dumps(rows, indent=2) + "\n")
    write_exam_sheet(rows)

    counts = {"compile": 0, "refuse": 0, "skip": 0}
    for row in rows:
        counts[row["expected"]] += 1
    print(
        f"{len(SECTIONS)} sections | {len(rows)} paragraphs | "
        f"{counts['compile']} compile / {counts['refuse']} refuse / "
        f"{counts['skip']} skip"
    )
    print(f"wrote {CORPUS_JSON}")
    print(f"wrote {EXAM_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
