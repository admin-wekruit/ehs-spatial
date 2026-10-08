"""Typed lookup functions behind spec/clauses-v0.json: ISO 13857:2019 Tables 2 / 4 / 7, ISO 13855:2010 S and C, ISO 13854:2017
Table 1. Every number is a vendor reproduction collected in docs/research/verdict-layer-rules-2026-10-07-survey-B-specs-harness.md
(Troax, Reer, Datalogic, ABB, Axelent, the ISO 13854 preview); nothing was compared with a purchased standard text.

Units: millimetres and milliseconds. Where the standard says that a value between two table entries takes the entry giving the
higher safety, the lookups do that (larger distance).
"""
from __future__ import annotations

import math

VERIFIED = False   # flip only after every number below was compared with the purchased text; clauses-v0 carries the same flag

# ---------------------------------------------------------------- ISO 13857:2019 Table 2 (reaching over, high risk)
# columns: structure height b; rows: hazard height a; cell: minimum horizontal distance c (all mm).
# Cells backed by survey B (Troax): (a=2000,b=1400)=1100 (1000,1400)=1000 (400,1400)=400 (1400,1800)=800 (1000,1800)=0
# (2600,2000)=500 (a<=1000,2000)=0 (2600,2200)=400 (a<=1600,2200)=0 (2600,2500)=100 and the whole b=2700 column = 0.
# The other cells are recalled from EN ISO 13857 Table 2 and are UNVERIFIED; tests assert only the survey cells and the shape.
_T2_B = (1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, 2700)
_T2_HIGH = {
    2700: (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    2600: (900, 800, 700, 600, 600, 500, 400, 300, 100, 0),
    2400: (1100, 1000, 900, 800, 700, 600, 400, 300, 100, 0),
    2200: (1300, 1200, 1000, 900, 800, 600, 400, 300, 0, 0),
    2000: (1400, 1300, 1100, 900, 800, 600, 400, 0, 0, 0),
    1800: (1500, 1400, 1100, 900, 800, 600, 0, 0, 0, 0),
    1600: (1500, 1400, 1100, 900, 800, 500, 0, 0, 0, 0),
    1400: (1500, 1400, 1100, 900, 800, 0, 0, 0, 0, 0),
    1200: (1500, 1400, 1100, 900, 700, 0, 0, 0, 0, 0),
    1000: (1500, 1400, 1000, 800, 0, 0, 0, 0, 0, 0),
    800: (1500, 1300, 900, 600, 0, 0, 0, 0, 0, 0),
    600: (1400, 1300, 800, 0, 0, 0, 0, 0, 0, 0),
    400: (1400, 1200, 400, 0, 0, 0, 0, 0, 0, 0),
    200: (1200, 900, 0, 0, 0, 0, 0, 0, 0, 0),
    0: (1100, 500, 0, 0, 0, 0, 0, 0, 0, 0),
}
MIN_STRUCTURE_HEIGHT_MM = 1400   # Table 2 note: lower structures only together with additional measures


def iso13857_table2(hazard_height_mm: float, structure_height_mm: float, risk_level: str = "high") -> int | None:
    """ISO 13857:2019 Table 2 (high risk): minimum horizontal distance c for reaching over a protective structure of height b
    towards a hazard zone at height a. Between entries the larger c applies; a and b above 2700 count as 2700.
    None when the table does not cover the input: risk_level 'low' (Table 1 was not extracted) or b below 1000."""
    if risk_level != "high" or structure_height_mm < _T2_B[0]:
        return None
    a = min(max(hazard_height_mm, 0), 2700)
    col = max(i for i, b in enumerate(_T2_B) if b <= min(structure_height_mm, 2700))
    lo = max(r for r in _T2_HIGH if r <= a)
    hi = min(r for r in _T2_HIGH if r >= a)
    return max(_T2_HIGH[lo][col], _T2_HIGH[hi][col])


# ---------------------------------------------------------------- ISO 13857:2019 Table 4 (reaching through regular openings, >= 14 years)
# (upper bound of opening e, sr slot, sr square, sr round); survey B / Troax; ABB applies it as 40x40 mesh -> 200 mm.
_T4 = ((4, 2, 2, 2), (6, 10, 5, 5), (8, 20, 15, 5), (10, 80, 25, 20), (12, 100, 80, 80), (20, 120, 120, 120),
       (30, 850, 120, 120), (40, 850, 200, 120), (120, 850, 850, 850))
_SHAPE_COL = {"slot": 1, "square": 2, "round": 3}


def iso13857_table4(opening_mm: float, shape: str) -> int:
    """ISO 13857:2019 Table 4: safety distance sr for a regular opening of size e (slot / square / round), 0 < e <= 120 mm.
    ValueError outside that range (no longer a 'regular opening' case) or for an unknown shape."""
    if shape not in _SHAPE_COL or not 0 < opening_mm <= 120:
        raise ValueError(f"ISO 13857 Table 4 does not cover opening {opening_mm} mm / shape {shape!r}")
    return next(row for row in _T4 if opening_mm <= row[0])[_SHAPE_COL[shape]]


# ---------------------------------------------------------------- ISO 13857:2019 Table 7 (openings for lower limbs), slot rows only
_T7_SLOT = ((60, 180), (80, 650), (95, 1100), (180, 1100))   # (upper bound of e, sr) for 35 < e <= upper
WHOLE_BODY_OPENING_MM = {"slot": 180, "square": 240, "round": 240}   # clause 4.4: above this a whole body can enter


def iso13857_table7(opening_mm: float, shape: str) -> int | None:
    """ISO 13857:2019 Table 7: safety distance sr for a lower-limb opening e. Only the slot rows 35 < e <= 180 were reported
    (survey B); None for e <= 35, for e > 180 (whole-body access, clause 4.4) and for square / round openings."""
    if shape != "slot" or not 35 < opening_mm <= 180:
        return None
    return next(sr for upper, sr in _T7_SLOT if opening_mm <= upper)


# ---------------------------------------------------------------- ISO 13855:2010 (the numbers vendors still print; 2024 ed. reformulated)
def iso13855_s(stop_time_ms: float, resolution_mm: float) -> int:
    """ISO 13855:2010, normal approach to an ESPE: S = K*T + C. K = 2000 mm/s when that gives at most 500 mm (S at least 100),
    otherwise recomputed with K = 1600 mm/s (S at least 500). C = 8*(d - 14) for detection capability d <= 40 mm, else 850.
    T in ms (overall stopping performance), result in mm rounded up."""
    t = stop_time_ms / 1000
    c = max(0.0, 8 * (resolution_mm - 14)) if resolution_mm <= 40 else 850
    s = 2000 * t + c
    if s > 500:
        s = max(1600 * t + c, 500)
    return math.ceil(max(s, 100))


def iso13855_horizontal_c(height_mm: float) -> int:
    """ISO 13855:2010, parallel approach (horizontal field at height H above the floor): C = 1200 - 0.4*H, at least 850.
    H may not exceed 1000 mm (ValueError); above 300 mm the standard asks for a crawl-under assessment (not encoded)."""
    if not 0 <= height_mm <= 1000:
        raise ValueError(f"ISO 13855 parallel approach: field height {height_mm} mm outside 0..1000")
    return math.ceil(max(850, 1200 - 0.4 * height_mm))


ISO13855_MULTIBEAM_HEIGHTS_MM = {4: (300, 600, 900, 1200), 3: (300, 700, 1100), 2: (400, 900), 1: (750,)}


def iso13855_multibeam_heights(n_beams: int) -> tuple[int, ...]:
    """ISO 13855:2010, multiple separate beams: beam heights above the floor for 1..4 beams (their S = 1600*T + 850).
    KeyError for other beam counts."""
    return ISO13855_MULTIBEAM_HEIGHTS_MM[n_beams]


# ---------------------------------------------------------------- ISO 13854:2017 Table 1 (minimum gaps to avoid crushing)
# body / head / leg were seen in the ISO preview; foot, toes, arm, hand, finger come from a secondary source (finger unverified).
ISO13854_GAP_MM = {"body": 500, "head": 300, "leg": 180, "foot": 120, "toes": 50, "arm": 120, "hand": 100, "finger": 25}
_BODY_PART_ALIAS = {"wrist": "hand", "fist": "hand", "toe": "toes"}


def iso13854_gap(body_part: str) -> int:
    """ISO 13854:2017 Table 1: minimum gap (mm) so that the named body part cannot be crushed. Crushing only, not impact or
    shearing; when several parts can enter, the caller takes the largest. KeyError for an unknown body part."""
    return ISO13854_GAP_MM[_BODY_PART_ALIAS.get(body_part, body_part)]
