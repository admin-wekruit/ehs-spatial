"""Spot values (the research note's numbers) and monotonicity of the l4_spec lookup functions."""
import pytest

from ehs_spatial.verdict.layers.l4_spec import tables as T


def test_nothing_verified_yet():
    assert T.VERIFIED is False


# ---------------------------------------------------------------- ISO 13857 Table 2
def test_table2_survey_b_cells():
    t2 = T.iso13857_table2
    assert (t2(2000, 1400), t2(1000, 1400), t2(400, 1400)) == (1100, 1000, 400)
    assert (t2(1400, 1800), t2(1000, 1800)) == (800, 0)
    assert t2(2600, 2000) == 500 and all(t2(a, 2000) == 0 for a in (0, 400, 800, 1000))
    assert t2(2600, 2200) == 400 and all(t2(a, 2200) == 0 for a in (0, 1000, 1400, 1600))
    assert t2(2600, 2500) == 100
    assert all(t2(a, 2700) == 0 for a in range(0, 2701, 100)) and t2(1000, 3000) == 0
    assert T.MIN_STRUCTURE_HEIGHT_MM == 1400


def test_table2_outside_extraction_is_none():
    assert T.iso13857_table2(2000, 1400, risk_level="low") is None   # Table 1 not extracted
    assert T.iso13857_table2(2000, 900) is None                      # below the table's lowest structure


def test_table2_distance_never_grows_with_structure_height():
    for a in range(0, 2701, 100):
        cs = [T.iso13857_table2(a, b) for b in range(1000, 2800, 50)]
        assert cs == sorted(cs, reverse=True), a


def test_table2_between_entries_takes_the_safer_value():
    t2 = T.iso13857_table2
    assert t2(1900, 1400) == max(t2(1800, 1400), t2(2000, 1400))
    assert t2(2000, 1500) == t2(2000, 1400)   # lower structure column
    assert t2(2650, 2450) == max(t2(2600, 2400), t2(2700, 2400))


# ---------------------------------------------------------------- ISO 13857 Table 4
def test_table4_spot_values():
    t4 = T.iso13857_table4
    assert t4(40, "square") == 200             # ABB: 40x40 mesh -> 200 mm
    assert (t4(4, "slot"), t4(6, "slot"), t4(25, "slot"), t4(120, "round"), t4(12, "round"), t4(30, "square")) == (2, 10, 850, 850, 80, 120)
    for bad in ((121, "slot"), (0, "slot"), (40, "hex")):
        with pytest.raises(ValueError):
            t4(*bad)


def test_table4_monotone_in_opening():
    for shape in ("slot", "square", "round"):
        srs = [T.iso13857_table4(e, shape) for e in range(1, 121)]
        assert srs == sorted(srs), shape


# ---------------------------------------------------------------- ISO 13857 Table 7
def test_table7_slot_rows():
    t7 = T.iso13857_table7
    assert (t7(36, "slot"), t7(60, "slot"), t7(61, "slot"), t7(80, "slot"), t7(95, "slot"), t7(180, "slot")) == (180, 180, 650, 650, 1100, 1100)
    assert t7(35, "slot") is None and t7(181, "slot") is None and t7(50, "square") is None
    srs = [t7(e, "slot") for e in range(36, 181)]
    assert srs == sorted(srs)
    assert T.WHOLE_BODY_OPENING_MM == {"slot": 180, "square": 240, "round": 240}


# ---------------------------------------------------------------- ISO 13855
def test_iso13855_s_branches_and_minimums():
    s = T.iso13855_s
    assert s(100, 14) == 200      # K = 2000, C = 0
    assert s(10, 14) == 100       # minimum for K = 2000
    assert s(300, 30) == 608      # 2000 branch gives 728 > 500 -> 1600*0.3 + 128
    assert s(200, 30) == 500      # 1600 branch gives 448 -> minimum 500
    assert s(100, 50) == 1010     # d > 40: C = 850
    assert s(0, 10) == 100        # C never negative


def test_iso13855_s_monotone():
    for d in (14, 30, 40, 41, 70):
        ss = [T.iso13855_s(t, d) for t in range(0, 3001, 10)]
        assert ss == sorted(ss), d
    for t in (50, 100, 400, 1000):
        ss = [T.iso13855_s(t, d) for d in range(14, 71)]
        assert ss == sorted(ss), t


def test_iso13855_horizontal_c():
    c = T.iso13855_horizontal_c
    assert (c(0), c(300), c(875), c(900), c(1000)) == (1200, 1080, 850, 850, 850)
    cs = [c(h) for h in range(0, 1001)]
    assert cs == sorted(cs, reverse=True)
    for bad in (-1, 1001):
        with pytest.raises(ValueError):
            c(bad)


def test_multibeam_heights():
    assert T.iso13855_multibeam_heights(4) == (300, 600, 900, 1200)
    assert T.iso13855_multibeam_heights(3) == (300, 700, 1100)
    assert T.iso13855_multibeam_heights(2) == (400, 900)
    assert T.iso13855_multibeam_heights(1) == (750,)
    with pytest.raises(KeyError):
        T.iso13855_multibeam_heights(5)


# ---------------------------------------------------------------- ISO 13854
def test_iso13854_gap():
    g = T.iso13854_gap
    assert [g(p) for p in ("body", "head", "leg", "foot", "toes", "arm", "hand", "finger")] == [500, 300, 180, 120, 50, 120, 100, 25]
    assert g("wrist") == g("fist") == 100
    assert max(g(p) for p in ("hand", "head", "body")) == 500   # several parts can enter -> the largest
    with pytest.raises(KeyError):
        g("tail")
