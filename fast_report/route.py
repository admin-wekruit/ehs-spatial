"""r5b (models): which display model an object card gets (route/jev's recommended rule, moved here from scripts/route_jev.py so
the report runs it; that script imports it back).

  route()      1) a card no generator takes (display_model.well_observed fails) keeps its observed surface or primitive, no
               question; 2) a class that fixes the shape decides (FIXED_SHAPE_POST_HOC: machines, tools, carts, furniture,
               cables -> generated; boxes, pallets, shelves, racks, signs, pipes, boards -> primitive); 3) Jev-Omni Q5 for the
               rest: P(none of the simple shapes) > P_COMPLEX_CUT -> generated. No VLM.
  soft()       bags and soft goods: never generated (their shape is the moment's), the observed surface stays.
  groups()     look-alike groups: the same type (the shown name, or a type-only card's type) and every sorted box
               extent within LOOK_ALIKE of the group's first (best-seen) card: one generated model per group, reused by the rest.

    python -m fast_report.route --self-check
"""
import numpy as np

from fast_report import cards, display_model

# Written from the user's definition before any object was labelled (route/jev 1b61ed1); POST HOC: minus the classes the labels
# showed to be shape-ambiguous (route-jev-001: bag -> pet-food bags are boxes; tool box / tool tray: boxes and trays; crate,
# display rack: mixed)
SIMPLE_CLASSES = {"shelf", "rack", "display rack", "cabinet", "locker", "refrigerator", "box", "stacked boxes", "pallet", "crate", "drum",
                  "can", "trash can", "first aid kit", "exit sign", "safety sign", "sign", "label", "floor marking", "bollard", "pipe", "duct",
                  "cable tray", "control panel", "electrical outlet", "switch", "door", "window", "column", "wall panel", "floor drain", "vent",
                  "wooden board", "metal sheet", "whiteboard", "paper", "clipboard", "book", "spill"}
COMPLEX_FAMILIES = {"machine", "tool", "handling", "furniture", "ppe"}
COMPLEX_CLASSES = {"bag", "ladder", "step stool", "work platform", "stairs", "fire extinguisher", "eyewash station", "emergency stop button",
                   "fire alarm", "guard", "railing", "safety cone", "cable", "hose", "fan", "keyboard", "printer", "phone", "rag", "cup",
                   "mannequin", "wrap"}
FIXED_SHAPE_POST_HOC = {"simple": SIMPLE_CLASSES - {"crate", "display rack"}, "drop": {"bag", "tool box", "tool tray"}}
P_COMPLEX_CUT = .5  # the cost knob: generate when P(complex) > cut (route-jev-001: 0.904 held out at 0.5, no fitted number)
JEV_STATE = ("This image is cropped from a video walk-through of a workplace (a workshop, warehouse or store). A white outline "
             "with a black edge, tagged 1, marks one region.")
Q5 = ("Which shape best represents the outlined object in a 3D model of the place?",
      ["a flat panel, board, sign, wall or door", "a plain box, carton, block or stack of boxes", "a cylinder: a pipe, pole, drum or roll",
       "an open frame: a shelf or rack", "none of these simple shapes: a machine, cart, tool, equipment, furniture or an irregular object"])
Q5_KIND = ["plane", "box", "cylinder", "open frame", None]
SOFT_CLASSES = {"bag", "rag", "wrap", "curtain", "glove"}
SOFT_WORDS = ("bag", "sack", "clothing", "garment", "towel", "pillow", "plush", "stuffed", "sock", "hat", "slipper", "blanket", "cushion",
              "bed pad", "training pad", "diaper", "tissue", "rug", "mat", "cloth", "rag", "wrap", "curtain", "glove", "bootie")
LOOK_ALIKE = 1.25  # every sorted box extent within x1.25 of the group's first card's (r5b: a priori, not fitted)


def class_route(cls, post_hoc=True):
    """'simple' | 'complex' | None (no prior: the decider decides)."""
    if not cls or (post_hoc and cls in FIXED_SHAPE_POST_HOC["drop"]):
        return None
    if cls in (FIXED_SHAPE_POST_HOC["simple"] if post_hoc else SIMPLE_CLASSES):
        return "simple"
    if cls in COMPLEX_CLASSES or cards.FAMILY.get(cls) in COMPLEX_FAMILIES:
        return "complex"
    return None


def card_class(card):
    """The naming cascade's canonical class (identity.canonical, else the detector word's), or None."""
    idn = card.get("identity") or {}
    c = idn.get("canonical") or cards.canonical(idn.get("proposed") or idn.get("name") or "")
    return None if c in (None, cards.NOT_OBJECT) else c


def route(card, q5=None, cut=P_COMPLEX_CUT):
    """-> (display model, why): 'primitive', 'generated' or 'ask jev' (then call again with Jev-Omni's Q5 probabilities: raw,
    one question on the card's outlined best view). 'primitive' means 'no generated model': display_model.for_card decides
    between the primitive and the observed surface."""
    score, why = display_model.well_observed(card)
    if score is None:
        return "primitive", f"no generator takes it ({why})"
    cls = card_class(card)
    r = class_route(cls)
    if r:
        return ("generated" if r == "complex" else "primitive"), f"the class fixes the shape ({cls})"
    if q5 is None:
        return "ask jev", f"no class fixes the shape ({cls}): Q5 on the outlined best view"
    return ("generated" if q5[4] > cut else "primitive"), f"jev q5: P(none of the simple shapes) {q5[4]:.2f} vs {cut}"


def soft(card):
    """Bags and soft goods (their shape is the moment's: a generated model would invent one)."""
    if card_class(card) in SOFT_CLASSES:
        return True
    name = cards.norm((card.get("identity") or {}).get("name") or "")
    return any((w in name) if " " in w else (w in name.split()) for w in SOFT_WORDS)


def type_key(card):
    """The look-alike key: the shown name, or a type-only card's type (its family's label: the size gate and the copy's outline
    check keep a group to look-alikes), normalised; None for an unnamed card (it is its own group)."""
    name = (card.get("identity") or {}).get("name") or ""
    if not name or name in (cards.UNIDENTIFIED, cards.NOT_OBJECT):
        return None
    return cards.norm(name[:-len(cards.TYPE_ONLY)] if name.endswith(cards.TYPE_ONLY) else name)


def extents(card):
    size = ((card.get("physical") or {}).get("box") or {}).get("size_m")
    return np.sort(np.asarray(size, float)) if size else None


def groups(cs):
    """cs: cards best-seen first -> [[card, ...]] look-alike groups, each led by its first (best-seen) card."""
    out, by = [], {}
    for c in cs:
        k, e = type_key(c), extents(c)
        home = None
        if k is not None and e is not None:
            for g in by.get(k, []):
                e0 = extents(g[0])
                if np.all(np.maximum(e, 1e-3) / np.maximum(e0, 1e-3) <= LOOK_ALIKE) and np.all(np.maximum(e0, 1e-3) / np.maximum(e, 1e-3) <= LOOK_ALIKE):
                    home = g
                    break
        if home is None:
            home = [c]
            out.append(home)
            if k is not None:
                by.setdefault(k, []).append(home)
        else:
            home.append(c)
    return out


def self_check():
    card = {"views": {"n": 5, "azimuth_spread_deg": 40, "distance_m": [1.5, 3.]}, "physical": {"top_above_floor": {"value": 1.}, "dropped_share": .05,
            "box": {"size_m": [.5, .4, .6]}}, "raw": {"size": {"longest": .6}}, "model": {"kind": "box"}}
    named = lambda n, **kw: {**card, "identity": {"name": n, "canonical": cards.canonical(n)}, **kw}  # noqa: E731
    assert route(named("milling machine"))[0] == "generated" and route(named("cardboard box"))[0] == "primitive"
    assert route(named("sneaker"))[0] == "ask jev" and route(named("sneaker"), [.1, .1, .1, .1, .6])[0] == "generated"
    assert route({**named("milling machine"), "views": {"n": 1}})[0] == "primitive"  # not well observed: no generator
    assert soft(named("plastic bag")) and soft(named("dog bed pillow")) and not soft(named("milling machine")) and not soft(named("shoe hanger"))
    a, b, c_, d = named("sneaker"), named("sneaker"), named("sneaker"), named("box sensor (type only)")
    b = {**b, "physical": {**b["physical"], "box": {"size_m": [.52, .41, .58]}}}
    c_ = {**c_, "physical": {**c_["physical"], "box": {"size_m": [1., .4, .6]}}}
    g = groups([a, b, c_, d, d, named("unidentified object")])
    assert [len(x) for x in g] == [2, 1, 2, 1], [len(x) for x in g]  # a type-only pair of one size groups; an unnamed card never
    print("route self-check ok: class tier, Jev tier, view gate, soft goods, look-alike groups")


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
