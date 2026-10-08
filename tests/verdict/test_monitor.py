"""L3 stpl@1 on a hand-built Scene: one view -> untrusted, floating floor_contact -> untrusted, implausible size -> untrusted."""
from pathlib import Path

from ehs_spatial.verdict import plugins
from ehs_spatial.verdict.contracts import Coverage, Fact, Facts, Obj, Scene
from ehs_spatial.verdict.layers.l3_monitor.stpl import Stpl


def box(oid, cls, size, bottom=0.0, views=("1", "2"), floor_contact=True):
    return Obj(id=oid, cls=cls, center_m=[0, 0, bottom + size[2] / 2], axes=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size_m=list(size),
               bottom_m=bottom, top_m=bottom + size[2], views=list(views), floor_contact=floor_contact)


SCENE = Scene(scene_id="t", ground_normal=[0, 0, 1], objects=[
    box("ok", "fence", (0.1, 3.0, 2.0)),
    box("one_view", "bollard", (0.1, 0.1, 0.9), views=("2",)),
    box("floating", "guard", (0.5, 1.0, 0.6), bottom=0.42, floor_contact=True),
    box("huge", "fence", (5.0, 5.0, 0.1)),
    box("unknown_class", "other", (9.0, 9.0, 9.0), views=("1", "2", "3")),
])
FACTS = Facts(scene_id="t", signature_version="1", facts=[Fact(pred="top_height", args=["one_view"], value=900, unit="mm"),
                                                           Fact(pred="top_height", args=["ok"], value=2000, unit="mm")])


def test_registered_and_flags_the_bad_objects(tmp_path: Path):
    assert plugins.get("L3", "stpl") is Stpl and plugins.tag(Stpl) == "stpl@1"
    facts = Stpl().run({"scene": SCENE, "facts": FACTS}, {}, tmp_path)["facts"]
    assert facts.quality["untrusted"] == ["one_view", "floating", "huge"]
    assert [(f.args[0], f.flags) for f in facts.facts if f.pred == "untrusted"] == [("one_view", ["views"]), ("floating", ["floor"]), ("huge", ["size"])]
    assert facts.facts[0].flags == ["untrusted"] and facts.facts[1].flags == []     # existing facts of untrusted objects are flagged
    assert facts.quality["coverage_ratio"] is None and facts.quality["checks"]["unknown_class"]["failed"] == []
    assert len(FACTS.facts) == 2 and FACTS.quality == {}                            # input not mutated


def test_cfg_and_coverage_ratio(tmp_path: Path):
    cov = Coverage(cell_m=0.1, origin_xy=[0, 0], basis=[[1, 0, 0], [0, 1, 0]], shape=[10, 10], observed=[[0, 0], [1, 1], [2, 2]])
    scene = SCENE.model_copy(update={"coverage": cov})
    facts = Stpl().run({"scene": scene, "facts": FACTS}, {"min_views": 1, "floor_tol_m": 0.5}, tmp_path)["facts"]
    assert facts.quality["untrusted"] == ["huge"] and facts.quality["coverage_ratio"] == 0.03
