"""The EHS safety-device taxonomy: what a reviewer expects to find in a
robot-cell photo, organized the way machine-safety audits organize it
(sensing / control / guards / impeding / information). The detection layer
walks this list item by item — every item is either located with a box or
reported missing, so coverage is a checklist, not a segmentation lottery.

ISO references are the well-established ones only; clause-level citations
are deliberately avoided.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeviceType:
    item_id: str
    category: str  # A/B/C/D/E
    zh: str
    en: str  # locate description for the VLM
    sam_label: str  # label used for SAM/scene entities
    iso: str = ""
    # 'pair' hints the detector to look left AND right; 'multi' means any
    # count; 'single' means at most one is expected
    expect: str = "single"
    aliases: tuple[str, ...] = field(default_factory=tuple)


CATEGORIES = {
    "A": ("感知防护 SENSING / AOPD", "#39c5cf"),
    "B": ("控制防护 CONTROL", "#f25c8a"),
    "C": ("防护罩/围护 GUARDS", "#4ad07a"),
    "D": ("阻挡与引导 IMPEDING", "#e8b93c"),
    "E": ("信息标识 INFORMATION", "#c9a0ff"),
    "F": ("物料/载具 PAYLOAD", "#ff8fa3"),
}


TAXONOMY: tuple[DeviceType, ...] = (
    # A — sensing protective devices
    DeviceType("a1", "A", "光幕立柱(左)", "vertical safety light curtain pillar on the LEFT side of the entrance (yellow/black striped column)", "safety sensor", "IEC 61496 / ISO 13855", "single"),
    DeviceType("a2", "A", "光幕立柱(右)", "vertical safety light curtain pillar on the RIGHT side of the entrance (yellow/black striped column)", "safety sensor", "IEC 61496 / ISO 13855", "single"),
    DeviceType("a3", "A", "贴地传感横杆", "horizontal floor-level sensor bar (yellow/black), often at the base of a sloped kick plate", "safety sensor", "IEC 61496", "multi"),
    # B — control protective devices
    DeviceType("b1", "B", "急停按钮", "emergency stop button (red mushroom head on yellow base)", "emergency stop button", "ISO 13850", "multi"),
    DeviceType("b2", "B", "复位/启动按钮", "reset or start push-button station near the entrance", "control button", "ISO 12100", "multi"),
    DeviceType("b3", "B", "门联锁开关", "safety door interlock switch or interlocked handle on a post or gate", "door interlock switch", "ISO 14119", "multi"),
    # C — guards
    DeviceType("c1", "C", "透明护板", "clear polycarbonate guard panel in an aluminium frame", "safety fence", "ISO 14120", "multi"),
    DeviceType("c2", "C", "铁网围栏", "wire mesh enclosing guard / fence panel", "safety fence", "ISO 14120", "multi"),
    DeviceType("c3", "C", "低位护栏/闸门", "low guard rail or gate section around the machine base", "safety fence", "ISO 14120", "multi"),
    # D — impeding & complementary
    DeviceType("d1", "D", "斜坡挡板(左)", "red sloped kick plate / deflector at floor level on the LEFT", "sloped surface", "ISO 12100", "single"),
    DeviceType("d2", "D", "斜坡挡板(右)", "red sloped kick plate / deflector at floor level on the RIGHT", "sloped surface", "ISO 12100", "single"),
    DeviceType("d3", "D", "防撞柱/护柱", "bollard or impact-protection post", "bollard", "ISO 12100", "optional"),
    # E — information for use
    DeviceType("e1", "E", "工位标识牌", "station ID placard or cell name sign (e.g. white sign with station code)", "warning sign", "ISO 3864", "multi"),
    DeviceType("e2", "E", "作业指导/警示牌", "work instruction placard or warning sign on a guard panel", "warning sign", "ISO 3864 / ISO 7010", "multi"),
    DeviceType("e3", "E", "地面警示胶带", "yellow/black hazard tape line on the floor", "floor marking", "ISO 3864", "multi"),
    DeviceType("e4", "E", "信号灯", "signal tower light or dome indicator light", "safety light", "ISO 12100", "multi"),
    # F — the movable payload that clearance policies measure against
    DeviceType("f1", "F", "物料容器", "large metal shipping container / pallet bin holding parts (often white with racks and branding), the main payload in the cell", "material cart", "", "multi"),
    DeviceType("f2", "F", "运载小车", "wheeled docking cart, AGV carrier or tug frame under/with the container (may have a red tow bar)", "material cart", "", "multi"),
    DeviceType("f3", "F", "导向挡板", "angled yellow/black striped guide or deflector plates funneling the carrier into the dock", "sloped surface", "ISO 12100", "multi"),
)


__all__ = ["TAXONOMY", "CATEGORIES", "DeviceType"]
