"""Parametric synthetic workcell -> contract Scene (metres, world frame, floor normal +z, every box axis-aligned).

cell(...) places, around a 1 x 1 x 2 m robot at the origin:
  four fence segments forming a closed ring whose inner faces are `robot_fixed_gap_m` from the robot's faces (east segment shortened by
  `opening_width_m` when enclosed=False), bottom at `floor_gap_m`, top at `fence_height_m`;
  two bollards and one light curtain east of the ring (far enough never to drive a crush-gap verdict), the light curtain's bottom at
  `lc_bottom_m`; one cart west of the ring: the trial rules never select 'cart', so it is the object the metamorphic
  'delete an unreferenced object' relation removes.
Every object carries the same 1-sigma per quantity (sigma_m: float or {'L','W','H','bottom'}); scale_rel_unc defaults to 0.0 (declared,
not None, so no default 2 % kicks in and U does not depend on the value). coverage='full' writes a Coverage on the same plan frame that
synth/facts.py uses, every cell observed; 'none' leaves Scene.coverage = None (topology rules can only say CANNOT_DETERMINE on a breach).
"""
from __future__ import annotations

from ehs_spatial.verdict.contracts import Coverage, Obj, Scene
from ehs_spatial.verdict.synth.facts import grid_frame

AXES = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
ROBOT_HALF_M = 0.5
ROBOT_TOP_M = 2.0
FENCE_T_M = 0.05


def box(id: str, cls: str, cx: float, cy: float, L: float, W: float, bottom: float, top: float, sigma: dict, label: str = "") -> Obj:
    return Obj(id=id, cls=cls, label=label or id, center_m=[cx, cy, (bottom + top) / 2], axes=AXES, size_m=[L, W, top - bottom],
               bottom_m=bottom, top_m=top, sigma_m=dict(sigma), confidence="high", floor_contact=bottom <= 0.0, views=["synthetic"])


def cell(fence_height_m: float = 2.0, floor_gap_m: float = 0.10, robot_fixed_gap_m: float = 1.0, lc_bottom_m: float = 0.25,
         enclosed: bool = True, opening_width_m: float = 0.8, coverage: str = "full", sigma_m: float | dict = 0.01,
         scale_rel_unc: float | None = 0.0, declared_inputs: dict | None = None, scene_id: str | None = None) -> Scene:
    if fence_height_m <= floor_gap_m:
        raise ValueError("fence_height_m must exceed floor_gap_m")
    if coverage not in ("full", "none"):
        raise ValueError("coverage is 'full' or 'none'")
    if not enclosed and opening_width_m <= 0:
        raise ValueError("an open cell needs opening_width_m > 0")
    s = sigma_m if isinstance(sigma_m, dict) else {"L": sigma_m, "W": sigma_m, "H": sigma_m, "bottom": sigma_m}
    d, t = ROBOT_HALF_M + robot_fixed_gap_m, FENCE_T_M          # inner face of the ring, fence thickness
    ring = 2 * d + 2 * t
    east_w = ring - (0.0 if enclosed else opening_width_m)
    objects = [
        box("robot", "robot", 0.0, 0.0, 2 * ROBOT_HALF_M, 2 * ROBOT_HALF_M, 0.0, ROBOT_TOP_M, s),
        box("fence_e", "fence", d + t / 2, (east_w - ring) / 2, t, east_w, floor_gap_m, fence_height_m, s, "east fence"),
        box("fence_w", "fence", -d - t / 2, 0.0, t, ring, floor_gap_m, fence_height_m, s, "west fence"),
        box("fence_n", "fence", 0.0, d + t / 2, 2 * d, t, floor_gap_m, fence_height_m, s, "north fence"),
        box("fence_s", "fence", 0.0, -d - t / 2, 2 * d, t, floor_gap_m, fence_height_m, s, "south fence"),
        box("bollard_1", "bollard", d + t + 1.5, 1.0, 0.1, 0.1, 0.0, 0.9, s),
        box("bollard_2", "bollard", d + t + 1.5, -1.0, 0.1, 0.1, 0.0, 0.9, s),
        box("lc", "light_curtain", d + t + 1.0, 0.0, 0.05, 0.05, lc_bottom_m, lc_bottom_m + 1.8, s, "light curtain"),
        box("cart", "cart", -d - t - 2.0, 0.0, 0.8, 1.2, 0.0, 1.0, s),
    ]
    cov = None
    if coverage == "full":
        frame = grid_frame(objects)
        nx, ny = frame["shape"]
        cov = Coverage(cell_m=frame["cell_m"], origin_xy=frame["origin_xy"], basis=frame["basis"], shape=frame["shape"],
                       observed=[[ix, iy] for ix in range(nx) for iy in range(ny)], method="synthetic: every cell observed")
    sid = scene_id or (f"synth-cell-h{fence_height_m:g}-g{floor_gap_m:g}-d{robot_fixed_gap_m:g}-lc{lc_bottom_m:g}-"
                       f"{'enc' if enclosed else f'open{opening_width_m:g}'}-cov{coverage}")
    return Scene(scene_id=sid, objects=objects, ground_normal=[0.0, 0.0, 1.0], scale_rel_unc=scale_rel_unc, coverage=cov,
                 declared_inputs=dict(declared_inputs or {}), producer="synth.scenes@1")
