% Trial rule pack v0 (2026-10-07). Integer millimetres. Facts come from engine.py:
%   obj(ID, Class).  bottom(ID, V, U).  top(ID, V, U).  dist(A, B, D, U).      V/D = measured, U = expanded uncertainty (k=2)
%   cell(C).  adj(C1, C2).  blocked(C).  outside(C).  hazard(C).                plan-view occupancy grid (relations.py)
% Thresholds are the numbers of docs/research/verdict-layer-rules-2026-10-07.md section C; verify against the purchased texts
% before any production use (several are vendor reproductions).

thr(fence_height, 1400).     % ISO 13857:2019 Table 2 note: structures under 1400 mm not used without additional measures
thr(floor_gap, 180).         % ISO 13857:2019 4.4: gap under a guard <= 180 mm (slot)
thr(lc_lowest_beam, 300).    % ISO 13855 (2010 numbers): lowest beam / field height; above 300 mm crawl-under must be assessed
thr(crush_gap, 500).         % ISO 13854:2017 Table 1 body 500 mm; ISO 10218-2 / R15.06 trapping clearance 500 mm (clause unverified)

dir(fence_height, ge).  dir(floor_gap, le).  dir(lc_lowest_beam, le).  dir(crush_gap, ge).

fixed(fence).  fixed(guard).  fixed(bollard).  fixed(light_curtain).

subject(fence_height, F)   :- obj(F, fence).
subject(floor_gap, F)      :- obj(F, fence).
subject(floor_gap, F)      :- obj(F, guard).
subject(lc_lowest_beam, L) :- obj(L, light_curtain).
subject(crush_gap, (R, X)) :- obj(R, robot), obj(X, C), fixed(C).

meas(fence_height, F, V, U)   :- subject(fence_height, F), top(F, V, U).
meas(floor_gap, F, V, U)      :- subject(floor_gap, F), bottom(F, V, U).
meas(lc_lowest_beam, L, V, U) :- subject(lc_lowest_beam, L), bottom(L, V, U).
meas(crush_gap, (R, X), D, U) :- subject(crush_gap, (R, X)), dist(R, X, D, U).

% Guard-banded decision (ILAC-G8 / ISO 14253-1 style): a side is taken only when the whole interval V +- U is on it.
status(R, S, pass)       :- meas(R, S, V, U), thr(R, T), dir(R, ge), V - U >= T.
status(R, S, fail)       :- meas(R, S, V, U), thr(R, T), dir(R, ge), V + U <  T.
status(R, S, needs_meas) :- meas(R, S, V, U), thr(R, T), dir(R, ge), V - U <  T, V + U >= T.
status(R, S, pass)       :- meas(R, S, V, U), thr(R, T), dir(R, le), V + U <= T.
status(R, S, fail)       :- meas(R, S, V, U), thr(R, T), dir(R, le), V - U >  T.
status(R, S, needs_meas) :- meas(R, S, V, U), thr(R, T), dir(R, le), V + U >  T, V - U <= T.
status(R, S, cannot_determine) :- subject(R, S), not meas(R, S, _, _).
margin(R, S, V - T) :- meas(R, S, V, _), thr(R, T), dir(R, ge).
margin(R, S, T - V) :- meas(R, S, V, _), thr(R, T), dir(R, le).

% Enclosure (topology): the outside reaches a hazard cell through cells no blocker covers.
reach(C)  :- outside(C), not blocked(C).
reach(C2) :- reach(C1), adj(C1, C2), not blocked(C2).
breach(C) :- hazard(C), reach(C).
opening(C) :- reach(C), adj(C, C2), hazard(C2), not hazard(C).
status(enclosure, hazard_zone, open)             :- breach(_).
status(enclosure, hazard_zone, pass)             :- hazard(_), not breach(_).
status(enclosure, hazard_zone, cannot_determine) :- not hazard(_).

#show status/3.
#show margin/3.
#show opening/1.
