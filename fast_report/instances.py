"""r4/instances: one real object = one 3D card. Rules on the lift's mask graph (segment.lift calls seam_labels) and on the
objects (cards.build calls relate); numpy only, on the CPU.

Seams (split wrongly merged touching objects): the lift joins masks of different keyframes whose 5 cm voxels overlap, so one
mask over two things ('the workbench' over the vise and the screwdrivers beside it) joins both into one object. SAM 3's own
separate masks for them on other keyframes are the evidence against it: two groups of masks that are separate masks on the
same keyframe (they share < SEP_MAX of the smaller mask's voxels) on >= SEAM_MIN keyframes stay apart, and the mask over
both ('several things') is left out of both.

    python -m fast_report.instances --self-check
"""
import sys

import numpy as np

SEAM_MIN = 2   # keyframes on which two groups are separate masks before they stay apart (1: one SAM 3 split of a thing splits it)
SEP_MAX = .2   # two masks of one keyframe sharing less than this share of the smaller one's voxels are separate things
HOLD = .5      # a link's larger mask 'holds' the smaller one when it has at least this share of the smaller one's voxels


class _Sets:
    def __init__(self, members):
        self.parent = {m: m for m in members}
        self.members = {m: [m] for m in members}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        if len(self.members[a]) < len(self.members[b]):
            a, b = b, a
        self.parent[b] = a
        self.members[a] += self.members.pop(b)
        return a


def seam_labels(n, fr, size, pairs, edges, seam_min=SEAM_MIN, sep_max=SEP_MAX):
    """n masks (keyframe fr, voxel count size). pairs: (a, b, c) arrays, voxels shared by masks a < b (any keyframes, c > 0);
    edges: (a, b, w) the lift's links between keyframes. -> (label per mask 0..k-1, record)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    fr, size = np.asarray(fr), np.asarray(size, float)
    ea, eb, ew = (np.asarray(x) for x in edges)
    ncomp, comp = connected_components(coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(n, n)), directed=False)
    pa, pb, pc = (np.asarray(x) for x in pairs)
    same = (fr[pa] == fr[pb]) & (pc >= sep_max * np.minimum(size[pa], size[pb]))
    together = set(zip(pa[same].tolist(), pb[same].tolist()))
    # cannot-links: separate masks of one keyframe inside one component
    order = np.lexsort((fr, comp))
    cl = {}
    key = comp[order] * (int(fr.max()) + 1 if n else 1) + fr[order]
    for grp in np.split(order, np.flatnonzero(np.diff(key)) + 1):
        if len(grp) < 2:
            continue
        g = sorted(grp.tolist())
        for i, x in enumerate(g):
            for y in g[i + 1:]:
                if (x, y) not in together:
                    cl.setdefault(x, []).append((y, int(fr[x])))
                    cl.setdefault(y, []).append((x, int(fr[x])))
    label = comp.copy()
    rec = {"components_split": 0, "bridges": 0, "refused_links": 0}
    if not cl:
        return label, rec
    suspect = np.unique(comp[list(cl)])
    next_label = ncomp
    cmap = {}
    for i, (a, b, w) in enumerate(zip(ea.tolist(), eb.tolist(), ew.tolist())):
        if comp[a] in suspect:
            cmap.setdefault(int(comp[a]), []).append((-w, i, a, b))
    inter = dict(zip(zip(pa.tolist(), pb.tolist()), pc.tolist()))
    for cid, es in cmap.items():
        mem = np.flatnonzero(comp == cid).tolist()
        sets = _Sets(mem)
        torn = []
        for _, _, a, b in sorted(es):
            ra, rb = sets.find(a), sets.find(b)
            if ra == rb:
                continue
            small = ra if len(sets.members[ra]) <= len(sets.members[rb]) else rb
            other = rb if small == ra else ra
            seams = {f for m in sets.members[small] for m2, f in cl.get(m, ()) if sets.find(m2) == other}
            if len(seams) >= seam_min:
                torn.append((a, b))
                continue
            sets.union(ra, rb)
        if not torn:
            continue
        rec["refused_links"] += len(torn)
        bridge = set()
        for a, b in torn:
            if sets.find(a) == sets.find(b):
                continue  # joined later through other links
            c = inter.get((min(a, b), max(a, b)), 0)
            big, sm = (a, b) if size[a] >= size[b] else (b, a)
            if c >= HOLD * size[sm]:
                bridge.add(big)
        rec["bridges"] += len(bridge)
        # the final groups: links honoured inside a group, bridges out
        keep = [(a, b) for _, _, a, b in es if sets.find(a) == sets.find(b) and a not in bridge and b not in bridge]
        loc = {m: j for j, m in enumerate(mem)}
        k, sub = connected_components(coo_matrix((np.ones(len(keep)), ([loc[a] for a, _ in keep], [loc[b] for _, b in keep])),
                                                 shape=(len(mem), len(mem))), directed=False)
        rec["components_split"] += int(k > 1)
        for m, s in zip(mem, sub):
            label[m] = next_label + s
        next_label += k
    _, label = np.unique(label, return_inverse=True)
    return label, rec


def self_check():
    # two tools X (masks 0, 2, 4) and Y (1, 3, 5) side by side on keyframes 0, 1, 2 (separate masks there), and mask 6 on
    # keyframe 3 over both: plain components join all seven; the seams keep X and Y apart and leave 6 out
    fr = np.array([0, 0, 1, 1, 2, 2, 3])
    size = np.array([10, 10, 10, 10, 10, 10, 20])
    pairs = ([0, 2, 0, 1, 3, 1, 0, 1, 2, 3, 4, 5], [2, 4, 4, 3, 5, 5, 6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9])
    edges = ([0, 2, 1, 3, 0, 1], [2, 4, 3, 5, 6, 6], [.7, .7, .7, .7, .4, .4])
    lab, rec = seam_labels(7, fr, size, pairs, edges)
    assert lab[0] == lab[2] == lab[4] and lab[1] == lab[3] == lab[5] and lab[0] != lab[1], lab
    assert lab[6] not in (lab[0], lab[1]) and rec["bridges"] == 1, (lab, rec)
    # one seam keyframe only (SAM 3 split the thing once): stays one object
    lab, rec = seam_labels(7, fr, size, pairs, edges, seam_min=4)
    assert len(set(lab.tolist())) == 1 and rec["bridges"] == 0, lab
    # a part inside a whole on one keyframe shares voxels: not a seam
    lab, _ = seam_labels(3, np.array([0, 0, 1]), np.array([10, 40, 40]), ([0, 0, 1], [1, 2, 2], [9, 9, 35]), ([0, 1], [2, 2], [.2, .8]))
    assert len(set(lab.tolist())) == 1, lab
    try:
        import torch
    except ImportError:
        print("instances self-check ok: seam split with a bridge left out, one seam keyframe kept whole, part inside whole not a seam "
              "(projection links and the join guard skipped: no torch here)")
        return
    from fast_report import segment
    # projection links: a 6 x 6 px tool 4 m away on two keyframes 10 cm apart; its two masks' points (depth off by 20 cm between the
    # keyframes: their 5 cm voxels never meet) still land on each other; a second tool elsewhere on keyframe 1 does not link
    H, W, fx = 280, 504, 262.
    K = torch.tensor([[fx, 0, 252], [0, fx, 140], [0, 0, 1]]).repeat(2, 1, 1)
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[1, 0, 3] = .1
    depth = torch.full((2, H, W), 8.)
    depth[0, 100:106, 200:206], depth[1, 100:106, 193:199] = 4., 4.2
    m = torch.zeros((3, H, W), dtype=torch.bool)
    m[0, 100:106, 200:206] = True
    m[1, 100:106, 193:199] = True
    m[2, 150:156, 300:306] = True
    ys, xs = torch.meshgrid(torch.arange(100, 106, 2).float(), torch.arange(200, 206, 2).float(), indexing="ij")
    pts0 = torch.stack([(xs.flatten() + .5 - 252) / fx * 4., (ys.flatten() + .5 - 140) / fx * 4., torch.full((9,), 4.)], 1)
    xs1 = xs - 7
    pts1 = torch.stack([(xs1.flatten() + .5 - 252) / fx * 4.2 + .1, (ys.flatten() + .5 - 140) / fx * 4.2, torch.full((9,), 4.2)], 1)
    pts2 = pts1 + torch.tensor([1., 1., 0])
    world, mid = torch.cat([pts0, pts1, pts2]), torch.tensor([0] * 9 + [1] * 9 + [2] * 9)
    a, b, w = segment.reproject_links(m[:, ::2, ::2], world, mid, torch.tensor([0, 1, 1]), depth, K, c2w)
    assert a.tolist() == [0] and b.tolist() == [1] and float(w[0]) > .2, (a, b, w)
    # join guard: a densify mask half in object 0 and half in object 1 joins neither; one mostly in object 0 joins it
    codes = torch.cat([segment.voxel_codes(torch.tensor([[.025 + .05 * i, 0., 0.] for i in range(8)])),
                       segment.voxel_codes(torch.tensor([[1.025 + .05 * i, 0., 0.] for i in range(8)]))])
    order = torch.argsort(codes)
    pts = torch.tensor([[.025 + .05 * i, 0., 0.] for i in range(6)] + [[1.025 + .05 * i, 0., 0.] for i in range(4)]
                       + [[.025 + .05 * i, 0., 0.] for i in range(5)])
    who, _ = segment.join({"lifted": torch.arange(2), "mid": torch.tensor([0] * 10 + [1] * 5), "world": pts}, codes[order],
                          torch.tensor([0] * 8 + [1] * 8)[order])
    assert who.tolist() == [-1, 0], who
    print("instances self-check ok: seam split with a bridge left out, one seam keyframe kept whole, part inside whole not a seam, "
          "projection links through a 20 cm depth error, the densify join refuses a mask over two objects")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
