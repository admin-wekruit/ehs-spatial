"""CPU check of scripts/workcell_guard_plates.py: the face rules on synthetic boards, and with --run a finished run:
every installed plate set beat (or matched within MARGIN) the RecGen part on the board's own masks.

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts python scripts/check_workcell_guard_plates.py [--run RUN ...]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import workcell_guard_plates as plates


def _sheet(origin, u, v, size, count, rng, noise=.0):
    a, b = rng.uniform(0, size[0], count), rng.uniform(0, size[1], count)
    normal = np.cross(u, v)
    return origin + np.outer(a, u) + np.outer(b, v) + np.outer(rng.normal(0, noise, count), normal)


def check_rules():
    rng = np.random.default_rng(0)
    up, x = np.array([0, 0, 1.]), np.array([1., 0, 0])
    # A board folded by 40 degrees along a vertical line at x = 0: two faces, the fold recovered, each face clipped at it.
    bent = np.array([np.cos(np.radians(40)), np.sin(np.radians(40)), 0])
    folded = np.r_[_sheet(np.zeros(3), x, up, (1., .8), 1500, rng, .004), _sheet(np.zeros(3), bent, up, (.7, .8), 1000, rng, .004)]
    faces = plates.fit_faces(folded, .02)
    assert len(faces) == 2, len(faces)
    angle = np.degrees(np.arccos(abs(faces[0][0] @ faces[1][0])))
    assert abs(angle - 40) < 2, angle
    for k in (0, 1):
        polygon = plates.face_polygon(faces[k], faces[1 - k])
        assert len(polygon) >= 3
        other_normal, other_centre, _ = faces[1 - k]
        side = np.sign(np.median((faces[k][2] - other_centre) @ other_normal))
        assert (side * ((polygon - other_centre) @ other_normal) >= -1e-6).all(), 'each face ends at the fold line'
    # A "fold" whose second face only one photo supports is the photos disagreeing about the tilt, not a bend.
    tags = np.r_[np.full(1500, 1), np.full(900, 2), np.full(100, 1)]  # the second face: 10 % from photo 1, under SHARED
    assert len(plates.fit_faces(folded, .02, photos=tags)) == 1, 'second face from one photo only'
    mixed = rng.permutation(tags)
    assert len(plates.fit_faces(folded, .02, photos=mixed)) == 2, 'a fold both photos see stays a fold'
    # Two parallel layers 3 cm apart (two photos' depth disagreement) are one sheet, not a fold.
    layers = np.r_[_sheet(np.zeros(3), x, up, (1., .8), 1200, rng), _sheet(np.array([0, .03, 0]), x, up, (1., .8), 1200, rng)]
    assert len(plates.fit_faces(layers, .05)) == 1
    # A small tilted patch (clutter caught in the mask) is not a second face: a fold needs SECOND of the points.
    tilted = np.array([np.cos(np.radians(60)), np.sin(np.radians(60)), 0])
    cluttered = np.r_[_sheet(np.zeros(3), x, up, (1., .8), 2000, rng, .004), _sheet(np.array([.2, 0, .2]), tilted, up, (.15, .15), 80, rng)]
    assert len(plates.fit_faces(cluttered, .02)) == 1
    # A strip (near-line support) faces its camera: the plane contains the strip and turns toward the camera.
    strip = _sheet(np.array([0, 0, 1.]), x, np.array([0, .3, .95]) / np.linalg.norm([0, .3, .95]), (2., .04), 600, rng, .01)
    camera = np.array([1., -5., 1.5])
    (normal, centre, _), = plates.fit_faces(strip, .05, camera)
    assert abs(normal @ x) < .02, 'the strip keeps its length in its plane (the length axis comes from noisy points)'
    view = (camera - centre) / np.linalg.norm(camera - centre)
    assert abs(normal @ view) > .99 * np.linalg.norm(view - (view @ x) * x), 'faces the camera as far as its length allows'
    # One bent sheet: a centre band and two wings at 30 degrees; folds shared, one bottom and top edge, wings their own width.
    rngs = np.random.default_rng(1)
    centre_pts = _sheet(np.array([-1., 0, .2]), x, up, (2., .6), 800, rngs)
    lw, rw = np.array([-np.cos(np.radians(30)), -np.sin(np.radians(30)), 0]), np.array([np.cos(np.radians(30)), -np.sin(np.radians(30)), 0])
    left_pts, right_pts = _sheet(np.array([-1., 0, .2]), lw, up, (.7, .6), 500, rngs), _sheet(np.array([1., 0, .2]), rw, up, (.7, .6), 500, rngs)
    plane = lambda pts: (np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][2], pts.mean(0))
    sheet = plates.assemble({'left': plane(left_pts), 'center': plane(centre_pts), 'right': plane(right_pts)},
                            {'left': left_pts, 'center': centre_pts, 'right': right_pts}, up, 0.)
    c, l, r = sheet['center'], sheet['left'], sheet['right']
    assert np.allclose(c[0], l[0]) and np.allclose(c[3], l[3]) and np.allclose(c[1], r[0]) and np.allclose(c[2], r[3]), 'wings share the folds'
    assert np.allclose(c[0], [-1, 0, .2 + .6 * .02], atol=.02) and np.allclose(c[2], [1, 0, .2 + .6 * .98], atol=.02), 'folds at the true edges'
    assert abs(np.linalg.norm(l[1] - l[0]) - .7 * .98) < .03, 'a wing is as wide as its support'
    # A bridge closes the gap between two plates without moving them.
    p1 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1.]])
    p2 = p1 + [1.1, .05, 0]
    tris = plates.bridge(p1, p2)
    assert len(tris) == 2 and plates.gap(p1, np.concatenate(tris)) < 1e-9 and plates.gap(p2, np.concatenate(tris)) < 1e-9
    assert plates.bridge(p1, p1 + [1, 0, 0]) == [], 'touching plates need no bridge'
    # Sutherland-Hodgman keeps exactly the half-plane.
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1.]])
    half = plates._clip(square, lambda q: .5 - q[0])
    assert np.allclose(sorted(map(tuple, half)), sorted([(0, 0), (.5, 0), (.5, 1), (0, 1)]))


def check_run(root):
    catalog = json.loads((Path(root) / 'objects.json').read_text())
    record = catalog['coverage'].get('guardPlates')
    assert record and record.get('boards') is not None, 'the run records its guard plate decision'
    items = {item['id']: item for item in catalog['objects']}
    sheet = record.get('candidates')
    if sheet and sheet.get('scores'):  # one connected sheet: decided as a whole, all boards or none
        kept = {bool(b.get('kept')) for b in record['boards']}
        assert len(kept) == 1, 'all three boards are replaced together or not at all'
        if kept == {True}:
            assert sheet['meanIouAllBoards'] >= plates.FLOOR_IOU and sheet['meanIouAllBoards'] >= sheet['recgenMeanIouAllBoards'] - plates.MARGIN, sheet
            assert sheet['chosen'] in sheet['scores'] and sheet['scores'][sheet['chosen']] == max(sheet['scores'].values()), sheet
    for board in record['boards']:
        item = items[board['objectId']]
        if board.get('kept'):
            if not (sheet and sheet.get('scores')):
                mean, part = (np.mean(list(board[k].values())) if board[k] else 0. for k in ('iouByPhoto', 'recgenPartIouByPhoto'))
                assert mean >= plates.FLOOR_IOU and mean >= part - plates.MARGIN, board
            assert item['model']['file'] == plates.FILE and item['guardPlates'] == board and item['recgenGuard']['model'], board['objectId']
        else:
            assert item['model']['file'] != plates.FILE, board
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, action='append', default=[])
    args = parser.parse_args()
    check_rules()
    for run in args.run:
        record = check_run(run)
        print(run.name, record['status'], [(b['objectId'], b.get('kept')) for b in record['boards']])
    print('check_workcell_guard_plates passed')
