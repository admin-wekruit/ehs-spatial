import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "recall_eval.py"
SPEC = importlib.util.spec_from_file_location("recall_eval", SCRIPT)
recall_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recall_eval)


def _mask(h, w, y1, y2, x1, x2):
    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


def test_bbox_scoring_counts_hits_and_false_positives():
    gt = [(10, 10, 30, 30), (60, 60, 80, 80)]
    preds = [
        _mask(100, 100, 10, 30, 10, 30),   # exact hit on gt[0]
        _mask(100, 100, 0, 5, 90, 100),    # no overlap -> false positive
    ]

    score = recall_eval.score_bbox_image(gt, preds)

    assert score == {
        "gt": 2,
        "hits_iou30": 1,
        "hits_iou50": 1,
        "present": 1,
        "false_positive_masks": 1,
    }


def test_bbox_scoring_no_predictions():
    score = recall_eval.score_bbox_image([(0, 0, 10, 10)], [])
    assert score["hits_iou30"] == 0
    assert score["present"] == 0
    assert score["false_positive_masks"] == 0


def test_mask_scoring_union_coverage_and_presence():
    gt = _mask(50, 50, 0, 50, 0, 10)          # 500 px column
    half = _mask(50, 50, 0, 25, 0, 10)        # covers half of gt

    score = recall_eval.score_mask_image(gt, [half])

    assert score["coverage"] == 0.5
    assert score["present"] == 1
    assert 0 < score["iou"] <= 0.5


def test_mask_scoring_below_presence_threshold():
    gt = _mask(50, 50, 0, 50, 0, 10)
    sliver = _mask(50, 50, 0, 2, 0, 10)       # 4% coverage < 0.2

    score = recall_eval.score_mask_image(gt, [sliver])

    assert score["present"] == 0


def test_bbox_iou_basic():
    assert recall_eval._bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert recall_eval._bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
