"""Run: python tests/check_video_object_model.py (uses installed Open3D)."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_lingbot_object_model import evaluate
from ehs_spatial.platform.recgen import source_grid_crop


def check():
    depth = np.full((40, 40), 2, np.float32)
    mask = np.ones(depth.shape, bool)
    k = np.array([[40., 0, 20], [0, 40, 20], [0, 0, 1]])
    camera_vertices = np.array([[-1,-1,2], [1,-1,2], [1,1,2], [-1,1,2.]])
    faces = np.array([[0,1,2], [0,2,3]])
    c2w = np.array([[0.,0,1,3], [0,1,0,-2], [-1,0,0,1], [0,0,0,1]])
    world = camera_vertices @ c2w[:3,:3].T + c2w[:3,3]
    geometry = (depth, np.full(depth.shape, 2), mask, mask, k, c2w)
    assert evaluate(world, faces, *geometry)['accepted_source_consistency']
    assert not evaluate(world, faces, *geometry[:-1], np.eye(4))['accepted_source_consistency']
    wrong_depth = (camera_vertices*1.3) @ c2w[:3,:3].T + c2w[:3,3]
    assert not evaluate(wrong_depth, faces, *geometry)['accepted_source_consistency']
    empty = list(geometry); empty[3] = np.zeros(depth.shape, bool)
    assert not evaluate(world, faces, *empty)['accepted_source_consistency']
    scaled = list(geometry); scaled[0] = depth*3; scaled[-1] = c2w.copy(); scaled[-1][:3,3] *= 3
    assert evaluate(world*3, faces, *scaled)['accepted_source_consistency']
    # Original-pixel cropping preserves the same camera ray under resize + crop.
    rgb = np.zeros((80,80,3), np.uint8); original_mask = np.zeros((80,80), bool); original_mask[30:60,30:60] = True
    mapping = np.array([[.5,0,-.25], [0,.5,-.25], [0,0,1.]])
    crop = source_grid_crop(rgb, original_mask, depth, k, mapping)
    point = np.array([45.,45,1]); translated = np.asarray(crop['pixelMapping']['matrix']) @ point
    assert np.allclose(np.linalg.inv(crop['K']) @ translated, np.linalg.inv(k) @ mapping @ point)
    print('Video object source rays, world pose, depth rejection and scale invariance: passed')


if __name__ == '__main__': check()
