"""Pages geometry helpers reused without numerical changes."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

def floor_transform(floor):
    """NumPy equivalent of Blender's shortest rotation from floor normal to +Z."""
    plane = np.asarray(floor['plane_native'], dtype=float)
    plane /= np.linalg.norm(plane[:3])
    normal, target = plane[:3], np.array([0., 0., 1.])
    cross = np.cross(normal, target)
    cosine = float(normal @ target)
    if cosine < -1 + 1e-12:
        rotation = np.diag([1., -1., -1.])
    else:
        x, y, z = cross
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + skew + skew @ skew / (1 + cosine)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = rotation @ (plane[3] * normal)
    assert np.allclose(rotation @ normal, target, atol=1e-8)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    return transform

def pose(parts):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('xyz', parts['rotation_deg'], degrees=True).as_matrix() @ np.diag(parts['scale'])
    result[:3, 3] = parts['position']
    return result

def contours(mask):
    assert mask.ndim == 2 and mask.dtype == bool and mask.any(), 'Empty or invalid canonical mask'
    # RETR_LIST retains inner rings; the report must use SVG fill-rule="evenodd".
    rings, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for ring in rings:
        polygon = cv2.approxPolyDP(ring, .5, True).reshape(-1, 2)
        if len(polygon) >= 3 and cv2.contourArea(polygon) > 0:
            polygons.append(polygon.tolist())
    assert polygons, 'No non-degenerate visible mask contour'
    yy, xx = np.nonzero(mask)
    return polygons, [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]

def metrics(comparison):
    def row(view):
        before, after = view['generated_initial'], view['generated_refined']
        return {'frame_id': view['frame_id'], 'before_iou': before['visible_iou'],
                'after_iou': after['visible_iou'], 'before_depth': before['relative_depth_p50'],
                'after_depth': after['relative_depth_p50'], 'before_boundary': before['boundary_error_image_height'],
                'after_boundary': after['boundary_error_image_height']}
    rows = [row(view) for view in comparison['views']] if comparison else []
    result = {}
    for key in ['before_iou', 'after_iou', 'before_depth', 'after_depth']:
        values = [r[key] for r in rows]
        # Do not silently omit a failed view with no depth overlap from its mean.
        result[key] = float(np.mean(values)) if values and all(v is not None for v in values) else None
    return dict(result, views=rows)
