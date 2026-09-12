"""Deterministic measurements of visible depth support; never complete-object truth.

Dimensions enclose all validated support points in the existing fitted-floor
basis. Orientation uses the central 95% by distance from the coordinate median
to limit isolated depth tails, with at least 20 fit points. A principal axis
requires largest/middle covariance eigenvalue >= 1.5 (the existing spatial-state
gate). A plane additionally requires middle/largest >= .05, smallest/middle <=
.02, and p95 plane residual / middle-axis span <= .05. These are evidence gates,
not calibrated probabilities or claims of a physical object's upright axis.
"""
from __future__ import annotations

import numpy as np


def _rotation_to_positive_z(normal: np.ndarray) -> np.ndarray:
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(normal, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(normal, target))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]],
         [-cross[1], cross[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value > 0


def measurement_scale(scene: dict, calibration: dict | None = None, *, scene_sha256=None) -> dict:
    source, factor = scene.get('scale_source'), scene.get('scale_factor')
    result = {'status':'uncalibrated', 'source':source, 'm_per_native':None,
              'source_sha256':scene_sha256, 'reason':'No validated metric scale source is recorded'}
    if not _positive(factor):
        return result
    if source == 'moge_anchor':
        result.update(status='estimated', m_per_native=float(factor),
                      reason='Model-estimated scale; not a surveyed or operator-calibrated dimension')
    elif source == 'camera_height' and calibration and calibration.get('provided') is True and _positive(calibration.get('camera_height_m')):
        result.update(status='operator_anchored', m_per_native=float(factor), reason=None,
                      camera_height_m=float(calibration['camera_height_m']),
                      calibration_source_sha256=calibration.get('source_sha256'))
    elif source == 'camera_height':
        result['reason']='Recorded camera-height scale has no evidence that a measured height was explicitly supplied'
    return result


def measure_observed_points(points_native, scene_record: dict, *, mask_pixels: int,
                            source: dict | None = None, calibration: dict | None = None) -> dict:
    """Return source-only measurements; callers supply already validated mask points."""
    source = dict(source or {})
    result = {'version':1, 'status':'unavailable', 'coverage':'observed_partial',
              'basis':None, 'dimensions_native':None,
              'scale':measurement_scale(scene_record,calibration,scene_sha256=source.get('scene_sha256')),
              'orientation':{name:{'status':'unavailable','value_deg':None,'reason':'No supported floor-relative fit',
                                  'method':'covariance of central 95% visible native support'}
                             for name in ['planar_slope','principal_axis_tilt']},
              'quality':{'supported_points':0,'mask_pixels':int(mask_pixels),'valid_fraction':0.0,
                         'fit_points':0,'eigenvalues':None,'plane_residual_p95_native':None,
                         'planarity_ratio':None,'elongation_ratio':None,
                         'warnings':['Visible surface only; hidden surfaces and complete-object dimensions are unknown']},
              'reason':None, 'source':source}
    points = np.asarray(points_native, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        result['reason']='Native support must be finite XYZ points';return result
    quality=result['quality'];quality['supported_points']=len(points)
    quality['valid_fraction']=len(points)/mask_pixels if mask_pixels else 0.0
    if len(points)<8:
        result['reason']='Fewer than eight independent valid depth samples';return result
    plane=np.asarray(scene_record.get('floor_plane'),dtype=float)
    if plane.shape!=(4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3])<1e-9:
        result['reason']='No valid saved native floor reference';return result
    up=plane[:3]/np.linalg.norm(plane[:3]);basis=_rotation_to_positive_z(up)
    projected=points@basis.T;low=projected.min(0);high=projected.max(0);extent=high-low
    if not np.any(extent > 1e-12):
        result['reason']='Observed support has no spatial extent';return result
    corners=np.array([[high[k] if (i>>k)&1 else low[k] for k in range(3)] for i in range(8)])@basis
    result.update(status='available',dimensions_native={'height':float(extent[2]),'width':float(extent[0]),'depth':float(extent[1])},
                  basis={'kind':'floor_aligned_native','axes_native':basis.tolist(),'corners_native':corners.tolist()},reason=None)
    quality['warnings'].extend(scene_record.get('warnings') or [])
    radii=np.linalg.norm(points-np.median(points,axis=0),axis=1)
    core=points[radii<=np.quantile(radii,.95)];quality['fit_points']=len(core)
    if len(core)<20:
        for value in result['orientation'].values():value['reason']='Fewer than twenty fit points for orientation'
        return result
    centered=core-core.mean(0)
    eigenvalues,eigenvectors=np.linalg.eigh(centered.T@centered/max(1,len(core)-1))
    eigenvalues=np.maximum(eigenvalues,0);quality['eigenvalues']=eigenvalues.tolist()
    if eigenvalues[2]<=1e-16:
        for value in result['orientation'].values():value['reason']='Observed support has no spatial extent'
        return result
    elongation=float(eigenvalues[2]/eigenvalues[1]) if eigenvalues[1]>1e-16 else None
    quality['elongation_ratio']=elongation
    axis=eigenvectors[:,2]
    if np.dot(axis,up)<0:axis=-axis
    tilt=result['orientation']['principal_axis_tilt']
    tilt.update(reference='saved_floor_normal',meaning='Dominant visible-support axis angle to vertical; physical object upright is not established')
    if elongation is None or elongation>=1.5:
        tilt.update(status='available',value_deg=float(np.degrees(np.arccos(np.clip(abs(axis@up),0,1)))),axis_native=axis.tolist(),reason=None)
    else:tilt['reason']='No dominant visible-support axis (largest/middle eigenvalue ratio < 1.5)'
    normal=eigenvectors[:,0]
    if normal@up<0:normal=-normal
    residual=float(np.quantile(np.abs(centered@normal),.95));middle_span=float(np.ptp(core@eigenvectors[:,1]))
    planarity=float(eigenvalues[0]/eigenvalues[1]) if eigenvalues[1]>1e-16 else None
    quality.update(plane_residual_p95_native=residual,planarity_ratio=planarity,
                   plane_residual_ratio=residual/middle_span if middle_span>1e-12 else None)
    slope=result['orientation']['planar_slope']
    slope.update(reference='saved_floor_plane',meaning='Fitted visible surface angle to the floor; not complete-object tilt')
    if eigenvalues[1]/eigenvalues[2]<.05 or planarity is None:
        slope['reason']='Support is line-like, so no stable two-dimensional plane is established'
    elif planarity>.02 or middle_span<=1e-12 or residual/middle_span>.05:
        slope['reason']='Visible support does not pass the planar residual and eigenvalue gates'
    else:slope.update(status='available',value_deg=float(np.degrees(np.arccos(np.clip(abs(normal@up),0,1)))),normal_native=normal.tolist(),reason=None)
    return result
