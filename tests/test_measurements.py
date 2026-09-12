"""A physical frame rotation must not alter dimensions or floor-relative angles."""
import numpy as np
from ehs_spatial.measurements import measure_observed_points, measurement_scale, _rotation_to_positive_z


def test_visible_dimensions_orientation_and_scale_provenance():
    up = np.array([0., -1., 0.])
    basis = _rotation_to_positive_z(up)
    scene = {'floor_plane': [*up, 20.], 'scale_source':'moge_anchor', 'scale_factor':2.}
    grid = np.array(np.meshgrid(np.linspace(-2,2,9), np.linspace(-1,1,7), np.linspace(4,7,5))).reshape(3,-1).T
    points = grid @ basis
    measured = measure_observed_points(points, scene, mask_pixels=len(points), source={'scene_sha256':'a'*64})
    assert measured['status']=='available'
    assert measured['dimensions_native']=={'height':3.,'width':4.,'depth':2.}
    corners=np.array(measured['basis']['corners_native'])
    assert np.allclose(np.linalg.norm(corners[[1,2,4]]-corners[0],axis=1),[4,2,3])
    assert measured['scale']['status']=='estimated' and measured['scale']['m_per_native']==2
    assert measured['orientation']['planar_slope']['status']=='unavailable'
    # An observed plane with 30-degree slope, independently rotated into the native frame.
    x,y=np.meshgrid(np.linspace(-2,2,31),np.linspace(-2,2,31))
    tilted=np.column_stack([x.ravel(),y.ravel(),(x*np.tan(np.pi/6)).ravel()]) @ basis
    fit=measure_observed_points(tilted,scene,mask_pixels=len(tilted))
    assert fit['orientation']['planar_slope']['status']=='available'
    assert np.isclose(fit['orientation']['planar_slope']['value_deg'],30.)
    # This line's dominant visible axis is 25 degrees from up, not global native Z.
    direction=np.array([np.sin(np.deg2rad(25)),0,np.cos(np.deg2rad(25))]) @ basis
    line=np.linspace(-2,2,101)[:,None]*direction
    fit=measure_observed_points(line,scene,mask_pixels=len(line))
    assert np.isclose(fit['orientation']['principal_axis_tilt']['value_deg'],25.)
    assert fit['orientation']['planar_slope']['status']=='unavailable'
    cube=np.array(np.meshgrid(*([np.linspace(-1,1,9)]*3))).reshape(3,-1).T
    fit=measure_observed_points(cube,scene,mask_pixels=len(cube))
    assert all(x['status']=='unavailable' for x in fit['orientation'].values())
    for bad, record in [(points[:7],scene),(points,{}),(np.ones((20,3)),scene)]:
        fit=measure_observed_points(bad,record,mask_pixels=len(bad))
        assert fit['status']=='unavailable' and fit['dimensions_native'] is None and fit['reason']
    legacy={'scale_source':'camera_height','scale_factor':1.5}
    assert measurement_scale(legacy)['m_per_native'] is None
    assert measurement_scale(legacy,{'provided':False,'camera_height_m':1.5})['status']=='uncalibrated'
    anchored=measurement_scale(legacy,{'provided':True,'camera_height_m':1.7,'source_sha256':'b'*64})
    assert anchored['status']=='operator_anchored' and anchored['calibration_source_sha256']=='b'*64
