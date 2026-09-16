from copy import deepcopy

import numpy as np
import pytest

from ehs_spatial.platform.coarse_model import coarse_open_frame, qualify_depth_views
from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.spatial import project_native, unproject_pixels


def views_and_points():
    views, points = [], {}
    for i in range(2):
        camera = np.eye(4); camera[0, 3] = .8*i
        view = {'observationId':str(i), 'observationRevision':1, 'imageId':str(i),
            'coordinateFrameId':'world', 'K':np.array([[80.,0,50],[0,80.,50],[0,0,1]]),
            'cameraToWorld':camera, 'sourceHashes':dict.fromkeys(['image','mask','camera','geometry'], 'a'*64),
            'mask':np.zeros((100,100),bool), 'valid':np.ones((100,100),bool), 'depth':np.full((100,100),4.)}
        # An explicitly slender open frame, with the principal axis along world Y.
        x = 50 - i*16
        view['mask'][30:71,x-2:x+3] = True
        y, xx = np.mgrid[:100,:100]
        points[str(i)] = unproject_pixels(np.stack((xx,y),-1).reshape(-1,2), view['depth'].ravel(), view).reshape(100,100,3)
        views.append(view)
    return views, points


def test_coarse_frame_has_openings_and_keeps_owned_axis_in_both_images():
    views, _ = views_and_points()
    mesh, pose, proof = coarse_open_frame(views,['0','1'],bar_fraction=.25,rung_count=7)
    assert len(mesh.faces) == 9*12 and np.linalg.det(pose[:3,:3]) == pytest.approx(1)
    for view in views:
        xy, z = project_native(pose[None,:3,3],view)
        assert z[0] > 0
        y,x = np.rint(xy[0][::-1]).astype(int)
        assert view['mask'][y,x]
    assert proof['assumptions']['hiddenThicknessAndRungsMeasured'] is False
    assert proof['physicalPlacementConfirmed'] is False


def test_depth_qualification_is_candidate_independent_and_retains_raw_evidence():
    views, points = views_and_points()
    original = deepcopy(views)
    # Background seen through a wire opening has wrong parallax in the other mask.
    views[0]['depth'][32:38,48:53] = 8
    y,x = np.mgrid[:100,:100]
    points['0'] = unproject_pixels(np.stack((x,y),-1).reshape(-1,2),views[0]['depth'].ravel(),views[0]).reshape(100,100,3)
    result, proof = qualify_depth_views(views,['0','1'],points)
    assert not result[0]['valid'][32:38,48:53].any()
    assert result[0]['valid'][40:60,48:53].all()
    assert np.array_equal(result[0]['depth'],views[0]['depth'])
    assert np.array_equal(result[0]['mask'],original[0]['mask'])
    assert views[0]['valid'].all() and result[0]['valid'][0,0]  # Keep non-target occluders.
    assert len(result) == len(views) and proof['perView'][0]['rejectedTargetDepthPixels'] == 30
    assert 'depthQualification' not in views[0]['sourceHashes']
    assert 'depthQualification' in result[0]['sourceHashes']
    assert proof['physicalAccuracyConfirmed'] is False


@pytest.mark.parametrize('error', ['unowned','same_image','same_camera','wrong_frame','empty','no_baseline'])
def test_invalid_frame_evidence_does_not_create_geometry(error):
    views,_ = views_and_points();ids=['0','1']
    if error == 'unowned':ids=['0','other']
    if error == 'same_image':views[1]['imageId']='0'
    if error == 'same_camera':views[1]['cameraToWorld']=views[0]['cameraToWorld'].copy()
    if error == 'wrong_frame':views[1]['coordinateFrameId']='unregistered'
    if error == 'empty':views[1]['mask'][:]=False
    if error == 'no_baseline':views[1]['mask']=views[0]['mask'].copy()
    with pytest.raises(PlatformError):
        coarse_open_frame(views,ids,bar_fraction=.25,rung_count=7)


def test_coarse_pose_fits_position_without_filling_small_openings():
    from ehs_spatial.platform.model_quality import assess_model, refine_model_pose
    from ehs_spatial.platform.spatial import MeshData
    from test_platform_model_quality import scene
    _, view = scene();view['maskComplete']=False
    vertices, faces = [], []
    for column in range(7):
        x = 5.5 + 3*column
        quad = unproject_pixels(np.array([[x,3.5],[x+2,3.5],[x+2,16.5],[x,16.5]]),np.full(4,4.),view)
        n=len(vertices);vertices.extend(quad);faces.extend([[n,n+1,n+2],[n,n+2,n+3]])
    mesh=MeshData(np.array(vertices),np.array(faces));pose=np.eye(4);pose[2,3]=.3
    assert assess_model(mesh,pose,[view],coarse=True)['status']=='observed_inconsistent'
    result=refine_model_pose(mesh,pose,[view],coarse=True,fit_scale=True,max_iterations=300)
    assert result['accepted'] and result['after']['status']=='observed_consistent'
    assert result['after']['perView'][0]['exactCoverage'] < .8
    assert result['after']['perView'][0]['relativeDepthP50'] <= .05
