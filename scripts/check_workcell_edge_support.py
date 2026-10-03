"""Run with PYTHONPATH=.:scripts python scripts/check_workcell_edge_support.py."""
import numpy as np

from check_workcell_button_bundle import fixture
from workcell_button_bundle import _project_raw
from workcell_photo_metrology import _line_fit, _match_edges


def main():
    frames, *_ = fixture()
    up = np.array([0., 0., 1.])

    def observation(photo, span, face_y=0.):
        xyz = np.array([[x, face_y, .6] for x in span])
        raw = _project_raw(xyz, frames[photo])[0]
        canonical = np.c_[raw, np.ones(2)] @ frames[photo]['A'].T
        return {'photo': photo, 'rawEnds': raw.tolist(), 'uv': canonical[:, :2].tolist(), 'score': 1., 'planeIndex': 0}

    rows = [observation(1, [-.1, .1]), observation(2, [-.1, .1]), observation(3, [1., 1.2])]
    try:
        _line_fit(rows, frames, up)
    except ValueError as error:
        assert 'common visible segment' in str(error), str(error)
    else:
        raise AssertionError('A disconnected third segment was counted as evidence for the first two views')
    edge = _match_edges(rows, frames, up)
    assert edge['sourcePhotos'] == [1, 2], edge
    assert edge['maxReprojectionErrorRawPx'] < 1e-8
    chain = [observation(1, [-.1, .1]), observation(2, [0., .2]), observation(3, [.15, .3])]
    joined = _line_fit(chain, frames, up)
    assert joined['sourcePhotos'] == [1, 2, 3]
    assert joined['maxReprojectionErrorRawPx'] < 1e-8
    weak = [{**row, 'identityEvidence': {'independentlySupported': False}} for row in rows]
    try:
        _match_edges(weak, frames, up, require_identity_anchor=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Two unidentified borders established a physical lower rail')
    anchored = [{**weak[0], 'identityEvidence': {'independentlySupported': True}}, *weak[1:]]
    transferred = _match_edges(anchored, frames, up, require_identity_anchor=True)
    assert transferred['sourcePhotos'] == [1, 2]
    assert transferred['identityAnchorPhotos'] == [1]
    assert transferred['physicalPromotionAllowed'] is False
    assert transferred['identityStatus'] == 'conditional_same_face_unverified'

    # Both real edges have height .6, but belong to different parallel faces.
    # Their viewing planes intersect at height .4171 with zero image error and
    # a finite shared span. A strong part label in view 1 cannot disambiguate it.
    wrong_face = [
        {**observation(1, [-1., 1.]), 'identityEvidence': {'independentlySupported': True}},
        {**observation(2, [-1., 1.], .3), 'identityEvidence': {'independentlySupported': False}},
    ]
    source_plane = {'index': 0, 'normal': [0., 1., 0.], 'offset': 0., 'residualP95Native': .01}
    wrong = _match_edges(wrong_face, frames, up, require_identity_anchor=True, fence_plane=source_plane)
    assert wrong['maxReprojectionErrorRawPx'] < 1e-8
    assert abs(np.asarray(wrong['pointNative']) @ up - .6) > .18
    assert wrong['physicalPromotionAllowed'] is False
    assert wrong['fencePlaneAssociation']['withinSourceScatter'] is False
    assert wrong['fencePlaneAssociation']['maxDistanceNative'] > .7
    # The saved depth plane is diagnostic, never perfect metric ground truth:
    # changing it must not move the transparent experimental RGB line.
    shifted = _match_edges(wrong_face, frames, up, require_identity_anchor=True,
                           fence_plane={**source_plane, 'offset': -.7620320855614973})
    assert np.allclose(shifted['pointNative'], wrong['pointNative'])
    assert shifted['fencePlaneAssociation']['withinSourceScatter'] is True
    assert shifted['physicalPromotionAllowed'] is False
    no_scatter = _match_edges(anchored, frames, up, require_identity_anchor=True,
                              fence_plane={k: v for k, v in source_plane.items() if k != 'residualP95Native'})
    assert no_scatter['fencePlaneAssociation']['withinSourceScatter'] is None
    assert no_scatter['physicalPromotionAllowed'] is False
    try:
        _match_edges(wrong_face, frames, up, require_identity_anchor=True,
                     fence_plane={**source_plane, 'index': 1})
    except ValueError:
        pass
    else:
        raise AssertionError('Candidates from another current fence instance were mixed into this fit')
    print('PASS: finite support and part anchor checks; zero-error wrong-face height cannot be physically promoted; saved plane is diagnostic only')


if __name__ == '__main__':
    main()
