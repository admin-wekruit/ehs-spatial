"""Run with PYTHONPATH=.:scripts python scripts/check_workcell_edge_support.py."""
import numpy as np

from check_workcell_button_bundle import fixture
from workcell_button_bundle import _project_raw
from workcell_photo_metrology import _line_fit, _match_edges


def main():
    frames, *_ = fixture()
    up = np.array([0., 0., 1.])

    def observation(photo, span):
        xyz = np.array([[x, 0., .6] for x in span])
        raw = _project_raw(xyz, frames[photo])[0]
        canonical = np.c_[raw, np.ones(2)] @ frames[photo]['A'].T
        return {'photo': photo, 'rawEnds': raw.tolist(), 'uv': canonical[:, :2].tolist(), 'score': 1.}

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
    print('PASS: disconnected view rejected; valid pair retained; connected partial visibility retained')


if __name__ == '__main__':
    main()
