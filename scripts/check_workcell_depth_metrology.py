"""Small source-identity and fresh-geometry checks for the depth ablation bridge."""
from copy import deepcopy
from pathlib import Path
import tempfile

from scripts.workcell_depth_metrology import match_fence_plane, match_reference, depth_run_path


def check():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        actual = root/'actual-volume'; (actual/'workcell-depth-resolution/run-a').mkdir(parents=True)
        mount = root/'mount'; mount.symlink_to(actual, target_is_directory=True)
        assert depth_run_path(mount, 'workcell-depth-resolution/run-a').resolve() == actual.resolve()/'workcell-depth-resolution/run-a'
        for invalid in ('../run', '/tmp/escape', 'other/run'):
            try:
                depth_run_path(mount, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('Escaping Volume path accepted')
    edge = [[10., 20.], [40., 21.]]
    old = {'fence': {'beams': [
        {'plane': 0, 'sourcePhoto': photo, 'rawEdges': [edge]}
        for photo in (2, 3)]}}
    new = deepcopy(old)
    for row in new['fence']['beams']:
        row['plane'] = 1
        row['rawEdges'] = [list(reversed(edge))]
    new['fence']['beams'].append({'plane': 0, 'sourcePhoto': 2,
                                 'rawEdges': [[[200., 30.], [400., 32.]]]})
    assert match_fence_plane(old, new)['planeIndex'] == 1
    assert match_fence_plane(old, new)['sourcePhotos'] == [2, 3]
    old['anchor'] = {'views': [{'photo': p, 'boxRaw': [10, 20, 50, 80]} for p in (2, 3, 4)]}
    new['anchor'] = deepcopy(old['anchor'])
    assert match_reference(old, new)['sourcePhotos'] == [2, 3, 4]
    new['anchor']['views'][0]['boxRaw'] = [100, 200, 150, 280]
    try:
        match_reference(old, new)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed physical button identity was accepted')
    for invalid in ('different', 'one-view', 'ambiguous'):
        value = deepcopy(new)
        if invalid == 'different':
            value['fence']['beams'] = value['fence']['beams'][-1:]
        elif invalid == 'one-view':
            value['fence']['beams'] = value['fence']['beams'][:1]
        else:
            for row in deepcopy(value['fence']['beams'][:2]):
                row['plane'] = 2
                value['fence']['beams'].append(row)
        try:
            match_fence_plane(old, value)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Unverified fence identity accepted: {invalid}')
    print('PASS: physical fence identity survives plane reorder; missing/ambiguous identity rejected')


if __name__ == '__main__':
    check()
