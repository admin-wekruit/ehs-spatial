"""Small source-identity and fresh-geometry checks for the depth ablation bridge."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import scripts.workcell_depth_metrology as bridge
from scripts.workcell_depth_metrology import match_fence_plane, match_reference, depth_run_path, refresh_catalog


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
    item = {'id': 'fence-0'}
    old = {'fence': {'planes': [{}, {}], 'beams': [
        {'plane': 0, 'sourcePhoto': photo, 'rawEdges': [edge]}
        for photo in (2, 3)]}}
    new = deepcopy(old)
    for row in new['fence']['beams']:
        row['plane'] = 1
        row['rawEdges'] = [list(reversed(edge))]
    new['fence']['beams'].append({'plane': 0, 'sourcePhoto': 2,
                                 'rawEdges': [[[200., 30.], [400., 32.]]]})
    assert match_fence_plane(old, new, item)['planeIndex'] == 1
    assert match_fence_plane(old, new, item)['sourcePhotos'] == [2, 3]
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
    for invalid in ('different', 'one-view', 'ambiguous', 'absent-plane'):
        value = deepcopy(new)
        if invalid == 'different':
            value['fence']['beams'] = value['fence']['beams'][-1:]
        elif invalid == 'one-view':
            value['fence']['beams'] = value['fence']['beams'][:1]
        elif invalid == 'ambiguous':
            for row in deepcopy(value['fence']['beams'][:2]):
                row['plane'] = 2
                value['fence']['beams'].append(row)
        else:
            value['fence']['planes'] = [{}]
        try:
            match_fence_plane(old, value, item)
        except ValueError as error:
            assert 'fence-0' in str(error), str(error)
        else:
            raise AssertionError(f'Unverified fence identity accepted: {invalid}')

    # Exercise the complete catalog caller: two physical sections swap plane
    # order in a new depth world, retaining all four targets and their own masks.
    with tempfile.TemporaryDirectory() as directory:
        baseline, fresh = Path(directory)/'baseline', Path(directory)/'fresh'
        baseline.mkdir(); fresh.mkdir()
        old['floor'] = {'normal': [0., 0., 1.], 'offset': 0., 'residualP95Native': .001}
        old['fence']['planes'] = [{'normal': [0., 1., 0.], 'offset': -depth} for depth in (0., 1.)]
        old['fence']['beams'].extend({'plane': 1, 'sourcePhoto': photo,
            'rawEdges': [[[200., 30.], [400., 32.]]]} for photo in (2, 3))
        catalog = {'objects': [{'id': ident, 'label': ident, 'kind': 'safety fence' if ident.startswith('fence-') else 'yellow safety post',
            'observations': [{'photo': photo, 'source': 'SAM instance 0'} for photo in (2, 3)]}
            for ident in bridge.TARGETS]}
        (baseline/'geometry.json').write_text(json.dumps(old))
        (baseline/'objects.json').write_text(json.dumps(catalog))
        fresh_geometry = deepcopy(old)
        fresh_geometry['fence']['planes'].reverse()
        for row in fresh_geometry['fence']['beams']:
            row['plane'] = 1-row['plane']
        (fresh/'geometry.json').write_text(json.dumps(fresh_geometry))
        (fresh/'sam3.json').write_text('{}')
        y, x = np.indices((20, 20))
        points = np.stack([x*.01, (y >= 10).astype(float), .3+y*.05], axis=-1)
        raw = {'pts3d': points, 'non_ambiguous_mask': np.ones((20, 20), bool)}
        with patch.object(bridge, '_frame', return_value=raw), \
             patch.object(bridge, '_array', side_effect=lambda array: array), \
             patch.object(bridge, '_response', return_value={'rle': ['unused']}), \
             patch.object(bridge, '_mask', side_effect=lambda *_: np.ones((20, 20), bool)):
            identities = refresh_catalog(baseline, fresh)
            rebuilt = {row['id']: row for row in json.loads((fresh/'objects.json').read_text())['objects']}
            assert set(rebuilt) == set(bridge.TARGETS)
            assert {key: row['planeIndex'] for key, row in identities.items()} == {'fence-0': 1, 'fence-1': 0}
            assert rebuilt['fence-0']['geometryPlaneIndex'] == 1 and rebuilt['fence-1']['geometryPlaneIndex'] == 0
            assert rebuilt['fence-0']['observations'][0]['box'] == [0, 0, 20, 10]
            assert rebuilt['fence-1']['observations'][0]['box'] == [0, 10, 20, 20]
            saved = (fresh/'objects.json').read_bytes()
            for failure in ('missing-section', 'merged-sections'):
                broken = deepcopy(fresh_geometry)
                if failure == 'missing-section':
                    broken['fence']['beams'] = [row for row in broken['fence']['beams'] if row['plane'] == 1]
                else:
                    for row in broken['fence']['beams']:
                        row['plane'] = 1
                (fresh/'geometry.json').write_text(json.dumps(broken))
                try:
                    refresh_catalog(baseline, fresh)
                except ValueError as error:
                    assert ('fence-1' in str(error) if failure == 'missing-section' else 'same fresh plane' in str(error)), str(error)
                else:
                    raise AssertionError(f'Invalid fresh fence coverage accepted: {failure}')
                assert (fresh/'objects.json').read_bytes() == saved, 'Failed identity matching overwrote the prior catalog'
    print('PASS: four-target refresh; independent fence plane reorder and source masks; missing/merged/ambiguous sections rejected without catalog writes')


if __name__ == '__main__':
    check()
