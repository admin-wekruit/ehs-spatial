"""Small check: evaluation cannot alter frozen predictions or invent unsupported values."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from scripts import workcell_metrology_report as report
from scripts.workcell_metrology_report import evaluate


def check():
    identities = ('fence-0', 'post-box-1', 'post-box-2')
    result = {'routes': {'C': {'label': 'source geometry', 'scaleMPerNative': .5,
              'objects': [{'id': identity, 'status': 'conditional', 'heightM': .1,
                           'heightNative': .2} for identity in identities]}}}
    original = deepcopy(result)
    config = {'evaluation': {'targets': [{'objectId': identity, 'groundTruthM': .11}
                                        for identity in identities]}}
    first = evaluate(result, config)
    for target in config['evaluation']['targets']:
        target['groundTruthM'] = 8
    second = evaluate(result, config)
    assert result == original
    assert [row['estimateM'] for row in first] == [row['estimateM'] for row in second]
    assert all(row['below3cm'] for row in first) and not any(row['below3cm'] for row in second)
    for mutation in ('unsupported', 'scale', 'duplicate', 'missing'):
        bad = deepcopy(result)
        route = bad['routes']['C']
        if mutation == 'unsupported':
            route['objects'][0]['status'] = 'unsupported'
        elif mutation == 'scale':
            route['scaleMPerNative'] = .6
        elif mutation == 'duplicate':
            route['objects'].append(deepcopy(route['objects'][0]))
        else:
            route['objects'].pop()
        try:
            evaluate(bad, config)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid {mutation} was accepted')
    result['routes']['C']['objects'][0].update(status='unsupported', heightM=None)
    assert evaluate(result, config)[0]['estimateM'] is None
    with tempfile.TemporaryDirectory(prefix='metrology-report-check-') as directory:
        root = Path(directory)
        baseline = root / 'baseline'; (baseline / 'page').mkdir(parents=True)
        (baseline / 'page/scene-report.json').write_text('{}')
        (root / 'web/dist-photo').mkdir(parents=True)
        (root / 'web/dist-photo/photo.html').write_text('<title>Synthetic UI</title>')
        prediction = {'cameraRoute': 'synthetic', 'calibration': {'new': {'status': 'unsupported'}, 'candidates': []},
                      'routes': {key: deepcopy(original['routes']['C']) for key in 'ABCD'}}
        old, current, failed = [root / name for name in ('old', 'current', 'failed')]
        for run in (old, current, failed):
            run.mkdir()
            (run / 'input-manifest.json').write_text('{}')
            (run / 'spend-ledger.json').write_text(json.dumps({'status': 'failed', 'functionSeconds': None,
                                                              'estimateUsd': None, 'callWindowEstimateUsd': .05}))
        (old / 'original-cameras').mkdir()
        (old / 'original-cameras/results.json').write_text(json.dumps(prediction))
        (current / 'refined-cameras').mkdir()
        (current / 'refined-cameras/results.json').write_text(json.dumps(prediction))
        with patch.object(report, '__file__', str(root / 'scripts/report.py')), patch.object(report, 'load_measurements', return_value=config):
            summary = report.build(baseline, [old, current], root / 'missing-original', root / 'unused.json')
            assert len(summary['reports']) == 2 and len(summary['ledger']) == 2
            assert '未输出预测' in (root / 'missing-original/page/metrology.html').read_text()
            (current / 'original-cameras').mkdir()
            (current / 'original-cameras/results.json').write_text(json.dumps(prediction))
            report.build(baseline, [current], root / 'empty-circle-candidates', root / 'unused.json')
            summary = report.build(baseline, [failed], root / 'all-failed', root / 'unused.json')
            assert not summary['reports'] and summary['functionSecondsTotal'] is None
            assert not summary['allThreeBelow3cm'] and summary['callWindowEstimateUsd'] == .05
            assert '未取得完整函数时间' in (root / 'all-failed/page/metrology.html').read_text()
    print('PASS: frozen predictions, no truth feedback, metric contract, failed/partial run reports')


if __name__ == '__main__':
    check()
