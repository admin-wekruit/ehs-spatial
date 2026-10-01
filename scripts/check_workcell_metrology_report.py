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
            joint_run = root / 'joint'; (joint_run / 'joint-cameras').mkdir(parents=True)
            (joint_run / 'input-manifest.json').write_text('{}')
            (joint_run / 'spend-ledger.json').write_text((failed / 'spend-ledger.json').read_text())
            joint = deepcopy(prediction)
            joint['jointReference'] = {'status': 'unsupported', 'mPerNative': None, 'reason': 'Synthetic held-out view fails',
                                       'heldOutPhotos': [{'photo': 2, 'maxErrorRawPx': 108.4}]}
            joint['calibration']['candidates'] = [{'feature': 'redActuatorDiameterM', 'heldOutPhotos': [{'maxErrorRawPx': 7.0}]}]
            joint['routes']['J'] = {'label': 'Joint three dimensions and cameras', 'scaleMPerNative': None,
                                    'objects': [{'id': identity, 'status': 'unsupported', 'heightM': None,
                                                 'reason': 'Synthetic held-out view fails'} for identity in identities]}
            # A/C happen to fit these disclosed checks; a rejected J must remain rejected.
            (joint_run / 'joint-cameras/results.json').write_text(json.dumps(joint))
            joint_config = deepcopy(config)
            for row in joint_config['evaluation']['targets']:
                row['groundTruthM'] = .1
            with patch.object(report, 'load_measurements', return_value=joint_config):
                summary = report.build(baseline, [old, joint_run], root / 'joint-rejected', root / 'unused.json')
            assert not summary['allThreeBelow3cm'], 'Joint trial acceptance must not select old comparison routes'
            markup = (root / 'joint-rejected/page/metrology.html').read_text()
            assert '三个已知尺寸' in markup and '联合求解未通过' in markup
            assert '整套按钮留出照片检查的最大偏差为 108.4' in markup and '偏差为 7.0' not in markup
            ablation = root / 'ablation'; ablation.mkdir()
            (ablation / 'input-manifest.json').write_text('{}')
            (ablation / 'spend-ledger.json').write_text(json.dumps({'functionSeconds': 120., 'estimateUsd': .2, 'callWindowEstimateUsd': .3}))
            for variant, value in (('cached', .1), ('default-518', .2), ('double-1036', .3)):
                destination = ablation / ('cached-joint-cameras' if variant == 'cached' else f'{variant}/measurements/joint-cameras')
                destination.mkdir(parents=True)
                candidate = deepcopy(prediction)
                candidate['jointReference'] = {'status': 'available', 'mPerNative': .5}
                candidate['routes']['J'] = {'label': 'Joint source-supported candidate', 'scaleMPerNative': .5,
                    'objects': [{'id': identity, 'status': 'conditional', 'heightM': value, 'heightNative': value/.5} for identity in identities]}
                (destination / 'results.json').write_text(json.dumps(candidate))
                if variant != 'cached':
                    fresh = ablation / variant / 'fresh-geometry'; fresh.mkdir()
                    (fresh / 'geometry.json').write_text(json.dumps({'floor': {'normal': [0, 1, 0], 'offset': 0}}))
                    for filename in ('floor-fitted.glb', 'fence-fitted.glb'):
                        (fresh / filename).write_bytes(b'synthetic model pointer, not a rendered model')
                model = destination.parent / 'joint-button/reference-model'; model.mkdir(parents=True, exist_ok=True)
                (model / 'reference-candidate.glb').write_bytes(b'synthetic candidate pointer')
                (model / 'reference-candidate.json').write_text(json.dumps({'status': 'unsupported', 'units': 'native', 'metricScaleMPerNative': None}))
            depth = root / 'depth'; depth.mkdir()
            (depth / 'spend-ledger.json').write_text(json.dumps({'functionSeconds': 65., 'estimateUsd': .1, 'callWindowEstimateUsd': .15}))
            (depth / 'run.json').write_text(json.dumps({'branches': {name: {'status': 'completed', 'timing': {'inferenceSeconds': seconds, 'modelLoadSeconds': 30., 'wallSecondsInContainer': 40., 'peakAllocatedGiB': 9.}}
                for name, seconds in (('default-518', .85), ('double-1036', 1.17))}}))
            prior_depth = root / 'prior-depth'; prior_depth.mkdir()
            (prior_depth / 'run.json').write_text('{"branches":{}}')
            (prior_depth / 'spend-ledger.json').write_text(json.dumps({'functionSeconds': 10., 'estimateUsd': .02, 'callWindowEstimateUsd': .03}))
            with patch.object(report, 'load_measurements', return_value=joint_config):
                summary = report.build(baseline, [ablation], root / 'paired-depth', root / 'unused.json', depth_runs=[prior_depth, depth])
            assert 'double-1036' in summary['primarySource'], 'Source-supported resolution priority must ignore better cached GT error'
            assert summary['functionSecondsTotal'] == 195. and len(summary['ledger']) == 3
            assert summary['depthRun']['branches'], 'Latest depth is shown while prior depth spend remains counted'
            assert 'Fresh paired neural depth' in summary['timingScope']
            assert {row['variant'] for row in summary['reports']} == {'cached', 'default-518', 'double-1036'}
            for option in summary['modelPreviews']:
                assert all(str(Path(asset).parent) == option['world'] for asset in option['assets'])
                if option['kind'] == 'joint-reference':
                    assert option['floor'] is None and len(option['assets']) == 1
            markup = (root / 'paired-depth/page/metrology.html').read_text()
            assert '0.85 s' in markup and '1.17 s' in markup and '已发布估计</th>' not in markup
            assert 'scene.remove(active);dispose(active)' in markup, 'Switching variants must remove the previous native world'
            assert '本次没有重跑神经深度推理' not in markup
            with patch.object(report, 'load_measurements', return_value=config):
                changed_truth = report.build(baseline, [ablation], root / 'paired-depth-truth-changed', root / 'unused.json', depth_runs=[depth])
            assert changed_truth['primarySource'] == summary['primarySource']
            (ablation / 'double-1036/measurements/joint-cameras/results.json').unlink()
            (ablation / 'double-1036/measurements/stages.json').write_text(json.dumps({
                'freshGeometry': {'status': 'completed'},
                'sourceIdentity': {'status': 'failed', 'error': 'Synthetic source-edge identity mismatch'}}))
            (ablation / 'cached-joint-cameras/results.json').unlink()
            (ablation / 'cached-joint-button').mkdir()
            (ablation / 'cached-joint-button/joint-reference.json').write_text(json.dumps({
                'status': 'unsupported', 'reason': 'Synthetic independent button contour rejected', 'diagnostics': {'stage': 'source boundary'}}))
            partial = report.build(baseline, [ablation], root / 'paired-depth-partial', root / 'unused.json', depth_runs=[depth])
            assert partial['executionRecords'][0]['branchStages'][0]['stages']['sourceIdentity']['status'] == 'failed'
            assert partial['executionRecords'][0]['jointCandidates'][0]['reason'] == 'Synthetic independent button contour rejected'
            markup = (root / 'paired-depth-partial/page/metrology.html').read_text()
            assert 'Synthetic source-edge identity mismatch' in markup and 'Synthetic independent button contour rejected' in markup
            assert '未导出受支持测量' in markup and '本轮独立测出的尺寸' in markup and '灰色底座' in markup
            assert '没有完整传播分割、相机和地面误差' in markup
    print('PASS: frozen predictions, truth-independent selection, new-depth timing/labels, isolated model worlds, failed/partial runs')


if __name__ == '__main__':
    check()
