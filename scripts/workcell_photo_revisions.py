"""Build the main revision and candidate-model revisions of one run; package one page.

Every revision is a separate run directory finished by the same
``workcell_photo_report.finalize`` tail, so its models, floor, scale, endpoint
measurements, semantic binding and exports belong to that revision alone. A
candidate never replaces the main models. The page loads one revision at a
time; identical files are shared by relative URL only when their bytes match.

python scripts/workcell_photo_revisions.py --root RUN --candidate HOUSING_EVIDENCE_DIR \
  --viewer-assets THREE_0_178_0_DIR --out NEW_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'scripts')]
from scripts.workcell_photo_oneshot import _build_page, _export_metric_scene, _freeze_report_ui  # noqa: E402
from scripts.workcell_photo_report import LINEAGE, finalize  # noqa: E402

REVISIONS = 'revisions'
PAGE_FILES = ('scene-report.json', 'measurement-evaluation.json', 'measurements.json', 'model-endpoint-estimate.json',
              'housing-models.json', 'workcell-conditional.glb', 'workcell-metric.glb', 'workcell-native.glb')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _metrics(root):
    """Timing of the run that produced these models; a replay keeps the original."""
    if (root / 'one-shot.json').is_file():
        return json.loads((root / 'one-shot.json').read_text())
    return json.loads((root / 'replay-manifest.json').read_text())['originalMetrics']


def _require_semantics(root, report):
    if (root / 'semantic-experiment').is_dir() and 'semanticExperiment' not in report:
        raise ValueError(report['revision']['id'] + ': semantic experiment not bound: ' + report['semanticBinding']['reason'])
    return report


def _choice(report, url, status):
    revision = report['revision']
    return {'id': revision['id'], 'label': revision['label'], 'branchId': revision['branchId'], 'status': status,
            'documentSha256': revision['documentSha256'], 'parentRevisionId': revision['parentRevisionId'], 'url': url}


def _package_candidate(page, root, report):
    """A candidate page directory: changed files copied, byte-identical files shared."""
    target = page / REVISIONS / report['revision']['id']
    target.mkdir(parents=True)
    urls = {}
    for asset_id, name in report['assetURLs'].items():
        shared = page / name
        if shared.is_file() and _sha(shared) == _sha(root / name):
            urls[asset_id] = '../../' + name
        else:
            shutil.copyfile(root / name, target / name)
            urls[asset_id] = name
    expected = {asset['id']: asset['sha256'] for asset in report['revision']['document']['assets']}
    for asset_id, url in urls.items():
        if _sha((target / url).resolve()) != expected[asset_id]:
            raise ValueError('Candidate asset does not match its document hash: ' + asset_id)
    report['assetURLs'] = urls
    from scripts.workcell_policy_evidence import policy_evidence
    report['policyEvidence'] = policy_evidence(report, json.loads((root / 'objects.json').read_text())['objects'])
    semantic = report.get('semanticExperiment')
    if semantic:
        for obj in semantic['objects']:
            for ref in obj['sourceRefs']:
                if _sha(page / ref['cropPath']) != _sha(root / 'semantic-experiment' / ref['cropPath'].removeprefix('semantic/')):
                    raise ValueError('Candidate semantic crop differs from the shared crop: ' + ref['cropPath'])
                ref['cropPath'] = '../../' + ref['cropPath']
    for name in PAGE_FILES[1:]:
        if (root / name).is_file():
            shutil.copyfile(root / name, target / name)
    return target


def build(root, out, viewer_assets, candidates=(), *, main_id, evidence_url=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Output must be a new directory')
    started = time.monotonic()
    out.mkdir(parents=True)
    replay = json.loads((root / 'replay-manifest.json').read_text()) if (root / 'replay-manifest.json').is_file() else None
    main = out / main_id
    shutil.copytree(root, main)
    lineage = {'branchId': 'oneshot', 'label': '主模型', 'parentRevisionId': replay['revisionId'] if replay else None,
               'role': 'main'}
    (main / LINEAGE).write_text(json.dumps(lineage, ensure_ascii=False, indent=2) + '\n')
    reports = {'main': _require_semantics(main, finalize(main))}
    _export_metric_scene(main, reports['main'])
    from scripts.workcell_bottom_models import install_candidate_models
    built = []
    for evidence, ident, label in candidates:
        evidence = Path(evidence).resolve()
        directory = out / ident
        shutil.copytree(root, directory)
        catalog = json.loads((directory / 'objects.json').read_text())
        volumes = json.loads((evidence / 'volume-candidates.json').read_text())
        install_candidate_models(directory, volumes, evidence, catalog)
        (directory / LINEAGE).write_text(json.dumps({
            'branchId': 'housing-volume-candidate', 'label': label, 'parentRevisionId': main_id, 'role': 'candidate',
            'candidateEvidence': {'file': 'volume-candidates.json', 'sha256': _sha(evidence / 'volume-candidates.json'),
                                  'acceptedForPhysicalUse': False, 'url': evidence_url}}, ensure_ascii=False, indent=2) + '\n')
        report = _require_semantics(directory, finalize(directory))
        _export_metric_scene(directory, report)
        built.append((directory, report))
    _freeze_report_ui(main, Path(viewer_assets).resolve())
    page = _build_page(main, _metrics(main))
    main_report = json.loads((page / 'scene-report.json').read_text())
    seconds = round(time.monotonic() - started, 2)
    update = {'kind': 'saved-geometry-replay' if replay else 'oneshot-revisions', 'modelInferenceRepeated': False,
              'revisionBuildSeconds': seconds, 'sourceRevisionId': replay['revisionId'] if replay else None}
    choices = [_choice(main_report, 'scene-report.json', 'main')]
    for directory, report in built:
        choices.append(_choice(report, f'{REVISIONS}/{report["revision"]["id"]}/scene-report.json', 'candidate_unaccepted'))
    for directory, report in built:
        target = _package_candidate(page, directory, report)
        report.update(timing=main_report['timing'], measurementUpdate=update,
                      revisionChoices=[{**row, 'url': '../../' + row['url']} for row in choices])
        (target / 'scene-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    main_report.update(measurementUpdate=update, revisionChoices=choices)
    if evidence_url and built:
        main_report['experiment'] = {'title': '光幕候选模型可切换核对',
                                     'summary': '页面上方可在主模型与候选模型之间切换；模型、离地测点、卡尺、语义空间证据和下载都只读取所选版本。候选未通过跨图严格门槛，不替换主模型。',
                                     'reportURL': evidence_url, 'timingLabel': '原始完整流程'}
    (page / 'scene-report.json').write_text(json.dumps(main_report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return {'page': str(page), 'revisions': choices, 'seconds': seconds}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='Finished or replayed run directory (read only)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--viewer-assets', type=Path, required=True)
    parser.add_argument('--main-id', required=True, help='Revision ID of the main model build')
    parser.add_argument('--candidate', nargs=3, action='append', default=[], metavar=('EVIDENCE_DIR', 'REVISION_ID', 'LABEL'),
                        help='Directory with volume-candidates.json and its GLBs, the candidate revision ID and its label')
    parser.add_argument('--evidence-url', help='Relative URL of the published candidate evidence page')
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.out, args.viewer_assets, args.candidate, main_id=args.main_id,
                           evidence_url=args.evidence_url), ensure_ascii=False, indent=2))
