"""The winning geometry route on the production path, Modal CPU, network blocked (2026-10-06).

Configuration (unchanged): DA3-BASE start -> RoMa v1 outdoor matches -> BA with one focal per photo -> two-view triangulation
of RoMa's dense warp at CERT 0.05 (mvs_route.THRESHOLDS), conf = 1 on kept pixels (+ the conf = certainty variant).
Production path: worktree geometry_clean_ab.refine with the numpy Schur LM (modal_apps/bundle_adjust.py; no pycolmap) and
geometry_clean_ab.roma_model (mirrored RoMa outdoor + DINOv2 from the Modal volume 'panoptes-geometry-weights', SHA-256
checked, passed explicitly). Image: docker/geometry-requirements.txt only (--no-deps; numpy 1.26.4, scipy 1.14.1, OpenCV
4.10.0.84, torch 2.5.1, romatch@77f8d68; no pycolmap / vggt / plyfile / Pi3) + the kit modules.
Ephemeral Modal only, every function with an explicit timeout, retries=0, min_containers=0.
  modal run prod_route_modal.py --stage route    per cell one CPU job, block_network=True:
      frozen  refine on the frozen L4 RoMa matches -> mvs on the frozen L4 dense warps (checks/clean-gpu: the inputs every
              scored MVS used) -> CELL-mvs-prod[-confcert]-padded. Must equal the scipyba exports' field values within 0.01 cm
      cpu     roma_model('cpu'); RoMa on the frozen content crops on CPU (fp32: romatch disables autocast off CUDA; the L4 run
              was fp16 autocast, its samples drawn by the CUDA RNG) -> warps compared with the frozen ones; the same refine +
              mvs -> CELL-mvs-prod-cpuroma-padded: the route end to end from the frames, offline
      + one job of the same image (network blocked): the self-tests on the pinned versions, a network probe, pip freeze
  modal run prod_route_modal.py --stage analyse  backbones.check_geometry, then the UNCHANGED fair_ab_modal.analyse -> OUT
  python prod_route_modal.py compare             field values vs the scipyba rows, BA terminations, cameras -> prod_route_results.json
  python prod_route_modal.py export CELL         backbone_ab_modal.export_run -> SCR/checks/bbab-export-CELL-mvs-prod, every link a
                                                 real copy, exportNote naming the solver that ran; backbones.check_geometry
"""
import io
from itertools import combinations
import json
import os
from pathlib import Path
import shutil
import sys
import time

import modal

NOTE = Path(__file__).resolve().parent
KIT = Path(os.environ.get('PANOPTES_WORKCELL', '/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed'))
sys.path[:0] = [str(NOTE), str(KIT / 'modal_apps'), str(KIT / 'scripts/onprem'), str(NOTE.parent / 'geometry-licence-ab-fair-2026-10-05')]
SCR = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
GEOM, OUT, ROMA_CPU = SCR / 'checks/prod-geom', SCR / 'checks/prod-analyse', SCR / 'checks/prod-cpuroma'
REF_GEOM, REF_OUT = SCR / 'checks/scipyba-geom', SCR / 'checks/scipyba-analyse'  # the scipyba run the exports came from
RATE = 8 * .0000131 + 32 * .00000222  # 8 CPU + 32 GiB list rate (USD/s); not an invoice
TEST_RATE = 2 * .0000131 + 4 * .00000222
FIELD_TOL_CM = .01

app = modal.App('geometry-prod-route')
kit_image = (modal.Image.debian_slim(python_version='3.11').apt_install('git', 'libgl1', 'libglib2.0-0')
             .pip_install_from_requirements(str(KIT / 'docker/geometry-requirements.txt'), extra_options='--no-deps')
             .env({'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'})
             .add_local_python_source('geometry_clean_ab', 'bundle_adjust', 'fetch_weights_geometry', 'fetch_weights', 'fetch_weights_da3',
                                      'fair_ab_modal', 'moge3_app', 'mvs_route', 'backbones', 'ba_scipy'))
weights = modal.Volume.from_name('panoptes-geometry-weights')
# block_network also blocks Modal's own blob upload of a large return value (first attempt, 2026-10-06): the arrays go through
# this volume instead (as onprem_image_proof.py does), the return value stays small
out_vol = modal.Volume.from_name('panoptes-geometry-prod-out', create_if_missing=True)


def environment() -> dict:
    """Network probe (must be blocked), versions, the GPL / research packages that must be absent, pip freeze."""
    import importlib.util
    import socket
    import subprocess
    net = {}
    for name, f in (('tcp 1.1.1.1:443', lambda: socket.create_connection(('1.1.1.1', 443), timeout=5).close()),
                    ('dns github.com', lambda: socket.getaddrinfo('github.com', 443))):
        try:
            f(); net[name] = 'REACHABLE'
        except Exception as error:  # noqa: BLE001 - only the type is kept
            net[name] = 'blocked: ' + type(error).__name__
    import cv2
    import numpy
    import scipy
    import torch
    return dict(network=net, numpy=numpy.__version__, scipy=scipy.__version__, opencv=cv2.__version__, torch=str(torch.__version__), python=sys.version.split()[0],
                absent={m: importlib.util.find_spec(m) is None for m in ('pycolmap', 'vggt', 'plyfile', 'pi3')},
                freeze=subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'], capture_output=True, text=True).stdout.split())


@app.function(image=kit_image, cpu=2, memory=4 * 1024, timeout=1200, retries=0, min_containers=0, block_network=True)
def selftests() -> dict:
    import contextlib
    import traceback
    import ba_scipy
    import bundle_adjust
    import fetch_weights_geometry
    import geometry_clean_ab
    t0, out = time.monotonic(), {}
    for name, fn in (('bundle_adjust', bundle_adjust._check), ('geometry_clean_ab', geometry_clean_ab._check), ('ba_scipy', ba_scipy._check),
                     ('fetch_weights_geometry', fetch_weights_geometry._check)):
        buf, t = io.StringIO(), time.monotonic()
        try:
            with contextlib.redirect_stdout(buf):
                fn()
            out[name] = dict(passed=True, output=buf.getvalue().strip()[-600:], seconds=time.monotonic() - t)
        except BaseException:  # noqa: BLE001 - reported
            out[name] = dict(passed=False, output=(buf.getvalue() + traceback.format_exc())[-1500:], seconds=time.monotonic() - t)
    return json.loads(json.dumps(dict(tests=out, environment=environment(), containerSeconds=time.monotonic() - t0), default=str))  # plain types only


def warp_diff(ref, new) -> dict:
    """Frozen L4 dense warp vs the CPU one, per direction: px difference where both certainties pass CERT, certainty agreement."""
    import numpy as np
    import mvs_route as mr
    out = {}
    for key, (u0, c0), (u1, c1) in (('AB', ref[:2], new[:2]), ('BA', ref[2:], new[2:])):
        k0, k1 = c0.astype(float) >= mr.ROMA_SAMPLE_THRESH, c1.astype(float) >= mr.ROMA_SAMPLE_THRESH; both = k0 & k1
        d = np.linalg.norm(u0[both].astype(float) - u1[both], axis=1)
        out[key] = dict(px=[float(np.median(d)), float(np.quantile(d, .95)), float(np.quantile(d, .99))], keptRef=float(k0.mean()), keptNew=float(k1.mean()),
                        keptDisagree=float((k0 ^ k1).mean()), certAbsDiffMedian=float(np.median(np.abs(c0.astype(float) - c1.astype(float)))))
    return out


@app.function(image=kit_image, cpu=8, memory=32 * 1024, timeout=3600, retries=0, min_containers=0, block_network=True,
              volumes={'/weights': weights, '/out': out_vol})
def route(job: dict) -> dict:
    import traceback
    import numpy as np
    import torch
    from PIL import Image
    import geometry_clean_ab as gc
    import mvs_route as mr
    t0 = time.monotonic(); gc.MVS.update(mr.THRESHOLDS)  # CERT 0.05, geometric checks unchanged (= mvs_route.run)
    frames = gc.unpack_frames(job['frames']); contents = [np.asarray(c) for c in job['contents']]

    dst = Path('/out') / job['run'] / job['cell']; dst.mkdir(parents=True)

    def once(tag, matches, dense):
        t = time.monotonic(); baf, rep = gc.refine(frames, contents, matches, focal=True); t_ba = time.monotonic() - t
        mv, st = gc.mvs(baf, dense)
        (dst / f'{tag}-baf.npz').write_bytes(gc.pack_frames(baf)); (dst / f'{tag}-mvs.npz').write_bytes(gc.pack_frames(mv))
        return dict(report=rep, mvsStats=st, baSeconds=t_ba, seconds=time.monotonic() - t)
    key = lambda k: tuple(map(int, k.split('-')))
    dense = {key(k): tuple(np.load(io.BytesIO(d))[n] for n in ('uvAB', 'certA', 'uvBA', 'certB')) for k, d in job['dense'].items()}
    out = dict(environment=environment(), frozen=once('frozen', {key(k): (np.asarray(a), np.asarray(b)) for k, (a, b) in job['matches'].items()}, dense))
    try:  # RoMa on CPU from the mirrored weights; the frozen result above stands even if this part fails
        t = time.monotonic(); torch.manual_seed(0); roma = gc.roma_model('cpu', '/weights'); out['romaLoadSeconds'] = time.monotonic() - t
        names = sorted(job['crops']); ims = [Image.open(io.BytesIO(job['crops'][n])).convert('RGB') for n in names]
        matches, dense2, cmp = {}, {}, {}
        for i, j in combinations(range(len(ims)), 2):
            t = time.monotonic(); r = gc.roma_pair(roma, ims[i], ims[j], 'cpu')
            matches[(i, j)] = (r['sparse']['uvA'], r['sparse']['uvB']); d = r['dense']; dense2[(i, j)] = (d['uvAB'], d['certA'], d['uvBA'], d['certB'])
            (dst / f'roma-{i}-{j}.npz').write_bytes(gc.npz(**r['sparse'])); (dst / f'dense-{i}-{j}.npz').write_bytes(gc.npz(**d))
            cmp[f'{i}-{j}'] = dict(warp_diff(dense[(i, j)], dense2[(i, j)]), seconds=time.monotonic() - t, samples=len(r['sparse']['uvA']))
        out.update(romaCpuVsL4=cmp, cpu=once('cpu', matches, dense2))
    except Exception:  # noqa: BLE001 - reported
        out['cpuError'] = traceback.format_exc()[-2000:]
    out['files'] = sorted(p.name for p in dst.iterdir()) + ['summary.json']; out['containerSeconds'] = time.monotonic() - t0
    summary = json.dumps(out, default=str); (dst / 'summary.json').write_text(summary + '\n'); out_vol.commit()
    return json.loads(summary)  # plain types only: the local side has no torch / numpy objects to unpickle into


def write_mvs(cell, name, frames, conf, manifest):
    import numpy as np
    import backbones as bb
    import geometry_clean_ab as gc
    out = [dict(f, conf=f['valid'].astype(np.float32) if conf == 'one' else f['conf'].astype(np.float32)) for f in frames]
    dst = GEOM / name / 'geometry'
    gc.write_geometry(dst, cell, out, dict(manifest, conf='1 on kept pixels (mvs_route)' if conf == 'one' else 'RoMa certainty (geometry_clean_ab.mvs)'))
    return bb.check_arrays(dst, len(out))


def ba_summary(rep):
    return dict(backend=rep['ba_backend'], converged=rep['converged'], usable=rep['usable'],
                passes=[{k: p[k] for k in ('pass_', 'termination', 'iterations', 'cost', 'gradientInf', 'converged', 'usable', 'points', 'dropped', 'seconds')} for p in rep['ba']])


@app.local_entrypoint()
def main(stage: str, cells: str = '090,030'):
    import geometry_clean_ab as gc
    import fair_ab_modal as fam
    import mvs_route as mr
    cells = cells.split(',')
    if stage == 'analyse':
        import backbone_ab_modal as bab
        bab.GEOM, bab.OUT = GEOM, OUT  # ponytail: analyse_local reads these module paths; pointed at this run's folders
        bab.analyse_local([])
        return
    if stage != 'route':
        raise ValueError(stage)
    names = [f'{c}-{n}' for c in cells for n in ('da3-base-prod-f', 'mvs-prod-padded', 'mvs-prod-confcert-padded', 'da3-base-prod-cpuroma-f', 'mvs-prod-cpuroma-padded')]
    clash = [n for n in names if (GEOM / n).exists()] + [str(ROMA_CPU / c) for c in cells if (ROMA_CPU / c).exists()]
    if clash:
        raise ValueError(f'{clash} exist; choose fresh names')
    jobs, run = [], time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    fetch = lambda c, n: b''.join(out_vol.read_file(f'{run}/{c}/{n}'))
    for c in cells:
        frames = gc.read_geometry(gc.INIT['da3-base'](c)); matches = gc.load_matches(c)
        jobs.append(dict(run=run, cell=c, frames=gc.pack_frames(frames), contents=gc.contents_for(c, frames), crops=fam.frames_for(c)['unpadded'],
                         matches={f'{i}-{j}': (u, v) for (i, j), (u, v) in matches.items()},
                         dense={f.stem.split('-', 1)[1]: f.read_bytes() for f in sorted((gc.GPUD / c).glob('dense-*.npz'))}))
    t = time.monotonic(); tests = selftests.spawn(); rows, total = [], 0.
    for job, r in zip(jobs, route.map(jobs, order_outputs=True, return_exceptions=True)):
        c = job['cell']
        if isinstance(r, Exception):
            print(c, 'FAILED', repr(r)[:800]); continue
        total += r['containerSeconds']; row = dict(cell=c, volumeRun=f'panoptes-geometry-prod-out:/{run}/{c}', containerSeconds=r['containerSeconds'], environment={k: v for k, v in r['environment'].items() if k != 'freeze'})
        for tag, sfx in (('frozen', ''), ('cpu', '-cpuroma')):
            if tag not in r:
                continue
            x = r[tag]; baf = GEOM / f'{c}-da3-base-prod{sfx}-f' / 'geometry'
            gc.write_geometry(baf, c, gc.unpack_frames(fetch(c, f'{tag}-baf.npz')), dict(base='da3-base', init=str(gc.INIT['da3-base'](c)), refinement='ba-f', ba=x['report']['ba_backend'],
                                                                     roma='frozen L4 run (checks/clean-gpu)' if tag == 'frozen' else 'CPU re-run from the mirrored weights (prod-cpuroma)',
                                                                     report=x['report']))
            common = dict(model_id=f'licence-clean MVS (da3-base BA-f cameras, numpy Schur LM; RoMa outdoor dense warp{"" if tag == "frozen" else " recomputed on CPU"}, midpoint triangulation)',
                          ba=x['report']['ba_backend'], baConverged=x['report']['converged'], baUsable=x['report']['usable'], cameras=str(baf),
                          thresholds=mr.THRESHOLDS, report=x['mvsStats'], created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            mv = gc.unpack_frames(fetch(c, f'{tag}-mvs.npz'))
            row[tag] = dict(ba=ba_summary(x['report']), baSeconds=x['baSeconds'], seconds=x['seconds'], focal=[p['focalAfter'][0] for p in x['report']['photos']],
                            contract={v: write_mvs(c, f'{c}-mvs-prod{sfx}{"-confcert" if v == "cert" else ""}-padded', mv, v, common) for v in (('one', 'cert') if tag == 'frozen' else ('one',))})
        if 'cpu' in r:
            (ROMA_CPU / c).mkdir(parents=True)
            for n in r['files']:
                if n.startswith(('roma-', 'dense-')):
                    (ROMA_CPU / c / n).write_bytes(fetch(c, n))
            row.update(romaLoadSeconds=r['romaLoadSeconds'], romaCpuVsL4=r['romaCpuVsL4'])
        else:
            row['cpuError'] = r['cpuError']
        (GEOM / f'environment-{c}.json').write_text(json.dumps(r['environment'], indent=1) + '\n'); rows.append(row)
        print(c, f"{r['containerSeconds']:.0f}s", json.dumps({k: row[k]['ba'] for k in ('frozen', 'cpu') if k in row}, default=float)[:1500], row.get('cpuError', '')[-400:])
    st = tests.get(); total_t = st['containerSeconds']
    (GEOM / 'selftests.json').write_text(json.dumps(st, indent=1) + '\n')
    print('self-tests', {k: v['passed'] for k, v in st['tests'].items()}, st['environment']['numpy'], st['environment']['network'])
    ledger = dict(stage='route', mode='ephemeral modal run, block_network', hardware='route: 8 CPU, 32 GiB per cell; self-tests: 2 CPU, 4 GiB', jobs=len(jobs) + 1,
                  functionSeconds=total + total_t, callSeconds=time.monotonic() - t, estimateUsd=RATE * total + TEST_RATE * total_t, rateSource='https://modal.com/pricing', rows=rows)
    (GEOM / f'spend-route-{int(time.time())}.json').write_text(json.dumps(ledger, indent=1, default=float) + '\n')
    print(json.dumps({k: v for k, v in ledger.items() if k != 'rows'}))


# ---------------------------------------------------------------- local: comparison + export
def compare():
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location('bbab_compile', NOTE / 'compile.py')  # this note's compile, under its own name
    bc = importlib.util.module_from_spec(spec); spec.loader.exec_module(bc)
    import geometry_clean_ab as gc
    runs = {}
    for d in (REF_OUT, OUT):
        for f in d.glob('*.json'):
            if not f.name.startswith('spend'):
                r = json.loads(f.read_text()); runs[(r['cell'], r['backbone'], r['variant'])] = r
    pairs = [('frozen', 'one', 'mvs-prod', 'mvs-scipyba'), ('frozen', 'certainty', 'mvs-prod-confcert', 'mvs-scipyba-confcert'), ('cpu', 'one', 'mvs-prod-cpuroma', 'mvs-scipyba')]
    for _, _, k, ref in pairs:
        for x in (k, ref):
            bc.fc.LICENCE.setdefault(x, 'MIT / BSD-3 / Apache-2.0 (numpy Schur LM BA, no GPL SuiteSparse)')
    row = lambda k: bc.clean.safe_config(runs, k, 'padded', 'maxInlier', 'heightCm')
    out = dict(fieldTolCm=FIELD_TOL_CM, gate=bc.GATE, configs=[], cameras={}, mvsArrays={})
    for roma, conf, k, ref in pairs:
        if (('090', k, 'padded') not in runs) or (('030', k, 'padded') not in runs):
            continue
        rs, rr = row(k), row(ref); dv = {n: rs['values'][n]['cm'] - rr['values'][n]['cm'] for n in bc.NAMES}
        out['configs'].append(dict(roma=roma, conf=conf, key=k, reference=ref, values={n: rs['values'][n]['cm'] for n in bc.NAMES}, maeCm=rs.get('maeCm'),
                                   maxAbsErrCm=rs.get('maxAbsErrCm'), verdict=bc.verdict(rs), referenceMaeCm=rr.get('maeCm'), fieldDiffCm=dv,
                                   gate={c: rs['gate'][c]['maxDeviationPct'] for c in ('090', '030')}, floorP95Cm=rs['floorP95Cm'], cameraHeightsCm=rs['cameraHeightsCm'],
                                   withinTol=max(map(abs, dv.values())) <= FIELD_TOL_CM))
    for c in ('090', '030'):
        m = runs[(c, 'mvs-scipyba', 'padded')]['floors']['maxInlier']['estop']['nativeToMeters'] * 100  # cm per native unit (scipyba run)
        for tag, sfx in (('frozen', ''), ('cpu', '-cpuroma')):
            f = GEOM / f'{c}-da3-base-prod{sfx}-f/geometry'
            if not f.exists():
                continue
            S, P = gc.read_geometry(f), gc.read_geometry(REF_GEOM / f'{c}-da3-base-scipyba-f/geometry')
            rp = json.loads((f / 'candidate_manifest.json').read_text())['report']; rr = json.loads((REF_GEOM / f'{c}-da3-base-scipyba-f/geometry/candidate_manifest.json').read_text())['report']
            out['cameras'][f'{c}-{tag}'] = dict(rotationDiffDeg=[gc.rot_deg(s['c2w'][:3, :3].T @ p['c2w'][:3, :3]) for s, p in zip(S, P)],
                                                centreDiffCm=[float(np.linalg.norm(s['c2w'][:3, 3] - p['c2w'][:3, 3]) * m) for s, p in zip(S, P)],
                                                focalDiffPx=[s['K'][0, 0] - p['K'][0, 0] for s, p in zip(S, P)], pairsEqual=rp['pairs'] == rr['pairs'],
                                                reprojPxInit=[rp['reprojPxInit'], rr['reprojPxInit']], ba=ba_summary(rp),
                                                referenceBa=[{k: p[k] for k in ('termination', 'iterations', 'cost', 'dropped', 'points')} for p in rr['ba']])
            ms, mr_ = gc.read_geometry(GEOM / f'{c}-mvs-prod{sfx}-padded/geometry'), gc.read_geometry(REF_GEOM / f'{c}-mvs-scipyba-padded/geometry')
            both = [a['valid'] & b['valid'] for a, b in zip(ms, mr_)]
            d = [np.linalg.norm(a['pts3d'][v].astype(float) - b['pts3d'][v], axis=1) * m for a, b, v in zip(ms, mr_, both)]
            out['mvsArrays'][f'{c}-{tag}'] = dict(validXor=[int((a['valid'] ^ b['valid']).sum()) for a, b in zip(ms, mr_)], validCount=[int(a['valid'].sum()) for a in ms],
                                                  pointDiffCmMedianP95Max=[[float(np.median(x)), float(np.quantile(x, .95)), float(x.max())] for x in d])
    spend = [json.loads(p.read_text()) for p in sorted(list(GEOM.glob('spend-*.json')) + list(OUT.glob('spend-ledger-*.json')))]
    out['spend'] = dict(totalEstimateUsd=sum(s['estimateUsd'] for s in spend), ledgers=[{k: v for k, v in s.items() if k != 'rows'} for s in spend])
    led = [x for x in (json.loads(p.read_text()) for p in sorted(GEOM.glob('spend-route-*.json'))) if 'rows' in x]
    out['route'] = [{k: v for k, v in r.items() if k not in ('contract',)} for r in (led[-1]['rows'] if led else [])]
    if (GEOM / 'selftests.json').exists():
        st = json.loads((GEOM / 'selftests.json').read_text())
        out['selftests'] = {k: dict(passed=v['passed'], output=v['output']) for k, v in st['tests'].items()}
        out['selftestEnvironment'] = {k: v for k, v in st['environment'].items() if k != 'freeze'}; out['imageFreeze'] = st['environment']['freeze']
    out['exports'] = {c: json.loads(f.read_text()) for c in ('090', '030') if (f := SCR / f'checks/prod-export-{c}-contract.json').exists()}
    out['passes'] = bool(out['configs']) and all(x['withinTol'] and x['verdict'] == 'yes' for x in out['configs'] if x['roma'] == 'frozen')
    (NOTE / 'prod_route_results.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    for x in out['configs']:
        print(x['roma'], x['conf'], {n: round(v, 4) for n, v in x['values'].items()}, 'MAE', round(x['maeCm'], 4), 'ref', round(x['referenceMaeCm'], 4),
              'diff', {n: f'{v:+.5f}' for n, v in x['fieldDiffCm'].items()}, 'within', x['withinTol'], x['verdict'])
    for k, v in out['cameras'].items():
        print(k, 'rot', [f'{a:.1e}' for a in v['rotationDiffDeg']], 'centre cm', [f'{a:.1e}' for a in v['centreDiffCm']], 'focal px', [f'{a:+.4f}' for a in v['focalDiffPx']],
              'pairsEqual', v['pairsEqual'], [(p['termination'], p['iterations']) for p in v['ba']['passes']])
    for k, v in out['mvsArrays'].items():
        print(k, v)
    print('passes', out['passes'], 'spend', round(out['spend']['totalEstimateUsd'], 4))


def export(cell):
    """backbone_ab_modal.export_run, unchanged, on this run's geometry; every link then becomes a real copy of its target."""
    import backbone_ab_modal as bab
    import backbones as bb
    bab.GEOM, bab.OUT = GEOM, OUT
    bab.export_run('mvs-prod', cell)
    dst = SCR / f'checks/bbab-export-{cell}-mvs-prod'; links = []
    for root, dirs, files in os.walk(dst):
        for n in dirs + files:
            p = Path(root) / n
            if p.is_symlink():
                target = p.resolve(); p.unlink(); links.append(str(p.relative_to(dst)))
                shutil.copytree(target, p) if target.is_dir() else shutil.copy2(target, p)
    left = [str(Path(r) / n) for r, d, f in os.walk(dst) for n in d + f if (Path(r) / n).is_symlink()]
    assert not left, left
    cm = json.loads((dst / 'geometry/candidate_manifest.json').read_text())
    man = json.loads((dst / 'manifest.json').read_text())
    man['exportNote'] = dict(ba=cm['ba'], baConverged=cm['baConverged'], baUsable=cm['baUsable'], geometry=str(GEOM / f'{cell}-mvs-prod-padded/geometry'),
                             route='prod_route_modal.py --stage route: geometry_clean_ab.refine + mvs (CERT 0.05, conf = 1 on kept pixels), Modal CPU, '
                                   'image from docker/geometry-requirements.txt (numpy 1.26.4), network blocked',
                             romaInputs='the frozen L4 RoMa outputs (checks/clean-gpu) every scored MVS used; the CPU re-run from the mirrored weights is the mvs-prod-cpuroma row',
                             imageSpaceFiles=f'{len(links)} links replaced by real copies of the source run files (input/, masks, rgba crops)',
                             inherited='created_at_utc, script_sha256 and privacy are the source run\'s')
    (dst / 'manifest.json').write_text(json.dumps(man, indent=2) + '\n')
    src = bab.RUNS / bab.harness().CELLS[cell]['run']
    contract = bb.check_geometry(dst / 'geometry', src)
    (SCR / f'checks/prod-export-{cell}-contract.json').write_text(json.dumps(dict(export=str(dst), contract=contract, linksReplaced=len(links)), indent=1) + '\n')
    print(dst, 'links replaced', len(links), 'contract', json.dumps(contract))


if __name__ == '__main__' and sys.argv[1:2] == ['compare']:
    compare()
if __name__ == '__main__' and sys.argv[1:2] == ['export']:
    export(sys.argv[2])
