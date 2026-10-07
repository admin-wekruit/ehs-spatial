"""Validate a job plan, then share one pinned RecGen pipeline across its jobs."""

import json
from pathlib import Path
import re
import sys
import time

import numpy as np


def validate_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    if not isinstance(plan, list) or not plan:
        raise ValueError('RecGen plan must be a nonempty job list')
    prepared, kinds, targets, outputs = [], set(), set(), set()
    for job in plan:
        if not isinstance(job, dict) or set(job) != {'kind', 'source', 'target', 'groups'}:
            raise ValueError('Each RecGen job needs kind, source, target and groups')
        kind = job['kind']
        if not isinstance(kind, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', kind) or kind in kinds:
            raise ValueError('RecGen job kinds must be unique nonempty names')
        kinds.add(kind)
        if not all(isinstance(job[k], str) and job[k] for k in ('source', 'target')):
            raise ValueError(f'{kind}: source and target must be paths')
        source, target = [(path.parent / job[k]).resolve() for k in ('source', 'target')]
        if not source.is_file():
            raise ValueError(f'{kind}: input NPZ does not exist: {source}')
        if target in targets or not target.parent.is_dir():
            raise ValueError(f'{kind}: targets must be distinct and have an existing parent directory')
        targets.add(target)
        groups = job['groups']
        if not isinstance(groups, list) or not groups:
            raise ValueError(f'{kind}: groups must be a nonempty list')
        names, indices = set(), set()
        for group in groups:
            if not isinstance(group, list) or len(group) != 2:
                raise ValueError(f'{kind}: expected [name, [view indices]]')
            name, views = group
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in names:
                raise ValueError(f'{kind}: model group names must be unique')
            if (not isinstance(views, list) or not views or any(type(i) is not int or i < 1 for i in views)
                    or len(set(views)) != len(views)):
                raise ValueError(f'{kind}/{name}: views must be distinct positive integers')
            output = Path(f'{target}-{name}.npz')
            if output in outputs:
                raise ValueError(f'{kind}/{name}: output path collides with another job')
            outputs.add(output); names.add(name); indices.update(views)
        frames = {}
        with np.load(source, allow_pickle=False) as saved:
            for i in sorted(indices):
                keys = [f'v{i}_{suffix}' for suffix in ('rgb', 'depth', 'mask', 'K')]
                if any(key not in saved for key in keys):
                    raise ValueError(f'{kind}: missing arrays for view {i}')
                rgb, depth, mask, intrinsics = [saved[key] for key in keys]
                if (rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]
                        or mask.shape != depth.shape or intrinsics.shape != (3, 3)):
                    raise ValueError(f'{kind}: invalid array shapes for view {i}')
                if (any(a.dtype.kind not in 'buif' or not np.isfinite(a).all() for a in (rgb, depth, mask, intrinsics))
                        or np.any((rgb < 0) | (rgb > 255)) or np.any(depth < 0)
                        or not np.isin(mask, [0, 1, 255]).all() or not ((mask > 0) & (depth > 0)).any()
                        or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0 or abs(np.linalg.det(intrinsics)) < 1e-12):
                    raise ValueError(f'{kind}: invalid values or empty foreground depth for view {i}')
                frames[i] = {'rgb': rgb, 'depth': depth, 'mask': mask, 'camera_intrinsics': intrinsics}
        prepared.append({'kind': kind, 'source': source, 'target': target, 'groups': groups, 'views': frames})
    if outputs & {job['source'] for job in prepared}:
        raise ValueError('RecGen output would overwrite a plan input')
    return prepared


def main(plan_path):
    jobs = validate_plan(plan_path)
    from fast_report import recgen_fast, x7

    started = time.monotonic()
    pipeline = x7.load_recgen()
    loaded = time.monotonic() - started
    records = {}
    for job in jobs:
        models = {}
        for name, indices in job['groups']:
            views = [job['views'][i] for i in indices]
            result = recgen_fast.run(pipeline, views, 42, recgen_fast.setting(**recgen_fast.FAST))
            np.savez_compressed(f"{job['target']}-{name}.npz", vertices=result['vertices'],
                                faces=result['faces'], colors=result['colors'])
            models[name] = {'views': indices, 'generationSeconds': result['seconds'],
                            'faces': result['faces_n'], 'stages': result['stages']}
        records[job['kind']] = {'models': models}
    torch = sys.modules.get('torch')  # loaded by RecGen on the GPU; absent in CPU plan checks
    report = {'modelLoadSeconds': loaded, 'jobs': records,
              'peakAllocatedGiB': torch.cuda.max_memory_allocated() / 2**30 if torch else None}
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: workcell_recgen_worker.py PLAN.json')
    main(sys.argv[1])
