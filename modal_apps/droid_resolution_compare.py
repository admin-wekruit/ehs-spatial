"""One higher-resolution inference; preserve all native arrays on Modal."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from droid_room import app, infer, sha, save, REV, CONTRACT_VERSION


def main(output, run_id, previous_run):
    if not run_id or not all(c.isalnum() or c in '-_' for c in run_id):
        raise ValueError('Invalid run ID')
    previous = json.loads((previous_run / 'run.json').read_text())
    assert previous['status'] == 'inference_complete' and previous['source_revision'] == REV
    assert sha(previous_run / 'input-manifest.json') == previous['input_manifest_sha256']
    # Cached cloud archive is checked by infer; do not reread or upload local RGB.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'input-manifest.json').write_bytes((previous_run / 'input-manifest.json').read_bytes())
    (output / 'runner-at-execution.py').write_bytes(Path(__file__).with_name('droid_room.py').read_bytes())
    state = {'status': 'prepared', 'run_id': run_id, 'contract_version': CONTRACT_VERSION,
             'source_revision': REV, 'script_sha256': sha(output / 'runner-at-execution.py'),
             'archive_sha256': previous['archive_sha256'], 'input_manifest_sha256': previous['input_manifest_sha256'],
             'cached_build_sha256': sha(previous_run / 'build.json'), 'resolution_scale': 2,
             'gpu_attempts': 1, 'reserved_usd': 2, 'cpu_build_invoked': False,
             'groundtruth_uploaded': False, 'sensor_depth_uploaded': False}
    save(output / 'run.json', state)
    with app.run():
        deadline = time.time() + 900
        call = infer.spawn(run_id, deadline, state['archive_sha256'], state['input_manifest_sha256'],
                           state['cached_build_sha256'], resolution_scale=2)
        state.update(status='gpu_running', gpu_call_id=call.object_id, app_id=app.app_id)
        save(output / 'run.json', state); print(json.dumps(state), flush=True)
        try:
            result = call.get(timeout=max(1, deadline-time.time()))
        except BaseException:
            call.cancel(terminate_containers=True)
            state['status'] = 'gpu_result_unknown_or_timeout'; save(output / 'run.json', state)
            raise
        save(output / 'remote-run.json', result)
        state['status'] = result['status']; save(output / 'run.json', state)
        print(json.dumps({k: v for k, v in result.items() if k != 'artifacts_sha256'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id', required=True)
    for name in ['output', 'previous-run']: p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); main(a.output, a.run_id, a.previous_run)
