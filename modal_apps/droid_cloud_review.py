"""Run the existing DROID exporter and source checks beside cloud-native arrays."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from droid_room import app, image, volume, ROOT, ART, DATASET, sha, save

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
review_image = image.apt_install('libgl1', 'libgomp1').pip_install('open3d==0.19.0', 'trimesh==5.1.0')
review_image = review_image.add_local_file(Path(__file__).with_name('droid_room.py'), '/root/droid_room.py')
for name in ['build_droid_replay.py', 'build_replay_scene.py', 'reconstruct_room_rgb.py', 'reconstruct_tum_room.py', 'video_motion.py']:
    review_image = review_image.add_local_file(SCRIPTS / name, '/review/' + name)
review_image = review_image.add_local_file(SCRIPTS.parent / 'tests/check_droid_display.py', '/review/check_droid_display.py')
for path in [ART / 'data/tum-fr1-room/source-rgb.mp4', ART / 'data/tum-fr1-room/manifest.json', DATASET / 'rgb.txt']:
    review_image = review_image.add_local_file(path, str(path))


@app.function(image=review_image, cpu=(4, 4), memory=(16384, 16384), timeout=900,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2, volumes={'/artifact': volume})
def review(run_id, local_run, runner, voxel):
    volume.reload()
    import runpy
    import numpy as np
    from types import SimpleNamespace
    sys.path[:0] = [str(ROOT / 'site'), '/review']
    import cv2
    import trimesh
    from build_droid_replay import build
    assert run_id and all(c.isalnum() or c in '-_' for c in run_id)
    root = ROOT / 'runs' / run_id
    run = root / 'result'
    remote = json.loads((run / 'remote-run.json').read_text())
    assert remote['status'] == local_run['status'] == 'inference_complete'
    assert remote['run_id'] == local_run['run_id'] == run_id
    assert sha(ROOT / 'input-manifest.json') == local_run['input_manifest_sha256']
    assert sha(ROOT / 'input-rgb.tar') == local_run['archive_sha256']
    (run / 'runner-at-execution.py').write_bytes(runner)
    assert sha(run / 'runner-at-execution.py') == local_run['script_sha256']
    shutil.copyfile(ROOT / 'input-manifest.json', run / 'input-manifest.json')
    save(run / 'run.json', local_run)
    manifest = json.loads((run / 'input-manifest.json').read_text())
    # Mirror original source paths so the shared source/time/pixel checks run unchanged.
    with tarfile.open(ROOT / 'input-rgb.tar') as archive:
        expected = {r['relative_path'] for r in manifest['frames']}
        members = archive.getmembers()
        assert len(members) == len(expected) and {m.name for m in members} == expected
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            path = DATASET / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open('xb') as dest:
                shutil.copyfileobj(source, dest)
    output = root / 'export'
    build(SimpleNamespace(run=run, output=output, depth_domain='final-fullres', voxel_length_native=voxel,
                          voxel_reason='Same native voxel size as previous DROID comparison; higher-resolution prediction only',
                          video=ART / 'data/tum-fr1-room/source-rgb.mp4', video_manifest=ART / 'data/tum-fr1-room/manifest.json'))
    sys.argv = ['check_droid_display.py', str(output / 'scene.json')]
    runpy.run_path('/review/check_droid_display.py', run_name='__main__')
    mesh = trimesh.load(output / 'predicted-scene.glb', force='mesh', process=False)
    components = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
    metrics = json.loads((output / 'metrics.json').read_text())
    metrics.update(components=len(components), largest_component_faces=max(map(len, components)),
                   source_display_check_passed=True, opencv=cv2.__version__, complete_room_accepted=False)
    save(output / 'metrics.json', metrics)
    preview = root / 'preview'; preview.mkdir(exist_ok=False)
    # Native arrays, full PLY and depth support stay in the volume; no local multi-GB collection.
    for name in ['scene.json', 'metrics.json', 'predicted-scene.glb', 'supported-keyframe-points.glb', 'depth-support.json', 'geometry-frames.json', 'command.json']:
        if name == 'scene.json':
            data = json.loads((output / name).read_text())
            (preview / name).write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
        else: shutil.copyfile(output / name, preview / name)
    files = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in preview.iterdir()]
    assert sum(f['bytes'] for f in files) <= 32 * 1024**2
    volume.commit()
    return {'summary': metrics, 'files': files}


def main(a):
    from lingbot_cloud_compare import collect_preview
    local = json.loads((a.run / 'run.json').read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    with app.run():
        result = review.remote(local['run_id'], local, (a.run / 'runner-at-execution.py').read_bytes(), a.voxel)
        save(a.output / 'collection.json', result)
        collect_preview('runs/' + local['run_id'], a.output, result, artifact_volume=volume)
        print(json.dumps(result['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'output']: p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--voxel', type=float, required=True)
    main(p.parse_args())
