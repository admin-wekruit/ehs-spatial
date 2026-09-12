"""Run one exact evidence candidate through the existing research pipeline."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .object_evidence import build_object_evidence
from .path_safety import validate_safe_path_segment
from .report_workspace import write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


class CacheInputMismatch(ValueError):
    """A different model input may use the explicitly authorized new-call budget."""


def validate_cached_generation(cache, candidate_id, payload, pins):
    """Cache identity is the exact model input and pinned model, never a label."""
    cache=Path(cache).resolve()
    record=json.loads((cache/'output.json').read_text())
    expected=hashlib.sha256(payload).hexdigest()
    if (record.get('status')!='complete' or record.get('object_id')!=candidate_id
            or record.get('source_payload_sha256')!=expected
            or any(record.get(key)!=value for key,value in pins.items())):
        raise CacheInputMismatch('Cached generation does not match this exact candidate payload/model')
    if digest(cache/'input.npz')!=expected:
        raise ValueError('Matching cached generation input bytes changed')
    if digest(cache/'source.json')!=record['source_json_sha256']:
        raise ValueError('Cached generation source record changed')
    files={'output.json','input.npz','source.json',*record['output_sha256'],*record['paths'].values()}
    for name in files:
        if not isinstance(name,str) or Path(name).name!=name or not (cache/name).resolve().is_relative_to(cache):
            raise ValueError('Cached output filename leaves the candidate directory')
        if not (cache/name).is_file():
            raise ValueError('Cached generation is incomplete: '+name)
    for name,sha in record['output_sha256'].items():
        if digest(cache/name)!=sha:
            raise ValueError('Cached generation asset changed: '+name)
    return record,sorted(files)


def generate_candidate(source_run, candidate_id, output_dir, research_repo, *, gpu_budget_seconds,
                       cached_generation_root=None, environment_record=None, python_executable=None,
                       viewer_repo=None, cached_generation_roots=()):
    """Persist status and immutable inputs/results. Zero budget means cache-only.

    output_dir is a new job outside the source run. The caller serializes its
    run's mutation and owns scheduling; this function never selects extra IDs.
    """
    validate_safe_path_segment(candidate_id,'candidate_id')
    if type(gpu_budget_seconds) is not int or gpu_budget_seconds<0:
        raise ValueError('gpu_budget_seconds must be an explicit nonnegative integer')
    source,output,research=map(lambda p:Path(p).resolve(),[source_run,output_dir,research_repo])
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Generation output must be outside the source run')
    if output.exists():
        raise FileExistsError('Preserve this generation job; choose a new output directory')
    output.mkdir(parents=True)
    status={'version':1,'state':'pending','source_run_id':source.name,'candidate_id':candidate_id,
            'gpu_budget_seconds':gpu_budget_seconds,'new_gpu_calls':0,'charged_gpu_seconds':0,
            'stages':[],'metric_scale_known':False}
    state_path=output/'status.json'
    def save(state,**fields):
        status.update(state=state,**fields); write_json(state_path,status)
    save('validating')
    scripts=research/'scripts/research'
    python=str(python_executable or (research/'.venv/bin/python' if (research/'.venv/bin/python').exists() else sys.executable))
    ehs=Path(__file__).resolve().parents[1]
    derived=output/'run'
    def step(name,script,*arguments):
        save(name)
        begin=time.monotonic()
        script_sha=digest(script)
        log=output/f'{name}.log'
        with log.open('w') as stream:
            process=subprocess.run([python,str(script),*map(str,arguments)],cwd=research,
                                   stdout=stream,stderr=subprocess.STDOUT)
        status['stages'].append({'stage':name,'wall_seconds':time.monotonic()-begin,
                                'script_sha256':script_sha,'log':log.name,'exit_code':process.returncode})
        if digest(script)!=script_sha:
            raise RuntimeError(f'{name} implementation changed during execution')
        if process.returncode:
            raise RuntimeError(f'{name} failed; see {log.name}')
    try:
        registry=build_object_evidence(source)
        matches=[c for c in registry['candidates'] if c['id']==candidate_id]
        if len(matches)!=1:
            raise ValueError('Unknown or duplicate evidence candidate ID')
        candidate=matches[0]
        status.update(frame_id=candidate['frame_id'],label=candidate['label'],source_sha256=registry['source_sha256'])
        if candidate['generation']['status']=='blocked':
            save('blocked',reason=candidate['generation']['reason']); return status
        # Reuse the official exporter and preflight. Its complete registry stays
        # available, while generation/assembly receive exactly the requested ID.
        step('exporting',scripts/'export_object_evidence.py','--source-run',source,'--output',derived,'--ehs-repo',ehs,
             '--candidate-id',candidate_id)
        manifest_path=derived/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        manifest['experiment']=output.name
        write_json(manifest_path,manifest)
        evidence_path=derived/'evidence/objects.json'
        evidence=json.loads(evidence_path.read_text())
        selected=[o for o in evidence['objects'] if o['object_id']==candidate_id]
        if len(selected)!=1:
            reason=next((o['reason'] for o in evidence['unavailable_objects'] if o['object_id']==candidate_id),'Candidate failed exact payload preflight')
            save('blocked',reason=reason); return status
        shutil.copyfile(evidence_path,evidence_path.with_name('all-objects.json'))
        step('fitting_floor',scripts/'prepare_product_floor.py','--root',derived,'--ehs-repo',ehs)
        write_json(evidence_path,{'source_run_id':source.name,'objects':selected,'unavailable_objects':[]})
        spec=importlib.util.spec_from_file_location('_candidate_payload',scripts/'generate_lucida_assets.py')
        generator=importlib.util.module_from_spec(spec);spec.loader.exec_module(generator)
        payload,payload_source=generator.payload_for_object(derived,selected[0])
        status['payload_sha256']=hashlib.sha256(payload).hexdigest()
        if status['payload_sha256']!=selected[0]['preflight']['payload_sha256']:
            raise ValueError('Selected payload changed after export preflight')
        write_json(output/'input-provenance.json',payload_source)
        tree=ast.parse((research/'modal_apps/lucida_assets.py').read_text())
        constants={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
                   and node.targets[0].id in {'CODE_REV','MODEL_REV'}}
        pins={'code_revision':constants['CODE_REV'],'model_revision':constants['MODEL_REV'],'seed':42}
        status['model_pins']=pins
        generation=derived/'generation';generation.mkdir(exist_ok=True)
        target=generation/candidate_id
        cache=None
        roots=([cached_generation_root] if cached_generation_root else [])+list(cached_generation_roots)
        for root in roots:
            previous=Path(root).resolve()/candidate_id
            if not (previous/'output.json').is_file():
                continue
            try:
                record,files=validate_cached_generation(previous,candidate_id,payload,pins)
            except CacheInputMismatch:
                continue
            cache=previous
            break
        if cache:
            target.mkdir()
            for name in files:shutil.copyfile(cache/name,target/name)
            save('cached',cache_output_sha256=digest(cache/'output.json'))
        else:
            if gpu_budget_seconds<630:
                save('blocked',reason='No exact cached output; one new object requires at least 630 GPU-budget seconds');return status
            if environment_record is None:
                save('blocked',reason='A configured successful pinned model environment check is required');return status
            environment_record=Path(environment_record).resolve()
            environment=json.loads(environment_record.read_text())
            if environment.get('status')!='complete' or any(environment.get(k)!=v for k,v in pins.items() if k!='seed'):
                raise ValueError('Model environment check does not match the pinned generator')
            (generation/'environment-recgen').mkdir()
            shutil.copyfile(environment_record,generation/'environment-recgen/output.json')
            write_json(generation/'gpu-budget.json',{'limit_seconds':gpu_budget_seconds,'timeout_per_call_seconds':600,
                'accounting':'This job only; reserve 630 seconds per object. Environment check is a pre-existing configured artifact.','calls':[]})
            status['new_gpu_calls']=1
            step('generating',scripts/'generate_lucida_assets.py','--root',derived,'--object-ids',candidate_id)
            validate_cached_generation(target,candidate_id,payload,pins)
        step('assembling',scripts/'assemble_lucida_scene.py','--run-dir',derived)
        step('rendering_comparisons',scripts/'render_lucida_comparisons.py','--root',derived)
        step('adding_context',scripts/'assemble_lucida_scene.py','--run-dir',derived,'--add-context','--ehs-repo',ehs)
        scene_path=derived/'result/scene.json'
        scene=json.loads(scene_path.read_text())
        if {o['id'] for o in scene['objects'] if o['source']=='generated'}!={candidate_id}:
            raise ValueError('Assembled scene contains an unexpected generated object')
        result=derived/'result'
        if viewer_repo:
            viewer=Path(viewer_repo).resolve()
            step('packing',viewer/'pack-model.py',result,result)
            for name in ('viewer.html','public-assets.js','i18n-catalog.js','i18n.js','site-config.js','site-client.js'):
                shutil.copyfile(viewer/name,result/name)
            shutil.copyfile(result/'viewer.html',result/'index.html')
        for name,sha in registry['source_sha256'].items():
            if digest(source/name)!=sha:
                raise ValueError('Source evidence changed during generation; this result is not current')
        status['artifacts']={p.relative_to(output).as_posix():digest(p) for p in result.rglob('*') if p.is_file()}
        save('complete',result_scene='run/result/scene.json',result_viewer='run/result/viewer.html' if viewer_repo else 'run/result/index.html',
             result_metrics='run/result/metrics.html',result_glb='run/result/scene.glb')
    except Exception as error:
        floor_validation=derived/'evidence/floor-validation.json'
        missing=json.loads(floor_validation.read_text()) if floor_validation.is_file() else {}
        reason=missing.get('reason') or f'{type(error).__name__}: {error}'
        save('blocked' if missing.get('state')=='blocked' else 'failed',
             reason=reason.replace(str(source),'<source run>').replace(str(output),'<generation job>'))
    finally:
        ledger=derived/'generation/gpu-budget.json'
        if ledger.is_file():
            calls=json.loads(ledger.read_text()).get('calls',[])
            status['charged_gpu_seconds']=sum(c['charged_seconds'] for c in calls)
            status['new_gpu_calls']=len(calls)
        write_json(state_path,status)
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['source-run','candidate-id','output-dir','research-repo']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--gpu-budget-seconds',type=int,required=True)
    for name in ['cached-generation-root','environment-record','python-executable','viewer-repo']:
        parser.add_argument('--'+name)
    args=vars(parser.parse_args())
    result=generate_candidate(**args)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['state']=='complete' else 1)
