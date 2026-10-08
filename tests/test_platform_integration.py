"""Real SQL + worker + Agent + policy checks without paid model calls."""
from uuid import uuid4
import json

import pytest

from test_platform_backend import repo, project, identity, capability, entity
from argus.platform.agent_service import AgentService, AgentResult
from argus.platform.contracts import PlatformError
from argus.platform.policy_service import PolicyService, templates, execute_jdm
from argus.platform.policy_repository import PostgresPolicyRepository
from argus.platform.storage import LocalBlobStore
from argus.platform.worker import run_job


def job_body(scene, kind):
    return {'requestId':identity(),'branchId':scene['branch']['id'],'baseRevisionId':scene['revision']['id'],'kind':kind,'inputs':{},'config':{}}


def test_worker_json_export_is_fixed_revision_and_repeat_is_free(repo, tmp_path):
    cap, scene = project(repo)
    blobs = LocalBlobStore(tmp_path)
    job = repo.create_job(scene['project']['id'],cap,job_body(scene,'export_json'))
    result = run_job(repo,blobs,job['id'],{})
    assert result['status'] == 'succeeded'
    assert result['resultRevisionId'] is None
    assert result['result']['newModelCalls'] == 0
    asset = result['result']['assets'][0]
    assert json.loads(blobs.get(asset['storageKey'],asset['sha256'],asset['sizeBytes'])) == scene['revision']['document']
    assert run_job(repo,blobs,job['id'],{})['result'] == result['result']
    assert repo.get_project(scene['project']['id'])['branches'][0]['headRevisionId'] == scene['revision']['id']


def test_mesh_export_retains_missing_entities_as_incomplete(repo, tmp_path):
    cap, scene = project(repo)
    added = entity()
    saved = repo.commit_edits(scene['project']['id'], cap, {'requestId': identity(),
        'branchId': scene['branch']['id'], 'baseRevisionId': scene['revision']['id'],
        'operations': [{'type':'addEntity','entity':added}]})
    scene['revision'] = saved['revision']
    job = repo.create_job(scene['project']['id'], cap, job_body(scene, 'export_glb'))
    result = run_job(repo, LocalBlobStore(tmp_path), job['id'], {})
    assert result['status'] == 'incomplete'
    assert result['result']['unplacedEntities'] == [{'entityId':added['id'],'reason':'no_representation'}]
    assert result['result']['newModelCalls'] == 0


class FakeAgent:
    name,model,max_call_cost,paid = 'contract-test','fixed-test',0,False
    def __init__(self, response):
        self.response, self.calls = response, []
    def respond(self,messages,context):
        self.calls.append((messages,context))
        return AgentResult(self.response, {'input_tokens':1,'output_tokens':1})


def turn_body(scene):
    return {**{k:v for k,v in job_body(scene,'agent_turn').items() if k not in ('kind','inputs','config')},
            'conversationId':identity(),'message':'Add an object','language':'en'}


def test_agent_proposes_once_then_normal_edit_is_only_writer(repo):
    cap,scene = project(repo)
    pid=scene['project']['id']
    added=entity()
    provider=FakeAgent({'kind':'proposal','message':'Proposed','operations':[{'type':'addEntity','entity':added}]})
    pending=repo.create_agent_turn(pid,cap,turn_body(scene))
    service=AgentService(repo,provider)
    result=service.run_turn(pending,cap)
    assert result['status']=='succeeded'
    assert result['response']['applied'] is False
    assert repo.get_revision(scene['revision']['id'])['document']['entities']==[]
    assert cap not in str(provider.calls)
    assert service.run_turn(pending,cap) is None and len(provider.calls)==1
    body={'requestId':identity(),'branchId':scene['branch']['id'],'baseRevisionId':scene['revision']['id'],
          'operations':result['response']['operations'],'agentTurnId':result['id']}
    saved=repo.commit_edits(pid,cap,body)
    assert saved['revision']['document']['entities'][0]['id']==added['id']
    assert repo.get_job(pending['jobId'])['status']=='succeeded'


def test_agent_missing_provider_and_unknown_outcome_leave_no_pending_charge(repo):
    cap,scene=project(repo)
    pending=repo.create_agent_turn(scene['project']['id'],cap,turn_body(scene))
    result=AgentService(repo,None).run_turn(pending,cap)
    assert result['status']=='failed' and result['response']['code']=='agent_not_configured'
    assert repo.get_job(pending['jobId'])['status']=='failed'
    class Broken(FakeAgent):
        def respond(self,*_): raise TimeoutError('must never expose provider secret')
    pending=repo.create_agent_turn(scene['project']['id'],cap,turn_body(scene))
    result=AgentService(repo,Broken({})).run_turn(pending,cap)
    assert result['status']=='outcome_unknown'
    assert 'secret' not in str(result)
    assert repo.get_job(pending['jobId'])['status']=='outcome_unknown'


def test_policy_zen_applicability_is_separate_from_machine_result():
    items=templates()
    if isinstance(items,dict):items=items['items']
    for item in items:
        for state in ('unknown','not_applicable','applicable'):
            result=execute_jdm(item['jdm'],state,[])
            assert result['applicability']==state


def test_policy_agent_proposal_is_pinned_and_uses_normal_policy_writer(repo):
    cap, scene = project(repo)
    pid = scene['project']['id']
    service = PolicyService(PostgresPolicyRepository(repo))
    template = templates()[0]
    policy = service.create_policy(pid, cap, {'requestId': identity(), **template})
    proposal = {'kind': 'proposal', 'proposalType': 'policy', 'message': 'Proposed check',
                'jdm': template['jdm'], 'tests': template['tests'],
                'limitations': ['Check visible walking surface hazards.'], 'sourceRefs': []}
    provider = FakeAgent(proposal)
    body = {**turn_body(scene), 'policyId': policy['policy']['id'],
            'policyRevisionId': policy['revision']['id']}
    pending = repo.create_agent_turn(pid, cap, body)
    result = AgentService(repo, provider, service).run_turn(pending, cap)
    assert result['status'] == 'succeeded'
    assert provider.calls[0][1]['policy']['id'] == policy['revision']['id']
    assert service.get_policy(policy['policy']['id'])['policy']['activeRevisionId'] is None
    proposed = result['response']
    edit = {key: proposed[key] for key in ('basePolicyRevisionId', 'jdm', 'tests', 'limitations', 'sourceRefs')}
    saved = service.save_revision(pid, policy['policy']['id'], cap,
                                  {'requestId': identity(), 'agentTurnId': result['id'], **edit})
    assert saved['parentRevisionId'] == policy['revision']['id']
    assert saved['agentTurnId'] == result['id']
    assert service.get_policy(policy['policy']['id'])['policy']['activeRevisionId'] is None
    assert repo.get_project(pid)['revision']['id'] == scene['revision']['id']
