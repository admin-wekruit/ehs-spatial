"""The browser and server consume this same absolute-pose family fixture."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.identity import model_family
from ehs_spatial.platform.repository import apply_operations
from ehs_spatial.platform.spatial import transform_matrix

FIXTURE = json.loads((Path(__file__).parents[1] / 'web/checks/fixtures/model-family.json').read_text())


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['name'])
def test_shared_model_family_fixture(case):
    document = deepcopy(FIXTURE['document'])
    for entity in document['entities']:
        assert [member['id'] for member in model_family(document, entity['id'])] == FIXTURE['families'][entity['id']]
        entity.update(deepcopy(case['entityOverrides'].get(entity['id'], {})))
    before = deepcopy(document)
    if 'errorCode' in case:
        with pytest.raises(PlatformError, match=case['errorCode']):
            apply_operations(document, case['operations'], base_revision_id=FIXTURE['baseRevisionId'])
    else:
        after, _ = apply_operations(document, case['operations'], base_revision_id=FIXTURE['baseRevisionId'])
        assert after['observations'] == before['observations'] and after['assets'] == before['assets']
        for entity in after['entities']:
            expected = case['expected'][entity['id']]
            if expected['matrix'] is None:
                assert entity['currentModelTransform'] is None
            else:
                assert np.allclose(transform_matrix(entity['currentModelTransform']).T.flatten(), expected['matrix'], atol=1e-6)
            assert entity.get('visible') == expected['visible'] and entity['label'] == expected['label']
            assert {rep['id']: {'material': rep.get('material'), 'placementState': rep['placementState']} for rep in entity['representations']} == expected['representations']
            if 'expectedRelations' in case:
                assert {key: entity.get(key, [] if key == 'lineage' else None) for key in ('parentEntityId', 'partRelation', 'lineage')} == case['expectedRelations'][entity['id']]
    assert document == before
