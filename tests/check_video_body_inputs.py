"""No provider calls: changed resume inputs must leave paid-result evidence intact."""
from pathlib import Path
import sys, tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_video_body_models import save_inputs

with tempfile.TemporaryDirectory() as temporary:
    folder = Path(temporary)
    image = np.zeros((8, 8, 3), np.uint8)
    mask = np.ones((8, 8), bool)
    source = {'sourceFrame': 1, 'entityId': 'person', 'source_video_sha256': 'a' * 64}
    save_inputs(folder, image, mask, source)
    original = {p.name: p.read_bytes() for p in folder.iterdir()}
    save_inputs(folder, image, mask, source)
    for modified in [(image + 1, mask, source), (image, ~mask, source), (image, mask, {**source, 'entityId': 'other'})]:
        try:
            save_inputs(folder, *modified)
            raise AssertionError('Changed body input was accepted')
        except ValueError as error:
            assert str(error) == 'Resumed body input differs'
        assert {p.name: p.read_bytes() for p in folder.iterdir()} == original
print('PASS: identical inputs resume; image, mask and identity changes never overwrite cached evidence')
