"""Offline recovery for saved full-image RGB resizes; never a runtime fallback.

Run only on a copy of a historical run. Every RGB channel must reproduce the
saved model input within one uint8 quantization level, or recovery is rejected.
"""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ehs_spatial.providers.map_anything import decode_encoded_array
from ehs_spatial.report_workspace import write_json


def recover(run: Path, frame_id: str = 'frame_0001'):
    provider = run / 'geometry/provider' / (frame_id + '.json')
    raw = provider.read_bytes()
    payload = json.loads(raw)
    if 'alpha_mask' in payload:
        raise ValueError('Recorded mapping already exists; do not overwrite it')
    originals = sorted((run / 'input').glob('image_*'))
    original = originals[int(frame_id.removeprefix('frame_')) - 1]
    rgb = ImageOps.exif_transpose(Image.open(original)).convert('RGB')
    saved = decode_encoded_array(payload['image'])
    if saved.dtype != np.uint8 or saved.ndim != 3 or saved.shape[2] != 3:
        raise ValueError('Expected saved RGB uint8 model input')
    h, w = saved.shape[:2]
    if payload['original_image'] != {'width': rgb.width, 'height': rgb.height}:
        raise ValueError('Source dimensions disagree')
    replay = np.asarray(rgb.resize((w, h), Image.Resampling.BICUBIC))
    error = abs(replay.astype(np.int16) - saved.astype(np.int16))
    if error.max() > 1 or not np.array_equal(saved, np.asarray(Image.open(run / 'geometry/frames' / frame_id / 'canonical.png'))):
        raise ValueError('RGB replay failed; no source-to-depth mapping can be recovered')
    sx, sy = w / rgb.width, h / rgb.height
    proof = {'method': 'full-image-bicubic-rgb-replay-v1', 'maxChannelErrorUint8': int(error.max()),
             'meanChannelErrorUint8': float(error.mean()), 'pixelChannelsChecked': int(error.size),
             'sourceImageSha256': hashlib.sha256(original.read_bytes()).hexdigest(),
             'originalProviderSha256': hashlib.sha256(raw).hexdigest()}
    payload['alpha_mask'] = {'shape': [h, w], 'dtype': 'uint8', 'data': base64.b64encode(np.ones((h, w), np.uint8).tobytes()).decode()}
    payload['input_mask_transform'] = {'resized_shape_hw': [h, w], 'crop_xyxy': [0, 0, w, h],
        'input_to_canonical_pixel_centres': [[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]], 'recovery': proof}
    # ponytail: supports verified full-image resize only. Cropped/padded or
    # differently filtered historical inputs require their actual preprocessing.
    backup = provider.with_suffix('.before-mapping.json')
    with backup.open('xb') as stream:
        stream.write(raw)
    write_json(provider, payload)
    return proof


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--frame-id', default='frame_0001')
    args = parser.parse_args()
    print(json.dumps(recover(args.run.resolve(), args.frame_id)))
