"""Request bodies of the v1 contract for tests and smoke runs: the synthetic SAM 3D input of modal_apps/sam3d_research.debug
(a red square on a grey plane two units away, its pointmap in the PyTorch3D camera) and a two-frame geometry-mvs cell."""
import base64
import hashlib
import io
import json

import numpy as np
from PIL import Image


def png(array) -> bytes:
    buf = io.BytesIO(); Image.fromarray(array).save(buf, format='PNG'); return buf.getvalue()


def npz(**arrays) -> bytes:
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays); return buf.getvalue()


def npy(array) -> bytes:
    buf = io.BytesIO(); np.save(buf, array); return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


def sha256_of(*parts) -> str:
    """The client-side idempotency key: sha256 over the canonical input bytes (here: every file + the JSON of the rest)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()


def sam3d_body(size=256, seed=42, salt='') -> dict:
    rgb = np.full((size, size, 3), 90, np.uint8); a, b = size * 88 // 256, size * 168 // 256
    rgb[a:b, a:b] = (200, 60, 40)
    mask = np.zeros((size, size), bool); mask[a:b, a:b] = True
    v, u = np.indices((size, size), dtype=np.float64)
    z = np.full((size, size), 2.); z[a:b, a:b] = 1.8
    c = (size - 1) / 2
    pointmap = np.stack([-(u - c) / 250 * z, -(v - c) / 250 * z, z], -1).astype(np.float32)
    image, mask_png, pm = png(rgb), png(mask.astype(np.uint8) * 255), npz(pointmap=pointmap)
    return dict(input_sha256=sha256_of(image, mask_png, pm, dict(seed=seed, salt=salt)), seed=seed,
                image_b64=b64(image), mask_b64=b64(mask_png), pointmap_npz_b64=b64(pm))


def geometry_body(cell='090', n=2, size=518, x0=63, width=392, salt='') -> dict:
    rng = np.random.default_rng(0); frames, parts = [], []
    for k in range(n):
        rgb = np.zeros((size, size, 3), np.uint8); rgb[:, x0:x0 + width] = rng.integers(0, 255, (size, width, 3), dtype=np.uint8)
        alpha = np.zeros((size, size), bool); alpha[:, x0:x0 + width] = True
        canon, al = png(rgb), npy(alpha); parts += [canon, al]
        frames.append(dict(frame_id=f'frame_{k + 1:04d}', canonical_png_b64=b64(canon), alpha_npy_b64=b64(al)))
    options = {'start': 'da3-base', 'roma': 'outdoor', 'pairs': 'all'}
    return dict(input_sha256=sha256_of(*parts, dict(cell=cell, options=options, salt=salt)), cell=cell, frames=frames, options=options)


if __name__ == '__main__':  # the smoke script: a sam3d body on stdout
    print(json.dumps(sam3d_body(size=64)))
