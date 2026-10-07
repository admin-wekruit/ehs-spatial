"""CPU-only check: PYTHONPATH=. python scripts/check_workcell_depth_resolution.py."""
import ast
import base64
from pathlib import Path
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image


def check():
    # Load only the pure metadata wrapper: no Modal image build, Torch, or weights.
    path = Path(__file__).resolve().parents[1] / 'modal_apps/mapanything_app.py'
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 ('_encode_array', 'load_images_with_metadata', '_relative_depth_gradient')]
    scope = {'base64': base64, 'CODE_REV': 'test'}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    image_module, crop_module = ModuleType('mapanything.utils.image'), ModuleType('mapanything.utils.cropping')
    requests = []

    def resize(image, shape):
        factor = max(shape[0] / image.width, shape[1] / image.height)
        return (image.resize(tuple(np.rint(np.array(image.size) * factor).astype(int))),)

    def load(paths, **kwargs):
        requests.append(kwargs)
        width, height = kwargs.get('size') or (392, 518)
        rows = []
        for filename in paths:
            original = Image.open(filename).convert('RGB')
            resized = resize(original, (width, height))[0]
            left, top = (resized.width - width) // 2, (resized.height - height) // 2
            raster = np.asarray(resized.crop((left, top, left + width, top + height)))
            rows.append({'true_shape': [[height, width]], 'img': [raster / 255.], 'data_norm_type': ['none']})
        return rows

    image_module.load_images, image_module.rgb = load, lambda array, _: array
    crop_module.rescale_image_and_other_optional_info = resize
    crop_module.crop_image_and_other_optional_info = lambda image, box: (image.crop(box),)
    with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {
        'mapanything.utils.image': image_module, 'mapanything.utils.cropping': crop_module}):
        photo = Path(temp) / 'source.jpg'
        Image.fromarray(np.random.default_rng(0).integers(0, 256, (1601, 1203, 3), dtype=np.uint8)).save(photo)
        metadata = []
        for dimensions in (None, (784, 1036)):
            _, rows = scope['load_images_with_metadata']([str(photo)], fixed_size=dimensions)
            row = rows[0]; metadata.append(row)
            A = np.array(row['input_mask_transform']['input_to_canonical_pixel_centres'])
            rh, rw = row['input_mask_transform']['resized_shape_hw']
            left, top, _, _ = row['input_mask_transform']['crop_xyxy']
            uv = np.array([723.5, 401.75, 1.])
            expected = [(uv[0]+.5)*rw/1203-.5-left, (uv[1]+.5)*rh/1601-.5-top]
            assert np.allclose((A @ uv)[:2], expected)
            raw_K = np.array([[1400., 0, 600.], [0, 1400., 800.], [0, 0, 1.]])
            camera_K = A @ raw_K
            assert np.allclose(np.linalg.inv(A) @ camera_K, raw_K), 'Intrinsics must return to exact raw pixels'
            assert row['input_mask_transform']['rgb_replay_pixel_exact']
        assert requests == [{}, {'resize_mode': 'fixed_size', 'size': (784, 1036)}]
        assert metadata[0]['_canonical_rgb'].shape == (518, 392, 3)
        assert metadata[1]['_canonical_rgb'].shape == (1036, 784, 3)
        for invalid in ((783, 1036), (784, 0), (True, 1036), (784,), '1036'):
            try:
                scope['load_images_with_metadata']([str(photo)], fixed_size=invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Invalid dimensions accepted: {invalid}')
    runner = path.with_name('workcell_depth_resolution.py')
    tree = ast.parse(runner.read_text())
    fetch = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'download_artifacts')
    fetch_scope = {'Path': Path, 'shutil': SimpleNamespace(disk_usage=lambda _: SimpleNamespace(free=7*2**30))}
    exec(compile(ast.Module(body=[fetch], type_ignores=[]), str(runner), 'exec'), fetch_scope)
    try:
        fetch_scope['download_artifacts']({'files': [{'bytes': 100}]}, Path('/tmp'))
    except ValueError as exc:
        assert '>8 GiB free' in str(exc)
    else:
        raise AssertionError('Low-disk download must reject before reading the Volume')
    geometry = path.parents[1] / 'scripts/workcell_photo_geometry.py'
    tree = ast.parse(geometry.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_unit', '_edge_depth_neighborhoods')]
    stencil = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(geometry), 'exec'), stencil)
    raw = np.array([[500., 1000.], [750., 1250.], [1000., 1500.]])
    recovered = []
    for scale, crop in ((518/4032, [0., -2.]), (1036/4032, [-3., -5.])):
        affine = np.array([[scale, 0, (scale-1)/2+crop[0]], [0, scale, (scale-1)/2+crop[1]], [0, 0, 1.]])
        neighborhoods = stencil['_edge_depth_neighborhoods'](raw, affine, (4032, 3024))
        back = np.c_[neighborhoods.reshape(-1, 2), np.ones(neighborhoods.size//2)] @ np.linalg.inv(affine).T
        recovered.append(back[:, :2].reshape(neighborhoods.shape))
        canonical = (np.c_[raw, np.ones(len(raw))] @ affine.T)[:, :2]
        assert np.allclose(np.linalg.norm(neighborhoods[0] - canonical, axis=1), 2*scale*4032/518)
    assert np.allclose(*recovered), '1x and 2x must sample the same original-image edge neighborhoods despite crop'
    assert np.allclose(np.linalg.norm(recovered[0][0]-raw, axis=1), 2*4032/518)
    raw_gradient = np.array([.2, .15])
    normalized_gradients = []
    for factor, crop in ((1., [0., -2.]), (2., [-3., -5.])):
        A = np.array([[factor*.1296, 0, crop[0]], [0, factor*.1294, crop[1]], [0, 0, 1.]])
        yy, xx = np.indices((24, 30))
        raw_uv = (np.c_[xx.ravel(), yy.ravel(), np.ones(xx.size)] @ np.linalg.inv(A).T)[:, :2]
        depth = (10 + raw_uv @ raw_gradient).reshape(xx.shape)
        metadata = {'original_image': {'height': 4032, 'width': 3024},
                    'input_mask_transform': {'input_to_canonical_pixel_centres': A.tolist()}}
        relative = scope['_relative_depth_gradient'](depth, metadata)
        expected = np.linalg.norm(raw_gradient)*4032/518/depth
        assert np.allclose(relative, expected), 'Chain rule must preserve raw gradients through anisotropic resize and crop'
        assert np.array_equal(relative < .08, expected < .08), 'No resolution-dependent acceptance threshold'
        normalized_gradients.append(relative*depth)
    assert np.allclose(*normalized_gradients)
    print('PASS: exact raster/K, same raw edge stencil and depth gradient at 1x/2x, crop, invalid dimensions, disk gate')


if __name__ == '__main__':
    check()
