"""Frozen-data alignment and COLMAP/LIMAP controls; run heavy work on Modal."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation
import trimesh

from scripts.workcell_photo_oneshot import _array, _frame


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def alignment(root, out):
    from fast_report.x7 import light, rays, refine, Caster, score_view
    from ehs_spatial.platform.scene_measurements import fitted_bend
    from ehs_spatial.platform.contracts import PlatformError
    out.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(root / 'guard-multi.glb', force='mesh')
    saved = json.loads((root / 'guard-placement.json').read_text())
    mesh.apply_transform(np.linalg.inv(saved['transform']))
    z = np.load(root / 'guard-input.npz')
    views = []
    for i in range(1, 5):
        depth = z[f'v{i}_depth'][::2, ::2]
        K = z[f'v{i}_K'].copy(); K[:2] /= 2
        views.append({'rays': rays(K, z[f'v{i}_c2w'], depth.shape[1], depth.shape[0]),
                      'target': z[f'v{i}_mask'][::2, ::2] > 0, 'depth': depth})
    v, f, _ = light(mesh.vertices, mesh.faces, mesh.visual.vertex_colors[:, :3])
    transform, result = refine(v, f, [views[i-1] for i in saved['generationViews']], uniform_scale=True)
    scales = np.linalg.svd(transform[:3, :3], compute_uv=False)
    assert np.ptp(scales) < 1e-10, 'Similarity alignment must preserve intrinsic angles'
    result.update(transform=transform.tolist(), singularValues=scales.tolist(), generationViews=saved['generationViews'],
                  sourceChecks=[score_view(Caster(v, f), transform, view) for view in views],
                  basis='same generated mesh and masks; uniform scale; model angles, not physical truth')
    parts = json.loads((root / 'guard-partition.json').read_text())['parts']
    result['boards'] = []
    for part in parts:
        m = mesh.submesh([part['sourceFaceIndices']], append=True, repair=False)
        before = None
        try: before = float(fitted_bend(m.triangles)['value'])
        except PlatformError: pass
        m.apply_transform(transform)
        m.export(out / part['file'])
        after = None
        try: after = float(fitted_bend(m.triangles)['value'])
        except PlatformError: pass
        if before is not None:
            assert abs(after-before) < 1e-5, 'Similarity transform changed the fitted fold'
        result['boards'].append({'side': part['side'], 'beforeDeg': before, 'angleDeg': after, 'file': part['file']})
    mesh.apply_transform(transform); mesh.export(out / 'guard-multi.glb')
    write(out / 'results.json', result)
    return result


def _export_reconstruction(recon, originals, out, *, native=False):
    """Rigid+uniform gauge alignment, never a coordinate-wise deformation."""
    ids = sorted(recon.reg_image_ids())
    poses, targets = [], []
    for image_id in ids:
        im = recon.images[image_id]
        p = np.eye(4); p[:3] = im.cam_from_world().inverse().matrix()
        poses.append(p); targets.append(originals[im.name]['pose'])
    a = np.array([p[:3, 3] for p in poses]); b = np.array([p[:3, 3] for p in targets])
    R, scale, t = np.eye(3), 1., np.zeros(3)
    if not native:
        # Fit orientation from camera orientations; scale/translation from centers.
        rotations = [target[:3, :3] @ pose[:3, :3].T for pose, target in zip(poses, targets)]
        U, _, V = np.linalg.svd(np.sum(rotations, axis=0))
        correction = np.eye(3); correction[-1, -1] = np.linalg.det(U @ V)
        R = U @ correction @ V
        ac, bc = a-a.mean(0), b-b.mean(0)
        scale = float(np.sum((ac @ R.T)*bc) / np.sum(ac**2))
        if not np.isfinite(scale) or scale <= 0: raise ValueError('Invalid COLMAP/native similarity gauge')
        t = b.mean(0)-scale*R@a.mean(0)
    frames = []
    for image_id, p in zip(ids, poses):
        im = recon.images[image_id]; cam = recon.cameras[im.camera_id]
        K = cam.calibration_matrix().copy(); K[:2, 2] -= .5
        K = np.linalg.inv(originals[im.name]['pixelTransform']) @ K
        p[:3, :3] = R @ p[:3, :3]; p[:3, 3] = scale*R@p[:3, 3]+t
        frames.append({'photo': originals[im.name]['photo'], 'K': K.tolist(), 'pose': p.tolist()})
    tracks = []
    for point_id, point in recon.points3D.items():
        obs = []
        for e in point.track.elements:
            im = recon.images[e.image_id]
            uv = im.points2D[e.point2D_idx].xy - .5
            uv = np.linalg.inv(originals[im.name]['pixelTransform']) @ np.r_[uv, 1.]
            obs.append({'photo': originals[im.name]['photo'], 'uv': uv[:2].tolist()})
        tracks.append({'id': point_id, 'xyz': (scale*R@point.xyz+t).tolist(), 'observations': obs,
                       'reprojectionErrorControlPixels': float(point.error)})
    prefix = 'native-' if native else ''
    write(out/(prefix+'cameras.json'), {'frames': frames, 'worldFrame': 'COLMAP incremental native' if native else 'MapAnything native',
          'gaugeScale': scale, 'gaugeRotation': R.tolist(), 'gaugeTranslation': t.tolist(),
          'cameraCenterAlignmentRmsNative': None if native else float(np.sqrt(np.mean(np.sum((scale*a@R.T+t-b)**2, axis=1)))),
          'gaugeBasis': 'unaltered incremental reconstruction' if native else 'mean camera orientation plus least-squares uniform scale and translation; remaining pose differences retained'})
    write(out/(prefix+'tracks.json'), {'tracks': tracks, 'basis': 'COLMAP SIFT tracks; all-view fit, not held-out truth'})


def colmap(root, out, sources, *, square_pixels=False):
    import pycolmap
    start = time.monotonic(); out.mkdir(parents=True, exist_ok=True)
    images = out/'images'; images.mkdir()
    database = out/'database.db'
    # Native source resolution is retained within the exact neural crop; 3x is
    # bounded at 1554 px, not interpolation of the 518 px prediction image.
    S = np.array([[3., 0, 1.], [0, 3., 1.], [0, 0, 1.]])
    originals = {}; camera_lines = []; image_lines = []; source_frames = []
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = 8
    extraction.max_image_size = 1800
    extraction.sift.max_num_features = 12000
    for i, source in enumerate(sources, 1):
        frame = _frame(root, i)
        A = np.array(frame['input_mask_transform']['input_to_canonical_pixel_centres'])
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(source)).convert('RGB'))
        h, w = frame['image']['shape'][:2]
        pixels = S
        size = (w*3, h*3)
        if square_pixels:
            # Fit square pixels in the supplied JPEG, before the neural raster's
            # anisotropic resize. Each view still has its own unknown focal length.
            factor = 1554 / max(rgb.shape[:2])
            raw_resize = np.array([[factor, 0, (factor-1)/2], [0, factor, (factor-1)/2], [0, 0, 1.]])
            pixels = raw_resize @ np.linalg.inv(A)
            size = tuple(np.ceil(np.array(rgb.shape[1::-1])*factor).astype(int))
        crop = cv2.warpAffine(rgb, (pixels@A)[:2], size, flags=cv2.INTER_AREA)
        name = f'photo-{i}.png'; Image.fromarray(crop).save(images/name)
        K = pixels @ _array(frame['intrinsics']); K[:2, 2] += .5
        pose = _array(frame['camera_poses']); originals[name] = {'photo': i, 'pose': pose, 'pixelTransform': pixels}
        source_frames.append({'name': name, 'photo': i, 'pose': pose.tolist(), 'pixelTransform': pixels.tolist(),
                              'inputToCanonicalPixelCentres': A.tolist(), 'sourceSha256': hashlib.sha256(Path(source).read_bytes()).hexdigest()})
        camera_model = 'SIMPLE_PINHOLE' if square_pixels else 'PINHOLE'
        parameters = [float(np.sqrt(K[0,0]*K[1,1])), K[0,2], K[1,2]] if square_pixels else [K[0,0], K[1,1], K[0,2], K[1,2]]
        reader = pycolmap.ImageReaderOptions(camera_model=camera_model, camera_params=','.join(map(str, parameters)))
        pycolmap.extract_features(database, images, image_names=[name], camera_mode=pycolmap.CameraMode.PER_IMAGE,
                                 reader_options=reader, extraction_options=extraction, device=pycolmap.Device.cpu)
    with pycolmap.Database.open(database) as db:
        for im in db.read_all_images():
            cam = db.read_camera(im.camera_id)
            camera_lines.append(f'{cam.camera_id} {camera_model} {cam.width} {cam.height} '+' '.join(map(str, cam.params)))
            world_to_cam = np.linalg.inv(originals[im.name]['pose'])
            q = Rotation.from_matrix(world_to_cam[:3,:3]).as_quat()[[3,0,1,2]]
            image_lines.extend([f'{im.image_id} '+' '.join(map(str, [*q, *world_to_cam[:3,3]]))+f' {cam.camera_id} {im.name}', ''])
    pycolmap.match_exhaustive(database, device=pycolmap.Device.cpu)
    write(out/'source-frames.json', {'frames': source_frames, 'cameraModel': camera_model,
                                    'poseUse': 'Gauge alignment after independent mapping only; never an incremental initialization.'})
    initialized = out/'initialized'; initialized.mkdir()
    (initialized/'cameras.txt').write_text('\n'.join(camera_lines)+'\n')
    (initialized/'images.txt').write_text('\n'.join(image_lines)+'\n')
    (initialized/'points3D.txt').write_text('')
    recon = pycolmap.Reconstruction(initialized)
    tri = out/'triangulated'; tri.mkdir()
    opts = pycolmap.IncrementalPipelineOptions()
    opts.num_threads = 8; opts.triangulation.ignore_two_view_tracks = False
    recon = pycolmap.triangulate_points(recon, database, images, tri, options=opts)
    before = {'points': recon.num_points3D(), 'meanErrorPx': recon.compute_mean_reprojection_error()}
    if recon.num_points3D() < 12: raise ValueError(f'Too few genuine COLMAP tracks: {before}')
    ba = pycolmap.BundleAdjustmentOptions()
    ba.refine_focal_length = True; ba.refine_principal_point = False; ba.refine_extra_params = False
    pycolmap.bundle_adjustment(recon, options=ba)
    model = out/'model'; model.mkdir(); recon.write(model)
    recon.export_PLY(out/'sparse.ply')
    _export_reconstruction(recon, originals, out)
    result = {'status': 'completed', 'version': pycolmap.__version__, 'seconds': time.monotonic()-start,
              'beforeBa': before, 'afterBa': {'points': recon.num_points3D(), 'meanErrorPx': recon.compute_mean_reprojection_error()},
              'registeredImages': recon.num_reg_images(), 'cameraModel': camera_model,
              'squareRawPixelsAssumed': square_pixels,
              'basis': 'SIFT matches; triangulation from MapAnything initial poses; camera+point BA with fixed principal points; no lens calibration truth'}
    write(out/'results.json', result)
    return result


def _incremental_worker(control, out):
    """Fresh pose estimation from the same feature DB; no initial reconstruction."""
    import pycolmap

    control, out = Path(control), Path(out)
    metadata = json.loads((control/'source-frames.json').read_text())
    if metadata['cameraModel'] != 'SIMPLE_PINHOLE':
        raise ValueError('Independent control requires the same raw-square-pixel camera model')
    originals = {row['name']: {**row, 'pose': np.asarray(row['pose']), 'pixelTransform': np.asarray(row['pixelTransform'])}
                 for row in metadata['frames']}
    expected = {f'photo-{p}.png' for p in range(1, 5)}
    if set(originals) != expected:
        raise ValueError('Independent control requires exactly the four source images')
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 8
    options.min_model_size = 4
    options.ba_refine_focal_length = True
    options.ba_refine_principal_point = False
    options.ba_refine_extra_params = False
    options.triangulation.ignore_two_view_tracks = False
    write(out/'options.json', {'numThreads': 8, 'minModelSize': 4, 'refineFocalLength': True,
                              'refinePrincipalPoint': False, 'refineExtraParams': False,
                              'ignoreTwoViewTracks': False, 'inputReconstruction': None,
                              'otherOptions': 'pycolmap defaults', 'version': pycolmap.__version__})
    models = out/'native-models'; models.mkdir()
    reconstructions = pycolmap.incremental_mapping(control/'database.db', control/'images', models, options=options)
    records, eligible = [], []
    for index, recon in sorted(reconstructions.items()):
        names = sorted(recon.images[i].name for i in recon.reg_image_ids())
        records.append({'model': index, 'registeredImages': names, 'points': recon.num_points3D(),
                        'meanReprojectionErrorControlPx': recon.compute_mean_reprojection_error() if recon.num_points3D() else None})
        if set(names) == expected and recon.num_points3D() >= 12:
            eligible.append((recon.num_points3D(), index, recon))
    result = {'status': 'unsupported', 'reason': 'No single incremental model registered all four source images with sufficient points',
              'models': records, 'initialPoseSource': 'None; incremental mapping from matched RGB images',
              'initialIntrinsicsSource': 'Same DB SIMPLE_PINHOLE focal/principal point initialization as the neural-pose control',
              'camerasFile': None, 'tracksFile': None, 'version': pycolmap.__version__}
    if eligible:
        _, index, recon = max(eligible, key=lambda row: (row[0], -row[1]))
        _export_reconstruction(recon, originals, out, native=True)
        _export_reconstruction(recon, originals, out)
        result.update(status='completed', reason=None, selectedModel=index,
                      camerasFile='cameras.json', tracksFile='tracks.json', nativeCamerasFile='native-cameras.json',
                      nativeTracksFile='native-tracks.json',
                      scope='Four-view registration is not physical calibration. Export gauge changes coordinates only; old depth/models are not transported.')
    write(out/'results.json', result)


def incremental_colmap(control, out, *, max_seconds=180):
    """Bounded same-DB pose-initialization control; preserves partial maps/logs."""
    control, out = Path(control).resolve(), Path(out).resolve()
    if isinstance(max_seconds, bool) or not 1 <= max_seconds <= 600:
        raise ValueError('Independent mapping needs a 1–600 second bound')
    if not all((control/name).is_file() for name in ('database.db', 'source-frames.json')):
        raise ValueError('Run colmap(square_pixels=True) once and retain its same feature DB')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    database_sha = hashlib.sha256((control/'database.db').read_bytes()).hexdigest()
    command = [sys.executable, '-c', 'import sys; from scripts.workcell_guard_controls import _incremental_worker; _incremental_worker(sys.argv[1],sys.argv[2])', str(control), str(out)]
    try:
        with (out/'incremental.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=max_seconds)
        if completed.returncode:
            raise RuntimeError(f'Incremental worker exit {completed.returncode}; see incremental.log')
        result = json.loads((out/'results.json').read_text())
    except (subprocess.TimeoutExpired, RuntimeError) as error:
        result = {'status': 'unsupported', 'reason': str(error), 'camerasFile': None, 'tracksFile': None,
                  'partialModels': sorted(p.name for p in (out/'native-models').glob('*') if p.is_dir())}
    result.update(seconds=time.monotonic()-started, maxSeconds=max_seconds, databaseSha256=database_sha,
                  sourceMetadataSha256=hashlib.sha256((control/'source-frames.json').read_bytes()).hexdigest(),
                  officialApi='https://colmap.github.io/pycolmap/pycolmap.html#pycolmap.incremental_mapping')
    write(out/'results.json', result)
    return result


def limap_control(colmap_dir, out, root, sources, *, guard_only=False):
    import pycolmap
    import pytlsd
    import limap.geometry
    import limap.scene
    import limap.sfm
    import limap.estimators.bundle_adjustment as ba
    start = time.monotonic(); out.mkdir(parents=True, exist_ok=True)
    recon = pycolmap.Reconstruction(colmap_dir/'model')
    originals, lines = {}, {}
    canonical = {row['photo']: np.asarray(row['K']) for row in json.loads((colmap_dir/'cameras.json').read_text())['frames']}
    for image_id, im in recon.images.items():
        i = int(im.name.split('-')[1].split('.')[0])
        camera = recon.cameras[im.camera_id]
        K = camera.calibration_matrix().copy(); K[:2, 2] -= .5
        S = K @ np.linalg.inv(canonical[i])
        frame = _frame(root, i)
        A = np.array(frame['input_mask_transform']['input_to_canonical_pixel_centres'])
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(sources[i-1])).convert('RGB'))
        h, w = frame['image']['shape'][:2]
        crop = cv2.warpAffine(rgb, (S@A)[:2], (camera.width,camera.height), flags=cv2.INTER_AREA)
        segments = pytlsd.lsd(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY))
        # Bound exhaustive matching; keep real lines longer than 25 control pixels.
        lengths = np.linalg.norm(segments[:,:2]-segments[:,2:4], axis=1)
        keep = np.flatnonzero(lengths >= 25)
        if guard_only:
            mask = np.load(root/'guard-input.npz')[f'v{i}_mask'] > 0
            uv = (segments[:,:2]+segments[:,2:4])/2
            uv = (np.c_[uv,np.ones(len(uv))]@np.linalg.inv(S).T)[:,:2]
            xy = np.rint(uv).astype(int)
            inside = (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
            hits = np.zeros(len(segments),bool)
            hits[inside] = mask[xy[inside,1],xy[inside,0]]
            keep = keep[hits[keep]]
        keep = keep[np.argsort(-lengths[keep])[:500]]
        # LSD uses pixel centers at integer coordinates; COLMAP uses +0.5.
        lines[image_id] = segments[keep,:4]+.5
        originals[im.name] = {'photo':i,'pose':_array(frame['camera_poses']),'pixelTransform':S}
    dbpath = out/'structure_database.db'
    limap.scene.create_structure_db(dbpath)
    with limap.scene.StructureDatabase.open(dbpath) as db:
        limap.scene.initialize_structures_from_reconstruction(db, recon)
        limap.scene.import_line_detections(db, limap.geometry.get_all_lines_2d(lines))
    ids = list(recon.images)
    opts = limap.sfm.GlobalLineTriangulationOptions()
    opts.num_threads = 8
    if guard_only: opts.min_visible_views = 2
    holistic = limap.sfm.pipelines.global_line_triangulation(recon, dbpath, opts, {i:[j for j in ids if j!=i] for i in ids})
    raw = out/'triangulated_model'; raw.mkdir(); holistic.write(raw); holistic.write_text(raw)
    line_count = holistic.structure_recon.num_lines3D()
    if line_count == 0:
        result = {'status':'unsupported','seconds':time.monotonic()-start,'lines3D':0,
                  'scope':'guard-only' if guard_only else 'scene','reason':'no supported 3D lines',
                  'detectedLinesPerPhoto':{str(originals[recon.images[i].name]['photo']):len(v) for i,v in lines.items()}}
        write(out/'results.json',result)
        return result
    options = ba.PointLineBundleAdjustmentOptions()
    options.refine_focal_length = True
    options.refine_principal_point = False; options.refine_extra_params = False
    config = ba.PointLineBundleAdjustmentConfig()
    for image_id in ids: config.add_image(image_id)
    for line_id in holistic.structure_recon.line3D_ids(): config.add_variable_line(line_id)
    config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    solver = ba.create_point_line_bundle_adjuster(options, config, holistic)
    summary = solver.solve()
    model = out/'model'; model.mkdir(); holistic.write(model); holistic.write_text(model)
    _export_reconstruction(holistic.point_recon, originals, out)
    result = {'status':'completed','version':'2.0.0','seconds':time.monotonic()-start,
              'detectedLinesPerPhoto':{str(originals[recon.images[i].name]['photo']):len(v) for i,v in lines.items()},
              'lines3D':line_count,'scope':'guard-only' if guard_only else 'scene','pointLineBaSummary':str(summary),
              'basis':'LIMAP native LSD line triangulation and point+line camera BA, existing COLMAP tracks; no CAD angle prior'}
    write(out/'results.json',result)
    return result
