"""Render frozen object/view silhouette evidence; never optimize or infer assets.

Run with the project Python and --root outputs/candidate-evaluation/lucida-replica-01.
The evaluation grid and scores come from the completed assembly, not a new protocol.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import trimesh

from assemble_lucida_scene import cast_depth, digest, load_view, score_view


def projected_mask(mesh, transform, view):
    depth = cast_depth(mesh, transform, view['rays'])
    # Identical foreground rule to score_view: target pixels are never occluded
    # away to hide a misplaced asset. Only observed non-target foreground hides it.
    hidden = (~view['target']) & (view['depth'] > 0) & (depth > view['depth'] * 1.04)
    return np.isfinite(depth) & (depth > 0) & ~hidden


def check_score(mesh, transform, view, saved):
    score = score_view(mesh, transform, view)
    visible = projected_mask(mesh, transform, view)
    target = view['target']
    drawn_iou = float((visible & target).sum() / max(1, (visible | target).sum()))
    assert abs(drawn_iou - score['visible_iou']) < 1e-12, 'Drawing/scoring foreground rules differ'
    assert abs(drawn_iou - saved['visible_iou']) < 1e-12, (
        f"{view['frame_id']}: recomputed IoU {drawn_iou} != recorded {saved['visible_iou']}")
    for key in ['target_pixels', 'visible_predicted_pixels', 'depth_comparison_pixels']:
        assert score[key] == saved[key], f"{view['frame_id']}: {key} changed"
    return visible


def panel(rgb, target, predicted=None):
    height, width = target.shape
    pixels = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA).copy()
    if predicted is not None:
        pixels[predicted] = np.round(.45 * pixels[predicted] + .55 * np.array([246, 123, 57])).astype(np.uint8)
    def edge(mask):
        return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if predicted is not None:
        pixels[edge(predicted)] = [255, 129, 64]
    pixels[edge(target)] = [52, 245, 178]
    return Image.fromarray(pixels).resize((width * 2, height * 2), Image.Resampling.NEAREST)


def render(root):
    root = root.resolve()
    aggregate_path = root / 'result/comparisons.json'
    aggregate_hash = digest(aggregate_path)
    aggregate = json.loads(aggregate_path.read_text())
    evidence_path = root / 'evidence/objects.json'
    evidence_hash = digest(evidence_path)
    evidence = {o['object_id']: o for o in json.loads(evidence_path.read_text())['objects']}
    comparisons = aggregate['objects']
    assert len({o['object_id'] for o in comparisons}) == len(comparisons), 'Duplicate object IDs'
    accounted = {o['object_id'] for o in comparisons} | {o['id'] for o in aggregate.get('unavailable_objects', [])}
    assert set(evidence) == accounted, 'Every candidate must have a result or an explicit unavailable record'
    size = aggregate['evaluation_grid_height']
    assert isinstance(size, int) and size > 0
    font, small = ImageFont.load_default(size=19), ImageFont.load_default(size=16)
    pair_count = 0
    for comparison in comparisons:
        size = comparison.get('evaluation_grid_height', aggregate['evaluation_grid_height'])
        object_id = comparison['object_id']
        assert Path(object_id).name == object_id and object_id not in {'.', '..'}
        comparison_path = root / 'result' / f'{object_id}-comparison.json'
        comparison_hash = digest(comparison_path)
        assert json.loads(comparison_path.read_text()) == comparison, 'Mixed assembly snapshots'
        record_path = root / comparison['native_record']
        assert digest(record_path) == comparison['native_record_sha256'], 'Native record changed'
        record = json.loads(record_path.read_text())
        assert record['status'] == 'complete'
        mesh_path = record_path.parent / record['paths']['mesh']
        assert digest(mesh_path) == record['output_sha256'][record['paths']['mesh']], 'Native mesh changed'
        mesh = trimesh.load(mesh_path, process=False, force='mesh')
        assert len(mesh.vertices) == comparison['vertices'] and len(mesh.faces) == comparison['faces']
        initial = np.asarray(comparison['initial_object_to_world'], float)
        final = np.asarray(comparison['final_object_to_world'], float)
        views = {v['frame_id']: v for v in evidence[object_id]['views']}
        assert set(views) == {v['frame_id'] for v in comparison['views']}, 'View evidence mismatch'
        rows = []
        for saved in comparison['views']:
            view = load_view(root, views[saved['frame_id']], size=size)
            before = check_score(mesh, initial, view, saved['generated_initial'])
            after = check_score(mesh, final, view, saved['generated_refined'])
            rows.append((saved, [panel(view['rgb'], view['target']),
                                  panel(view['rgb'], view['target'], before),
                                  panel(view['rgb'], view['target'], after)]))
            pair_count += 1
        width, height = rows[0][1][0].size
        gap, left, top, row_head, footer = 14, 20, 90, 60, 72
        canvas = Image.new('RGB', (left * 2 + width * 3 + gap * 2,
                                   top + len(rows) * (height + row_head + gap) + footer), '#10171e')
        draw = ImageDraw.Draw(canvas)
        draw.text((left, 14), f"{object_id} | {comparison['label']}", fill='#edf2f7', font=font)
        draw.text((left, 44), f"Evaluation: {size}px height | {comparison['faces']:,} native triangles | no asset replacement", fill='#bdcbd4', font=small)
        for row_index, (saved, panels) in enumerate(rows):
            y = top + row_index * (height + row_head + gap)
            draw.text((left, y), f"{saved['frame_id']} | {saved['role']}", fill='#edf2f7', font=font)
            labels = ['Input + target contour',
                      f"Native pose | IoU {saved['generated_initial']['visible_iou']:.4f}",
                      f"Aligned pose | IoU {saved['generated_refined']['visible_iou']:.4f}"]
            for column, (label, image) in enumerate(zip(labels, panels)):
                assert image.size == (width, height), 'Mixed view dimensions'
                x = left + column * (width + gap)
                draw.text((x, y + 29), label, fill='#bdcbd4', font=font)
                canvas.paste(image, (x, y + row_head))
        y = canvas.height - footer + 8
        draw.text((left, y), 'Green: target mask. Orange: projected mesh silhouette after the recorded foreground occlusion rule.', fill='#bdcbd4', font=small)
        draw.text((left, y + 26), 'Available object views constrain pose refinement. Source depth is estimated, not ground truth. Full grid retained.', fill='#bdcbd4', font=small)
        assert digest(comparison_path) == comparison_hash and digest(aggregate_path) == aggregate_hash, 'Assembly changed during rendering'
        assert digest(evidence_path) == evidence_hash, 'Evidence changed during rendering'
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text('comparison_sha256', comparison_hash)
        metadata.add_text('evidence_sha256', evidence_hash)
        metadata.add_text('evaluation_grid_height', str(size))
        output = root / 'result' / f'{object_id}-alignment.png'
        canvas.save(output, pnginfo=metadata)
        print(output, flush=True)
    print(f'Checked {len(comparisons)} objects, {pair_count} views, {pair_count * 2} exact IoU matches.', flush=True)


def self_check():
    mesh = trimesh.creation.box(extents=[2, 2, .2])
    transform = np.eye(4)
    transform[2, 3] = 3
    yy, xx = np.indices((8, 8))
    rays = np.stack([np.zeros_like(xx), np.zeros_like(xx), np.zeros_like(xx),
                     (xx - 3.5) / 8, (yy - 3.5) / 8, np.ones_like(xx)], -1).astype(np.float32)
    target = np.zeros((8, 8), bool)
    target[2:6, 2:6] = True
    view = dict(frame_id='synthetic', target=target, depth=np.ones((8, 8), np.float32), rays=rays)
    visible = check_score(mesh, transform, view, score_view(mesh, transform, view))
    assert np.array_equal(visible, target), 'Target must remain visible behind inconsistent target depth'
    view['depth'].fill(0)
    score = score_view(mesh, transform, view)
    assert score['visible_predicted_pixels'] == 36 and abs(score['visible_iou'] - 16 / 36) < 1e-12
    assert check_score(mesh, transform, view, score).sum() == 36
    print('Self-check passed: native rays, target preservation, foreground occlusion and exact scoring.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.root:
        render(args.root)
    else:
        parser.error('Pass --root or --self-check')
