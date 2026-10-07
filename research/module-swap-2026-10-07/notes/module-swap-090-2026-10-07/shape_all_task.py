"""Task for modal_apps/workcell_layer_trial.py: the standard multi-photo shape check of EVERY displayed model, exactly as
modal_apps/workcell_shape_check.py's remote run() does it (same sampling, masks eroded 7x7, scales .75-1.25, tolerance .004),
for a locally served (unpublished) report. Returns the same rows (+ plot bytes in files)."""
import hashlib
import io

import cv2
import numpy as np
import shape_core as wsc


def run(ctx, opts):
    cams, images, S, layer = ctx['cams'], ctx['images'], ctx['S'], ctx['layer']
    points = int(opts.get('points', 12000))
    displayed = {o['id']: o['mesh'] for o in ctx['objects']}
    scales = np.round(np.linspace(.75, 1.25, 51), 4)
    rows = []
    for o in ctx['objects']:
        masks = {k: cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) for k, m in o['masks'].items()}
        others = wsc.raycast_scene([displayed[i] for i in displayed if i != o['id']])
        variants = [('displayed', o['mesh'])] + ([('september', o['original'])] if o['mesh'] is not o['original'] else [])
        for name, (V, F) in variants:
            rng = np.random.default_rng(int(hashlib.sha256((o['id'] + name).encode()).hexdigest()[:8], 16))
            P, N = wsc.sample_surface(V, F, points, rng)
            row = wsc.check_entity(o['id'], P, N, wsc.raycast_scene([(V, F)]), others, cams, images, masks, scales, .004)
            row.update(label=o.get('label'), model=name, maskPhotos=sorted(k + 1 for k in masks), transferredMaskPhotos=[], verdict=wsc.verdict(row),
                       depthChangeM=(row['bestScale'] - 1) * row['distanceToReference'] * S, referencePhoto=row['referencePhoto'] + 1)
            rows.append(row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n, cols = len(rows), 6
    fig, axes = plt.subplots(-(-n // cols), cols, figsize=(3.2 * cols, 2.3 * -(-n // cols)), squeeze=False)
    for ax, r in zip(axes.ravel(), rows):
        xs, ys = zip(*r['curve'])
        ax.plot(xs, ys, color='#2a6f4e' if r['verdict'] == 'ok' else '#c0392b' if r['verdict'] in ('depth_off', 'not_photo_consistent') else '#888')
        ax.axvline(1, color='#999', lw=.6); ax.axhline(r['shiftedControl'], color='#bbb', lw=.6, ls='--')
        ax.set_title(f"{(r['label'] or '')[:18]}\n{r['verdict']} s*={r['bestScale']:.2f}", fontsize=7); ax.tick_params(labelsize=6)
    for ax in axes.ravel()[n:]:
        ax.axis('off')
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=110)
    return dict(rows=rows, skipped=ctx['skipped'], nativeToMeters=S, layerRevision=(layer or {}).get('revisionId'), files={'curves.png': buf.getvalue()})
