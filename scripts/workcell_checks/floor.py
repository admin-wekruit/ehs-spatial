"""Floor contact: how far each model's lowest point sits above (+) or below (-) the report's floor plane, in metres.

Nothing real goes through the floor, so a model sinking more than TOL is wrong; a model whose bottom hovers a little above
the floor (TOL..NEAR) most likely should stand on it. Objects higher up (signs, lamps) are only reported. TOL = 3 cm: the
reference scale's +-1.7 % over a ~1 m tall object plus the floor plane's +-0.6 cm spread."""
import numpy as np

TOL, NEAR = .03, .15


def status(bottom):
    return 'sinks' if bottom < -TOL else 'hovers' if TOL < bottom < NEAR else 'on_floor' if bottom <= TOL else 'above'


def assess(heights_m):
    bottom = float(np.percentile(heights_m, .5))  # robust lowest point: ignore a stray vertex
    return dict(bottomM=bottom, topM=float(np.percentile(heights_m, 99.5)), status=status(bottom))


def run(ctx, opts):
    if not ctx['floor']:
        return dict(status='no_floor', objects={})
    n, d = ctx['floor']
    return dict(status='ok', tolM=TOL, nearM=NEAR,
                objects={o['id']: assess((o['mesh'][0] @ n + d) * ctx['S']) for o in ctx['objects']})


def _check():
    assert assess(np.r_[np.zeros(10), np.linspace(0, 1, 100)])['status'] == 'on_floor'
    assert assess(np.linspace(-.05, 1, 200))['status'] == 'sinks'
    assert assess(np.linspace(.06, 1, 200))['status'] == 'hovers'
    assert assess(np.linspace(1.5, 2, 200))['status'] == 'above'
