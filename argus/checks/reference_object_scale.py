"""Metric scale from a reference object of known size, accepted only when every feature in every photo agrees with it.

Each feature seen in each photo (e.g. the e-stop's red head, 4 cm, and yellow body, 8 cm) gives one metres-per-native estimate,
specM / measuredNative. The scale is their geometric mean. It is accepted only when every feature, measured at that scale, is
within LIMIT of its specification; otherwise the scene gets no metric scale.

python scripts/reference_object_scale.py                 # self-check
python scripts/reference_object_scale.py FEATURES.json   # [{"photo": ..., "feature": ..., "specM": ..., "measuredNative": ...}, ...]
"""
import json
import math
import sys

LIMIT = .04  # user, 2026-10-04: the reference e-stop must be reproduced within 4 %


def joint_scale(rows, limit=LIMIT):
    if not rows or any(not (r['specM'] > 0 and r['measuredNative'] > 0) for r in rows):
        raise ValueError('reference features need positive specified and measured sizes')
    scale = math.exp(sum(math.log(r['specM'] / r['measuredNative']) for r in rows) / len(rows))
    features = [{**r, 'measuredM': r['measuredNative'] * scale, 'deviation': r['measuredNative'] * scale / r['specM'] - 1} for r in rows]
    worst = max(abs(f['deviation']) for f in features)
    return {'nativeToMeters': scale, 'limit': limit, 'maxDeviation': worst, 'passed': worst < limit, 'features': features}


def _check():
    # The September e-stop: head and body differ by 3.5 % from their 4 / 8 cm specification; both stay inside 4 %.
    ok = joint_scale([{'photo': 1, 'feature': 'red head', 'specM': .04, 'measuredNative': .03038},
                      {'photo': 1, 'feature': 'yellow body', 'specM': .08, 'measuredNative': .06268}])
    assert ok['passed'] and 1.27 < ok['nativeToMeters'] < 1.30 and ok['maxDeviation'] < .02, ok
    # A body read 10 % too wide breaks the limit, so no metric scale is granted.
    bad = joint_scale([{'photo': 1, 'feature': 'red head', 'specM': .04, 'measuredNative': .03038},
                       {'photo': 1, 'feature': 'yellow body', 'specM': .08, 'measuredNative': .0670}])
    assert not bad['passed'] and bad['maxDeviation'] > .04, bad
    try:
        joint_scale([])
    except ValueError:
        pass
    else:
        raise AssertionError('an empty reference must be rejected')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        _check()
        print('reference_object_scale check passed')
    else:
        print(json.dumps(joint_scale(json.load(open(sys.argv[1]))), indent=1))
