"""Measure observed torso displacement in a metric map, never infer hidden motion."""
from collections import defaultdict
import numpy as np


POLICY = {'method': 'two robust windows of the same visible torso anchor',
          'history_seconds': 1., 'max_observation_gap_seconds': .15,
          'endpoint_window_seconds': .15, 'min_endpoint_samples': 3,
          'min_elapsed_seconds': .5, 'displacement_resolution_m': .1,
          'interpretation': 'Observed surface torso displacement, not gait, body center of mass, or a safety verdict'}


def annotate_motion(frames, units):
    if units != 'meters':
        raise ValueError('Motion in metres requires a metric scene')
    histories = defaultdict(list)
    for frame in frames:
        now = frame['timeSec']
        present = {o['entityId'] for o in frame['objects']}
        for identity in list(histories):
            if identity not in present:
                del histories[identity]
        for obj in frame['objects']:
            obj['world_motion'] = 'insufficient_evidence'
            obj.pop('motionEstimate', None)
            joints = obj.get('keypoints3d', [])
            anchor = next(((name, np.mean([joints[a], joints[b]], axis=0))
                           for name, a, b in [('hips', 11, 12), ('shoulders', 5, 6)]
                           if len(joints) > max(a, b) and joints[a] is not None and joints[b] is not None), None)
            history = histories[obj['entityId']]
            if anchor is None:
                history.clear()
                continue
            name, point = anchor
            if not np.isfinite(point).all():
                raise ValueError('Non-finite world anchor')
            if history and (history[-1][2] != name
                            or frame['sourceFrame'] != history[-1][3] + 1
                            or not 0 < now-history[-1][0] <= POLICY['max_observation_gap_seconds']):
                history.clear()
            history.append((now, point, name, frame['sourceFrame']))
            history[:] = [r for r in history if now-r[0] <= POLICY['history_seconds']]
            old = [r for r in history if r[0]-history[0][0] <= POLICY['endpoint_window_seconds']]
            new = [r for r in history if now-r[0] <= POLICY['endpoint_window_seconds']]
            if min(len(old), len(new)) < POLICY['min_endpoint_samples']:
                continue
            elapsed = float(np.median([r[0] for r in new])-np.median([r[0] for r in old]))
            if elapsed < POLICY['min_elapsed_seconds']:
                continue
            delta = np.median([r[1] for r in new], axis=0)-np.median([r[1] for r in old], axis=0)
            distance = float(np.linalg.norm(delta))
            obj['world_motion'] = 'observed_displacement' if distance >= POLICY['displacement_resolution_m'] else 'below_resolution'
            obj['motionEstimate'] = {'anchor': name, 'elapsedSeconds': elapsed,
                                     'displacementM': distance, 'averageSpeedMps': distance/elapsed,
                                     'sourceFrames': [old[0][3], new[-1][3]],
                                     'status': 'model_estimate_not_ground_truth'}
    return POLICY.copy()


def self_check():
    def scene(speed):
        frames = []
        for i in range(31):
            points = [None]*17
            points[11], points[12] = [i/30*speed, 0, 1], [i/30*speed, .2, 1]
            frames.append({'sourceFrame': i, 'timeSec': i/30,
                           'objects': [{'entityId': 'a', 'keypoints3d': points}]})
        return frames
    moving, still = scene(1), scene(0)
    annotate_motion(moving, 'meters'); annotate_motion(still, 'meters')
    assert moving[-1]['objects'][0]['world_motion'] == 'observed_displacement'
    assert abs(moving[-1]['objects'][0]['motionEstimate']['averageSpeedMps']-1) < 1e-6
    assert still[-1]['objects'][0]['world_motion'] == 'below_resolution'
    gap = scene(1); gap[20]['objects'] = []
    annotate_motion(gap, 'meters')
    assert gap[-1]['objects'][0]['world_motion'] == 'insufficient_evidence'
    # Missing native camera frames are omitted, not emitted as empty objects.
    missing = scene(1); del missing[20]
    annotate_motion(missing, 'meters')
    assert missing[-1]['objects'][0]['world_motion'] == 'insufficient_evidence'
    assert 'motionEstimate' not in missing[-1]['objects'][0]
    try:
        annotate_motion(scene(1), 'uncalibrated_monocular')
    except ValueError:
        pass
    else:
        raise AssertionError('Uncalibrated speed must be rejected')
    print('video motion check passed: translation, static world anchor, visibility gap, metric units')


if __name__ == '__main__':
    self_check()
