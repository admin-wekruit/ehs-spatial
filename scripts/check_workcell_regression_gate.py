"""CPU check of scripts/workcell_regression_gate.py compare(): the rules on synthetic scores (no run needed).

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts python scripts/check_workcell_regression_gate.py
"""
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.workcell_regression_gate import CONNECTED, IOU_DROP, compare

BASE = {'objects': {'robot': {'model': True, 'meanIou': .70}, 'lamp': {'model': True, 'meanIou': .65}},
        'sheet': [{'boards': ['v-guard-left', 'v-guard-center'], 'gapNative': 0.}], 'contactsStatus': 'resolved', 'contactsUnresolved': []}


def variant(**changes):
    out = copy.deepcopy(BASE)
    for path, value in changes.items():
        head, _, tail = path.partition('__')
        if tail:
            out['objects'][head][tail] = value
        else:
            out[head] = value
    return out


assert compare(BASE, BASE) == ([], []), 'identical runs pass'
assert compare(BASE, variant(robot__meanIou=.70 - IOU_DROP + .01))[0] == [], 'a drop within the tolerance passes'
assert compare(BASE, variant(robot__meanIou=.70 - IOU_DROP - .01))[0], 'a larger drop fails'
assert compare(BASE, variant(lamp__model=False))[0], 'a lost model fails'
failures, notes = compare(BASE, variant(lamp__model=False), {'lamp': 'deliberately removed'})
assert not failures and notes, 'an accepted, explained change is reported, not failed'
assert compare(BASE, variant(sheet=[{'boards': ['v-guard-left', 'v-guard-center'], 'gapNative': CONNECTED + .01}]))[0], 'a broken sheet fails'
assert compare(BASE, variant(contactsStatus='partly resolved'))[0], 'unrecorded interpenetration fails'
print('check_workcell_regression_gate passed')
