"""One-off: make every machine path in the notes scripts an environment lookup whose default is the original path, so the same
files run unchanged on the original machine and on a clone (env.sh). Exact replacements, each asserted to occur once.
    python _parametrise.py            (idempotent: a second run finds nothing to replace and reports it)
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent / 'notes'
OLD_SCR = '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'
OLD_RN = '/Users/adam/Desktop/panoptes-public/research-notes'
OLD_SERV = '/Users/adam/Desktop/panoptes-public/panoptes-serving'
OLD_WT = '/Users/adam/.codex/worktrees/panoptes-workcell-photo-speed'
OLD_PLAT = '/Users/adam/Desktop/Tesla/panoptes-platform'
OLD_PAGES = '/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/workcell-photo-direct/report/measurement-layer'
OLD_RUNS = OLD_SERV + '/outputs/candidate-evaluation'
E = lambda var, old: f"os.environ.get('{var}', '{old}')"

PY = {  # file -> [(old, new)]
    'module-swap-090-2026-10-07/fill_geometry.py': [(f"SP = Path('{OLD_SCR}')", f"SP = Path({E('SWAP_SCRATCH', OLD_SCR)})")],
    'module-swap-090-2026-10-07/field_values_fill.py': [(f"SP = Path('{OLD_SCR}')", f"SP = Path({E('SWAP_SCRATCH', OLD_SCR)})")],
    'module-swap-090-2026-10-07/compare_json.py': [(f"SP = Path('{OLD_SCR}')", f"SP = Path({E('SWAP_SCRATCH', OLD_SCR)})"),
                                                   (f"RN = Path('{OLD_RN}')", f"RN = Path({E('SWAP_NOTES', OLD_RN)})"),
                                                   (f"LUCIDA = Path('{OLD_RUNS}/lucida-replica-01')", f"LUCIDA = Path({E('PANOPTES_RUNS', OLD_RUNS)}) / 'lucida-replica-01'")],
    'module-swap-090-2026-10-07/compare_layers.py': [(f"SP = Path('{OLD_SCR}')", f"SP = Path({E('SWAP_SCRATCH', OLD_SCR)})"),
                                                     (f"RN = Path('{OLD_RN}')", f"RN = Path({E('SWAP_NOTES', OLD_RN)})"),
                                                     (f"PAGES = Path('{OLD_PAGES}')", f"PAGES = Path({E('PANOPTES_PAGES', OLD_PAGES)})")],
    'module-swap-090-2026-10-07/tables_030.py': [(f"SP = Path('{OLD_SCR}')", f"SP = Path({E('SWAP_SCRATCH', OLD_SCR)})"),
                                                 (f"PLATFORM = Path('{OLD_PLAT}')", f"PLATFORM = Path({E('PANOPTES_PLATFORM', OLD_PLAT)})")],
    'module-swap-090-2026-10-07/build_swap_layer.py': [(f"SP = Path('{OLD_SCR}')", f"SP = Path({E('SWAP_SCRATCH', OLD_SCR)})"),
                                                       (f"PLATFORM = Path('{OLD_PLAT}')", f"PLATFORM = Path({E('PANOPTES_PLATFORM', OLD_PLAT)})"),
                                                       (f"PAGES = Path('{OLD_PAGES}')", f"PAGES = Path({E('PANOPTES_PAGES', OLD_PAGES)})")],
    'module-swap-090-2026-10-07/serve_export.py': [(f"SP = Path('{OLD_SCR}')", f"SP = Path({E('SWAP_SCRATCH', OLD_SCR)})"),
                                                   (f"PLATFORM = Path('{OLD_PLAT}')", f"PLATFORM = Path({E('PANOPTES_PLATFORM', OLD_PLAT)})")],
    'module-swap-090-2026-10-07/swap_generation.py': [(f"SP = Path('{OLD_SCR}')", f"SP = Path({E('SWAP_SCRATCH', OLD_SCR)})"),
                                                      (f"LUCIDA = Path('{OLD_RUNS}/lucida-replica-01')", f"LUCIDA = Path({E('PANOPTES_RUNS', OLD_RUNS)}) / 'lucida-replica-01'"),
                                                      (f"sys.path.insert(0, '{OLD_SERV}/scripts/research')", f"sys.path.insert(0, {E('PANOPTES_SERVING', OLD_SERV)} + '/scripts/research')")],
    'module-swap-090-2026-10-07/run_stages.py': [(f"SP = Path('{OLD_SCR}')", f"SP = Path({E('SWAP_SCRATCH', OLD_SCR)})"),
                                                 (f"RN = Path('{OLD_RN}')", f"RN = Path({E('SWAP_NOTES', OLD_RN)})"),
                                                 (f"WT = Path('{OLD_WT}')", f"WT = Path({E('PANOPTES_WORKCELL', OLD_WT)})"),
                                                 (f"MODAL = '{OLD_PLAT}/.venv/bin/modal'", f"MODAL = '{OLD_PLAT}/.venv/bin/modal'\nMODAL_RUN = os.environ.get('MODAL_RUN', MODAL + ' run').split()   # on-prem: 'python WT/scripts/onprem/run_stage.py --weights DIR'"),
                                                 ("cmd = [MODAL, 'run', str(WT / 'modal_apps/workcell_layer_trial.py'),", "cmd = MODAL_RUN + [str(WT / 'modal_apps/workcell_layer_trial.py'),")],
    'geometry-backbone-ab-2026-10-06/backbone_ab_modal.py': [(f"SERV = Path('{OLD_SERV}')", f"SERV = Path({E('PANOPTES_SERVING', OLD_SERV)})"),
                                                              (f"SCR = Path('{OLD_SCR}')", f"SCR = Path({E('SWAP_SCRATCH', OLD_SCR)})"),
                                                              ("RUNS = SERV / 'outputs/candidate-evaluation'", f"RUNS = Path({E('PANOPTES_RUNS', OLD_RUNS)})")],
    'geometry-backbone-ab-2026-10-06/prod_route_modal.py': [(f"KIT = Path('{OLD_WT}')", f"KIT = Path({E('PANOPTES_WORKCELL', OLD_WT)})"),
                                                             (f"SCR = Path('{OLD_SCR}')", f"SCR = Path({E('SWAP_SCRATCH', OLD_SCR)})")],
    'geometry-backbone-ab-2026-10-06/compile.py': [(f"RUNS = Path('{OLD_RUNS}')", f"RUNS = Path({E('PANOPTES_RUNS', OLD_RUNS)})")],
    'geometry-backbone-ab-2026-10-06/mvs_route.py': [(f"sys.path[:0] = [str(NOTE), '{OLD_WT}/modal_apps']", f"sys.path[:0] = [str(NOTE), {E('PANOPTES_WORKCELL', OLD_WT)} + '/modal_apps']")],
    'geometry-backbone-ab-2026-10-06/ba_scipy.py': [(f"sys.path[:0] = [str(Path(__file__).resolve().parent), '{OLD_WT}/modal_apps']", f"sys.path[:0] = [str(Path(__file__).resolve().parent), {E('PANOPTES_WORKCELL', OLD_WT)} + '/modal_apps']")],
    'geometry-licence-ab-fair-2026-10-05/fair_ab_modal.py': [(f"SERV = Path('{OLD_SERV}')", f"SERV = Path({E('PANOPTES_SERVING', OLD_SERV)})"),
                                                              (f"WT = Path('{OLD_WT}')", f"WT = Path({E('PANOPTES_WORKCELL', OLD_WT)})"),
                                                              (f"CELL030 = Path('{OLD_RN}/cell030-sept-pipeline-2026-10-05')", f"CELL030 = Path({E('SWAP_NOTES', OLD_RN)}) / 'cell030-sept-pipeline-2026-10-05'"),
                                                              (f"CLEARB = Path('{OLD_RN}/workcell-clearance-b-2026-10-05/clearance_b.py')", f"CLEARB = Path({E('SWAP_NOTES', OLD_RN)}) / 'workcell-clearance-b-2026-10-05/clearance_b.py'"),
                                                              (f"GEOMAB = Path('{OLD_RN}/geometry-licence-ab-2026-10-05/geometry_ab.py')", f"GEOMAB = Path({E('SWAP_NOTES', OLD_RN)}) / 'geometry-licence-ab-2026-10-05/geometry_ab.py'"),
                                                              (f"SCR = Path('{OLD_SCR}')", f"SCR = Path({E('SWAP_SCRATCH', OLD_SCR)})")],
    'geometry-licence-ab-fair-2026-10-05/compile.py': [(f"SCR = Path('{OLD_SCR}')", f"SCR = Path({E('SWAP_SCRATCH', OLD_SCR)})")],
    'licence-clean-stack-2026-10-06/geometry/compile.py': [(f"FAIR = Path('{OLD_RN}/geometry-licence-ab-fair-2026-10-05')", f"FAIR = Path({E('SWAP_NOTES', OLD_RN)}) / 'geometry-licence-ab-fair-2026-10-05'")],
    'completion-licence-ab-2026-10-05/completion_ab.py': [(f"WT = Path('{OLD_WT}')", f"WT = Path({E('PANOPTES_WORKCELL', OLD_WT)})"),
                                                           (f"SERVING = Path('{OLD_SERV}')", f"SERVING = Path({E('PANOPTES_SERVING', OLD_SERV)})"),
                                                           (f"    '{OLD_SCR}/checks/'", f"    {E('SWAP_SCRATCH', OLD_SCR)} + '/checks/'")],
    'completion-ab-090-2026-10-06/compare.py': [(f"CHECKS = Path('{OLD_SCR}/checks')", f"CHECKS = Path({E('SWAP_SCRATCH', OLD_SCR)}) / 'checks'"),
                                                 (f"else Path('{OLD_RUNS}/lucida-replica-01')", f"else Path({E('PANOPTES_RUNS', OLD_RUNS)}) / 'lucida-replica-01'"),
                                                 (f"    sys.path.insert(0, '{OLD_WT}/scripts')", f"    sys.path.insert(0, {E('PANOPTES_WORKCELL', OLD_WT)} + '/scripts')")],
    'completion-ab-090-2026-10-06/build_layer.py': [(f"sys.path.insert(0, '{OLD_SERV}/scripts/research')", f"sys.path.insert(0, {E('PANOPTES_SERVING', OLD_SERV)} + '/scripts/research')")],
}
SH = {  # shell: VAR=old -> VAR=${ENV:-old}; modal invocations -> ${=MODAL_RUN}
    'run_fill_chain.sh': True, 'run_v2_chain.sh': True, 'run_030_chain.sh': True, 'publish_site.sh': True,
}
SH_VARS = {'SP': ('SWAP_SCRATCH', OLD_SCR), 'PY': ('PY', OLD_SERV + '/.venv/bin/python'), 'MS': ('MS', OLD_SERV + '/.venv/bin/modal'),
           'M': ('M', OLD_PLAT + '/.venv/bin/modal'), 'RN': ('SWAP_NOTES', OLD_RN), 'SERV': ('PANOPTES_SERVING', OLD_SERV),
           'PLAT': ('PANOPTES_PLATFORM', OLD_PLAT), 'PG': ('PG', '/opt/homebrew/opt/postgresql@16/bin'), 'WT': ('PANOPTES_WORKCELL', OLD_WT)}


def patch_py(rel, pairs):
    p = ROOT / rel; s = p.read_text(); n = 0
    for old, new in pairs:
        c = s.count(old)
        if c == 0 and s.count(new.split('\n')[0]) >= 1:
            continue   # already patched
        assert c == 1, (rel, c, old[:80])
        s = s.replace(old, new); n += 1
    if 'os.environ' in s and not re.search(r'^import os\b|^from os\b', s, re.M):
        s = s.replace('from pathlib import Path\n', 'import os\nfrom pathlib import Path\n', 1)
        assert re.search(r'^import os\b', s, re.M), rel
    p.write_text(s); return n


def patch_sh(name):
    p = ROOT / 'module-swap-090-2026-10-07' / name; s = p.read_text(); n = 0
    for var, (env, old) in SH_VARS.items():
        for line in (f'{var}={old}\n', f'{var}=$PLAT/.venv/bin/python\n'):
            if line in s:
                s = s.replace(line, f'{var}=${{{env}:-{old}}}\n' if '$PLAT' not in line else f'{var}=${{PY:-$PLAT/.venv/bin/python}}\n'); n += 1
    if '$M run ' in s or '$MS run ' in s:
        if 'MODAL_RUN=' not in s:
            s = s.replace('filt() {', 'MODAL_RUN=${MODAL_RUN:-$M run}   # on-prem: "python $WT/scripts/onprem/run_stage.py --weights DIR"\nfilt() {', 1); n += 1
        s = s.replace('$M run ', '${=MODAL_RUN} ').replace('$MS run ', '${=MODAL_RUN} ')
    s = s.replace('-D /opt/homebrew/var/postgresql@16', '-D ${PGDATA_DIR:-/opt/homebrew/var/postgresql@16}').replace('-l /private/tmp/claude-501/pg55432.log', '-l ${SWAP_SCRATCH:-/private/tmp/claude-501}/pg55432.log')
    p.write_text(s); return n


if __name__ == '__main__':
    for rel, pairs in PY.items():
        print(rel, 'replacements', patch_py(rel, pairs))
    for name in SH:
        print(name, 'replacements', patch_sh(name))
