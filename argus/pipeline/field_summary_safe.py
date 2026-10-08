"""Preserve field scoring when a surface has insufficient points."""
import argus.pipeline.field_summary as fc

def safe_config(runs, bb, var, rule, s):
    """fc.config; when a scored value is missing (too few surface points) the row keeps its values but no aggregates."""
    try:
        return fc.config(runs, bb, var, rule, s)
    except TypeError:
        import copy
        tmp = {k: copy.deepcopy(v) if k[1] == bb else v for k, v in runs.items()}
        for c in ('090', '030'):
            tmp[(c, bb, var)]['floors'][rule]['gatePassed'] = False
        row = fc.config(tmp, bb, var, rule, s)
        for c in ('090', '030'):
            row['gate'][c]['passed'] = runs[(c, bb, var)]['floors'][rule]['gatePassed']
        row['gatePassedBothCells'] = all(g['passed'] for g in row['gate'].values())
        errs = [abs(x['errCm']) for x in row['values'].values() if x['errCm'] is not None]
        row.update(incomplete=True, maeAvailableCm=sum(errs) / len(errs), nAvailable=len(errs))
        return row
