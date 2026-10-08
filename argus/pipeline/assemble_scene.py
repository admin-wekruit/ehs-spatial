"""Assemble the selected capture run on the pipeline machine."""
import argparse
import json
from pathlib import Path
import time
from argus.pipeline.assemble_lucida_scene import assemble

def main(run: str, iterations: int = 100, eval_size: int = 288):
    root = Path(run).resolve()
    if (root / 'result').exists():
        raise ValueError('RUN/result already exists')
    start = time.monotonic()
    assemble(root, iterations, eval_size)
    ledger = {'mode': 'local CPU', 'functionSeconds': time.monotonic() - start, 'estimateUsd': 0.0}
    (root / 'result/assembly-spend-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps(ledger))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--eval-size', type=int, default=288)
    args = parser.parse_args()
    main(args.run, args.iterations, args.eval_size)
