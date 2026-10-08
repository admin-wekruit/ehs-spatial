#!/usr/bin/env python3
"""Verify a publication catalog and pre-encode HTTP responses before deployment."""
import argparse
from pathlib import Path
import json
import time
from argus.platform.publication_site import compile_catalog

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    result = compile_catalog(args.catalog, args.output)
    print(json.dumps({**result, 'seconds': round(time.perf_counter()-start, 3)}))
