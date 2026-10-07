"""Container healthcheck: GET /healthz on 127.0.0.1:PORT with the X-API-Key from the environment (stdlib only, exit 1 on failure).
    python healthcheck.py 8805            v1 services (X-API-Key required)
    python healthcheck.py 8801 --no-key   the docs/BACKENDS.md services (no auth)"""
import os
import sys
import urllib.request

port = sys.argv[1]
headers = {} if '--no-key' in sys.argv else {'X-API-Key': os.environ.get('PANOPTES_SERVICE_API_KEY', '')}
try:
    with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/healthz', headers=headers), timeout=8) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception as error:  # noqa: BLE001 - any failure is unhealthy
    print(error, file=sys.stderr)
    sys.exit(1)
