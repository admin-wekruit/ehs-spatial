"""Contract tests of docs/BACKENDS-v1.md: the fake backends in-process (test_v1_services.py) or a live service (live.py)."""
import os
from pathlib import Path
import sys

os.environ.setdefault('PANOPTES_FAKE_MODEL', '1')
os.environ.setdefault('PANOPTES_SERVICE_API_KEY', 'contract-test-key')
os.environ.setdefault('PANOPTES_SERVICE_MAX_QUEUE', '4')
os.environ.pop('PANOPTES_SERVICE_CACHE_DIR', None)
os.environ.pop('PANOPTES_RESULT_BUCKET', None)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'serving'))
