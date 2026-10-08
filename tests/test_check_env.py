"""scripts/check_env.py: env.template passes offline; the rules that protect the handoff fail loudly."""
import subprocess
import sys
from pathlib import Path

from argus import ROOT
TEMPLATE = (ROOT / "env.template").read_text()


def run(tmp_path, text, *flags):
    env_file = tmp_path / ".env"
    env_file.write_text(text)
    return subprocess.run([sys.executable, "-m", "scripts.check_env", str(env_file), *flags], capture_output=True, text=True)


def test_template_passes_offline(tmp_path):
    r = run(tmp_path, TEMPLATE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 errors" in r.stdout


def test_strict_rejects_missing_data_path(tmp_path):
    assert run(tmp_path, TEMPLATE.replace("PANOPTES_DATA_ROOT=./data", f"PANOPTES_DATA_ROOT={tmp_path / 'missing'}"), "--strict").returncode == 1


def test_missing_required_and_bad_values_fail(tmp_path):
    text = TEMPLATE.replace("PANOPTES_DATA_ROOT=", "# PANOPTES_DATA_ROOT=").replace("SAM3D_BACKEND=modal", "SAM3D_BACKEND=cloud")
    r = run(tmp_path, text)
    assert r.returncode == 1
    assert "PANOPTES_DATA_ROOT: required" in r.stdout and "SAM3D_BACKEND: 'cloud'" in r.stdout


def test_hf_token_and_missing_url_refused(tmp_path):
    text = TEMPLATE.replace("SAM3D_BACKEND=modal", "SAM3D_BACKEND=http") + "\nHF_TOKEN=hf_x\n"
    r = run(tmp_path, text)
    assert r.returncode == 1
    assert "HF_TOKEN" in r.stdout and "SAM3D_HTTP_URLS: required when SAM3D_BACKEND=http" in r.stdout


def test_database_credentials_are_not_printed(tmp_path):
    text = TEMPLATE.replace("PANOPTES_DATABASE_URL=postgresql://panoptes@127.0.0.1:55432/panoptes", "PANOPTES_DATABASE_URL=wrong://user:private-secret@example.test/db")
    r = run(tmp_path, text)
    assert r.returncode == 1 and "private-secret" not in r.stdout + r.stderr
