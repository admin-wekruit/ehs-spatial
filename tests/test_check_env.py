"""scripts/check_env.py: env.template passes offline; the rules that protect the handoff fail loudly."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "env.template").read_text()


def run(tmp_path, text, *flags):
    env_file = tmp_path / ".env"
    env_file.write_text(text)
    return subprocess.run([sys.executable, str(ROOT / "scripts/check_env.py"), str(env_file), *flags], capture_output=True, text=True)


def test_template_passes_offline(tmp_path):
    r = run(tmp_path, TEMPLATE)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 errors" in r.stdout


def test_template_fails_strict_until_filled(tmp_path):
    assert run(tmp_path, TEMPLATE, "--strict").returncode == 1   # placeholder key, example paths


def test_missing_required_and_bad_values_fail(tmp_path):
    text = TEMPLATE.replace("PANOPTES_SERVICE_API_KEY=", "# PANOPTES_SERVICE_API_KEY=").replace("SAM3D_BACKEND=http", "SAM3D_BACKEND=cloud")
    r = run(tmp_path, text)
    assert r.returncode == 1
    assert "PANOPTES_SERVICE_API_KEY: required" in r.stdout and "SAM3D_BACKEND: 'cloud'" in r.stdout


def test_hf_token_and_missing_url_refused(tmp_path):
    text = TEMPLATE.replace("SAM3D_HTTP_URLS=", "# SAM3D_HTTP_URLS=") + "\nHF_TOKEN=hf_x\n"
    r = run(tmp_path, text)
    assert r.returncode == 1
    assert "HF_TOKEN" in r.stdout and "SAM3D_HTTP_URLS: required when SAM3D_BACKEND=http" in r.stdout
