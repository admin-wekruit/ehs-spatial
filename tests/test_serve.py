"""The standalone report page: one URL per run carrying the full report
plus the review and agent panels, served by the same process as the
workbench."""

import os
from pathlib import Path

import pytest

os.environ["PANOPTES_NO_RESUME"] = "1"

from fastapi.testclient import TestClient  # noqa: E402

BOR1 = Path("runs/user-bor1-02")
needs_bor1 = pytest.mark.skipif(
    not (BOR1 / "inventory" / "inventory.json").exists(), reason="BOR1 run missing"
)


@pytest.fixture(scope="module")
def client():
    from ehs_spatial.serve import create_server

    return TestClient(create_server())


@needs_bor1
def test_report_page_is_one_page_with_hub(client):
    response = client.get("/report/user-bor1-02")
    assert response.status_code == 200
    page = response.text
    # the full 14-section report ...
    assert page.count('data-section="') == 14
    # ... with review + agent panels on the same page, not in an iframe
    assert 'id="hub"' in page and 'id="rv-save"' in page and 'id="ag-send"' in page
    assert "<iframe" not in page.split('data-section="viewer"')[0]


@needs_bor1
def test_report_file_download_has_no_live_panels(client):
    response = client.get("/report/user-bor1-02/file")
    assert response.status_code == 200
    assert 'id="hub"' not in response.text


@needs_bor1
def test_chat_endpoint_returns_list(client):
    response = client.get("/api/chat/user-bor1-02")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_unknown_run_is_404_and_bad_id_is_400(client):
    assert client.get("/report/does-not-exist").status_code == 404
    assert client.get("/report/..%2Fetc").status_code in (400, 404)


@needs_bor1
def test_review_requires_reviewer(client):
    response = client.post(
        "/api/review",
        json={"run_id": "user-bor1-02", "reviewer": "", "decision": "confirmed"},
    )
    assert response.status_code == 400


def test_workbench_still_mounted(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "gradio" in response.text.lower()
