"""ehs_spatial.verdict.llm: cache hit answers without the API; fake mode never calls out; the key covers model, schema and both prompts."""
import json

import pytest
from pydantic import BaseModel

from ehs_spatial.verdict import llm


class Answer(BaseModel):
    threshold_mm: int
    operator: str


def test_cache_hit_and_fake_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("PANOPTES_FAKE_MODEL", "1")
    path = llm.cache_path(tmp_path, "claude-haiku-5-5", Answer, "sys", "fence >= 1400 mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": "claude-haiku-5-5", "response": {"threshold_mm": 1400, "operator": ">="}, "usage": {"input_tokens": 3}}))
    out, meta = llm.complete("sys", "fence >= 1400 mm", Answer, model="claude-haiku-5-5", cache_dir=tmp_path)
    assert out == Answer(threshold_mm=1400, operator=">=") and meta["cached"] is True and meta["usage"] == {"input_tokens": 3}
    with pytest.raises(LookupError):
        llm.complete("sys", "fence >= 1500 mm", Answer, model="claude-haiku-5-5", cache_dir=tmp_path)
    assert llm.cache_path(tmp_path, "claude-haiku-5-5", Answer, "sys", "x") != llm.cache_path(tmp_path, "claude-sonnet-5-5", Answer, "sys", "x")
