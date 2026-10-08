"""Claude access of the lab (L4 clause extraction, L5 check synthesis): one function, one response cache, one fake mode.

complete(system, user, schema, model=None, effort=None, cache_dir=None) -> (parsed schema instance, meta)
  model      PANOPTES_VERDICT_MODEL or MODEL (Claude Haiku: the plan's cheap model; `claude-haiku-5-5`). Credentials are resolved by
             the SDK: ANTHROPIC_API_KEY, or the profile `ant auth login` stores under ~/.config/anthropic/. Nothing is read or
             logged here; never write a key into a file.
  schema     a pydantic model; the API is asked for it through structured outputs (client.messages.parse, output_format=schema),
             so the answer is validated before it reaches a plugin. Thinking stays at the model's adaptive default.
  cache      cache_dir/<sha256 of model + schema + system + user>.json -- the ONLY hash of this package (lab rule 5): a hit
             never calls the API, so a matrix re-run or a second variant on the same clause is free. Default dir: $PANOPTES_LLM_CACHE
             or runs/llm-cache next to the run directories (a plugin passes workdir.parents[1] / 'llm-cache').
  fake       PANOPTES_FAKE_MODEL=1 answers from the cache only; a miss raises LookupError. Tests either ship cached responses
             or monkeypatch `complete`. meta = {'model', 'cached', 'usage', 'stop_reason'}; a plugin counts llm_calls when cached is False.
The system prompt carries cache_control so the long, stable part (vocabulary + conventions) is cached on the API side too.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import BaseModel

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000


def cache_path(cache_dir: Path, model: str, schema: type[BaseModel], system: str, user: str) -> Path:
    key = hashlib.sha256("\n".join([model, json.dumps(schema.model_json_schema(), sort_keys=True), system, user]).encode()).hexdigest()
    return Path(cache_dir) / f"{key}.json"


def complete(system: str, user: str, schema: type[BaseModel], model: str | None = None, effort: str | None = None,
             cache_dir: str | Path | None = None) -> tuple[BaseModel, dict]:
    model = model or os.environ.get("PANOPTES_VERDICT_MODEL") or MODEL
    cache_dir = Path(cache_dir or os.environ.get("PANOPTES_LLM_CACHE") or "runs/llm-cache")
    path = cache_path(cache_dir, model, schema, system, user)
    if path.exists():
        hit = json.loads(path.read_text())
        return schema.model_validate(hit["response"]), {"model": hit["model"], "cached": True, "usage": hit.get("usage", {}), "stop_reason": hit.get("stop_reason")}
    if os.environ.get("PANOPTES_FAKE_MODEL") == "1":
        raise LookupError(f"PANOPTES_FAKE_MODEL=1 and no cached response at {path} (model {model}, schema {schema.__name__})")
    import anthropic

    client = anthropic.Anthropic()
    kwargs = {} if effort is None else {"output_config": {"effort": effort}}
    msg = client.messages.parse(model=model, max_tokens=MAX_TOKENS, output_format=schema,
                                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                                messages=[{"role": "user", "content": user}], **kwargs)
    if msg.stop_reason == "refusal":
        details = msg.stop_details
        raise RuntimeError(f"{model} refused ({getattr(details, 'category', None)}): {getattr(details, 'explanation', '')}")
    if msg.parsed_output is None:
        raise RuntimeError(f"{model} returned no parsable {schema.__name__} (stop_reason {msg.stop_reason})")
    usage = msg.usage.model_dump(exclude_none=True) if msg.usage else {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model, "schema": schema.__name__, "system": system, "user": user, "stop_reason": msg.stop_reason,
                                "usage": usage, "response": msg.parsed_output.model_dump(mode="json")}, indent=1, ensure_ascii=False, sort_keys=True))
    return msg.parsed_output, {"model": model, "cached": False, "usage": usage, "stop_reason": msg.stop_reason}
