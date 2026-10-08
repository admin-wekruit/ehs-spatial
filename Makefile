# Argus: install with uv sync, configure .env, then run the Modal pipeline.
UV ?= uv
ENV_FILE ?= .env
CELL ?= 090
ARGS ?=

.PHONY: env check-env test run publications

env:
	@test -f $(ENV_FILE) && echo "$(ENV_FILE) exists" || { cp env.template $(ENV_FILE); echo "wrote $(ENV_FILE)"; }

check-env:
	$(UV) run --no-project python -m scripts.check_env $(ENV_FILE) --strict

test:
	PANOPTES_FAKE_MODEL=1 HF_HUB_OFFLINE=1 EHS_LIVE_BENCHMARK=0 $(UV) run --extra dev pytest tests/ -q -p no:cacheprovider

run:
	$(UV) run --env-file $(ENV_FILE) panoptes run --cell $(CELL) $(ARGS)

publications:
	$(UV) run --env-file $(ENV_FILE) modal deploy argus/platform/publication_modal.py
