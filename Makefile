# Panoptes: the handoff targets (pipeline machine, needs uv) and the model services on the GPU boxes (docker compose only;
# docs/ENTERPRISE-PLAN-2026-10-07.md §2.1, docs/BACKENDS-v1.md). The config is .env; env.template is the one template.
#   make env / check-env / check-env-live   .env from env.template; validate it (--strict); + GET /healthz on every service URL
#   make lint / test          ruff + shell syntax; the whole offline test suite with the fake model backend (what CI runs)
#   make run CELL=090         panoptes run --cell 090 [ARGS="--from S4"]
#   make test-contract        contract tests against the fake backends (no GPU, no weights)
#   make context              the docker build context: ehs-spatial ($(PANOPTES_WORKCELL)) + this repo -> $(CONTEXT)
#   make images GPU=a|b       the base images (ehs-spatial recipes, unchanged) then the service images of that card
#   make up GPU=a|b           start that card's stack: the services serving/registry.yaml lists for it
#   make down / logs / smoke  GPU=a|b
GPU ?= a
PY ?= python3
CONTEXT ?= build-context
PANOPTES_WORKCELL ?= .
VERSION ?= v1
COMPOSE = docker compose --env-file .env -f deploy/compose.gpu-$(GPU).yml
SERVICES = $(shell $(PY) serving/registry.py services $(GPU) | tr '\n' ' ')
CODE_SHA = $(shell git rev-parse HEAD)
PORT_a = 8805
PORT_b = 8804
UV ?= uv
ENV_FILE ?= .env
CELL ?= 090
ARGS ?=

.PHONY: env check-env check-env-live lint test run test-contract context images up down logs smoke

env:            ## copy env.template -> .env once (never overwrites)
	@test -f $(ENV_FILE) && echo "$(ENV_FILE) exists, leaving it" || { cp env.template $(ENV_FILE); echo "wrote $(ENV_FILE): fill sections 1-4, then make check-env"; }

check-env:      ## validate .env: required keys, backend values, URL forms, no HF_TOKEN; placeholders and missing paths fail (--strict)
	$(UV) run --no-project python scripts/check_env.py $(ENV_FILE) --strict

check-env-live: ## check-env + GET /healthz through the jump on every *_HTTP_URL(S)
	$(UV) run --no-project python scripts/check_env.py $(ENV_FILE) --strict --live

lint:           ## ruff (ruff.toml) + shell syntax by shebang
	$(UV)x --from 'ruff==0.2.1' ruff check .
	@git ls-files '*.sh' | while read -r f; do if head -1 "$$f" | grep -q zsh; then zsh -n "$$f"; else bash -n "$$f"; fi || exit 1; done; echo "shell syntax ok"

test:           ## offline test suite, fake model backend (what CI runs)
	PANOPTES_FAKE_MODEL=1 HF_HUB_OFFLINE=1 $(UV) run pytest tests/ -q

run:            ## panoptes run --cell $(CELL) $(ARGS)   e.g. make run CELL=030 ARGS="--from S4"
	$(UV) run --env-file $(ENV_FILE) panoptes run --cell $(CELL) $(ARGS)

test-contract:
	$(PY) -m pytest tests/contract -q

context:
	rm -rf $(CONTEXT) && deploy/context.sh $(CONTEXT) $(PANOPTES_WORKCELL)

images: context
ifeq ($(GPU),a)
	docker build -f $(CONTEXT)/workcell/docker/sam3d.Dockerfile -t panoptes-sam3d $(CONTEXT)
	docker build -f $(CONTEXT)/serving/deploy/Dockerfile.sam3d --build-arg CODE_SHA=$(CODE_SHA) -t panoptes-sam3d-service:$(VERSION) $(CONTEXT)
else
	docker build -f $(CONTEXT)/workcell/docker/da3.Dockerfile -t panoptes-workcell-da3 $(CONTEXT)
	docker build -f $(CONTEXT)/workcell/docker/geometry.Dockerfile -t panoptes-workcell-geometry $(CONTEXT)
	docker build -f $(CONTEXT)/workcell/docker/workcell-gpu.Dockerfile -t panoptes-workcell-gpu $(CONTEXT)
	docker build -f $(CONTEXT)/serving/deploy/Dockerfile.geometry --build-arg CODE_SHA=$(CODE_SHA) -t panoptes-geometry-service:$(VERSION) $(CONTEXT)
endif
	docker build -f $(CONTEXT)/serving/deploy/Dockerfile.serving -t panoptes-serving:$(VERSION) $(CONTEXT)

up:
	@test -f .env || { echo ".env missing: make env, then fill PANOPTES_SERVICE_API_KEY and WEIGHTS (env.template section 1)"; exit 1; }
	$(COMPOSE) up -d $(SERVICES)

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200 $(SERVICES)

smoke:
	set -a && . ./.env && set +a && serving/smoke_v1.sh http://127.0.0.1:$(PORT_$(GPU))
