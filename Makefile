# Panoptes model services on the GPU boxes (docs/ENTERPRISE-PLAN-2026-10-07.md §2.1, docs/BACKENDS-v1.md). The config is .env.
#   make test                 contract tests against the fake backends (no GPU, no weights)
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

.PHONY: test-contract context images up down logs smoke

include Makefile.handoff   # env, check-env, check-env-live, lint, test (whole offline suite), run CELL=…

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
	@test -f .env || { echo ".env missing: cp .env.example .env and fill PANOPTES_SERVICE_API_KEY, WEIGHTS"; exit 1; }
	$(COMPOSE) up -d $(SERVICES)

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200 $(SERVICES)

smoke:
	set -a && . ./.env && set +a && serving/smoke_v1.sh http://127.0.0.1:$(PORT_$(GPU))
