# `make help` lists every target. Variables are overridable: `make train-local PRESET=e4b`.
.DEFAULT_GOAL := help

PRESET  ?= e4b-instruct
RELEASE ?= $(PRESET)
NAME    ?=
REPO    ?=
OUT     ?= data/mixture
EVAL    ?= data/eval
S1      ?= data/s1bench
LIMIT   ?= 20000
STEPS   ?= 20
SOURCES ?=
URL     ?= http://localhost:8000
FRESH   ?=
RESUME  ?=
GPUS    ?= 1
MLFLOW  ?=

help: ## Show this help
	@awk 'BEGIN{FS=":.*?## "} \
		/^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5)} \
		/^[a-zA-Z0-9_-]+:.*?## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

##@ Setup

.PHONY: setup setup-train setup-mlflow setup-serve setup-modal
setup: ## Install the workspace
	uv sync

setup-train: ## Install with the training extras (torch, transformers, peft)
	uv sync --extra train

setup-mlflow: ## Training extras plus MLflow
	uv sync --extra mlflow

setup-serve: ## Install with the serving extras (adds fastapi, uvicorn)
	uv sync --extra serve

setup-modal: ## Install with the Modal client
	uv sync --extra modal

##@ Develop

.PHONY: test lint fmt check clean
test: ## Run every test
	uv run pytest

lint: ## Check formatting and lint rules
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Apply formatting and autofixes
	uv run ruff check --fix .
	uv run ruff format .

check: lint test ## Lint then test — run before pushing

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache dist build

##@ Data

.PHONY: plan check-data data eval-set s1bench
plan: ## Print the PRESET's training budget without spending it
	uv run gev plan --preset $(PRESET)

check-data: ## Contamination guard over a source list (SOURCES=path)
	uv run gev check-data $(SOURCES)

data: ## Build the training mixture locally (LIMIT=rows per source)
	uv run gev data build --out $(OUT) --limit-per-source $(LIMIT)

eval-set: ## Export the held-out split as gevbench task files
	uv run gev data eval --data $(OUT) --out $(EVAL)

s1bench: ## Export the S1Bench eval subsets as gevbench task files
	uv run gev s1bench export --out $(S1)

##@ Train

.PHONY: smoke-local train-local calibrate-local mlflow-ui smoke train calibrate
smoke-local: ## Train Gemma 4 E2B for a few steps — proves the path without Modal
	uv run gev data build --out /tmp/gev-mix --limit-per-source 2000 --n-examples 4000
	uv run gev train --preset smoke --data /tmp/gev-mix \
		--output-dir /tmp/gev-smoke --max-steps $(STEPS)

# GPUS>1 launches one data-parallel process per GPU through torchrun.
TRAIN_CMD = $(if $(filter-out 1,$(GPUS)),uv run torchrun --standalone --nproc_per_node $(GPUS) -m gev.cli,uv run gev)

train-local: ## Train PRESET here; resumes the newest checkpoint (GPUS=2, MLFLOW=1, FRESH=1, RESUME=path)
	$(TRAIN_CMD) train --preset $(PRESET) --data $(OUT) $(if $(FRESH),--fresh,) $(if $(RESUME),--resume $(RESUME),) $(if $(MLFLOW),--mlflow,)

calibrate-local: ## Fit per-bucket temperatures for PRESET's newest local checkpoint (MLFLOW=1)
	uv run gev calibrate --preset $(PRESET) --data $(OUT) $(if $(MLFLOW),--mlflow,)

mlflow-ui: ## Browse local MLflow runs at http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

smoke: ## Modal: exercise the whole training path on Gemma 4 E2B
	modal run modal/app.py::smoke

train: ## Modal: the real run; resumes the newest checkpoint (FRESH=1 to start over, RESUME=path)
	modal run modal/app.py::train --preset $(PRESET) $(if $(FRESH),--fresh,) $(if $(RESUME),--resume $(RESUME),)

calibrate: ## Modal: fit per-bucket temperatures after training
	modal run modal/app.py::calibrate --preset $(PRESET)

##@ Serve and release

.PHONY: serve deploy release weights publish
serve: ## Modal: dev server on an H100 (ephemeral; any `modal run` on this app steals its URL)
	GEV_SERVE_PRESET=$(PRESET) modal serve modal/app.py

deploy: ## Modal: deploy /v1/systemone with a stable URL (PRESET selects the checkpoint)
	GEV_SERVE_PRESET=$(PRESET) modal deploy modal/app.py

release: ## Modal: package the newest PRESET checkpoint into a release dir on the volume
	modal run modal/app.py::export_checkpoint --preset $(PRESET) $(if $(NAME),--name $(NAME),)

weights: ## Pull a packaged release to weights/RELEASE (RELEASE=name, default PRESET)
	mkdir -p weights
	modal volume get --force gev-checkpoints releases/$(RELEASE) weights/

publish: ## Upload weights/RELEASE to the Hub (REPO=org/name; needs HF_TOKEN)
	uv run gev release publish weights/$(RELEASE) --repo $(REPO)

##@ Benchmark

.PHONY: bench bench-local sweep snake
bench: s1bench ## Benchmark Jev on the S1Bench subsets (needs TYPESAFE_API_KEY)
	uv run gevbench eval --backend jev --tasks $(S1)

bench-local: s1bench ## Benchmark a /v1/systemone server on the S1Bench subsets (URL=server)
	uv run gevbench eval --backend gev --base-url $(URL) --tasks $(S1)

sweep: ## Measure shared-state batching economics
	uv run gevbench sweep --backend jev

snake: ## The decision model plays Snake over /v1/systemone (URL=server; --record in artifacts/)
	uv run gevbench snake --backend gev --base-url $(URL) --record artifacts/snake/run.jsonl
