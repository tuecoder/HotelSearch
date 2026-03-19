# ============================================================
#  HotelSearch — project task runner
#  Usage:  make <target>
#          make          (defaults to `help`)
# ============================================================

PYTHON  := hotelenv/Scripts/python
PIP     := hotelenv/Scripts/pip
PYTEST  := hotelenv/Scripts/pytest
RUFF    := hotelenv/Scripts/ruff

APP_MODULE  := src.app.streamlit_app:app
APP_PORT    := 8080
DATA_DIR    := data
MODELS_DIR  := models
SRC_DIR     := src
TESTS_DIR   := tests

.DEFAULT_GOAL := help

# ── Colours (no-op on terminals that don't support ANSI) ────────────────────
CYAN  := \033[0;36m
RESET := \033[0m

# ============================================================
#  HELP
# ============================================================
.PHONY: help
help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS=":.*##"}; {printf "  $(CYAN)%-22s$(RESET) %s\n", $$1, $$2}'

# ============================================================
#  ENVIRONMENT
# ============================================================
.PHONY: install install-dev venv

venv:          ## Create the hotelenv virtual environment
	python -m venv hotelenv

install:       ## Install production dependencies into hotelenv
	$(PIP) install -r requirements.txt

install-dev:   ## Install dev extras (pytest, ruff, jupyter)
	$(PIP) install pytest pytest-asyncio httpx ruff jupyter

# ============================================================
#  APPLICATION
# ============================================================
.PHONY: run duckling dev

run:           ## Start the FastAPI app on port $(APP_PORT)
	$(PYTHON) -m uvicorn $(APP_MODULE) --reload --port $(APP_PORT)

duckling:      ## Start the Duckling date-parser container (port 8000)
	docker-compose up duckling

dev:           ## Start Duckling + Elasticsearch in background, then launch the app
	docker-compose up -d duckling elasticsearch
	$(PYTHON) -m uvicorn $(APP_MODULE) --reload --port $(APP_PORT)

# ============================================================
#  TRAINING
# ============================================================
.PHONY: train

train:         ## Train the budget classifier, track with MLflow, export to ONNX
	$(PYTHON) -m src.query_understanding.budget_classifier

index:         ## Build the Elasticsearch hotels index (run once after ES starts)
	$(PYTHON) -m src.retrieval.es_indexer

mlflow-ui:     ## Open the MLflow tracking UI (http://localhost:5000)
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ============================================================
#  NOTEBOOK
# ============================================================
.PHONY: notebook

notebook:      ## Launch Jupyter Lab in the notebooks/ directory
	$(PYTHON) -m jupyter lab notebooks/

# ============================================================
#  TESTS
# ============================================================
.PHONY: test test-v

test:          ## Run the test suite
	$(PYTEST) $(TESTS_DIR) -q

test-v:        ## Run tests with verbose output
	$(PYTEST) $(TESTS_DIR) -v

# ============================================================
#  CODE QUALITY
# ============================================================
.PHONY: lint format check

lint:          ## Lint with ruff
	$(RUFF) check $(SRC_DIR) $(TESTS_DIR)

format:        ## Auto-format with ruff
	$(RUFF) format $(SRC_DIR) $(TESTS_DIR)

check: lint    ## Alias for lint

# ============================================================
#  DATA & MODELS
# ============================================================
.PHONY: data-info model-info

data-info:     ## Print a quick summary of the raw CSV dataset
	$(PYTHON) -c "\
import pandas as pd; \
df = pd.read_csv('$(DATA_DIR)/raw/hotels.csv', encoding='latin-1', nrows=5); \
df.columns = df.columns.str.strip(); \
print('Columns:', df.columns.tolist()); \
full = pd.read_csv('$(DATA_DIR)/raw/hotels.csv', encoding='latin-1'); \
full.columns = full.columns.str.strip(); \
print('Rows:', len(full)); \
print('Countries:', full['countyName'].nunique()); \
"

model-info:    ## List ONNX models in models/
	@ls -lh $(MODELS_DIR)/*.onnx 2>/dev/null || echo "No ONNX models found -- run: make train"

# ============================================================
#  CLEAN
# ============================================================
.PHONY: clean clean-all

clean:         ## Remove Python cache files
	find . -type d -name __pycache__ ! -path ./hotelenv/\* -exec rm -rf {} + 2>/dev/null; \
	find . -type f -name "*.pyc"     ! -path ./hotelenv/\* -delete 2>/dev/null; \
	echo "Cache cleaned."

clean-all: clean  ## Remove cache AND hotelenv (full reset)
	rm -rf hotelenv
	@echo "hotelenv removed. Run: make venv && make install"
