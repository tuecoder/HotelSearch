# Hotel Search Engine

Portfolio-scale hotel search and ranking system inspired by Booking.com.

## Status
- UI: FastAPI query parser with mock ranking.
- Query understanding: preprocessing, budget classification (ONNX), flexibility detection.
- Data and ML pipelines: placeholders.

## Quick Start

```bash
make dev       # start Duckling (Docker) in background + FastAPI app
```

Then open: http://127.0.0.1:8080

## Make Commands

| Command | Description |
|---|---|
| `make run` | Start the FastAPI app on port 8080 |
| `make duckling` | Start the Duckling date-parser container (port 8000) |
| `make dev` | Start Duckling in background, then launch the app |
| `make train` | Retrain the budget classifier and export to ONNX |
| `make install` | `pip install -r requirements.txt` |
| `make install-dev` | Install dev extras: pytest, ruff, jupyter |
| `make venv` | Create the `hotelenv` virtual environment |
| `make notebook` | Open Jupyter Lab in `notebooks/` |
| `make test` | Run the test suite |
| `make test-v` | Run tests with verbose output |
| `make lint` | Lint source with ruff |
| `make format` | Auto-format source with ruff |
| `make data-info` | Print a summary of the raw CSV dataset |
| `make model-info` | List saved ONNX models |
| `make clean` | Remove `__pycache__` and `.pyc` files |
| `make clean-all` | Remove cache and delete `hotelenv` entirely |

Run `make help` to see this list at any time.

## Manual Setup (without make)

```bash
# 1. Create venv and install dependencies
python -m venv hotelenv
hotelenv/Scripts/pip install -r requirements.txt

# 2. Start the Duckling date parser
docker-compose up -d duckling

# 3. Train the budget classifier (first time only)
hotelenv/Scripts/python -m src.query_understanding.budget_classifier

# 4. Start the app
hotelenv/Scripts/python -m uvicorn src.app.streamlit_app:app --reload --port 8080
```

## Requirements

- Python 3.12+
- Docker (for Duckling)
- `make` (Git Bash / WSL on Windows)
