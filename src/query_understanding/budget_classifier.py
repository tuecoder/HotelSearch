"""
Budget Tier Classifier — Component 1c of Query Understanding
=============================================================
Architecture:
  TfidfVectorizer  (8 000 unigram+bigram features, sublinear TF)
       |   <- "small LLM encoder": dense bag-of-bigrams text representation
       v
  MLPClassifier   (256 -> 128 -> 4, ReLU, early stopping)
       |   <- "modified classification head"
       v
  Label  {0=LUXURY, 1=MID_RANGE, 2=BUDGET, 3=UNSPECIFIED}

Training logs params + per-class metrics to MLflow, registers the sklearn
pipeline under the name "budget-classifier", and saves the ONNX artefact
alongside it.  Inference uses only the ONNX model via onnxruntime.

Usage
-----
Training (run once):
    python -m src.query_understanding.budget_classifier

Inference (in app):
    clf = BudgetClassifier()
    tier: BudgetTier = clf.predict("cheap family hotel in amsterdam")

MLflow UI:
    mlflow ui --backend-store-uri mlruns/
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _ROOT / "models" / "budget_classifier.onnx"
_DATA_PATH = _ROOT / "data" / "processed" / "budget_classification.csv"
_MLFLOW_TRACKING_URI = f"sqlite:///{_ROOT / 'mlflow.db'}"
_REGISTRY_NAME = "budget-classifier"

LABEL_MAP: dict[int, str] = {
    0: "LUXURY",
    1: "MID_RANGE",
    2: "BUDGET",
    3: "UNSPECIFIED",
}


class BudgetTier(IntEnum):
    LUXURY = 0
    MID_RANGE = 1
    BUDGET = 2
    UNSPECIFIED = 3


# ---------------------------------------------------------------------------
# Training, evaluation, and MLflow tracking
# ---------------------------------------------------------------------------

def train_and_export(output_path: Path | None = None) -> Path:
    """Train TF-IDF + MLP, track with MLflow, register model, export to ONNX.

    What gets logged to MLflow
    --------------------------
    Parameters:
        tfidf_max_features, tfidf_ngram_range, tfidf_sublinear_tf,
        mlp_hidden_layer_sizes, mlp_activation, mlp_max_iter,
        mlp_early_stopping, test_size, random_state

    Metrics (overall + per-class):
        accuracy, precision_macro, recall_macro, f1_macro,
        precision_<class>, recall_<class>, f1_<class>

    Artifacts:
        budget_classifier.onnx  (ONNX model for runtime inference)

    Model Registry:
        sklearn pipeline registered as "budget-classifier"

    Returns the path to the saved .onnx file.
    """
    import mlflow
    import mlflow.sklearn
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType

    out = output_path or MODEL_PATH
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── Hyperparameters ─────────────────────────────────────────────────────
    TFIDF_PARAMS = dict(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
    MLP_PARAMS = dict(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # ── Data ────────────────────────────────────────────────────────────────
    df = pd.read_csv(_DATA_PATH)
    X, y = df["text"].tolist(), df["label"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # ── Pipeline ────────────────────────────────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("clf",   MLPClassifier(**MLP_PARAMS)),
    ])

    # ── MLflow run ──────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment("budget-classifier")

    with mlflow.start_run():

        # -- Train -----------------------------------------------------------
        print("Training budget classifier ...")
        pipeline.fit(X_train, y_train)

        # -- Evaluate --------------------------------------------------------
        y_pred = pipeline.predict(X_test)
        report = classification_report(
            y_test, y_pred,
            target_names=list(LABEL_MAP.values()),
            output_dict=True,
        )
        print(classification_report(
            y_test, y_pred, target_names=list(LABEL_MAP.values())
        ))

        # -- Log parameters --------------------------------------------------
        mlflow.log_params({
            "tfidf_max_features":    TFIDF_PARAMS["max_features"],
            "tfidf_ngram_range":     str(TFIDF_PARAMS["ngram_range"]),
            "tfidf_sublinear_tf":    TFIDF_PARAMS["sublinear_tf"],
            "mlp_hidden_layer_sizes": str(MLP_PARAMS["hidden_layer_sizes"]),
            "mlp_activation":        MLP_PARAMS["activation"],
            "mlp_max_iter":          MLP_PARAMS["max_iter"],
            "mlp_early_stopping":    MLP_PARAMS["early_stopping"],
            "test_size":             TEST_SIZE,
            "random_state":          RANDOM_STATE,
            "train_samples":         len(X_train),
            "test_samples":          len(X_test),
        })

        # -- Log overall metrics ---------------------------------------------
        mlflow.log_metrics({
            "accuracy":        round(report["accuracy"], 4),
            "precision_macro": round(report["macro avg"]["precision"], 4),
            "recall_macro":    round(report["macro avg"]["recall"], 4),
            "f1_macro":        round(report["macro avg"]["f1-score"], 4),
        })

        # -- Log per-class metrics -------------------------------------------
        for label_name in LABEL_MAP.values():
            mlflow.log_metrics({
                f"precision_{label_name}": round(report[label_name]["precision"], 4),
                f"recall_{label_name}":    round(report[label_name]["recall"], 4),
                f"f1_{label_name}":        round(report[label_name]["f1-score"], 4),
                f"support_{label_name}":   int(report[label_name]["support"]),
            })

        # -- Export ONNX and log as artifact ---------------------------------
        initial_type = [("text_input", StringTensorType([None, 1]))]
        onx = convert_sklearn(pipeline, initial_types=initial_type, target_opset=17)
        with open(out, "wb") as f:
            f.write(onx.SerializeToString())

        mlflow.log_artifact(str(out), artifact_path="onnx")
        print(f"ONNX model saved: {out}")

        # -- Register sklearn pipeline in Model Registry ---------------------
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="sklearn_pipeline",
            registered_model_name=_REGISTRY_NAME,
        )
        print(f"Model registered: {_REGISTRY_NAME}  run_id={model_info.run_id}")

    return out


# ---------------------------------------------------------------------------
# Inference  (onnxruntime only — no MLflow dependency at request time)
# ---------------------------------------------------------------------------

class BudgetClassifier:
    """Loads the ONNX budget classifier and classifies query strings.

    Parameters
    ----------
    model_path : Path, optional
        Override the default models/budget_classifier.onnx path.

    Examples
    --------
    >>> clf = BudgetClassifier()
    >>> clf.predict("cheap hostel amsterdam")
    <BudgetTier.BUDGET: 2>
    >>> clf.predict("luxury five-star spa hotel")
    <BudgetTier.LUXURY: 0>
    """

    def __init__(self, model_path: Path | None = None) -> None:
        import onnxruntime as rt

        path = model_path or MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {path}. "
                "Run: python -m src.query_understanding.budget_classifier"
            )
        opts = rt.SessionOptions()
        opts.log_severity_level = 3  # suppress INFO logs
        self._session = rt.InferenceSession(str(path), sess_options=opts)
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, text: str) -> BudgetTier:
        """Return the predicted BudgetTier for a query string."""
        inp = np.array([[text]])
        label_idx = int(self._session.run(None, {self._input_name: inp})[0][0])
        return BudgetTier(label_idx)

    def predict_label(self, text: str) -> str:
        """Return the string label e.g. 'BUDGET'."""
        return LABEL_MAP[self.predict(text).value]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_and_export()
