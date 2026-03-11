"""
Budget Tier Classifier — Component 1c of Query Understanding
=============================================================
Architecture:
  TfidfVectorizer  (8 000 unigram+bigram features, sublinear TF)
       │   ← "small LLM encoder" substitute: dense text representation
       ▼
  MLPClassifier   (256 → 128 → 4, ReLU, early stopping)
       │   ← "modified classification head"
       ▼
  Label  {0=LUXURY, 1=MID_RANGE, 2=BUDGET, 3=UNSPECIFIED}

The fitted pipeline is exported to ONNX via skl2onnx and served at
inference time by onnxruntime — no PyTorch required.

Usage
-----
Training (run once):
    python -m src.query_understanding.budget_classifier

Inference (in app):
    clf = BudgetClassifier()
    tier: BudgetTier = clf.predict("cheap family hotel in amsterdam")
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
# Training & ONNX export
# ---------------------------------------------------------------------------

def train_and_export(output_path: Path | None = None) -> Path:
    """Train TF-IDF + MLP on budget_classification.csv and export to ONNX.

    Returns the path to the saved .onnx file.
    """
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

    # ── Load data ──────────────────────────────────────────────────────────
    df = pd.read_csv(_DATA_PATH)
    X, y = df["text"].tolist(), df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Pipeline: TF-IDF encoder → MLP head ────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )),
    ])

    print("Training budget classifier …")
    pipeline.fit(X_train, y_train)

    # ── Evaluate ────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    print(classification_report(
        y_test, y_pred, target_names=list(LABEL_MAP.values())
    ))

    # ── Export to ONNX ──────────────────────────────────────────────────────
    # Input: 2-D string tensor  shape=(batch, 1)
    initial_type = [("text_input", StringTensorType([None, 1]))]
    onx = convert_sklearn(pipeline, initial_types=initial_type, target_opset=17)
    with open(out, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"ONNX model saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Inference
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
        # shape: (1, 1) — batch of one, one text column
        inp = np.array([[text]])
        label_idx = int(self._session.run(None, {self._input_name: inp})[0][0])
        return BudgetTier(label_idx)

    def predict_label(self, text: str) -> str:
        """Return the string label e.g. 'BUDGET'."""
        return LABEL_MAP[self.predict(text).value]


# ---------------------------------------------------------------------------
# Entry point — run training from command line
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_and_export()
