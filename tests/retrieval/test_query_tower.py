from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "models" / "query_tower.onnx"


def test_query_tower_onnx_exists():
    assert ONNX_PATH.exists(), "models/query_tower.onnx not found — run: make train-towers"


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="models/query_tower.onnx not found — run train_query_tower")
def test_query_tower_onnx_output_shape():
    import onnxruntime as rt
    sess = rt.InferenceSession(str(ONNX_PATH))
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 24), dtype=np.float32)
    out = sess.run(None, {input_name: dummy})[0]
    assert out.size == 24, f"Expected 24 output values, got shape {out.shape}"


def test_train_query_tower_produces_onnx(tmp_path):
    """End-to-end smoke test: train on tiny synthetic data, verify ONNX output shape."""
    import pandas as pd
    import numpy as np
    from src.retrieval.train_query_tower import train_query_tower

    # Build minimal synthetic pairs — needs enough rows for sklearn's early_stopping
    # internal 90/10 val split on the 75% train portion: 50*0.75*0.1 ≈ 3 val samples (>=2)
    n = 50
    rng = np.random.default_rng(0)
    pairs_rows = []
    for i in range(n):
        pairs_rows.append({
            "q__amenity_req": [],
            "q__budget_tier": "MID_RANGE",
            "q__guests": 2,
            "q__stay_nights": 3,
            "q__flexible": 1,
            "i__hotel_id": i + 1,
            "label": 1,
        })
    pairs_df = pd.DataFrame(pairs_rows)
    pairs_path = tmp_path / "pairs.parquet"
    pairs_df.to_parquet(pairs_path, index=False)

    # Fake embeddings: n hotels, 24-dim
    embeddings = rng.random((n, 24)).astype(np.float32)
    emb_path = tmp_path / "embeddings.npy"
    np.save(emb_path, embeddings)

    # Fake index parquet
    index_df = pd.DataFrame({
        "row_idx": list(range(n)),
        "hotel_id": list(range(1, n + 1)),  # hotel_ids 1..n
        "hotel_name": [f"Hotel {i}" for i in range(n)],
        "city": ["Amsterdam"] * n,
    })
    idx_path = tmp_path / "index.parquet"
    index_df.to_parquet(idx_path, index=False)

    model_out = tmp_path / "query_tower.onnx"

    # Patch mlflow so the test doesn't write to the real DB
    import unittest.mock as mock
    with mock.patch("src.retrieval.train_query_tower.mlflow"):
        mean_sim = train_query_tower(
            pairs_path=pairs_path,
            items_path=emb_path,
            index_path=idx_path,
            model_out=model_out,
        )

    assert model_out.exists(), "ONNX file was not created"
    assert isinstance(mean_sim, float)

    import onnxruntime as rt
    sess = rt.InferenceSession(str(model_out))
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 24), dtype=np.float32)
    out = sess.run(None, {input_name: dummy})[0]
    # skl2onnx ≥1.17 produces (n_targets, 1) for multi-output regressors;
    # verify by total size rather than exact shape.
    assert out.size == 24, f"Expected 24 output values, got shape {out.shape}"
