from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _make_fake_artifacts(tmp_path):
    """Create fake embeddings and index parquet for testing."""
    emb = np.array([[0.1] * 24, [0.2] * 24, [0.3] * 24], dtype=np.float32)
    emb_path = tmp_path / "item_tower_embeddings.npy"
    np.save(emb_path, emb)

    idx = pd.DataFrame({
        "row_idx":    [0, 1, 2],
        "hotel_id":   [1001, 1002, 1003],
        "hotel_name": ["Hotel Alpha", "Hotel Beta", "Hotel Gamma"],
        "city":       ["Amsterdam", "Rotterdam", "Utrecht"],
    })
    idx_path = tmp_path / "item_tower_index.parquet"
    idx.to_parquet(idx_path, index=False)

    return emb, emb_path, idx_path


def test_attach_embeddings_by_name(tmp_path):
    from src.retrieval.es_indexer import _attach_embeddings

    emb, emb_path, idx_path = _make_fake_artifacts(tmp_path)
    hotels = [
        {"id": 1, "name": "Hotel Alpha"},
        {"id": 2, "name": "Hotel Beta"},
        {"id": 3, "name": "Hotel Gamma"},
    ]
    _attach_embeddings(hotels, emb_path, idx_path)

    assert "tower_embedding" in hotels[0]
    assert len(hotels[0]["tower_embedding"]) == 24
    np.testing.assert_allclose(hotels[0]["tower_embedding"], [0.1] * 24, rtol=1e-5)
    np.testing.assert_allclose(hotels[1]["tower_embedding"], [0.2] * 24, rtol=1e-5)
    np.testing.assert_allclose(hotels[2]["tower_embedding"], [0.3] * 24, rtol=1e-5)


def test_attach_embeddings_missing_files_is_noop(tmp_path):
    from src.retrieval.es_indexer import _attach_embeddings

    hotels = [{"id": 1, "name": "Hotel Alpha"}]
    _attach_embeddings(hotels, tmp_path / "missing.npy", tmp_path / "missing.parquet")
    assert "tower_embedding" not in hotels[0]


def test_attach_embeddings_unmatched_name_skipped(tmp_path):
    from src.retrieval.es_indexer import _attach_embeddings

    emb, emb_path, idx_path = _make_fake_artifacts(tmp_path)
    hotels = [{"id": 99, "name": "Unknown Hotel XYZ"}]
    _attach_embeddings(hotels, emb_path, idx_path)
    # No match by name — hotel should have no tower_embedding
    assert "tower_embedding" not in hotels[0]
