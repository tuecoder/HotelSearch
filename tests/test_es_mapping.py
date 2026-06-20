import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPING_PATH = ROOT / "configs" / "es_mapping.json"


def test_tower_embedding_field_present():
    mapping = json.loads(MAPPING_PATH.read_text())
    props = mapping["mappings"]["properties"]
    assert "tower_embedding" in props, "tower_embedding missing from es_mapping.json"


def test_tower_embedding_is_dense_vector():
    mapping = json.loads(MAPPING_PATH.read_text())
    field = mapping["mappings"]["properties"]["tower_embedding"]
    assert field["type"] == "dense_vector"
    assert field["dims"] == 24
    assert field["index"] is True
    assert field["similarity"] == "cosine"
