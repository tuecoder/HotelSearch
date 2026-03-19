"""
Elasticsearch Hotel Indexer
===========================
Builds and populates the `hotels` index from the processed Netherlands
Excel file.  Run once after Elasticsearch is up:

    python -m src.retrieval.es_indexer
    # or
    make index
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_ROOT        = Path(__file__).resolve().parents[2]
_EXCEL_PATH  = _ROOT / "data" / "processed" / "netherlands_hotels.xlsx"
_MAPPING_PATH = _ROOT / "configs" / "es_mapping.json"

ES_HOST  = "http://localhost:9200"
INDEX    = "hotels"

# ── Rating helpers (mirrored from streamlit_app so we don't import it) ──────

_RATING_TO_STARS: dict[str, int] = {
    "fivestar": 5, "fourstar": 4, "threestar": 3,
    "twostar": 2,  "onestar": 1,  "all": 3,
}
_RATING_TO_SCORE: dict[str, float] = {
    "fivestar": 9.5, "fourstar": 8.5, "threestar": 7.5,
    "twostar": 6.5,  "onestar": 6.0,  "all": 7.0,
}
_AMENITY_KEYWORDS: dict[str, list[str]] = {
    "pool":         ["pool", "swimming"],
    "gym":          ["gym", "fitness", "health club"],
    "wifi":         ["wifi", "wi-fi", "internet"],
    "spa":          ["spa", "wellness"],
    "breakfast":    ["breakfast"],
    "parking":      ["parking"],
    "family rooms": ["family room", "family suite"],
    "restaurant":   ["restaurant"],
    "bar":          ["bar", "lounge"],
    "bike rental":  ["bicycle", "bike rental"],
}


def _clean_description(raw: str) -> str:
    text = raw.strip()
    text = re.sub(
        r"^HeadLine\s*:\s*.+?Location\s*:\s*", "", text, flags=re.IGNORECASE
    )
    for bad, good in {
        "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"',
        "\x96": "-", "\x97": "-",
    }.items():
        text = text.replace(bad, good)
    return " ".join(text.split())


def _parse_amenities(facilities: str) -> list[str]:
    if not facilities or facilities.lower() in ("nan", "n/a", ""):
        return []
    fac_lower = facilities.lower()
    return [
        amenity
        for amenity, keywords in _AMENITY_KEYWORDS.items()
        if any(kw in fac_lower for kw in keywords)
    ]


def _load_hotels() -> list[dict[str, Any]]:
    import pandas as pd
    df = pd.read_excel(_EXCEL_PATH)
    hotels = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        rating_key = str(row.HotelRating).strip().lower()
        hotels.append({
            "id":          idx,
            "name":        str(row.HotelName).strip(),
            "city":        str(row.cityName).strip(),
            "country":     str(row.countyName).strip(),
            "stars":       _RATING_TO_STARS.get(rating_key, 3),
            "rating":      _RATING_TO_SCORE.get(rating_key, 7.0),
            "address":     str(row.Address).strip(),
            "attractions": str(row.Attractions).strip(),
            "description": _clean_description(str(row.Description)),
            "fax":         str(row.FaxNumber).strip(),
            "phone":       str(row.PhoneNumber).strip(),
            "amenities":   _parse_amenities(str(row.HotelFacilities)),
        })
    return hotels


def build_index(es_host: str = ES_HOST, index: str = INDEX) -> None:
    """Create (or recreate) the hotels index and bulk-index all properties."""
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk

    es = Elasticsearch(es_host)

    # Verify connectivity
    if not es.ping():
        raise ConnectionError(
            f"Cannot reach Elasticsearch at {es_host}. "
            "Run: docker-compose up -d elasticsearch"
        )

    # Load mapping
    mapping = json.loads(_MAPPING_PATH.read_text())

    # Recreate index (drop stale data)
    if es.indices.exists(index=index):
        es.indices.delete(index=index)
        print(f"Dropped existing index '{index}'")

    es.indices.create(index=index, mappings=mapping["mappings"])
    print(f"Created index '{index}'")

    # Load and bulk-index
    hotels = _load_hotels()
    actions = [
        {"_index": index, "_id": h["id"], "_source": h}
        for h in hotels
    ]
    success, errors = bulk(es, actions, raise_on_error=False)
    if errors:
        print(f"  {len(errors)} errors during bulk index")
    print(f"Indexed {success} documents into '{index}'")


if __name__ == "__main__":
    build_index()
