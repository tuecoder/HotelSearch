"""
Build (query, hotel) training pairs for two-tower model.

For each synthetic query, samples up to 2 positive hotels (match budget + amenities)
and up to 4 negative hotels (do not match). Skips queries with no positive match.

Output:
  data/processed/query_item_pairs.parquet

Run:
    python -m src.data_pipeline.build_query_item_pairs
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_QUERY_PARQUET = _ROOT / "data" / "processed" / "synthetic_queries.parquet"
_ITEM_PARQUET = _ROOT / "data" / "processed" / "item_features.parquet"
_OUT_DIR = _ROOT / "data" / "processed"

# rating_num bands per budget tier (inclusive on both ends)
BUDGET_RATING: dict[str, tuple[int, int]] = {
    "LUXURY":    (4, 5),
    "MID_RANGE": (2, 4),
    "BUDGET":    (0, 3),
}

# item columns to carry into the pairs file (drop string meta-columns)
_ITEM_FEATURE_COLS = [
    "hotel_id", "rating_num", "lat", "lon",
    "has_description", "description_length", "has_attractions",
    "amenity_wifi", "amenity_pool", "amenity_spa", "amenity_gym",
    "amenity_parking", "amenity_breakfast", "amenity_restaurant", "amenity_bar",
    "amenity_airport_shuttle", "amenity_air_conditioning", "amenity_front_desk_24h",
    "amenity_non_smoking", "amenity_garden", "amenity_family_rooms",
    "amenity_bike_rental", "amenity_pets_allowed", "amenity_room_service",
    "amenity_laundry", "amenity_concierge", "amenity_lift",
]


def build_pairs(
    query_parquet: Path = _QUERY_PARQUET,
    item_parquet: Path = _ITEM_PARQUET,
    out_dir: Path = _OUT_DIR,
) -> pd.DataFrame:
    query_df = pd.read_parquet(query_parquet)
    item_df = pd.read_parquet(item_parquet)[_ITEM_FEATURE_COLS]

    interactions: list[dict] = []

    for _, query in query_df.iterrows():
        lo, hi = BUDGET_RATING[query["budget_tier"]]

        # hotels in the right rating band
        candidates = item_df[
            (item_df["rating_num"] >= lo) & (item_df["rating_num"] <= hi)
        ]

        # keep only hotels that have ALL requested amenities
        for amenity in query["amenity_req"]:
            col = f"amenity_{amenity}"
            candidates = candidates[candidates[col] == 1]

        if candidates.empty:
            continue

        pos = candidates.sample(min(2, len(candidates)), random_state=42)
        neg_pool = item_df[~item_df["hotel_id"].isin(candidates["hotel_id"])]
        neg = neg_pool.sample(min(4, len(neg_pool)), random_state=42)

        # query feature dict (prefixed q__)
        query_features = {
            "q__amenity_req": query["amenity_req"],  
            "q__budget_tier": query["budget_tier"],
            "q__guests":      query["guests"],
            "q__stay_nights": query["stay_nights"],
            "q__flexible":    query["flexible"],
        }

        for _, hotel in pos.iterrows():
            hotel_features = {f"i__{col}": hotel[col] for col in _ITEM_FEATURE_COLS}
            interactions.append({**query_features, **hotel_features, "label": 1})

        for _, hotel in neg.iterrows():
            hotel_features = {f"i__{col}": hotel[col] for col in _ITEM_FEATURE_COLS}
            interactions.append({**query_features, **hotel_features, "label": 0})

    result = pd.DataFrame(interactions)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_dir / "query_item_pairs.parquet", index=False)
    print(f"Pairs: {len(result)}")
    print(result["label"].value_counts().to_string())
    return result


def main() -> None:
    build_pairs()


if __name__ == "__main__":
    main()
