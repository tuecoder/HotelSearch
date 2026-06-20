"""
Encode raw query fields from query_item_pairs.parquet into a 24-dim float32 vector.

Query vector layout (24 dims total):
  [0:20]  amenity multi-hot  (1 per canonical amenity)
  [20]    budget_tier ordinal (BUDGET=0, MID_RANGE=1, LUXURY=2)
  [21]    guests
  [22]    stay_nights
  [23]    flexible

This module is imported by train_query_tower.py. It can also be run standalone
to verify the encoding on the pairs file.

Run:
    python -m src.data_pipeline.build_query_features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_PAIRS_PARQUET = _ROOT / "data" / "processed" / "query_item_pairs.parquet"

# Must match order in generate_queries.py / build_item_features.py exactly
AMENITIES: list[str] = [
    "wifi", "pool", "spa", "gym", "parking", "breakfast", "restaurant",
    "bar", "airport_shuttle", "air_conditioning", "front_desk_24h",
    "non_smoking", "garden", "family_rooms", "bike_rental", "pets_allowed",
    "room_service", "laundry", "concierge", "lift",
]

BUDGET_MAP: dict[str, int] = {"BUDGET": 0, "MID_RANGE": 1, "LUXURY": 2}

QUERY_DIM = 24  # 20 amenity + budget + guests + stay_nights + flexible


def encode_query(df: pd.DataFrame) -> np.ndarray:
    """
    Encode query columns (q__*) from a pairs DataFrame into a (N, 24) float32 array.

    Args:
        df: DataFrame with columns q__amenity_req, q__budget_tier,
            q__guests, q__stay_nights, q__flexible

    Returns:
        np.ndarray of shape (N, 24), dtype float32
    """
    n = len(df)

    # 20-dim multi-hot amenity vector
    amenity_matrix = np.zeros((n, len(AMENITIES)), dtype=np.float32)
    for i, amenity_list in enumerate(df["q__amenity_req"]):
        for amenity in amenity_list:
            if amenity in AMENITIES:
                amenity_matrix[i, AMENITIES.index(amenity)] = 1.0

    budget  = np.array(df["q__budget_tier"].map(BUDGET_MAP).fillna(0), dtype=np.float32).reshape(-1, 1)
    guests  = np.array(df["q__guests"], dtype=np.float32).reshape(-1, 1)
    nights  = np.array(df["q__stay_nights"], dtype=np.float32).reshape(-1, 1)
    flex    = np.array(df["q__flexible"], dtype=np.float32).reshape(-1, 1)

    return np.hstack([amenity_matrix, budget, guests, nights, flex])


def main() -> None:
    df = pd.read_parquet(_PAIRS_PARQUET)
    X = encode_query(df)
    print(f"Input pairs:   {len(df)}")
    print(f"Query matrix:  {X.shape}")   # (N, 24)
    print(f"dtype:         {X.dtype}")   # float32
    print(f"NaN count:     {np.isnan(X).sum()}")
    print(f"Sample row 0:  {X[0]}")


if __name__ == "__main__":
    main()
