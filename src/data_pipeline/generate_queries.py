"""
Generate 20,000 synthetic search queries for two-tower training.

Each query represents a parsed SearchQuery with structured fields only
(no raw text / destination — city filtering is handled by ES upstream).

Output:
  data/processed/synthetic_queries.parquet

Run:
    python -m src.data_pipeline.generate_queries
or:
    make gen-queries
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "data" / "processed" / "synthetic_queries.parquet"

N_QUERIES = 20000
RANDOM_SEED = 42

# Must match keys in build_item_features.AMENITY_PATTERNS exactly
AMENITIES = [
    "wifi",
    "pool",
    "spa",
    "gym",
    "parking",
    "breakfast",
    "restaurant",
    "bar",
    "airport_shuttle",
    "air_conditioning",
    "front_desk_24h",
    "non_smoking",
    "garden",
    "family_rooms",
    "bike_rental",
    "pets_allowed",
    "room_service",
    "laundry",
    "concierge",
    "lift",
]

BUDGET_TIERS = ["BUDGET", "MID_RANGE", "LUXURY"]
BUDGET_WEIGHTS = [0.40, 0.40, 0.20]


def generate_queries(n: int = N_QUERIES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a DataFrame of n synthetic queries."""
    random.seed(seed)
    rows = []
    for _ in range(n):

        k = random.randint(0,4)
        amenity_req = random.sample(AMENITIES, k)

        budget_tier = random.choices(BUDGET_TIERS, weights=BUDGET_WEIGHTS)[0]

        guests = random.randint(1, 8)

        stay_nights = random.randint(1, 14)

        flexible = random.choices([0, 1], weights=(40, 60))[0]

        rows.append(
            {
                "amenity_req": amenity_req,   # list[str], e.g. ["wifi", "pool"]
                "budget_tier": budget_tier,   # str
                "guests": guests,             # int
                "stay_nights": stay_nights,   # int
                "flexible": flexible,         # int 0 or 1
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = generate_queries()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT, index=False)
    print(f"Saved {len(df)} queries -> {_OUT}")
    print(df.dtypes)
    print(df.head())


if __name__ == "__main__":
    main()
