"""
Item Feature Pipeline — input for the Two-Tower item tower
===========================================================
Reads  : data/raw/hotels.csv  (Netherlands hotels, latin-1)
Writes :
  data/processed/item_features.parquet      — tabular features
  data/processed/description_embeddings.npy — sentence vectors (N, 384)

Run:
    python -m src.data_pipeline.build_item_features
or:
    make features
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_RAW_CSV = _ROOT / "data" / "raw" / "hotels.csv"
_OUT_DIR = _ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Amenity canonical map
# Each key becomes a binary column  amenity_<key>  (0 / 1).
# Values are regex patterns matched case-insensitively against HotelFacilities.
# ---------------------------------------------------------------------------
AMENITY_PATTERNS: dict[str, list[str]] = {
    "wifi":             [r"wi.?fi", r"internet", r"wireless"],
    "pool":             [r"pool", r"swimming"],
    "spa":              [r"\bspa\b"],
    "gym":              [r"\bgym\b", r"fitness", r"workout"],
    "parking":          [r"parking"],
    "breakfast":        [r"breakfast"],
    "restaurant":       [r"restaurant"],
    "bar":              [r"\bbar\b"],
    "airport_shuttle":  [r"airport.shuttle", r"airport.transfer", r"airport.transport"],
    "air_conditioning": [r"air.condition"],
    "front_desk_24h":   [r"24.hour", r"24-hour", r"front.desk"],
    "non_smoking":      [r"non.smoking", r"no.smoking"],
    "garden":           [r"\bgarden\b"],
    "family_rooms":     [r"family.room"],
    "bike_rental":      [r"bike.rental", r"bicycle"],
    "pets_allowed":     [r"\bpet\b", r"\bdog\b"],
    "room_service":     [r"room.service"],
    "laundry":          [r"laundry", r"dry.clean"],
    "concierge":        [r"concierge"],
    "lift":             [r"\blift\b", r"elevator"],
}

# Pre-compile each pattern group into one regex for speed
_AMENITY_RE: dict[str, re.Pattern] = {
    name: re.compile("|".join(patterns), re.IGNORECASE)
    for name, patterns in AMENITY_PATTERNS.items()
}

RATING_MAP: dict[str, int] = {
    "FiveStar": 5,
    "FourStar": 4,
    "ThreeStar": 3,
    "TwoStar": 2,
    "OneStar": 1,
    "All": 0,
}


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_amenities(facilities: pd.Series) -> pd.DataFrame:
    """Return a DataFrame with one binary column per canonical amenity."""
    fac = facilities.fillna("").astype(str)
    rows: dict[str, pd.Series] = {}
    for name, regex in _AMENITY_RE.items():
        rows[f"amenity_{name}"] = fac.str.contains(regex).astype(int)
    return pd.DataFrame(rows, index=facilities.index)


def _extract_coords(map_col: pd.Series) -> pd.DataFrame:
    """Split 'lat|lon' strings into float columns."""
    split = map_col.str.split("|", expand=True)
    lat = pd.to_numeric(split[0], errors="coerce")
    lon = pd.to_numeric(split[1], errors="coerce")
    return pd.DataFrame({"lat": lat, "lon": lon}, index=map_col.index)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_item_features(
    raw_csv: Path = _RAW_CSV,
    out_dir: Path = _OUT_DIR,
    embed_model: str = "BAAI/bge-small-en-v1.5",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & filter ────────────────────────────────────────────────────
    print("Loading raw CSV ...")
    df = pd.read_csv(raw_csv, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    df = df[df["countyName"].str.strip() == "Netherlands"].reset_index(drop=True)
    print(f"Loaded {len(df):,} Netherlands hotels")

    # ── 2. Scalar features ──────────────────────────────────────────────────
    df["rating_num"] = (
        df["HotelRating"].str.strip().map(RATING_MAP).fillna(0).astype(int)
    )
    coords = _extract_coords(df["Map"])
    df["lat"] = coords["lat"]
    df["lon"] = coords["lon"]
    df["has_description"]    = df["Description"].notna().astype(int)
    df["description_length"] = df["Description"].fillna("").str.len()
    df["has_attractions"]    = df["Attractions"].notna().astype(int)

    # ── 3. Amenity one-hot ──────────────────────────────────────────────────
    print("Extracting amenities ...")
    amenity_df = _extract_amenities(df["HotelFacilities"])
    print(f"Extracted {len(amenity_df.columns)} amenity columns for {len(df):,} hotels")

    # ── 4. Assemble tabular output ──────────────────────────────────────────
    keep = [
        "HotelCode", "HotelName", "cityName", "cityCode", "countyCode",
        "rating_num", "lat", "lon",
        "has_description", "description_length", "has_attractions",
    ]
    features = pd.concat([df[keep], amenity_df], axis=1).rename(columns={
        "HotelCode":  "hotel_id",
        "HotelName":  "hotel_name",
        "cityName":   "city",
        "cityCode":   "city_code",
        "countyCode": "country_code",
    })

    parquet_path = out_dir / "item_features.parquet"
    features.to_parquet(parquet_path, index=False)
    print(f"Saved: {parquet_path}  ({features.shape[0]:,} rows x {features.shape[1]} cols)")

    # ── 5. Description embeddings via fastembed (ONNX, no PyTorch) ──────────
    print(f"Generating description embeddings with {embed_model} ...")
    from fastembed import TextEmbedding  # imported late — large optional dep

    model = TextEmbedding(model_name=embed_model)
    texts = df["Description"].fillna("").tolist()
    embeddings = np.array(list(model.embed(texts)), dtype=np.float32)

    emb_path = out_dir / "description_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"Generated embeddings shape {embeddings.shape}")
    print(f"Saved: {emb_path}")


if __name__ == "__main__":
    build_item_features()
