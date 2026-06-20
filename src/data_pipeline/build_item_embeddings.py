from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as rt

FEATURE_COLS = [
    "rating_num", "lat", "lon",
    "has_description", "description_length", "has_attractions",
    "amenity_wifi", "amenity_pool", "amenity_spa", "amenity_gym",
    "amenity_parking", "amenity_breakfast", "amenity_restaurant", "amenity_bar",
    "amenity_airport_shuttle", "amenity_air_conditioning", "amenity_front_desk_24h",
    "amenity_non_smoking", "amenity_garden", "amenity_family_rooms",
    "amenity_bike_rental", "amenity_pets_allowed", "amenity_room_service",
    "amenity_laundry", "amenity_concierge", "amenity_lift",
]

_ROOT = Path(__file__).resolve().parents[2]
_ITEM_PARQUET  = _ROOT / "data" / "processed" / "item_features.parquet"
_MODEL_PATH    = _ROOT / "models" / "item_tower.onnx"
_EMB_OUT       = _ROOT / "data" / "processed" / "item_tower_embeddings.npy"
_INDEX_OUT     = _ROOT / "data" / "processed" / "item_tower_index.parquet"

def make_embeddings(item_features: Path = _ITEM_PARQUET,
                    model: Path = _MODEL_PATH,
                    embeddings_path: Path = _EMB_OUT,
                    index_path: Path = _INDEX_OUT ):
    
    df = pd.read_parquet(item_features)
    X = df[FEATURE_COLS].fillna(0).astype(np.float32).values

    sess = rt.InferenceSession(str(model))
    input_name = sess.get_inputs()[0].name   # "float_input"
    embeddings = sess.run(None, {input_name: X})[0]  
    np.save(embeddings_path, embeddings)

    index_df = pd.DataFrame({
        "row_idx": np.arange(len(df)),
        "hotel_id": df["hotel_id"].values,
        "hotel_name": df["hotel_name"].values,
        "city": df["city"]
    })
    index_df.to_parquet(index_path, index = False)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved: {embeddings_path}")
    print(f"Saved {index_path}")

    from numpy.linalg import norm

    # pick two Amsterdam hotels and one non-Amsterdam hotel
    ams_rows = index_df[index_df["city"] == "Amsterdam"].index[:2].tolist()
    other_row = index_df[index_df["city"] != "Amsterdam"].index[0]

    def cosine(a, b):
        return np.dot(a, b) / (norm(a) * norm(b))

    sim_same = cosine(embeddings[ams_rows[0]], embeddings[ams_rows[1]])
    sim_diff = cosine(embeddings[ams_rows[0]], embeddings[other_row])
    print(f"Same city similarity:  {sim_same:.3f}")
    print(f"Cross city similarity: {sim_diff:.3f}")
    # same city should be higher


if __name__ == "__main__":
    make_embeddings()
