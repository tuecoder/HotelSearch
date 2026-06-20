from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

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
_FEATURE_PARQUET = _ROOT / "data" / "processed" / "item_features.parquet"
_OUT_DIR = _ROOT / "models" / "item_tower.onnx"


def build_item_tower(item_parquet: Path = _FEATURE_PARQUET,
                     model_path: Path = _OUT_DIR
                     ):
    df = pd.read_parquet(item_parquet)
    X = df[FEATURE_COLS].fillna(0).astype(np.float32)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svd", TruncatedSVD(n_components=24, random_state=42))
    ])
    pipeline.fit(X)

    explained = pipeline["svd"].explained_variance_ratio_.sum()
    print(f"Explained variance: {explained:.3f}")

    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
    onnx_model = convert_sklearn(pipeline, initial_types=initial_type)

    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    mlflow.set_tracking_uri(f"sqlite:///{_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("item-tower")

    with mlflow.start_run():
        mlflow.log_param("n_components", 24)
        mlflow.log_metric("explained_variance", explained)
        mlflow.log_artifact(str(model_path))
        mlflow.sklearn.log_model(pipeline, "item_tower_sklearn",
                                registered_model_name="item-tower")
        

if __name__ == "__main__":
    build_item_tower()
