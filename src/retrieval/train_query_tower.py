from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.data_pipeline.build_query_features import encode_query

_ROOT = Path(__file__).resolve().parents[2]
_PAIRS_PARQUET  = _ROOT / "data" / "processed" / "query_item_pairs.parquet"
_ITEM_EMBEDDING = _ROOT / "data" / "processed" / "item_tower_embeddings.npy"
_INDEX_PARQUET  = _ROOT / "data" / "processed" / "item_tower_index.parquet"
_MODEL_OUT      = _ROOT / "models" / "query_tower.onnx"


def train_query_tower(
    pairs_path: Path = _PAIRS_PARQUET,
    items_path: Path = _ITEM_EMBEDDING,
    index_path: Path = _INDEX_PARQUET,
    model_out:  Path = _MODEL_OUT,
) -> float:
    """Train MLP query tower, export to ONNX, log to MLflow.

    Returns mean cosine similarity on the validation set.
    """
    pairs_df   = pd.read_parquet(pairs_path)
    index_df   = pd.read_parquet(index_path)
    embeddings = np.load(items_path)

    pos = pairs_df[pairs_df["label"] == 1].reset_index(drop=True)
    X   = encode_query(pos)

    id_to_row = dict(zip(index_df["hotel_id"], index_df["row_idx"]))
    hotel_ids = pos["i__hotel_id"].values
    y = np.array([embeddings[id_to_row[hid]] for hid in hotel_ids], dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    sims = [cosine_similarity(y_pred[i:i+1], y_val[i:i+1])[0][0] for i in range(len(y_val))]
    mean_sim = float(np.mean(sims))
    print(f"Mean cosine similarity on val: {mean_sim:.4f}")

    # ONNX export
    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    model_out.parent.mkdir(exist_ok=True)
    with open(model_out, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX model -> {model_out}")

    # MLflow
    mlflow.set_tracking_uri(f"sqlite:///{_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("query-tower")
    with mlflow.start_run():
        mlflow.log_param("hidden_layer_sizes", "128,64")
        mlflow.log_param("activation", "relu")
        mlflow.log_param("max_iter", 500)
        mlflow.log_metric("mean_cosine_sim_val", mean_sim)
        mlflow.log_artifact(str(model_out))
        mlflow.sklearn.log_model(model, "query_tower_sklearn",
                                 registered_model_name="query-tower")

    # Quick verify
    import onnxruntime as rt
    sess = rt.InferenceSession(str(model_out))
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, X.shape[1]), dtype=np.float32)
    out = sess.run(None, {input_name: dummy})[0]
    assert out.shape == (1, 24), f"ONNX output shape mismatch: {out.shape}"
    print(f"ONNX verify OK — output shape: {out.shape}")

    return mean_sim


if __name__ == "__main__":
    train_query_tower()
