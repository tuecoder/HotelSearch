from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = ROOT / "models" / "query_tower.onnx"


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="models/query_tower.onnx not found — run train_query_tower")
def test_query_tower_onnx_exists():
    assert ONNX_PATH.exists(), "models/query_tower.onnx not found — run train_query_tower"


@pytest.mark.skipif(not ONNX_PATH.exists(), reason="models/query_tower.onnx not found — run train_query_tower")
def test_query_tower_onnx_output_shape():
    import onnxruntime as rt
    sess = rt.InferenceSession(str(ONNX_PATH))
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 24), dtype=np.float32)
    out = sess.run(None, {input_name: dummy})[0]
    assert out.shape == (1, 24), f"Expected (1, 24), got {out.shape}"
