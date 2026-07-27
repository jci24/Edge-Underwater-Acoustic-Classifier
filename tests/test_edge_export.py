from pathlib import Path

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from edge_underwater.cnn import SmallCnn
from edge_underwater.edge_benchmark import (
    create_onnx_session,
    export_onnx_model,
    validate_onnx_parity,
)


def test_static_onnx_export_has_embedded_weights_and_matches_pytorch(tmp_path: Path):
    torch.manual_seed(42)
    model = SmallCnn().eval()
    output = tmp_path / "small_cnn.onnx"
    example = torch.zeros(1, 1, 64, 155)

    export_onnx_model(model, output, example, opset=18)
    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)

    input_shape = [
        dimension.dim_value
        for dimension in onnx_model.graph.input[0].type.tensor_type.shape.dim
    ]
    output_shape = [
        dimension.dim_value
        for dimension in onnx_model.graph.output[0].type.tensor_type.shape.dim
    ]
    assert input_shape == [1, 1, 64, 155]
    assert output_shape == [1, 4]
    assert onnx_model.graph.initializer
    assert all(
        initializer.data_location != onnx.TensorProto.EXTERNAL
        for initializer in onnx_model.graph.initializer
    )

    session = create_onnx_session(output, threads=1)
    features = [
        torch.zeros(1, 64, 155),
        torch.randn(1, 64, 155),
    ]
    parity = validate_onnx_parity(model, session, features)
    assert parity["window_count"] == 2
    assert parity["matching_prediction_count"] == 2
    assert parity["all_predictions_match"]
    assert np.isfinite(parity["maximum_absolute_error"])
