from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from edge_underwater.cnn import SmallCnn
from edge_underwater.edge_benchmark import create_onnx_session, export_onnx_model
from edge_underwater.onnx_quantization import (
    OnnxQuantizationConfig,
    TrainingCalibrationReader,
    create_quantized_models,
    select_deployment_variant,
    strip_intermediate_value_info,
    variant_qualifies,
)


def config() -> OnnxQuantizationConfig:
    return OnnxQuantizationConfig(
        source_checkpoint="checkpoint.pt",
        source_checkpoint_sha256="a" * 64,
        preprocessing_config_hash="preprocessing",
        model_config_hash="model",
        training_config_hash="training",
        selected_run_config_hash="run",
    )


def test_configuration_hash_and_round_trip_are_deterministic(tmp_path: Path):
    first = config()
    second = config()
    assert first.config_hash == second.config_hash
    output = tmp_path / "config.json"
    first.save(output)
    assert OnnxQuantizationConfig.load(output) == first
    assert first.input_shape == (1, 1, 64, 155)
    assert first.output_shape == (1, 4)
    assert first.dynamic_operators == ("Gemm", "MatMul")
    assert first.static_operators == ("Conv", "Gemm", "MatMul")
    assert first.warmup_calls == 50
    assert first.measured_calls == 1_000
    assert first.intra_op_threads == 1


def calibration_rows(split: str = "train"):
    rows = [
        {
            "window_id": f"window-{index}",
            "split": split,
            "class": ("Cargo", "Passengership", "Tanker", "Tug")[index % 4],
            "vessel_group": f"vessel-{index % 43}",
        }
        for index in range(829)
    ]
    features = [torch.zeros(1, 64, 155) for _ in rows]
    return rows, features


def test_calibration_reader_is_training_only_complete_and_rewindable():
    rows, features = calibration_rows()
    reader = TrainingCalibrationReader(rows, features)

    assert reader.coverage["window_count"] == 829
    assert reader.coverage["unique_window_count"] == 829
    assert reader.coverage["classes"] == [
        "Cargo",
        "Passengership",
        "Tanker",
        "Tug",
    ]
    assert reader.coverage["vessel_group_count"] == 43
    assert reader.get_next()["features"].shape == (1, 1, 64, 155)
    reader.rewind()
    assert reader.position == 0

    validation_rows, _ = calibration_rows("validation")
    with pytest.raises(ValueError, match="training"):
        TrainingCalibrationReader(validation_rows, features)


def simple_metrics(macro_f1: float, recalls: tuple[float, ...]):
    classes = ("Cargo", "Passengership", "Tanker", "Tug")
    return {
        "macro_f1": macro_f1,
        "per_class": {
            class_name: {"recall": recalls[index]}
            for index, class_name in enumerate(classes)
        },
    }


def test_qualification_and_selection_use_quality_latency_and_size():
    settings = config()
    baseline = simple_metrics(0.50, (0.5, 0.5, 0.5, 0.5))
    good = simple_metrics(0.495, (0.5, 0.49, 0.5, 0.5))
    bad_recall = simple_metrics(0.495, (0.5, 0.4, 0.5, 0.5))

    qualifies, checks = variant_qualifies(
        baseline, good, 1.0, 1.02, 100_000, 80_000, settings
    )
    assert qualifies
    assert checks["qualifies"]
    assert not variant_qualifies(
        baseline, bad_recall, 1.0, 1.02, 100_000, 80_000, settings
    )[0]

    rows = [
        {
            "variant": "fp32",
            "qualification": {"qualifies": True},
            "timing": {"p99_ms": 1.0},
            "size_bytes": 100_000,
        },
        {
            "variant": "dynamic_int8",
            "qualification": {"qualifies": True},
            "timing": {"p99_ms": 0.9},
            "size_bytes": 90_000,
        },
        {
            "variant": "static_int8",
            "qualification": {"qualifies": True},
            "timing": {"p99_ms": 0.8},
            "size_bytes": 80_000,
        },
    ]
    assert select_deployment_variant(rows) == "static_int8"
    rows[1]["qualification"]["qualifies"] = False
    rows[2]["qualification"]["qualifies"] = False
    assert select_deployment_variant(rows) == "fp32"


def test_dynamic_and_static_quantization_have_the_intended_graph_scope(
    tmp_path: Path,
):
    torch.manual_seed(42)
    model = SmallCnn().eval()
    fp32_path = tmp_path / "fp32.onnx"
    source_path = tmp_path / "source.onnx"
    dynamic_path = tmp_path / "dynamic.onnx"
    static_path = tmp_path / "static.onnx"
    export_onnx_model(
        model,
        fp32_path,
        torch.zeros(1, 1, 64, 155),
        opset=18,
    )
    strip_intermediate_value_info(fp32_path, source_path)

    rows = [
        {
            "window_id": f"window-{index}",
            "split": "train",
            "class": ("Cargo", "Passengership", "Tanker", "Tug")[index],
            "vessel_group": f"vessel-{index}",
        }
        for index in range(4)
    ]
    features = [torch.randn(1, 64, 155) for _ in rows]
    reader = TrainingCalibrationReader(rows, features, expected_count=4)
    create_quantized_models(
        source_path,
        dynamic_path,
        static_path,
        reader,
        config(),
    )

    for path in (dynamic_path, static_path):
        graph = onnx.load(path)
        onnx.checker.check_model(graph)
        assert graph.graph.input[0].name == "features"
        assert graph.graph.output[0].name == "logits"
        assert graph.graph.initializer
        assert [
            dimension.dim_value
            for dimension in graph.graph.input[0].type.tensor_type.shape.dim
        ] == [1, 1, 64, 155]
        assert [
            dimension.dim_value
            for dimension in graph.graph.output[0].type.tensor_type.shape.dim
        ] == [1, 4]
        session = create_onnx_session(path, threads=1)
        logits = session.run(
            ["logits"],
            {"features": np.zeros((1, 1, 64, 155), dtype=np.float32)},
        )[0]
        assert logits.shape == (1, 4)
        assert np.isfinite(logits).all()

    dynamic_operators = [
        node.op_type for node in onnx.load(dynamic_path).graph.node
    ]
    assert dynamic_operators.count("Conv") == 3
    assert "MatMulInteger" in dynamic_operators

    static_graph = onnx.load(static_path)
    static_operators = [node.op_type for node in static_graph.graph.node]
    assert static_operators.count("Conv") == 3
    assert "QuantizeLinear" in static_operators
    assert "DequantizeLinear" in static_operators
    assert any(
        initializer.data_type == onnx.TensorProto.INT8
        for initializer in static_graph.graph.initializer
    )
