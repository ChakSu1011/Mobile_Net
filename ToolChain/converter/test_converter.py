"""模块：Converter 单元与集成测试。

职责：
  验证 RecordingGraph 代理一致性、元数据记录、Qwen3 Adapter 计数、
  权重映射、JSON 往返与错误路径。

边界：
  依赖 Graph 原生扩展与 Qwen3 config.json；
  不依赖真实权重、GPU、torch 或 transformers。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ToolChain.common.errors import BuildError
from ToolChain.common.graph_api import load_graph_api
from ToolChain.common.model_ir import ModelGraph
from ToolChain.converter.converter import convert
from ToolChain.converter.qwen3_adapter import huggingface_weight_name
from ToolChain.converter.recording_graph import RecordingGraph
from ToolChain.testing import build_tiny_model

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_QWEN3_CONFIG = (
    _REPOSITORY_ROOT / "Models" / "Qwen3" / "config.json")


def _valid_inference_config(**overrides) -> dict:
    """返回合法的引擎输入配置字典。

    参数：
      overrides：需要覆盖的字段。

    返回：
      扁平配置字典，含全部必填字段。
    """
    data = {
        "max_num_tokens": 256,
        "max_new_tokens": 2048,
        "max_batch_size": 1,
        "repetition_penalty": 1.0,
        "seed": 0,
    }
    data.update(overrides)
    return data


def _build_demo_graph(graph, api) -> None:
    """在给定 Graph 或 RecordingGraph 上构建演示图。

    参数：
      graph：原生 Graph 或 RecordingGraph；
      api：Graph 模块，提供枚举。

    返回：
      无。
    """
    source = graph.create_input(
        "input.x", api.Precision.kFloat16, api.Layout.kRowMajor)
    weight = graph.create_constant(
        "weight.w",
        dims=[4, 4],
        pitch=[4, 4],
        precision=api.Precision.kFloat16,
        layout=api.Layout.kRowMajor,
        byte_size=32,
        data_ptr=0,
    )
    output = graph.create_op(
        "linear",
        [source, weight],
        1,
        api.OpType.kLinear,
        {
            "precision": "float16",
            "layout": "row_major",
            "in_features": 4,
            "out_features": 4,
        },
    )[0]
    graph.create_output("output.y", output)
    graph.sort_op()


class RecordingGraphTest(unittest.TestCase):
    """代理记录行为测试。"""

    def setUp(self) -> None:
        """创建 Graph 模块与原生图。"""
        self.api = load_graph_api()
        self.native = self.api.Graph("demo")
        _build_demo_graph(self.native, self.api)
        self.recording = RecordingGraph(
            self.api.Graph("demo"), name="demo", adapter="test")
        _build_demo_graph(self.recording, self.api)
        self.model = self.recording.finish()

    def test_proxy_ir_matches_native(self) -> None:
        """UT-CONV-001：代理图与原图 IR 完全一致。"""
        self.assertEqual(
            self.native.dump_ir(), self.recording.dump_ir())

    def test_input_record(self) -> None:
        """UT-CONV-002：输入 kind/precision/layout 正确。"""
        tensor = self.model.tensor("input.x")
        self.assertEqual(tensor.kind, "input")
        self.assertEqual(tensor.precision, "float16")
        self.assertEqual(tensor.layout, "row_major")
        self.assertFalse(tensor.reusable)

    def test_constant_record(self) -> None:
        """UT-CONV-003：常量 dims/pitch/byte_size/data_ptr 正确。"""
        tensor = self.model.tensor("weight.w")
        self.assertEqual(tensor.kind, "constant")
        self.assertEqual(tensor.static_shape, (4, 4))
        self.assertEqual(tensor.pitch, (4, 4))
        self.assertEqual(tensor.byte_size, 32)
        self.assertEqual(tensor.data_ptr, 0)

    def test_multi_output_names(self) -> None:
        """UT-CONV-004：多输出按 name.out.<i> 记录。"""
        graph = RecordingGraph(
            self.api.Graph("multi"), name="multi", adapter="test")
        first = graph.create_input(
            "input.a", self.api.Precision.kFloat16,
            self.api.Layout.kRowMajor)
        second = graph.create_input(
            "input.b", self.api.Precision.kFloat16,
            self.api.Layout.kRowMajor)
        outputs = graph.create_op(
            "add",
            [first, second],
            2,
            self.api.OpType.kAdd,
            {"precision": "float16", "layout": "row_major"},
        )
        self.assertEqual(
            [tensor.name for tensor in outputs],
            ["add.out.0", "add.out.1"],
        )
        graph.create_output("output.a", outputs[0])
        graph.sort_op()
        model = graph.finish()
        self.assertEqual(
            model.ops[0].outputs, ("add.out.0", "add.out.1"))

    def test_invalid_attr(self) -> None:
        """UT-CONV-005：非 JSON 安全属性抛 BuildError。"""
        graph = RecordingGraph(
            self.api.Graph("bad"), name="bad", adapter="test")
        source = graph.create_input(
            "input.x", self.api.Precision.kFloat16,
            self.api.Layout.kRowMajor)
        with self.assertRaises(BuildError):
            graph.create_op(
                "linear",
                [source],
                1,
                self.api.OpType.kLinear,
                {"bad": object()},
            )

    def test_output_binding(self) -> None:
        """UT-CONV-006：输出绑定正确且输出张量不可复用。"""
        self.assertEqual(
            [binding.name for binding in self.model.outputs],
            ["output.y"],
        )
        self.assertEqual(
            self.model.outputs[0].tensor, "linear.out")
        self.assertFalse(
            self.model.tensor("linear.out").reusable)


class InferenceConfigTest(unittest.TestCase):
    """运行时输入配置测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        """执行一次 Qwen3 转换并缓存结果。"""
        cls.conversion = convert("qwen3", _QWEN3_CONFIG)

    def _write_model_dir(
        self,
        directory: str,
        *,
        inference: dict | None = None,
        generation: dict | None = None,
    ) -> Path:
        """在临时目录写出最小模型配置。

        参数：
          directory：临时目录；
          inference：可选输入配置字典；
          generation：可选 generation_config 字典。

        返回：
          config.json 路径。
        """
        root = Path(directory)
        data = json.loads(
            _QWEN3_CONFIG.read_text(encoding="utf-8"))
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps(data), encoding="utf-8")
        if generation is not None:
            (root / "generation_config.json").write_text(
                json.dumps(generation), encoding="utf-8")
        if inference is not None:
            input_dir = root / "input"
            input_dir.mkdir()
            (input_dir / "inference_config.json").write_text(
                json.dumps(inference), encoding="utf-8")
        return config_path

    def test_merged_values(self) -> None:
        """UT-CONV-013：合并官方 config 与引擎输入配置。"""
        config = self.conversion.inference_config
        self.assertIsNotNone(config)
        data = config.to_dict()
        self.assertEqual(data["context_length"], 40960)
        self.assertEqual(data["vocab_size"], 151936)
        self.assertEqual(data["max_num_tokens"], 256)
        self.assertEqual(data["max_new_tokens"], 2048)
        self.assertEqual(data["max_batch_size"], 1)
        self.assertTrue(data["do_sample"])
        self.assertEqual(data["temperature"], 0.6)
        self.assertEqual(data["top_k"], 20)
        self.assertEqual(data["top_p"], 0.95)
        self.assertEqual(data["repetition_penalty"], 1.0)
        self.assertEqual(data["seed"], 0)
        self.assertEqual(data["stop_token_ids"], [151645, 151643])
        self.assertEqual(data["bos_token_id"], 151643)
        self.assertEqual(data["pad_token_id"], 151643)
        self.assertEqual(
            self.conversion.model_graph.metadata[
                "inference_config"],
            data,
        )

    def test_missing_inference_config(self) -> None:
        """UT-CONV-014：缺少输入配置时保持兼容。"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_model_dir(directory)
            conversion = convert("qwen3", config_path)
        self.assertIsNone(conversion.inference_config)
        self.assertIsNone(
            conversion.model_graph.metadata["inference_config"])

    def test_repeated_official_field(self) -> None:
        """UT-CONV-015：官方参数不得在新配置重复声明。"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_model_dir(
                directory,
                inference=_valid_inference_config(temperature=0.5),
            )
            with self.assertRaises(BuildError) as context:
                convert("qwen3", config_path)
        self.assertIn("must be read from", str(context.exception))

    def test_out_of_range(self) -> None:
        """UT-CONV-016：max_num_tokens 超出上下文时报错。"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_model_dir(
                directory,
                inference=_valid_inference_config(
                    max_num_tokens=999999),
            )
            with self.assertRaises(BuildError) as context:
                convert("qwen3", config_path)
        self.assertIn("max_num_tokens", str(context.exception))

    def test_null_generation_values(self) -> None:
        """UT-CONV-019：官方采样字段为 null 时使用缺省值。"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_model_dir(
                directory,
                inference=_valid_inference_config(),
                generation={
                    "do_sample": None,
                    "temperature": None,
                    "top_k": None,
                    "top_p": None,
                },
            )
            conversion = convert("qwen3", config_path)
        config = conversion.inference_config
        self.assertFalse(config.do_sample)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_k, 0)
        self.assertEqual(config.top_p, 1.0)

    def test_invalid_generation_config(self) -> None:
        """UT-CONV-017：官方采样参数非法时报错。"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_model_dir(
                directory,
                inference=_valid_inference_config(),
                generation={"top_p": 2.0},
            )
            with self.assertRaises(BuildError) as context:
                convert("qwen3", config_path)
        self.assertIn("top_p", str(context.exception))

    def test_unknown_field(self) -> None:
        """UT-CONV-018：未知字段报错。"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_model_dir(
                directory,
                inference=_valid_inference_config(extra=1),
            )
            with self.assertRaises(BuildError) as context:
                convert("qwen3", config_path)
        self.assertIn("unknown field", str(context.exception))


class Qwen3ConversionTest(unittest.TestCase):
    """Qwen3 转换集成测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        """执行一次 Qwen3 转换并缓存结果。"""
        cls.conversion = convert("qwen3", _QWEN3_CONFIG)
        cls.model = cls.conversion.model_graph

    def test_qwen3_counts(self) -> None:
        """UT-CONV-007：图计数与 IR 计数符合预期。"""
        self.assertEqual(len(self.model.inputs), 2)
        self.assertEqual(len(self.model.constants), 310)
        self.assertEqual(len(self.model.ops), 311)
        self.assertEqual(len(self.model.outputs), 1)
        self.assertEqual(len(self.model.tensors), 623)
        ir_text = self.conversion.ir_text
        self.assertIn("\nops 314\n", ir_text)

    def test_weight_mapping(self) -> None:
        """UT-CONV-008：所有常量都有 HF 映射与静态形状。"""
        for name in self.model.constants:
            tensor = self.model.tensor(name)
            self.assertIsNotNone(tensor.weight_source, name)
            self.assertIsNotNone(tensor.static_shape, name)
            self.assertEqual(
                tensor.weight_source,
                huggingface_weight_name(name),
            )

    def test_tied_weights(self) -> None:
        """UT-CONV-009：tied 配置不登记 lm_head 常量。"""
        tied = self.model.metadata["tied_weights"]
        self.assertEqual(
            tied.get("lm_head.weight"), "embedding.weight")
        self.assertNotIn("lm_head.weight", self.model.constants)
        self.assertEqual(
            self.model.tensor("embedding.weight").static_shape,
            (151936, 1024),
        )

    def test_untied_weights(self) -> None:
        """UT-CONV-009：untied 配置登记独立 lm_head 常量。"""
        with tempfile.TemporaryDirectory() as directory:
            data = json.loads(
                _QWEN3_CONFIG.read_text(encoding="utf-8"))
            data["tie_word_embeddings"] = False
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(data), encoding="utf-8")
            conversion = convert("qwen3", config_path)
        self.assertIn(
            "lm_head.weight", conversion.model_graph.constants)
        self.assertEqual(
            len(conversion.model_graph.constants), 311)
        self.assertEqual(
            conversion.model_graph.tensor(
                "lm_head.weight").weight_source,
            "lm_head.weight",
        )

    def test_json_roundtrip(self) -> None:
        """UT-CONV-010：ModelGraph JSON 往返一致。"""
        roundtrip = ModelGraph.from_dict(self.model.to_dict())
        self.assertEqual(self.model.to_dict(), roundtrip.to_dict())

    def test_missing_weights(self) -> None:
        """UT-CONV-011：权重目录缺失时不报错且全部不可用。"""
        conversion = convert(
            "qwen3",
            _QWEN3_CONFIG,
            weights_path=Path("does-not-exist"),
        )
        model = conversion.model_graph
        self.assertFalse(model.metadata["weights_supplied"])
        self.assertTrue(all(
            not model.tensor(name).weight_available
            for name in model.constants
        ))

    def test_invalid_model_graph(self) -> None:
        """UT-CONV-012：引用缺失时自校验失败并含对象名。"""
        data = build_tiny_model().to_dict()
        data["tensors"] = [
            item for item in data["tensors"]
            if item["name"] != "head.weight"
        ]
        with self.assertRaises(BuildError) as context:
            ModelGraph.from_dict(data)
        self.assertIn("head.weight", str(context.exception))


if __name__ == "__main__":
    unittest.main()
