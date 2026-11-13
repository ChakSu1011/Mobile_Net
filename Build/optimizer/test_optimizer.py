"""模块：Optimizer 单元与集成测试。

职责：
  覆盖 Precision、Shape、Memory 三个 Pass 的正常、边界与错误路径，
  并验证 Qwen3 全图优化结果、显存不变量与确定性。

边界：
  单元测试使用合成 ``ModelGraph``；集成测试依赖 Graph 与
  Qwen3 config.json，不依赖真实权重。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from Build.common.errors import BuildError
from Build.common.model_ir import ModelGraph
from Build.converter.converter import convert
from Build.optimizer import OptimizeConfig, Optimizer
from Build.optimizer.memory import MemoryPass
from Build.testing import (
    build_add_model,
    build_attention_model,
    build_rms_norm_model,
    build_tiny_model,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_QWEN3_CONFIG = (
    _REPOSITORY_ROOT / "Models" / "Qwen3" / "config.json")
_SMALL_CONFIG = OptimizeConfig(max_num_tokens=4)


class PrecisionPassTest(unittest.TestCase):
    """精度推导测试。"""

    def test_precision_conversion(self) -> None:
        """UT-OPT-001：BF16 转 FP16 且字节数不变。"""
        model = build_tiny_model("bfloat16")
        before = {
            name: tensor.byte_size
            for name, tensor in model.tensors.items()
            if tensor.kind == "constant"
        }
        result = Optimizer(_SMALL_CONFIG).run(model)
        for name, tensor in result.model_graph.tensors.items():
            if tensor.precision == "int64":
                continue
            self.assertEqual(tensor.precision, "float16", name)
        for name, byte_size in before.items():
            self.assertEqual(
                result.model_graph.tensor(name).byte_size,
                byte_size,
            )
        linear = next(
            op for op in result.model_graph.ops
            if op.op_type == "linear")
        self.assertEqual(
            linear.attrs["accumulate_precision"], "float32")

    def test_input_precision_preserved(self) -> None:
        """UT-OPT-002：int64 输入保持 int64。"""
        result = Optimizer(_SMALL_CONFIG).run(
            build_tiny_model())
        self.assertEqual(
            result.model_graph.tensor("input.input_ids").precision,
            "int64")

    def test_disable_precision(self) -> None:
        """UT-OPT-003：target_precision=keep 时精度不变。"""
        config = OptimizeConfig(
            target_precision="keep", max_num_tokens=4)
        result = Optimizer(config).run(
            build_tiny_model("bfloat16"))
        self.assertEqual(
            result.model_graph.tensor("head.out").precision,
            "bfloat16")
        self.assertEqual(
            result.model_graph.tensor("head.out").static_shape,
            (1, 4, 32),
        )

    def test_unsupported_conversion(self) -> None:
        """UT-OPT-004：FP32→FP16 改变元素大小时报错。"""
        with self.assertRaises(BuildError) as context:
            Optimizer(_SMALL_CONFIG).run(
                build_tiny_model("float32"))
        self.assertIn("precision conversion", str(context.exception))


class ShapePassTest(unittest.TestCase):
    """形状推导测试。"""

    def test_gather_shape(self) -> None:
        """UT-OPT-005：gather 输出 indices + weight[-1]。"""
        result = Optimizer(_SMALL_CONFIG).run(
            build_tiny_model())
        self.assertEqual(
            result.model_graph.tensor(
                "embedding.out").static_shape,
            (1, 4, 16),
        )

    def test_linear_shape(self) -> None:
        """UT-OPT-006：linear 输出末维替换为 out_features。"""
        result = Optimizer(_SMALL_CONFIG).run(
            build_tiny_model())
        self.assertEqual(
            result.model_graph.tensor("head.out").static_shape,
            (1, 4, 32),
        )

    def test_gqa_ratio(self) -> None:
        """UT-OPT-007：GQA head 不可整除时报错。"""
        with self.assertRaises(BuildError) as context:
            Optimizer(_SMALL_CONFIG).run(
                build_attention_model(16, 7))
        self.assertIn("num_kv_heads", str(context.exception))

    def test_rms_norm_divisibility(self) -> None:
        """UT-OPT-008：RMSNorm 末维不可整除时报错。"""
        with self.assertRaises(BuildError) as context:
            Optimizer(_SMALL_CONFIG).run(
                build_rms_norm_model(4, 10))
        self.assertIn("divisible", str(context.exception))

    def test_add_shape_mismatch(self) -> None:
        """UT-OPT-009：Add 输入不同形时报错。"""
        with self.assertRaises(BuildError) as context:
            Optimizer(_SMALL_CONFIG).run(
                build_add_model((1, 4), (1, 5)))
        self.assertIn("add input", str(context.exception))


class MemoryPassTest(unittest.TestCase):
    """显存规划测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        """缓存 Qwen3 转换与默认优化结果。"""
        conversion = convert("qwen3", _QWEN3_CONFIG)
        cls.qwen3_result = Optimizer(
            OptimizeConfig()).run(conversion.model_graph)

    def test_lifetimes(self) -> None:
        """UT-OPT-010：线性链 birth/last_use 正确。"""
        result = Optimizer(_SMALL_CONFIG).run(
            build_tiny_model())
        plan = result.memory_plan
        embedding = plan.entry("embedding.out")
        head = plan.entry("head.out")
        self.assertEqual((embedding.birth, embedding.last_use), (0, 1))
        self.assertEqual((head.birth, head.last_use), (1, 2))
        self.assertTrue(embedding.reusable)
        self.assertFalse(head.reusable)

    def test_reuse_no_overlap(self) -> None:
        """UT-OPT-011：生命周期重叠的张量不共享字节。"""
        entries = self.qwen3_result.memory_plan.entries
        for index, first in enumerate(entries):
            for second in entries[index + 1:]:
                lifetime_overlap = (
                    first.birth <= second.last_use
                    and second.birth <= first.last_use
                )
                if not lifetime_overlap:
                    continue
                byte_overlap = (
                    first.offset < second.end
                    and second.offset < first.end
                )
                self.assertFalse(
                    byte_overlap, f"{first.name}/{second.name}")

    def test_disable_reuse(self) -> None:
        """UT-OPT-012：关闭复用后 arena 变大且偏移单调。"""
        conversion = convert("qwen3", _QWEN3_CONFIG)
        config = OptimizeConfig(enable_reuse=False)
        result = Optimizer(config).run(conversion.model_graph)
        self.assertGreater(
            result.memory_plan.arena_bytes,
            self.qwen3_result.memory_plan.arena_bytes,
        )
        entries = result.memory_plan.entries
        end = 0
        for entry in entries:
            self.assertGreaterEqual(entry.offset, end)
            end = entry.end

    def test_alignment(self) -> None:
        """UT-OPT-013：所有偏移按 256 字节对齐。"""
        result = Optimizer(_SMALL_CONFIG).run(
            build_tiny_model())
        for entry in result.memory_plan.entries:
            self.assertEqual(entry.offset % 256, 0, entry.name)

    def test_determinism(self) -> None:
        """UT-OPT-014：连续两次优化结果完全一致。"""
        first = Optimizer(_SMALL_CONFIG).run(build_tiny_model())
        second = Optimizer(_SMALL_CONFIG).run(build_tiny_model())
        self.assertEqual(
            first.model_graph.to_dict(),
            second.model_graph.to_dict(),
        )
        self.assertEqual(
            first.memory_plan.to_dict(),
            second.memory_plan.to_dict(),
        )
        self.assertEqual(first.report, second.report)


class Qwen3OptimizerTest(unittest.TestCase):
    """Qwen3 全图优化测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        """执行一次 Qwen3 优化并缓存结果。"""
        conversion = convert("qwen3", _QWEN3_CONFIG)
        cls.result = Optimizer(OptimizeConfig()).run(
            conversion.model_graph)

    def test_shape_completeness(self) -> None:
        """UT-OPT-015：423 个 activation 全部有静态形状。"""
        activations = self.result.model_graph.activation_tensors
        self.assertEqual(len(activations), 423)
        for tensor in activations:
            self.assertIsNotNone(tensor.static_shape, tensor.name)
            self.assertTrue(tensor.static_shape, tensor.name)
            self.assertGreater(tensor.byte_size, 0, tensor.name)

    def test_report_json(self) -> None:
        """UT-OPT-016：报告可 JSON 序列化且不变量全部通过。"""
        encoded = json.dumps(
            self.result.report, ensure_ascii=False, sort_keys=True)
        self.assertGreater(len(encoded), 0)
        self.assertTrue(all(
            item["passed"]
            for item in self.result.report["invariants"]
        ))
        self.assertEqual(
            self.result.report["memory"]["entries"], 423)
        self.assertEqual(
            self.result.report["memory"]["constants_bytes"],
            1192099840,
        )


class MemoryPassIsolationTest(unittest.TestCase):
    """MemoryPass 独立运行测试。"""

    def test_memory_requires_shapes(self) -> None:
        """形状未推导时 MemoryPass 报错。"""
        with self.assertRaises(BuildError):
            MemoryPass(OptimizeConfig()).run(build_tiny_model())

    def test_metadata_unmodified(self) -> None:
        """Optimizer 不修改调用方传入的模型图。"""
        model = build_tiny_model()
        before = model.to_dict()
        Optimizer(_SMALL_CONFIG).run(model)
        self.assertEqual(model.to_dict(), before)
        self.assertIsInstance(model, ModelGraph)


if __name__ == "__main__":
    unittest.main()
