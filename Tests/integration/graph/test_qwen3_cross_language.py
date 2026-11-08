"""模块：Qwen3-0.6B 双端一致性集成测试。

职责：
  构建 C++ 侧（``at::Tensor``）与 Python 侧（``torch.Tensor``）的
  Qwen3-0.6B 参考图，逐字节比较 Canonical JSON，并检查节点数、边数、
  权重数、融合规则命中与 torch 原生属性。

边界：
  - 只使用 meta/CPU 元数据张量，不加载真实权重；
  - 不依赖网络或模型目录；固定超参数来自 Qwen3-0.6B config.json；
  - 金样文件缺失时由测试首次运行生成，之后只读比较。

依赖：
  ``_graph``、``mobilenet_graph.graph_tools``、
  ``mobilenet_graph.builders.qwen3``、``torch``。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_EXTENSION_DIR = os.environ.get(
    "MOBILENET_GRAPH_PYTHON_DIR",
    str(REPOSITORY_ROOT / "Build" / "graph" / "python"),
)
sys.path.insert(0, PYTHON_EXTENSION_DIR)
sys.path.insert(0, str(REPOSITORY_ROOT / "Graph" / "python"))

import torch  # noqa: E402

import _graph  # noqa: E402
from mobilenet_graph import graph_tools as tools  # noqa: E402
from mobilenet_graph.builders import qwen3  # noqa: E402

GOLDEN_PATH = REPOSITORY_ROOT / "Tests" / "golden" / "qwen3_0_6b_graph.json"
EXPECTED_NODES = 204
EXPECTED_EDGES = 570
EXPECTED_WEIGHTS = 311


class Qwen3CrossLanguageTest(unittest.TestCase):
    """双端图一致性测试。"""

    @classmethod
    def setUpClass(cls):
        """构建两端图，避免每个用例重复构图。"""
        cls.cpp_graph = _graph.build_qwen3_reference_graph()
        cls.python_graph = qwen3.build_qwen3_python_graph()
        cls.cpp_json = cls.cpp_graph.to_canonical_json()
        cls.python_json = cls.python_graph.to_canonical_json()

    def test_counts(self):
        """IT-GRAPH-008：节点/边/权重统计一致且符合预期。"""
        for graph in (self.cpp_graph, self.python_graph):
            self.assertEqual(len(graph.nodes), EXPECTED_NODES)
            self.assertEqual(len(graph.edges), EXPECTED_EDGES)
            self.assertEqual(len(graph.weights), EXPECTED_WEIGHTS)
            self.assertEqual(len(graph.topo_order), EXPECTED_NODES)
            self.assertEqual(len(graph.input_nodes), 3)
            self.assertEqual(len(graph.output_nodes), 1)

    def test_canonical_json_byte_identical(self):
        """UT-GRAPH-018/IT-GRAPH-001：两端 Canonical JSON 逐字节一致。"""
        difference = tools.first_json_difference(
            self.python_json, self.cpp_json
        )
        self.assertIsNone(difference)
        self.assertEqual(self.python_json, self.cpp_json)
        self.assertGreater(len(self.python_json), 100000)

    def test_python_graph_validates(self):
        """IT-GRAPH-001：Python 构图通过 V1~V16 校验。"""
        report = self.python_graph.validate()
        self.assertTrue(report.ok, report.to_string())
        self.assertEqual(len(report), 0)

    def test_runtime_conversion_matches(self):
        """UT-GRAPH-018：Graph IR 与 RuntimeGraph 规范 JSON 一致。"""
        self.assertEqual(
            self.cpp_json, _graph.convert_graph_to_runtime_json(self.cpp_graph)
        )
        self.assertEqual(
            self.python_json,
            _graph.convert_graph_to_runtime_json(self.python_graph),
        )

    def test_native_torch_attributes(self):
        """UT-GRAPH-026：Qwen3 图属性为 torch 原生对象。"""
        edge = self.python_graph.edge("layer.0.q")
        self.assertIsInstance(edge.tensor, torch.Tensor)
        self.assertEqual(edge.tensor.dtype, torch.bfloat16)
        self.assertEqual(tuple(edge.tensor.shape), (1, 8, 2048))
        node = self.python_graph.node("layer.0.qkv")
        self.assertIsInstance(node.precision, torch.dtype)
        self.assertEqual(node.precision, torch.bfloat16)
        self.assertEqual(node.attributes["num_heads"], 16)
        weight = self.python_graph.weight("embedding.weight")
        self.assertIsInstance(weight.tensor, torch.Tensor)
        self.assertEqual(tuple(weight.tensor.shape), (151936, 1024))
        config = self.python_graph.model_config
        self.assertIsInstance(config.dtype, torch.dtype)
        self.assertEqual(config.dtype, torch.bfloat16)
        self.assertEqual(config.model_family, "qwen3")

    def test_weight_tie_shares_tensor(self):
        """IT-GRAPH-001：lm_head 与 embedding 共享张量。"""
        embedding = self.python_graph.weight("embedding.weight").tensor
        lm_head = self.python_graph.weight("lm_head.weight").tensor
        self.assertEqual(embedding.data_ptr(), lm_head.data_ptr())
        self.assertEqual(embedding.untyped_storage().data_ptr(),
                         lm_head.untyped_storage().data_ptr())

    def test_fusion_rules_hit_every_compute_node(self):
        """IT-GRAPH-001：Qwen3 计算节点的融合规则均可查询。"""
        table = _graph.FusionRuleTable.instance()
        rule_ids = {rule.rule_id for rule in table.rules}
        for layer in (0, 13, 27):
            for suffix in (
                "attn_norm",
                "qkv",
                "attn",
                "attn_residual",
                "mlp_norm",
                "mlp_gate_up",
                "mlp_down_residual",
            ):
                node = self.python_graph.node(f"layer.{layer}.{suffix}")
                self.assertIn(node.fusion_rule_id, rule_ids)
                self.assertIsNotNone(table.find(node.fusion_rule_id))

    def test_golden_snapshot(self):
        """IT-GRAPH-008：与金样快照逐字节一致。"""
        if not GOLDEN_PATH.exists():
            GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(self.cpp_json, encoding="utf-8")
        expected = GOLDEN_PATH.read_text(encoding="utf-8")
        difference = tools.first_json_difference(self.cpp_json, expected)
        self.assertIsNone(difference)

    def test_canonical_json_stable_across_calls(self):
        """UT-GRAPH-017：重复调用输出稳定。"""
        self.assertEqual(self.cpp_json, self.cpp_graph.to_canonical_json())
        self.assertEqual(
            self.python_json, self.python_graph.to_canonical_json()
        )


if __name__ == "__main__":
    unittest.main()
