"""模块：显存排布可视化工具测试。

职责：
  验证包读取、Activation/权重区块、Pitch、planned offset、Layer 折叠与 SVG。

边界：
  使用仓库内 Qwen3 模型包；不读取权重内容、不执行推理。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[2]
_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for _path in (_REPOSITORY_ROOT, _TOOLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from graph_memory.layout import layout_memory  # noqa: E402
from graph_memory.package import load_model_package  # noqa: E402
from graph_memory.repeats import analyze_layer_runs  # noqa: E402
from graph_memory.svg import render_memory_svg  # noqa: E402
from graph_memory.view_model import build_memory_view  # noqa: E402

_QWEN3_PACKAGE = (
    _REPOSITORY_ROOT / "Models" / "Qwen3" / "artifacts" / "qwen3_0_6b")


class MemoryVisualizerTest(unittest.TestCase):
    """显存视图与 SVG 行为测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        """加载一次 Qwen3 包并构建视图。"""
        cls.package = load_model_package(_QWEN3_PACKAGE)
        cls.view = build_memory_view(cls.package)
        cls.layout = layout_memory(cls.view)

    def test_counts(self) -> None:
        """UT-MEM-001：Activation/权重计数与 summary 正确。"""
        self.assertEqual(len(self.view.activation_blocks), 312)
        self.assertEqual(len(self.view.weight_blocks), 310)
        self.assertEqual(self.view.arena_bytes, 78317568)
        self.assertEqual(self.view.peak_bytes, 78315520)
        self.assertEqual(self.view.constants_bytes, 1192099840)

    def test_activation_fields(self) -> None:
        """UT-MEM-002：Activation 字段与 pitch 推导。"""
        block = next(
            item for item in self.view.activation_blocks
            if item.tensor == "layer.0.q_path.out")
        self.assertEqual(block.owner_op, "layer.0.q_path")
        self.assertEqual(block.shape, (1, 256, 2048))
        self.assertEqual(block.pitch, (1, 256, 2048))
        self.assertTrue(block.pitch_derived)
        self.assertEqual(block.size, 1048576)
        self.assertEqual(block.offset, 1048576)

    def test_weight_fields(self) -> None:
        """UT-MEM-003：权重 owner、shape、pitch 与 planned offset。"""
        block = next(
            item for item in self.view.weight_blocks
            if item.name == "layer.0.q_proj.weight")
        self.assertEqual(block.owner_ops, ("layer.0.q_path",))
        self.assertEqual(block.shape, (2048, 1024))
        self.assertEqual(block.pitch, (2048, 1024))
        self.assertEqual(block.size, 4194304)
        self.assertFalse(self.view.planned_weights)
        self.assertEqual(block.offset % 256, 0)

    def test_shared_weight_owner(self) -> None:
        """UT-MEM-004：tied embedding 显示多个 owner。"""
        block = next(
            item for item in self.view.weight_blocks
            if item.name == "embedding.weight")
        self.assertEqual(
            set(block.owner_ops), {"embedding", "lm_head"})

    def test_layer_runs(self) -> None:
        """UT-MEM-006：Qwen3 的 Layer 布局折叠为 ×28。"""
        self.assertEqual(len(self.view.activation_runs), 1)
        self.assertEqual(self.view.activation_runs[0].count, 28)
        self.assertFalse(self.view.activation_runs[0].is_variant)
        self.assertEqual(len(self.layout.strips), 28)

    def test_synthetic_variant(self) -> None:
        """UT-MEM-007：签名不同的层标记为差异。"""
        signatures = {
            0: (("a",), ("w",)),
            1: (("a",), ("w",)),
            2: (("b",), ("w",)),
        }
        runs = analyze_layer_runs(signatures)
        self.assertEqual(len(runs), 2)
        self.assertEqual(runs[0].count, 2)
        self.assertTrue(runs[1].is_variant)

    def test_layout_rows(self) -> None:
        """UT-MEM-008：代表层明细行数正确。"""
        self.assertEqual(len(self.layout.activation_rows), 15)
        self.assertEqual(len(self.layout.weight_rows), 13)
        self.assertEqual(len(self.layout.weight_groups), 3)

    def test_render_deterministic(self) -> None:
        """UT-MEM-009：两次渲染 SVG 字节一致。"""
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.svg"
            second = Path(directory) / "second.svg"
            render_memory_svg(self.view, self.layout, first)
            render_memory_svg(self.view, self.layout, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            text = first.read_text(encoding="utf-8")
            self.assertIn("Activation Memory Arena", text)
            self.assertIn("Parameter / Weight Memory", text)
            self.assertIn("pitch", text)
            self.assertIn("REUSE", text)
            self.assertIn("LayerWeights ×28", text)
            self.assertIn("weights 310", text)


if __name__ == "__main__":
    unittest.main()
