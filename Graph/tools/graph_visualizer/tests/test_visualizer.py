"""模块：Graph 静态图可视化工具测试。

职责：
  验证模型包读取、视图模型、融合算子描述、Layer 折叠、差异层检测与 SVG 渲染。

边界：
  - 使用仓库内 Qwen3 模型包和合成 ModelGraph；
  - 不执行推理、不读取权重内容、不依赖外部绘图库。
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

from ToolChain.common.model_ir import (  # noqa: E402
    IOBinding,
    ModelGraph,
    OpMeta,
)
from ToolChain.testing import make_tensor  # noqa: E402

from graph_visualizer.composition import describe_op  # noqa: E402
from graph_visualizer.formatting import format_shape  # noqa: E402
from graph_visualizer.layout import layout_view  # noqa: E402
from graph_visualizer.package import (  # noqa: E402
    LoadedPackage,
    load_model_package,
)
from graph_visualizer.repeats import analyze_layer_runs  # noqa: E402
from graph_visualizer.svg import render_svg  # noqa: E402
from graph_visualizer.view_model import build_view_model  # noqa: E402

_QWEN3_PACKAGE = (
    _REPOSITORY_ROOT / "Models" / "Qwen3" / "artifacts" / "qwen3_0_6b")


def _layer_model(variant: bool) -> ModelGraph:
    """构造三层合成图，用于 Layer 折叠测试。

    参数：
      variant：为 True 时第二层属性不同。

    返回：
      通过校验的 ``ModelGraph``。
    """
    tensors = {
        "input.x": make_tensor(
            "input.x", "input", (1, 4), "float16", reusable=False),
        "layer.0.add.out": make_tensor(
            "layer.0.add.out", "activation", (1, 4), "float16",
            reusable=False),
        "layer.1.add.out": make_tensor(
            "layer.1.add.out", "activation", (1, 4), "float16",
            reusable=False),
        "layer.2.add.out": make_tensor(
            "layer.2.add.out", "activation", (1, 4), "float16",
            reusable=False),
    }
    ops = []
    previous = "input.x"
    for index in range(3):
        attrs = {"precision": "float16"}
        if variant and index == 1:
            attrs["variant"] = True
        output = f"layer.{index}.add.out"
        ops.append(OpMeta(
            name=f"layer.{index}.add",
            op_type="add",
            inputs=(previous,),
            outputs=(output,),
            attrs=attrs,
        ))
        previous = output
    return ModelGraph(
        name="synthetic_layers",
        adapter="test",
        inputs=("input.x",),
        constants=(),
        outputs=(IOBinding(name="output.y", tensor=previous),),
        tensors=tensors,
        ops=tuple(ops),
        metadata={},
    )


class VisualizerTest(unittest.TestCase):
    """可视化核心行为测试。"""

    @classmethod
    def setUpClass(cls) -> None:
        """加载一次 Qwen3 包并构建视图。"""
        cls.package = load_model_package(_QWEN3_PACKAGE)
        cls.view = build_view_model(cls.package)
        cls.layout = layout_view(cls.view)

    def test_package_counts(self) -> None:
        """UT-VIS-001：Qwen3 包计数与 manifest 一致。"""
        graph = self.package.model_graph
        self.assertEqual(len(graph.inputs), 2)
        self.assertEqual(len(graph.constants), 310)
        self.assertEqual(len(graph.ops), 423)
        self.assertEqual(len(graph.outputs), 1)
        self.assertEqual(len(graph.tensors), 735)

    def test_layer_runs(self) -> None:
        """UT-VIS-007：Qwen3 的连续相同层折叠为 ×28。"""
        runs = analyze_layer_runs(
            self.package.model_graph.ops,
            set(self.package.model_graph.constants),
        )
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].repeat_count, 28)
        self.assertFalse(runs[0].is_variant)

    def test_synthetic_same_layers(self) -> None:
        """UT-VIS-007：三层同构图折叠为 ×3。"""
        graph = _layer_model(variant=False)
        runs = analyze_layer_runs(graph.ops, set(graph.constants))
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].repeat_count, 3)

    def test_synthetic_variant_layer(self) -> None:
        """UT-VIS-008：第二层不同时单独标记差异。"""
        graph = _layer_model(variant=True)
        runs = analyze_layer_runs(graph.ops, set(graph.constants))
        self.assertEqual(len(runs), 3)
        self.assertTrue(any(run.is_variant for run in runs))
        self.assertTrue(any(
            run.start == 1 and run.end == 1 and run.is_variant
            for run in runs))
        package = LoadedPackage(
            package_dir=Path("."),
            model_graph=graph,
            manifest_sha256="0" * 64,
            ir_sha256="0" * 64,
            format_version=1,
            model_name=graph.name,
        )
        view = build_view_model(package)
        layout = layout_view(view)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "variant.svg"
            render_svg(view, layout, output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("差异", text)
            self.assertIn("Layer 1", text)

    def test_composition(self) -> None:
        """UT-VIS-003：融合算子组成正确。"""
        description = describe_op("linear_swi_glu", {})
        self.assertEqual(description.category, "mlp")
        self.assertEqual(len(description.composition), 4)

    def test_unknown_op(self) -> None:
        """UT-VIS-004：未知 op 类型不崩溃。"""
        description = describe_op("not_a_real_op", {})
        self.assertEqual(description.category, "other")
        self.assertEqual(description.composition, ())

    def test_shape_format(self) -> None:
        """UT-VIS-006：shape 使用 × 连接。"""
        self.assertEqual(format_shape((1, 256, 1024)), "1×256×1024")
        self.assertEqual(format_shape(()), "[]")

    def test_view_model(self) -> None:
        """UT-VIS-002：Qwen3 视图包含关键节点和 tensor 菱形。"""
        names = {node.id for node in self.view.nodes}
        self.assertIn("embedding", names)
        self.assertIn("layer.0.mlp_gate_up", names)
        self.assertIn("final_norm", names)
        self.assertIn("lm_head", names)
        self.assertEqual(len(self.view.runs), 1)
        self.assertEqual(self.view.runs[0].run.repeat_count, 28)
        kinds = [tensor.kind for tensor in self.view.tensors]
        self.assertEqual(kinds.count("input"), 2)
        self.assertEqual(kinds.count("output"), 1)
        self.assertEqual(kinds.count("parameter"), 14)
        parameter_links = [
            link for link in self.view.links
            if link.src.startswith("tensor:") and any(
                tensor.id == link.src and tensor.kind == "parameter"
                for tensor in self.view.tensors)
        ]
        self.assertEqual(len(parameter_links), 14)
        self.assertFalse(self.view.warnings)

    def test_parameter_labels(self) -> None:
        """UT-VIS-005：参数菱形只有精度与 shape。"""
        parameters = [
            tensor for tensor in self.view.tensors
            if tensor.kind == "parameter"]
        self.assertTrue(parameters)
        for tensor in parameters:
            self.assertEqual(tensor.precision, "float16")
            self.assertTrue(tensor.shape)
            self.assertIsNone(tensor.caption)

    def test_render_deterministic(self) -> None:
        """UT-VIS-009：两次渲染生成字节一致的 SVG。"""
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.svg"
            second = Path(directory) / "second.svg"
            render_svg(self.view, self.layout, first)
            render_svg(self.view, self.layout, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            text = first.read_text(encoding="utf-8")
            self.assertIn("DecoderLayer ×28", text)
            self.assertIn("kLinearSwiGlu", text)
            self.assertIn("1024×1024", text)
            self.assertIn("fp16", text)
            self.assertNotIn("available", text)
            self.assertNotIn(".weight", text)


if __name__ == "__main__":
    unittest.main()
