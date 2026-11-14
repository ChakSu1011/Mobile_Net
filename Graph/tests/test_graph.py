"""模块：Runtime Graph Python 绑定单元测试。

职责：
  验证 ``_graph`` 暴露的 Graph/Tensor/枚举、create_* 构图接口、
  sort_op、dump_ir 以及主要错误路径；测试不加载模型、不访问 GPU。

边界：
  - 只使用 ctypes 分配的 CPU 元数据缓冲区，不读取权重内容；
  - Tensor 句柄生命周期依赖 Graph，测试全程持有 Graph 对象；
  - 不测试真实 kernel 执行和显存规划。

依赖：
  ``_graph``、Python 3.10+ 标准库。
"""

from __future__ import annotations

import ctypes
import os
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PYTHON_EXTENSION_DIR = os.environ.get(
    "GRAPH_PYTHON_DIR",
    str(REPOSITORY_ROOT / "Build" / "graph" / "python"),
)
sys.path.insert(0, PYTHON_EXTENSION_DIR)

import _graph as gf  # noqa: E402


class GraphPythonTest(unittest.TestCase):
    """验证 Python 侧 runtime Graph 的构图、排序、dump 和错误路径。"""

    def setUp(self) -> None:
        """构建一个 input -> linear -> add -> output 的小图。"""
        self.weight_buffer = ctypes.create_string_buffer(12)
        self.graph = gf.Graph("unit_graph")
        self.weight = self.graph.create_constant(
            "weight",
            dims=[2, 3],
            pitch=[2, 3],
            precision=gf.Precision.kFloat16,
            layout=gf.Layout.kRowMajor,
            byte_size=12,
            data_ptr=ctypes.addressof(self.weight_buffer),
        )
        self.input_tensor = self.graph.create_input(
            "input", gf.Precision.kFloat16, gf.Layout.kRowMajor
        )
        self.linear_output = self.graph.create_op(
            "linear",
            [self.input_tensor, self.weight],
            1,
            gf.OpType.kLinear,
            {"axis": 1, "transpose": False},
        )[0]
        self.add_output = self.graph.create_op(
            "add", [self.linear_output], 1, gf.OpType.kAdd
        )[0]
        self.graph.create_output("output", self.add_output)
        self.graph.sort_op()
        self.dump = self.graph.dump_ir()
        self.lines = self.dump.splitlines()

    def test_graph_sections(self) -> None:
        """dump_ir 的段标题、计数和顺序符合预期。"""
        self.assertEqual(self.lines[0], "graph unit_graph")
        self.assertIn("inputs 1", self.lines)
        self.assertIn("constants 1", self.lines)
        self.assertIn("ops 4", self.lines)
        self.assertIn("outputs 1", self.lines)

    def test_topological_order(self) -> None:
        """op 序列为 input、linear、add、output，constant 不进入序列。"""
        op_lines = [line for line in self.lines if line.startswith("  [")]
        self.assertEqual(len(op_lines), 4)
        self.assertIn("  [0] input input", op_lines[0])
        self.assertIn("  [1] linear linear", op_lines[1])
        self.assertIn("  [2] add add", op_lines[2])
        self.assertIn("  [3] output output", op_lines[3])
        self.assertIn("src=[input,weight]", op_lines[1])
        self.assertIn("dst=[linear.out]", op_lines[1])
        self.assertIn("src=[linear.out]", op_lines[2])
        self.assertIn("dst=[add.out]", op_lines[2])
        self.assertIn("output <- [add.out]", self.dump)

    def test_constant_dump(self) -> None:
        """constant tensor 的 storage 文本包含精度、形状和不可复用标记。"""
        self.assertIn(
            "weight storage(precision=float16,layout=row_major,"
            "device=host,shape=[2,3],offset=0,bytes=12,data=set,"
            "reusable=false)",
            self.dump,
        )

    def test_input_attr_storage(self) -> None:
        """显式 create_input 把 precision/layout 写入 Input attr storage。"""
        self.assertIn(
            "storage=storage(precision=float16,layout=row_major,"
            "device=unknown,shape=[],offset=0,bytes=0,data=unset,"
            "reusable=true)",
            self.dump,
        )

    def test_tensor_properties(self) -> None:
        """Tensor 暴露名称、constant 标记和复用标记，且不暴露 Storage 对象。"""
        self.assertEqual(self.input_tensor.name, "input")
        self.assertFalse(self.input_tensor.is_constant)
        self.assertTrue(self.weight.is_constant)
        self.assertFalse(self.weight.reusable)
        self.linear_output.set_reusable(False)
        self.assertFalse(self.linear_output.reusable)
        self.linear_output.set_reusable(True)
        self.assertTrue(self.linear_output.reusable)
        self.assertFalse(hasattr(self.input_tensor, "storage"))

    def test_legacy_create_input(self) -> None:
        """legacy create_input 写入默认 Storage，排序与 dump 仍可用。"""
        graph = gf.Graph("legacy_graph")
        input_tensor = graph.create_input("x")
        output_tensor = graph.create_op(
            "copy", [input_tensor], 1, gf.OpType.kReshape
        )[0]
        graph.create_output("y", output_tensor)
        graph.sort_op()
        dump = graph.dump_ir()
        self.assertIn("storage(precision=unknown,layout=row_major", dump)
        self.assertIn("ops 3", dump.splitlines())

    def test_attribute_types(self) -> None:
        """Python dict 的 bool/int/float/str/数组属性可写入并稳定输出。"""
        graph = gf.Graph("attr_graph")
        input_tensor = graph.create_input("x")
        outputs = graph.create_op(
            "op",
            [input_tensor],
            1,
            gf.OpType.kAdd,
            {
                "flag": True,
                "count": 7,
                "scale": 0.25,
                "mode": "fast",
                "axes": [1, 2],
                "scales": [1.5, 2.5],
                "names": ["a", "b"],
            },
        )
        graph.create_output("y", outputs[0])
        graph.sort_op()
        dump = graph.dump_ir()
        self.assertIn("axes=[1,2]", dump)
        self.assertIn("count=7", dump)
        self.assertIn("flag=true", dump)
        self.assertIn("mode=fast", dump)
        self.assertIn("names=[a,b]", dump)
        self.assertIn("scale=0.25", dump)
        self.assertIn("scales=[1.5,2.5]", dump)

    def test_multiple_outputs(self) -> None:
        """out_cnt > 1 时生成稳定命名的多个输出 tensor。"""
        graph = gf.Graph("multi_output_graph")
        input_tensor = graph.create_input("x")
        outputs = graph.create_op(
            "split", [input_tensor], 2, gf.OpType.kReshape
        )
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].name, "split.out.0")
        self.assertEqual(outputs[1].name, "split.out.1")
        graph.create_output("y", outputs[0])
        graph.sort_op()
        dump = graph.dump_ir()
        self.assertIn("dst=[split.out.0,split.out.1]", dump)

    def test_public_module(self) -> None:
        """公共模块可加载扩展，并导出与扩展相同的图类型和枚举。"""
        package_dir = REPOSITORY_ROOT / "Graph" / "python"
        sys.path.insert(0, str(package_dir))
        import mobilenet_runtime_graph as package

        self.assertIs(package.Graph, gf.Graph)
        self.assertIs(package.Tensor, gf.Tensor)
        self.assertIs(package.Precision, gf.Precision)
        self.assertIs(package.Layout, gf.Layout)
        self.assertIs(package.DeviceType, gf.DeviceType)
        self.assertIs(package.OpType, gf.OpType)

    def test_errors(self) -> None:
        """重复名称、外部 tensor、空图排序和非法参数抛出异常。"""
        with self.assertRaises(RuntimeError):
            self.graph.create_op(
                "linear",
                [self.input_tensor],
                1,
                gf.OpType.kLinear,
            )
        with self.assertRaises(RuntimeError):
            self.graph.create_constant(
                "weight",
                [2, 3],
                [2, 3],
                gf.Precision.kFloat16,
                gf.Layout.kRowMajor,
                12,
                0,
            )
        with self.assertRaises(RuntimeError):
            self.graph.create_input("input")

        other = gf.Graph("other")
        foreign_tensor = other.create_input("foreign")
        with self.assertRaises(RuntimeError):
            self.graph.create_output("bad_output", foreign_tensor)

        empty_graph = gf.Graph("empty")
        with self.assertRaises(RuntimeError):
            empty_graph.sort_op()

        with self.assertRaises(ValueError):
            self.graph.create_constant(
                "too_many_dims",
                [1] * 9,
                [1] * 9,
                gf.Precision.kFloat16,
                gf.Layout.kRowMajor,
                2,
                0,
            )
        with self.assertRaises(ValueError):
            self.graph.create_constant(
                "pitch_mismatch",
                [2, 3],
                [2],
                gf.Precision.kFloat16,
                gf.Layout.kRowMajor,
                12,
                0,
            )
        with self.assertRaises(ValueError):
            self.graph.create_op(
                "bad_attr",
                [self.input_tensor],
                1,
                gf.OpType.kAdd,
                {"unsupported": object()},
            )
        with self.assertRaises(RuntimeError):
            self.graph.create_op(
                "zero_out", [self.input_tensor], 0, gf.OpType.kAdd
            )


if __name__ == "__main__":
    unittest.main()
