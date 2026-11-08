"""模块：Graph IR Python 单元测试。

职责：
  通过 Python 绑定验证 GraphBuilder、GraphValidator、Canonical JSON、
  ToTensorDesc、融合规则表与 torch 原生属性的行为，覆盖设计文档
  UT-GRAPH-001~011、016~018、021~026 的主要路径。

边界：
  - 只构造小规模合成图，不加载真实权重；
  - 不测试 RuntimeGraph.Forward（由 C++ test_runtime_graph 覆盖）；
  - 使用标准库 unittest，不引入第三方测试框架。

依赖：
  ``_graph``、``mobilenet_graph.graph_tools``、``torch``。
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


def make_model_config(family: str = "test") -> "_graph.ModelConfig":
    """构造最小合法模型配置。"""
    config = _graph.ModelConfig()
    config.model_family = family
    config.num_layers = 1
    config.hidden_size = 4
    config.intermediate_size = 8
    config.num_attention_heads = 1
    config.num_kv_heads = 1
    config.head_dim = 4
    config.vocab_size = 8
    config.max_context_length = 8
    config.norm_eps = 1e-6
    config.activation = "silu"
    config.tie_word_embeddings = False
    config.dtype = torch.float32
    config.rope.theta = 10000.0
    config.rope.layout = _graph.RopeLayout.kNeox
    config.rope.scaling_factor = 1.0
    config.rope.original_max_position_embeddings = 8
    return config


def device_tensor(shape=(2,), dtype=torch.int64) -> torch.Tensor:
    """创建 meta 设备张量，作为设备边。"""
    return torch.empty(shape, dtype=dtype, device="meta")


def host_tensor(shape=(2,), dtype=torch.int64) -> torch.Tensor:
    """创建 CPU 主机张量，作为 I/O host_tensor。"""
    return torch.zeros(shape, dtype=dtype, device="cpu")


def input_node(name="input.x", edge="x", binding="x", dtype=torch.int64):
    """构造输入节点。"""
    return tools.input_node(name, edge, binding, host_tensor((2,), dtype))


def output_node(name="output.y", edge="y", binding="y_out",
                dtype=torch.int64):
    """构造输出节点。"""
    return tools.output_node(name, edge, binding, host_tensor((2,), dtype))


def compute_node(name="cast", op_type=_graph.OpType.kCast,
                 inputs=("x",), outputs=("y",), dtype=torch.int64,
                 op_kind="cast", **kwargs):
    """构造计算节点。"""
    return _graph.Node(
        name=name,
        op_type=op_type,
        fused_ops=[tools.fused_op("test." + op_kind, op_kind)],
        input_edges=list(inputs),
        output_edges=list(outputs),
        precision=dtype,
        kernel_key="cast",
        **kwargs,
    )


def activation_edge(name, shape=(2,), dtype=torch.int64,
                    symbolic_shape=None):
    """构造激活边。"""
    resolved_shape = (
        tools.concrete_shape(shape) if symbolic_shape is None
        else symbolic_shape
    )
    return tools.activation_edge(name, resolved_shape, dtype)


def build_minimal_graph():
    """构造最小合法图：Input -> Cast -> Output。"""
    builder = _graph.GraphBuilder()
    builder.set_model_config(make_model_config())
    x_id = builder.add_edge(activation_edge("x"))
    y_id = builder.add_edge(activation_edge("y"))
    input_id = builder.add_node(input_node())
    builder.add_node(compute_node())
    output_id = builder.add_node(output_node())
    builder.set_input_nodes([input_id])
    builder.set_output_nodes([output_id])
    graph = builder.finalize()
    return builder, graph, x_id, y_id


class GraphIrTest(unittest.TestCase):
    """Graph IR 行为测试。"""

    def test_minimal_graph_and_canonical_json(self):
        """UT-GRAPH-001/017：最小图构图与规范 JSON 确定性。"""
        _, graph, x_id, y_id = build_minimal_graph()
        self.assertEqual([node.name for node in graph.nodes],
                         ["input.x", "cast", "output.y"])
        self.assertEqual(graph.topo_order, ["input.x", "cast", "output.y"])
        self.assertEqual(graph.input_nodes, ["input.x"])
        self.assertEqual(graph.output_nodes, ["output.y"])
        self.assertTrue(graph.validate().ok)
        self.assertEqual(x_id, 0)
        self.assertEqual(y_id, 1)
        first = graph.to_canonical_json()
        second = graph.to_canonical_json()
        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_multi_branch_topology(self):
        """UT-GRAPH-002：多分支拓扑序按 NodeId 升序确定。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        for name in ("x", "a", "b", "c"):
            builder.add_edge(activation_edge(name))
        input_id = builder.add_node(input_node())
        builder.add_node(
            compute_node(name="cast.a", inputs=("x",), outputs=("a",))
        )
        builder.add_node(
            compute_node(name="cast.b", inputs=("x",), outputs=("b",))
        )
        builder.add_node(
            _graph.Node(
                name="add",
                op_type=_graph.OpType.kResidualAdd,
                fused_ops=[tools.fused_op("test.add", "add")],
                input_edges=["a", "b"],
                output_edges=["c"],
                precision=torch.int64,
                kernel_key="add",
            )
        )
        output_id = builder.add_node(output_node(edge="c"))
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        graph = builder.finalize()
        self.assertEqual(graph.topo_order,
                         ["input.x", "cast.a", "cast.b", "add", "output.y"])

    def test_cycle_rejected(self):
        """UT-GRAPH-003：成环图被拒绝。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        for name in ("x", "cycle_a", "cycle_b", "out"):
            builder.add_edge(activation_edge(name))
        input_id = builder.add_node(input_node())
        builder.add_node(compute_node(name="cast.0", inputs=("x",),
                                      outputs=("out",)))
        output_id = builder.add_node(output_node(edge="out"))
        builder.add_node(compute_node(name="cycle.a", inputs=("cycle_b",),
                                      outputs=("cycle_a",)))
        builder.add_node(compute_node(name="cycle.b", inputs=("cycle_a",),
                                      outputs=("cycle_b",)))
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        with self.assertRaises(_graph.GraphError) as context:
            builder.finalize()
        self.assertEqual(context.exception.code, "INVALID_ARGUMENT")
        self.assertIn("cycle", context.exception.detail.lower())

    def test_dangling_edge_rejected(self):
        """UT-GRAPH-004：悬空边与重复端口被拒绝。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.add_edge(activation_edge("x"))
        input_id = builder.add_node(input_node())
        builder.add_node(
            _graph.Node(
                name="cast",
                op_type=_graph.OpType.kCast,
                fused_ops=[tools.fused_op("test.cast", "cast")],
                input_edges=["x", "x"],
                output_edges=["missing"],
                precision=torch.int64,
                kernel_key="cast",
            )
        )
        output_id = builder.add_node(output_node(edge="missing"))
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        with self.assertRaises(_graph.GraphError) as context:
            builder.finalize()
        self.assertEqual(context.exception.code, "INVALID_ARGUMENT")

    def test_duplicate_names_rejected(self):
        """UT-GRAPH-005：节点/边/权重重名被拒绝。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.add_edge(activation_edge("x"))
        builder.add_node(input_node())
        with self.assertRaises(_graph.GraphError) as context:
            builder.add_node(input_node())
        self.assertEqual(context.exception.code, "INVALID_ARGUMENT")

        builder = _graph.GraphBuilder()
        builder.add_edge(activation_edge("x"))
        with self.assertRaises(_graph.GraphError):
            builder.add_edge(activation_edge("x"))

        builder = _graph.GraphBuilder()
        tensor = device_tensor()
        builder.add_weight(tools.weight_desc("w", "model.w", tensor))
        with self.assertRaises(_graph.GraphError):
            builder.add_weight(tools.weight_desc("w", "model.w2", tensor))

    def test_missing_io_nodes_rejected(self):
        """UT-GRAPH-006：缺少 Input 或 Output 节点被拒绝。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.add_edge(activation_edge("x"))
        builder.add_edge(activation_edge("y"))
        input_id = builder.add_node(input_node())
        builder.add_node(compute_node())
        builder.set_input_nodes([input_id])
        with self.assertRaises(_graph.GraphError) as context:
            builder.finalize()
        self.assertIn("output", context.exception.detail)

    def test_io_edge_rules_rejected(self):
        """UT-GRAPH-007：Input 有入边、Output 有出边、I/O 权重边被拒绝。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.add_edge(activation_edge("x"))
        builder.add_edge(activation_edge("y"))
        builder.add_edge(activation_edge("z"))
        input_id = builder.add_node(
            _graph.Node(
                name="input.x",
                op_type=_graph.OpType.kInput,
                input_edges=["z"],
                output_edges=["x"],
                precision=torch.int64,
                binding_name="x",
                copy_kind=_graph.CopyKind.kH2D,
                host_tensor=host_tensor(),
            )
        )
        builder.add_node(compute_node())
        output_id = builder.add_node(
            _graph.Node(
                name="output.y",
                op_type=_graph.OpType.kOutput,
                input_edges=["y"],
                output_edges=["z"],
                precision=torch.int64,
                binding_name="y_out",
                copy_kind=_graph.CopyKind.kD2H,
                host_tensor=host_tensor(),
            )
        )
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        with self.assertRaises(_graph.GraphError):
            builder.finalize()

    def test_io_dtype_mismatch_rejected(self):
        """UT-GRAPH-008：I/O host 与 device dtype 不一致且无 Cast。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.add_edge(activation_edge("x"))
        builder.add_edge(activation_edge("y"))
        input_id = builder.add_node(
            _graph.Node(
                name="input.x",
                op_type=_graph.OpType.kInput,
                output_edges=["x"],
                precision=torch.int32,
                binding_name="x",
                copy_kind=_graph.CopyKind.kH2D,
                host_tensor=host_tensor(dtype=torch.int32),
            )
        )
        builder.add_node(compute_node())
        output_id = builder.add_node(output_node())
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        with self.assertRaises(_graph.GraphError):
            builder.finalize()

    def test_weight_edge_rules_rejected(self):
        """UT-GRAPH-009：权重边必须是权重边且引用有效。"""
        builder = _graph.GraphBuilder()
        with self.assertRaises(_graph.GraphError) as context:
            builder.add_edge(
                _graph.Edge(
                    name="weight.x",
                    tensor=device_tensor(),
                    symbolic_shape=tools.concrete_shape((2,)),
                    is_weight=True,
                    weight_ref="",
                )
            )
        self.assertEqual(context.exception.code, "INVALID_ARGUMENT")
        with self.assertRaises(_graph.GraphError):
            builder.add_edge(
                _graph.Edge(
                    name="y",
                    tensor=device_tensor(),
                    symbolic_shape=tools.concrete_shape((2,)),
                    is_weight=False,
                    weight_ref="w",
                )
            )

    def test_dtype_and_shape_mismatch_rejected(self):
        """UT-GRAPH-010：生产/消费 dtype、rank、常量维不匹配。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.add_edge(activation_edge("x"))
        builder.add_edge(activation_edge("y", dtype=torch.bfloat16))
        input_id = builder.add_node(input_node())
        builder.add_node(compute_node(dtype=torch.int64))
        output_id = builder.add_node(output_node(dtype=torch.bfloat16))
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        with self.assertRaises(_graph.GraphError):
            builder.finalize()

        builder = _graph.GraphBuilder()
        with self.assertRaises(_graph.GraphError):
            builder.add_edge(
                _graph.Edge(
                    name="bad",
                    tensor=device_tensor((2,)),
                    symbolic_shape=tools.concrete_shape((2, 1)),
                    is_weight=False,
                )
            )

    def test_fusion_rule_mismatch_rejected(self):
        """UT-GRAPH-011：融合规则阶段/顺序不匹配被拒绝并报告 rule_id。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config("qwen3"))
        builder.add_edge(activation_edge("x"))
        builder.add_edge(activation_edge("y"))
        input_id = builder.add_node(input_node())
        builder.add_node(
            _graph.Node(
                name="layer.0.qkv",
                op_type=_graph.OpType.kQkvProjection,
                fused_ops=[
                    tools.fused_op("model.layers.0.self_attn.q_proj",
                                   "q_linear"),
                ],
                input_edges=["x"],
                output_edges=["y"],
                precision=torch.int64,
                kernel_key="qkv_projection",
                fusion_rule_id="qwen3.block.qkv.v1",
            )
        )
        output_id = builder.add_node(output_node())
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        with self.assertRaises(_graph.GraphError) as context:
            builder.finalize()
        self.assertIn("qwen3.block.qkv.v1", context.exception.detail)

    def test_version_mismatch(self):
        """UT-GRAPH-016：schema 主版本不兼容被拒绝。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.set_schema_version(2, 0)
        builder.add_edge(activation_edge("x"))
        builder.add_edge(activation_edge("y"))
        input_id = builder.add_node(input_node())
        builder.add_node(compute_node())
        output_id = builder.add_node(output_node())
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        with self.assertRaises(_graph.GraphError) as context:
            builder.finalize()
        self.assertEqual(context.exception.code, "VERSION_MISMATCH")

    def test_sampling_constraints(self):
        """UT-GRAPH-021：LM Head/Sampling 数量与顺序约束。"""
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.add_edge(activation_edge("x"))
        builder.add_edge(activation_edge("y"))
        input_id = builder.add_node(input_node())
        builder.add_node(
            _graph.Node(
                name="lm_head",
                op_type=_graph.OpType.kLmHead,
                fused_ops=[tools.fused_op("lm_head", "linear")],
                input_edges=["x"],
                output_edges=["y"],
                precision=torch.int64,
                kernel_key="lm_head",
            )
        )
        output_id = builder.add_node(output_node())
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        with self.assertRaises(_graph.GraphError) as context:
            builder.finalize()
        self.assertIn("sampling", context.exception.detail)

    def test_symbolic_shape_rules(self):
        """UT-GRAPH-022：符号维示例值校验与内部符号边。"""
        symbolic = tools.symbolic_shape(
            [tools.dim_symbolic("batch", 1), tools.dim_constant(2)]
        )
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.add_edge(
            tools.activation_edge(
                "x", tools.concrete_shape((2,)), torch.float32
            )
        )
        builder.add_edge(
            tools.activation_edge("mid", symbolic, torch.float32)
        )
        builder.add_edge(
            tools.activation_edge(
                "y", tools.concrete_shape((2,)), torch.float32
            )
        )
        input_id = builder.add_node(
            tools.input_node("input.x", "x", "x",
                             host_tensor((2,), torch.float32))
        )
        builder.add_node(
            compute_node(name="cast.0", dtype=torch.float32,
                         inputs=("x",), outputs=("mid",))
        )
        builder.add_node(
            compute_node(name="cast.1", dtype=torch.float32,
                         inputs=("mid",), outputs=("y",))
        )
        output_id = builder.add_node(
            tools.output_node("output.y", "y", "y_out",
                              host_tensor((2,), torch.float32))
        )
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        graph = builder.finalize()
        self.assertTrue(graph.validate().ok)
        self.assertEqual(graph.edge("mid").symbolic_shape.dims[0].symbol,
                         "batch")
        self.assertEqual(graph.edge("mid").symbolic_shape.dims[0].value, 1)

        builder = _graph.GraphBuilder()
        with self.assertRaises(_graph.GraphError):
            builder.add_edge(
                _graph.Edge(
                    name="bad",
                    tensor=device_tensor((2,)),
                    symbolic_shape=tools.symbolic_shape(
                        [tools.dim_symbolic("batch", 3)]
                    ),
                )
            )

    def test_quantization_metadata_roundtrip(self):
        """UT-GRAPH-023：量化元数据可构造并在转换后保留。"""
        quantization = _graph.QuantizationDesc(
            "int8", "per_tensor", [0.5], [0], 0
        )
        tensor = device_tensor((2,), torch.bfloat16)
        edge = _graph.Edge(
            name="q",
            tensor=tensor,
            symbolic_shape=tools.concrete_shape((2,)),
            quantization=quantization,
        )
        self.assertEqual(edge.quantization.storage_dtype, "int8")
        self.assertEqual(edge.quantization.scales, [0.5])
        descriptor = _graph.to_tensor_desc(tensor, edge.symbolic_shape,
                                           quantization)
        self.assertEqual(descriptor["dtype"], "bfloat16")
        self.assertEqual(descriptor["quantization"].storage_dtype, "int8")

    def test_fusion_rule_table(self):
        """UT-GRAPH-024：多模型族查询、未知规则与唯一 rule_id。"""
        table = _graph.FusionRuleTable.instance()
        self.assertEqual(table.version, "1.0")
        self.assertEqual(len(table.rules), 7)
        rule_ids = [rule.rule_id for rule in table.rules]
        self.assertEqual(len(rule_ids), len(set(rule_ids)))
        self.assertIsNotNone(table.find("qwen3.block.qkv.v1"))
        self.assertIsNone(table.find("unknown.rule"))
        matched = table.find_by_stage(
            "qwen3",
            _graph.BlockStage.kStage2Qkv,
            ["q_linear", "k_linear", "v_linear", "qk_rmsnorm", "rope"],
        )
        self.assertIsNotNone(matched)
        self.assertEqual(matched.rule_id, "qwen3.block.qkv.v1")
        self.assertIsNone(
            table.find_by_stage("unknown", _graph.BlockStage.kStage2Qkv, [])
        )

    def test_to_tensor_desc_rules(self):
        """UT-GRAPH-025：非连续、列主序与符号维转换规则。"""
        base = torch.empty((4, 6), dtype=torch.float32)
        non_contiguous = base.t()
        descriptor = _graph.to_tensor_desc(
            non_contiguous,
            tools.symbolic_shape(
                [tools.dim_symbolic("rows", 6),
                 tools.dim_symbolic("cols", 4)]
            ),
        )
        self.assertEqual(descriptor["stride"], [1, 6])
        self.assertEqual(descriptor["layout"], "column_major")
        self.assertEqual(descriptor["device"], "host")
        self.assertEqual(descriptor["shape"][0]["symbol"], "rows")

        symbolic = _graph.to_tensor_desc(
            torch.empty((1, 8, 16), dtype=torch.float32),
            tools.symbolic_shape(
                [
                    tools.dim_symbolic("batch", 1),
                    tools.dim_symbolic("sequence", 8),
                    tools.dim_constant(16),
                ]
            ),
        )
        self.assertEqual(symbolic["dtype"], "float32")
        self.assertEqual(symbolic["stride"], [128, 16, 1])

        with self.assertRaises(_graph.GraphError):
            _graph.to_tensor_desc(
                torch.empty((2,), dtype=torch.float32),
                tools.concrete_shape((3,)),
            )

    def test_python_attribute_types(self):
        """UT-GRAPH-026：Node/Edge/ModelConfig 暴露 torch 原生类型。"""
        _, graph, _, _ = build_minimal_graph()
        edge = graph.edge("x")
        self.assertIsInstance(edge.tensor, torch.Tensor)
        self.assertIsInstance(edge.tensor.dtype, torch.dtype)
        node = graph.node("cast")
        self.assertIsInstance(node.precision, torch.dtype)
        self.assertEqual(node.precision, torch.int64)
        self.assertIsInstance(node.host_tensor, type(None))
        input_node_view = graph.node("input.x")
        self.assertIsInstance(input_node_view.host_tensor, torch.Tensor)
        config = graph.model_config
        self.assertIsInstance(config.dtype, torch.dtype)
        self.assertEqual(config.dtype, torch.float32)
        self.assertIsInstance(graph.nodes[0], _graph.Node)

    def test_runtime_conversion_json_equality(self):
        """UT-GRAPH-018：最小图 IR 与 RuntimeGraph 规范 JSON 一致。"""
        _, graph, _, _ = build_minimal_graph()
        self.assertEqual(graph.to_canonical_json(),
                         _graph.convert_graph_to_runtime_json(graph))

    def test_unknown_io_binding_and_error_fields(self):
        """UT-GRAPH-017：查找缺失对象与 GraphError 字段。"""
        _, graph, _, _ = build_minimal_graph()
        self.assertIsNone(graph.node("missing"))
        self.assertIsNone(graph.edge("missing"))
        self.assertIsNone(graph.weight("missing"))
        self.assertEqual(graph.find_node("missing"), -1)
        self.assertEqual(graph.find_edge("missing"), -1)
        self.assertEqual(graph.find_node("cast"), 1)
        self.assertTrue(graph.validate().ok)

        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        builder.add_edge(activation_edge("x"))
        builder.add_node(input_node())
        builder.add_node(compute_node(outputs=("missing",)))
        output_id = builder.add_node(output_node(edge="missing"))
        builder.add_input_node(0)
        builder.add_output_node(output_id)
        with self.assertRaises(_graph.GraphError) as context:
            builder.finalize()
        error = context.exception
        self.assertTrue(error.detail)
        self.assertEqual(error.code, "INVALID_ARGUMENT")
        self.assertTrue(hasattr(error, "node_name"))
        self.assertTrue(hasattr(error, "edge_name"))

    def test_large_graph_validation(self):
        """UT-GRAPH-019：1000 节点 DAG 校验不崩溃且拓扑序完整。"""
        node_count = 1000
        builder = _graph.GraphBuilder()
        builder.set_model_config(make_model_config())
        edge_names = []
        for index in range(node_count + 1):
            name = f"e{index}"
            builder.add_edge(
                tools.activation_edge(
                    name, tools.concrete_shape((2,)), torch.int64
                )
            )
            edge_names.append(name)
        input_id = builder.add_node(input_node(edge="e0"))
        for index in range(node_count):
            builder.add_node(
                _graph.Node(
                    name=f"n{index}",
                    op_type=_graph.OpType.kCast,
                    fused_ops=[tools.fused_op(f"src{index}", "cast")],
                    input_edges=[edge_names[index]],
                    output_edges=[edge_names[index + 1]],
                    precision=torch.int64,
                    kernel_key="cast",
                )
            )
        output_id = builder.add_node(output_node(edge=f"e{node_count}"))
        builder.set_input_nodes([input_id])
        builder.set_output_nodes([output_id])
        graph = builder.finalize()
        self.assertTrue(graph.validate().ok)
        self.assertEqual(len(graph.nodes), node_count + 2)
        self.assertEqual(len(graph.topo_order), node_count + 2)

    def test_runtime_headers_have_no_torch_dependency(self):
        """UT-GRAPH-027：RuntimeGraph 头文件不包含 torch/ATen/c10。"""
        roots = (
            REPOSITORY_ROOT / "Graph" / "cpp" / "runtime",
            REPOSITORY_ROOT / "Graph" / "cpp" / "common",
        )
        forbidden = (
            "<ATen/",
            "<torch/",
            "<c10/",
            '"ATen/',
            '"torch/',
            '"c10/',
        )
        for root in roots:
            for path in sorted(root.rglob("*.h")):
                text = path.read_text(encoding="utf-8")
                for line in text.splitlines():
                    if line.strip().startswith("#include"):
                        for token in forbidden:
                            self.assertNotIn(token, line, f"{path}: {line}")


if __name__ == "__main__":
    unittest.main()
