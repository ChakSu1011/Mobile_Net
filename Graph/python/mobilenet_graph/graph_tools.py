"""模块：Graph Python 构图辅助工具。

职责：
  在 Python 侧使用 ``torch.Tensor`` / ``torch.dtype`` 组装 Graph IR 时提供
  命名、形状、节点、边与差异定位的轻量辅助函数。所有对象仍由 C++ 绑定
  （``_graph`` 模块）创建，本文件不定义第二套 Node/Edge 结构。

边界：
  - 只做参数整理与诊断，不复制、不修改 C++ 校验规则；
  - 不执行张量运算，不读取权重数值；
  - 不提供 Forward 或 RuntimeGraph 执行入口。

依赖：
  ``mobilenet_graph`` 原生扩展 ``_graph``（由 graph_python.cc 编译产物
  导入）、``torch``、标准库。
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional, Sequence

import torch

import _graph as graph


def dim_constant(value: int) -> "graph.Dim":
    """构造常量维。

    参数：
      value：维度值，必须非负。
    返回：
      ``graph.Dim`` 常量维对象。
    异常：
      ValueError：value 为负数时抛出。
    """
    if value < 0:
        raise ValueError("constant dim must be non-negative")
    return graph.Dim.constant(int(value))


def dim_symbolic(symbol: str, example_value: int) -> "graph.Dim":
    """构造符号维。

    参数：
      symbol：符号名称，例如 ``batch``、``sequence``；
      example_value：该维在示例张量中的具体取值。
    返回：
      ``graph.Dim`` 符号维对象。
    异常：
      ValueError：符号为空或示例值为负时抛出。
    """
    if not symbol:
        raise ValueError("symbol must not be empty")
    if example_value < 0:
        raise ValueError("example value must be non-negative")
    return graph.Dim.symbolic(symbol, int(example_value))


def symbolic_shape(dims: Sequence["graph.Dim"]) -> "graph.SymbolicShape":
    """由维度列表构造符号形状。

    参数：
      dims：常量维/符号维序列，顺序即张量维度顺序。
    返回：
      ``graph.SymbolicShape``。
    """
    return graph.SymbolicShape(list(dims))


def concrete_shape(values: Iterable[int]) -> "graph.SymbolicShape":
    """构造全常量形状。

    参数：
      values：各维具体取值。
    返回：
      全常量 ``graph.SymbolicShape``。
    """
    return graph.SymbolicShape([dim_constant(value) for value in values])


def hidden_shape(
    batch: int, sequence: int, hidden_size: int
) -> "graph.SymbolicShape":
    """构造 ``[batch, sequence, hidden]`` 激活形状。

    参数：
      batch：batch 示例值；
      sequence：sequence 示例值；
      hidden_size：隐藏层宽度。
    返回：
      batch/sequence 为符号维、末维为常量维的形状。
    """
    return symbolic_shape(
        [
            dim_symbolic("batch", batch),
            dim_symbolic("sequence", sequence),
            dim_constant(hidden_size),
        ]
    )


def sequence_shape(
    batch: int, sequence: int, last_dimension: int
) -> "graph.SymbolicShape":
    """构造 ``[batch, sequence, last_dimension]`` 激活形状。

    参数：
      batch：batch 示例值；
      sequence：sequence 示例值；
      last_dimension：末维具体宽度。
    返回：
      batch/sequence 为符号维、末维为常量维的形状。
    """
    return symbolic_shape(
        [
            dim_symbolic("batch", batch),
            dim_symbolic("sequence", sequence),
            dim_constant(last_dimension),
        ]
    )


def logits_shape(batch: int, vocab_size: int) -> "graph.SymbolicShape":
    """构造 ``[batch, vocab]`` logits 形状。

    参数：
      batch：batch 示例值；
      vocab_size：词表大小。
    返回：
      batch 为符号维、末维为常量维的形状。
    """
    return symbolic_shape(
        [dim_symbolic("batch", batch), dim_constant(vocab_size)]
    )


def fused_op(
    source_name: str,
    op_kind: str,
    attributes: Optional[dict[str, Any]] = None,
) -> "graph.FusedOp":
    """构造融合前散算子描述。

    参数：
      source_name：原始模块/参数路径；
      op_kind：原始算子种类；
      attributes：可选属性字典。
    返回：
      ``graph.FusedOp``。
    """
    return graph.FusedOp(source_name, op_kind, attributes or {})


def input_node(
    name: str,
    edge_name: str,
    binding_name: str,
    host_tensor: torch.Tensor,
    copy_kind: Optional["graph.CopyKind"] = None,
) -> "graph.Node":
    """构造输入节点。

    参数：
      name：节点名；
      edge_name：该节点产生的输入边名；
      binding_name：I/O 绑定名；
      host_tensor：主机侧示例张量，须为 CPU 张量；
      copy_kind：默认 ``CopyKind.kH2D``。
    返回：
      ``graph.Node``；op_type 为 ``OpType.kInput``。
    """
    return graph.Node(
        name=name,
        op_type=graph.OpType.kInput,
        input_edges=[],
        output_edges=[edge_name],
        precision=host_tensor.dtype,
        binding_name=binding_name,
        copy_kind=graph.CopyKind.kH2D if copy_kind is None else copy_kind,
        host_tensor=host_tensor,
    )


def output_node(
    name: str,
    edge_name: str,
    binding_name: str,
    host_tensor: torch.Tensor,
    copy_kind: Optional["graph.CopyKind"] = None,
) -> "graph.Node":
    """构造输出节点。

    参数：
      name：节点名；
      edge_name：该节点消费的输出边名；
      binding_name：I/O 绑定名；
      host_tensor：主机侧示例张量，须为 CPU 张量；
      copy_kind：默认 ``CopyKind.kD2H``。
    返回：
      ``graph.Node``；op_type 为 ``OpType.kOutput``。
    """
    return graph.Node(
        name=name,
        op_type=graph.OpType.kOutput,
        input_edges=[edge_name],
        output_edges=[],
        precision=host_tensor.dtype,
        binding_name=binding_name,
        copy_kind=graph.CopyKind.kD2H if copy_kind is None else copy_kind,
        host_tensor=host_tensor,
    )


def activation_edge(
    name: str,
    shape: "graph.SymbolicShape",
    dtype: torch.dtype,
    tensor: Optional[torch.Tensor] = None,
) -> "graph.Edge":
    """构造激活边。

    参数：
      name：边名；
      shape：符号形状；
      dtype：元素类型；
      tensor：可选显式张量；缺省按示例形状创建 meta 张量。
    返回：
      非权重 ``graph.Edge``。
    """
    if tensor is None:
        tensor = torch.empty(shape.example_shape, dtype=dtype, device="meta")
    return graph.Edge(name=name, tensor=tensor, symbolic_shape=shape)


def weight_edge(
    name: str, tensor: torch.Tensor, weight_ref: str
) -> "graph.Edge":
    """构造权重边。

    参数：
      name：边名；
      tensor：权重张量，须与权重表条目共享存储；
      weight_ref：权重表引用名。
    返回：
      ``is_weight=True`` 的 ``graph.Edge``。
    """
    return graph.Edge(
        name=name,
        tensor=tensor,
        symbolic_shape=concrete_shape(tuple(tensor.shape)),
        is_weight=True,
        weight_ref=weight_ref,
    )


def weight_desc(
    ref: str,
    source_name: str,
    tensor: torch.Tensor,
) -> "graph.WeightDesc":
    """构造权重表条目。

    参数：
      ref：权重引用名；
      source_name：safetensors 中的参数名；
      tensor：权重张量（可为 meta 张量）。
    返回：
      ``graph.WeightDesc``。
    """
    return graph.WeightDesc(ref=ref, source_name=source_name, tensor=tensor)


def first_json_difference(left: str, right: str) -> Optional[str]:
    """定位两段 Canonical JSON 的首个差异路径。

    参数：
      left：第一段 JSON 文本；
      right：第二段 JSON 文本。
    返回：
      完全相同时返回 None；否则返回形如 ``nodes[3].name: 'a' != 'b'``
      的首个差异描述；文本级差异但结构相同时返回首个不同字符位置。
    """
    if left == right:
        return None
    try:
        left_value = json.loads(left)
        right_value = json.loads(right)
    except json.JSONDecodeError:
        return _first_text_difference(left, right)
    path: list[str] = []
    difference = _compare_json(left_value, right_value, path)
    if difference is not None:
        return difference
    return _first_text_difference(left, right)


def _compare_json(left: Any, right: Any, path: list[str]) -> Optional[str]:
    """递归比较 JSON 值并返回首个差异路径。"""
    location = _format_path(path)
    if type(left) is not type(right):
        return (
            f"{location}: type {type(left).__name__} != "
            f"{type(right).__name__}"
        )
    if isinstance(left, dict):
        left_keys = list(left.keys())
        right_keys = list(right.keys())
        if left_keys != right_keys:
            return f"{location}.keys: {left_keys} != {right_keys}"
        for key in left_keys:
            path.append(str(key))
            difference = _compare_json(left[key], right[key], path)
            path.pop()
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{location}.length: {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            path.append(f"[{index}]")
            difference = _compare_json(left_item, right_item, path)
            path.pop()
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{location}: {left!r} != {right!r}"
    return None


def _format_path(path: list[str]) -> str:
    """把路径片段拼成可读字符串。"""
    if not path:
        return "<root>"
    result = path[0]
    for item in path[1:]:
        result += item if item.startswith("[") else "." + item
    return result


def _first_text_difference(left: str, right: str) -> str:
    """返回两段文本首个不同字符的位置。"""
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            begin = max(0, index - 30)
            end = index + 30
            return (
                f"text offset {index}: "
                f"{left[begin:end]!r} != {right[begin:end]!r}"
            )
    return f"text length: {len(left)} != {len(right)}"


def load_model_config(
    config: dict[str, Any],
    model_family: str,
    torch_dtype: torch.dtype,
) -> "graph.ModelConfig":
    """由 HuggingFace config.json 构造 Graph ModelConfig。

    参数：
      config：``config.json`` 解析结果；
      model_family：融合规则表键，例如 ``qwen3``；
      torch_dtype：模型权重 dtype。
    返回：
      填充结构参数的 ``graph.ModelConfig``。
    """
    model_config = graph.ModelConfig()
    model_config.model_family = model_family
    model_config.num_layers = int(config["num_hidden_layers"])
    model_config.hidden_size = int(config["hidden_size"])
    model_config.intermediate_size = int(config["intermediate_size"])
    model_config.num_attention_heads = int(config["num_attention_heads"])
    model_config.num_kv_heads = int(
        config.get("num_key_value_heads", config["num_attention_heads"])
    )
    model_config.head_dim = int(
        config.get(
            "head_dim",
            config["hidden_size"] // config["num_attention_heads"],
        )
    )
    model_config.vocab_size = int(config["vocab_size"])
    model_config.max_context_length = int(config["max_position_embeddings"])
    model_config.norm_eps = float(config["rms_norm_eps"])
    model_config.activation = str(config.get("hidden_act", "silu"))
    model_config.tie_word_embeddings = bool(
        config.get("tie_word_embeddings", False)
    )
    model_config.dtype = torch_dtype
    model_config.rope.theta = float(config.get("rope_theta", 10000.0))
    model_config.rope.layout = graph.RopeLayout.kNeox
    model_config.rope.scaling_type = "none"
    model_config.rope.scaling_factor = 1.0
    model_config.rope.original_max_position_embeddings = int(
        config["max_position_embeddings"]
    )
    return model_config
