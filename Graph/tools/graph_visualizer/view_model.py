"""模块：静态图视图模型构建。

职责：
  把 ``ModelGraph`` 转换为可视化节点、边和 Layer 区间；折叠连续相同层，
  为每个显示算子生成融合组成、精度和参数边。

边界：
  - 不读取权重内容、不执行推理；
  - 不负责布局、SVG 绘制或命令行；
  - 参数边只保留精度与 shape，不保留权重数值。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ToolChain.common.model_ir import ModelGraph, OpMeta, TensorMeta

from graph_visualizer.composition import (
    describe_op,
    is_known_op_type,
)
from graph_visualizer.package import LoadedPackage
from graph_visualizer.repeats import LayerRun, analyze_layer_runs

_LAYER_OP = re.compile(r"^layer\.(?P<index>\d+)\.(?P<suffix>.*)$")
_LAYER_REF = re.compile(r"layer\.(?P<index>\d+)\.")


@dataclass(frozen=True)
class Node:
    """可视化节点。

    属性：
      id：稳定节点 ID；
      name：显示名称；
      op_type：稳定 op 类型名；
      category：显示类别；
      precision：节点精度；
      attrs：算子属性；
      composition：散算子组成；
      node_type：compute/input/output。
    """

    id: str
    name: str
    op_type: str
    category: str
    precision: str
    shape: tuple[int | str, ...]
    attrs: dict[str, Any]
    composition: tuple[str, ...]
    node_type: str


@dataclass(frozen=True)
class Edge:
    """可视化边。

    属性：
      id：稳定边 ID；
      src：源节点 ID；参数边为 None；
      dst：目标节点 ID；
      tensor：张量名；
      precision：张量精度；
      shape：张量形状；
      kind：input/activation/output/parameter。
    """

    id: str
    src: str | None
    dst: str
    tensor: str
    precision: str
    shape: tuple[int | str, ...]
    kind: str


@dataclass(frozen=True)
class LayerRunView:
    """带显示节点列表的 Layer 区间。

    属性：
      run：Layer 区间元数据；
      node_ids：代表性子图的节点 ID。
    """

    run: LayerRun
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class ViewModel:
    """完整可视化视图模型。

    属性：
      title：图标题；
      model_name：模型图名；
      manifest_sha256：manifest 哈希；
      ir_sha256：IR 哈希；
      nodes：节点元组；
      edges：边元组；
      runs：Layer 区间视图；
      input_ids：输入边界节点 ID；
      output_ids：输出边界节点 ID；
      warnings：构建过程中产生的警告。
    """

    title: str
    model_name: str
    manifest_sha256: str
    ir_sha256: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    runs: tuple[LayerRunView, ...]
    input_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    warnings: tuple[str, ...]


def _shape_of(meta: TensorMeta | None) -> tuple[int | str, ...]:
    """返回张量形状的值元组。

    参数：
      meta：张量元数据；None 时返回空元组。

    返回：
      维度值元组。
    """
    if meta is None:
        return ()
    return tuple(dim.value for dim in meta.shape)


def _precision_of(meta: TensorMeta | None, fallback: str) -> str:
    """返回张量精度。

    参数：
      meta：张量元数据；
      fallback：元数据缺失时的回退精度。

    返回：
      精度名。
    """
    if meta is None or not meta.precision:
        return fallback
    return meta.precision


def _layer_index(name: str) -> int | None:
    """解析 op 名称中的层号。

    参数：
      name：op 名称。

    返回：
      层号；不是 layer op 时返回 None。
    """
    match = _LAYER_OP.match(name)
    if match is None:
        return None
    return int(match.group("index"))


def _map_op_name(name: str, representative: dict[int, int]) -> str:
    """把层内 op 名映射到代表层。

    参数：
      name：原始 op 名；
      representative：层号到代表层号的映射。

    返回：
      映射后的 op 名；非层 op 原样返回。
    """
    match = _LAYER_OP.match(name)
    if match is None:
        return name
    index = int(match.group("index"))
    target = representative.get(index, index)
    return f"layer.{target}.{match.group('suffix')}"


def _map_tensor_name(name: str, representative: dict[int, int]) -> str:
    """把层内张量名映射到代表层。

    参数：
      name：原始张量名；
      representative：层号到代表层号的映射。

    返回：
      映射后的张量名；非层张量原样返回。
    """
    match = _LAYER_REF.match(name)
    if match is None:
        return name
    index = int(match.group("index"))
    target = representative.get(index, index)
    return f"layer.{target}." + name[match.end():]


def _node_shape(op: OpMeta, graph: ModelGraph) -> tuple[int | str, ...]:
    """返回节点输出形状。

    参数：
      op：算子元数据；
      graph：模型图。

    返回：
      第一个输出张量的形状；缺失时返回空元组。
    """
    if not op.outputs:
        return ()
    return _shape_of(_meta_for(graph, op.outputs[0]))


def _node_precision(op: OpMeta, graph: ModelGraph) -> str:
    """计算节点显示的精度。

    参数：
      op：算子元数据；
      graph：模型图。

    返回：
      单输出精度、多个输出精度集合或属性精度。
    """
    values = []
    for name in op.outputs:
        try:
            values.append(graph.tensor(name).precision)
        except KeyError:
            continue
    unique = sorted({value for value in values if value})
    if unique:
        return "/".join(unique)
    fallback = op.attrs.get("precision")
    return str(fallback) if fallback else "unknown"


def _meta_for(graph: ModelGraph, name: str) -> TensorMeta | None:
    """安全查询张量元数据。

    参数：
      graph：模型图；
      name：张量名。

    返回：
      ``TensorMeta``；不存在时返回 None。
    """
    try:
        return graph.tensor(name)
    except KeyError:
        return None


def _display_ops(
    graph: ModelGraph,
    representative: dict[int, int],
) -> list[tuple[str, OpMeta]]:
    """返回需要显示的算子及其显示名。

    参数：
      graph：模型图；
      representative：层号到代表层号的映射。

    返回：
      ``(display_name, op)`` 列表，保持 ModelGraph 顺序。
    """
    display_indices = set(representative.values())
    result = []
    for op in graph.ops:
        index = _layer_index(op.name)
        if index is not None and index not in display_indices:
            continue
        result.append((_map_op_name(op.name, representative), op))
    return result


def _edge_meta(
    graph: ModelGraph,
    original: str,
    mapped: str,
) -> TensorMeta | None:
    """查询边的张量元数据。

    参数：
      graph：模型图；
      original：原始张量名；
      mapped：映射后的张量名。

    返回：
      ``TensorMeta``；优先使用映射后的名称。
    """
    return _meta_for(graph, mapped) or _meta_for(graph, original)


def build_view_model(package: LoadedPackage) -> ViewModel:
    """构建静态图视图模型。

    参数：
      package：已校验的模型包。

    返回：
      ``ViewModel``；已折叠连续相同层并生成参数边。

    异常：
      ValueError：模型图缺少输出或张量引用不完整时由 ModelGraph 校验抛出。
    """
    graph = package.model_graph
    runs = analyze_layer_runs(graph.ops, set(graph.constants))
    representative: dict[int, int] = {}
    for run in runs:
        for index in range(run.start, run.end + 1):
            representative[index] = run.start

    warnings = []
    display_ops = _display_ops(graph, representative)
    display_names = {name for name, _ in display_ops}

    nodes: list[Node] = []
    input_ids: list[str] = []
    output_ids: list[str] = []

    for name in graph.inputs:
        meta = _meta_for(graph, name)
        node_id = f"input:{name}"
        input_ids.append(node_id)
        nodes.append(Node(
            id=node_id,
            name=f"INPUT {name}",
            op_type="input",
            category="io",
            precision=_precision_of(meta, "unknown"),
            shape=_shape_of(meta),
            attrs={},
            composition=(),
            node_type="input",
        ))

    for display_name, op in display_ops:
        description = describe_op(op.op_type, op.attrs)
        if not is_known_op_type(op.op_type):
            warnings.append(
                f"unknown op type: {op.op_type} ({display_name})")
        nodes.append(Node(
            id=display_name,
            name=display_name,
            op_type=op.op_type,
            category=description.category,
            precision=_node_precision(op, graph),
            shape=_node_shape(op, graph),
            attrs=dict(op.attrs),
            composition=description.composition,
            node_type="compute",
        ))

    for binding in graph.outputs:
        meta = _meta_for(graph, binding.tensor)
        producer = next(
            (op for op in graph.ops if binding.tensor in op.outputs),
            None,
        )
        if producer is None:
            warnings.append(
                f"output tensor has no producer: {binding.tensor}")
            continue
        node_id = f"output:{binding.name}"
        output_ids.append(node_id)
        nodes.append(Node(
            id=node_id,
            name=f"OUTPUT {binding.name}",
            op_type="output",
            category="io",
            precision=_precision_of(meta, "unknown"),
            shape=_shape_of(meta),
            attrs={},
            composition=(),
            node_type="output",
        ))

    producer_by_tensor: dict[str, str] = {}
    for op in graph.ops:
        for output in op.outputs:
            producer_by_tensor[output] = op.name

    edges: list[Edge] = []
    seen: set[tuple[str | None, str, str, str]] = set()

    def add_edge(
        src: str | None,
        dst: str,
        tensor: str,
        meta: TensorMeta | None,
        kind: str,
        fallback_precision: str,
    ) -> None:
        """追加去重后的边。

        参数：
          src：源节点 ID；参数边为 None；
          dst：目标节点 ID；
          tensor：显示用张量名；
          meta：张量元数据；
          kind：边类型；
          fallback_precision：元数据缺失时的精度。
        """
        key = (src, dst, tensor, kind)
        if key in seen:
            return
        seen.add(key)
        edges.append(Edge(
            id=f"e{len(edges)}",
            src=src,
            dst=dst,
            tensor=tensor,
            precision=_precision_of(meta, fallback_precision),
            shape=_shape_of(meta),
            kind=kind,
        ))

    for display_name, op in display_ops:
        for source in op.inputs:
            mapped = _map_tensor_name(source, representative)
            meta = _edge_meta(graph, source, mapped)
            if source in graph.constants:
                add_edge(
                    None,
                    display_name,
                    mapped,
                    meta,
                    "parameter",
                    "unknown",
                )
                continue
            if source in graph.inputs:
                add_edge(
                    f"input:{source}",
                    display_name,
                    source,
                    meta,
                    "input",
                    _precision_of(meta, "unknown"),
                )
                continue
            producer = producer_by_tensor.get(source)
            if producer is None:
                warnings.append(
                    f"tensor has no producer: {source}")
                continue
            producer_id = _map_op_name(producer, representative)
            if producer_id not in display_names:
                warnings.append(
                    f"producer is not displayed: {producer}")
                continue
            add_edge(
                producer_id,
                display_name,
                mapped,
                meta,
                "activation",
                _precision_of(meta, "unknown"),
            )

    for binding in graph.outputs:
        producer = producer_by_tensor.get(binding.tensor)
        if producer is None:
            continue
        producer_id = _map_op_name(producer, representative)
        if producer_id not in display_names:
            continue
        meta = _meta_for(graph, binding.tensor)
        add_edge(
            producer_id,
            f"output:{binding.name}",
            binding.tensor,
            meta,
            "output",
            _precision_of(meta, "unknown"),
        )

    run_views = []
    for run in runs:
        prefix = f"layer.{run.start}."
        node_ids = tuple(
            _map_op_name(op.name, representative)
            for op in graph.ops
            if op.name.startswith(prefix)
        )
        if run.is_variant:
            warnings.append(
                f"layer {run.start}-{run.end} differs from base layers")
        run_views.append(LayerRunView(run=run, node_ids=node_ids))

    return ViewModel(
        title=f"{package.model_name} Static Graph",
        model_name=package.model_name,
        manifest_sha256=package.manifest_sha256,
        ir_sha256=package.ir_sha256,
        nodes=tuple(nodes),
        edges=tuple(edges),
        runs=tuple(run_views),
        input_ids=tuple(input_ids),
        output_ids=tuple(output_ids),
        warnings=tuple(warnings),
    )
