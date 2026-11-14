"""模块：静态图视图模型构建。

职责：
  把 ``ModelGraph`` 转换为融合算子节点、数据边平行四边形和 Layer 区间；折叠连续
  相同层，为输入/输出/激活/参数生成唯一的 tensor 平行四边形。

边界：
  - 不读取权重内容、不执行推理；
  - 不负责布局、SVG 绘制或命令行；
  - 参数平行四边形只保留精度与 shape，不保留权重名称或数值。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

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
    """融合算子节点。

    属性：
      id：稳定节点 ID；
      name：显示名称；
      op_type：稳定 op 类型名；
      category：显示类别；
      precision：节点精度；
      composition：散算子组成。
    """

    id: str
    name: str
    op_type: str
    category: str
    precision: str
    composition: tuple[str, ...]


@dataclass(frozen=True)
class TensorNode:
    """数据边平行四边形节点。

    属性：
      id：稳定 ID；
      name：张量名；
      precision：张量精度；
      shape：张量形状；
      kind：input/activation/output/parameter；
      caption：输入/输出平行四边形外部显示的 tensor 名称。
    """

    id: str
    name: str
    precision: str
    shape: tuple[int | str, ...]
    kind: str
    caption: str | None


@dataclass(frozen=True)
class Link:
    """节点或数据边之间的连接。

    属性：
      id：稳定连接 ID；
      src：源节点/数据边 ID；
      dst：目标节点/数据边 ID。
    """

    id: str
    src: str
    dst: str


@dataclass(frozen=True)
class LayerRunView:
    """带显示节点列表的 Layer 区间。

    属性：
      run：Layer 区间元数据；
      node_ids：代表性子图的融合算子节点 ID。
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
      nodes：融合算子节点；
      tensors：输入/输出/激活/参数平行四边形；
      links：连接；
      runs：Layer 区间视图；
      warnings：构建警告。
    """

    title: str
    model_name: str
    manifest_sha256: str
    ir_sha256: str
    nodes: tuple[Node, ...]
    tensors: tuple[TensorNode, ...]
    links: tuple[Link, ...]
    runs: tuple[LayerRunView, ...]
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


def _numel(shape: tuple[int | str, ...]) -> int:
    """返回静态形状的元素数量。

    参数：
      shape：静态形状。

    返回：
      元素数量；不能计算时返回 0。
    """
    if not shape:
        return 1
    try:
        return math.prod(int(value) for value in shape)
    except (TypeError, ValueError):
        return 0


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


def build_view_model(package: LoadedPackage) -> ViewModel:
    """构建静态图视图模型。

    参数：
      package：已校验的模型包。

    返回：
      ``ViewModel``；包含融合算子节点、tensor 平行四边形和连接。
    """
    graph = package.model_graph
    runs = analyze_layer_runs(graph.ops, set(graph.constants))
    representative: dict[int, int] = {}
    for run in runs:
        for index in range(run.start, run.end + 1):
            representative[index] = run.start

    warnings: list[str] = []
    display_ops = _display_ops(graph, representative)
    display_names = {name for name, _ in display_ops}

    nodes: list[Node] = []
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
            composition=description.composition,
        ))

    tensors: dict[str, TensorNode] = {}
    links: list[Link] = []
    link_seen: set[tuple[str, str]] = set()

    def ensure_tensor(
        name: str,
        kind: str,
        caption: str | None = None,
        *,
        key_suffix: str = "",
    ) -> str | None:
        """创建或更新一个 tensor 平行四边形。

        参数：
          name：原始张量名；
          kind：input/activation/output/parameter；
          caption：输入/输出平行四边形外部说明；
          key_suffix：参数平行四边形按消费者去重使用的后缀。

        返回：
          tensor 平行四边形 ID；元数据缺失时返回 None。
        """
        mapped = _map_tensor_name(name, representative)
        meta = _edge_meta(graph, name, mapped)
        if meta is None:
            return None
        tensor_id = f"tensor:{mapped}{key_suffix}"
        existing = tensors.get(tensor_id)
        final_kind = kind
        final_caption = caption
        if existing is not None:
            if kind == "output":
                final_kind = "output"
            elif existing.kind == "output":
                final_kind = "output"
                final_caption = existing.caption
            elif existing.kind == "input":
                final_kind = existing.kind
            if final_caption is None:
                final_caption = existing.caption
        tensors[tensor_id] = TensorNode(
            id=tensor_id,
            name=mapped,
            precision=_precision_of(meta, "unknown"),
            shape=_shape_of(meta),
            kind=final_kind,
            caption=final_caption,
        )
        return tensor_id

    def add_link(src: str, dst: str) -> None:
        """追加去重连接。

        参数：
          src：源 ID；
          dst：目标 ID。
        """
        key = (src, dst)
        if key in link_seen:
            return
        link_seen.add(key)
        links.append(Link(id=f"l{len(links)}", src=src, dst=dst))

    for name in graph.inputs:
        ensure_tensor(name, "input", caption=name)

    producer_by_tensor: dict[str, str] = {}
    for op in graph.ops:
        for output in op.outputs:
            producer_by_tensor[output] = op.name

    for display_name, op in display_ops:
        for source in op.inputs:
            if source in graph.constants:
                meta = _edge_meta(
                    graph,
                    source,
                    _map_tensor_name(source, representative),
                )
                if meta is None:
                    warnings.append(
                        f"constant metadata missing: {source}")
                    continue
                shape = _shape_of(meta)
                if _numel(shape) <= 1:
                    warnings.append(
                        f"scalar parameter not drawn: {source}")
                    continue
                tensor_id = ensure_tensor(
                    source,
                    "parameter",
                    key_suffix=f"@{display_name}",
                )
                if tensor_id is not None:
                    add_link(tensor_id, display_name)
                continue
            if source in graph.inputs:
                tensor_id = ensure_tensor(
                    source, "input", caption=source)
                if tensor_id is not None:
                    add_link(tensor_id, display_name)
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
            tensor_id = ensure_tensor(source, "activation")
            if tensor_id is not None:
                add_link(tensor_id, display_name)

        for output in op.outputs:
            tensor_id = ensure_tensor(output, "activation")
            if tensor_id is not None:
                add_link(display_name, tensor_id)

    for binding in graph.outputs:
        producer = producer_by_tensor.get(binding.tensor)
        if producer is None:
            warnings.append(
                f"output tensor has no producer: {binding.tensor}")
            continue
        producer_id = _map_op_name(producer, representative)
        if producer_id not in display_names:
            continue
        tensor_id = ensure_tensor(
            binding.tensor, "output", caption=binding.name)
        if tensor_id is not None:
            add_link(producer_id, tensor_id)

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
        tensors=tuple(tensors.values()),
        links=tuple(links),
        runs=tuple(run_views),
        warnings=tuple(warnings),
    )
