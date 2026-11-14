"""模块：视图模型的至上而下布局。

职责：
  计算融合算子节点、tensor 数据边和 Layer 背景框的坐标。

边界：
  - 不做 SVG 绘制；
  - 布局失败时回退到顺序布局。
"""

from __future__ import annotations

from dataclasses import dataclass

from graph_visualizer.composition import op_type_label
from graph_visualizer.formatting import (
    display_width,
    format_shape,
    precision_label,
    wrap_lines,
)
from graph_visualizer.view_model import (
    LayerRunView,
    Node,
    TensorNode,
    ViewModel,
)

_NODE_GAP = 190.0
_TENSOR_HALF_HEIGHT = 34.0


@dataclass(frozen=True)
class ComputeBox:
    """融合算子节点的布局结果。

    属性：
      node：节点；
      x：左边界；
      y：上边界；
      width：宽度；
      height：高度；
      lines：显示文本行。
    """

    node: Node
    x: float
    y: float
    width: float
    height: float
    lines: tuple[str, ...]


@dataclass(frozen=True)
class TensorBox:
    """tensor 数据边布局结果。

    属性：
      tensor：tensor 节点；
      cx：中心横坐标；
      cy：中心纵坐标；
      half_width：水平半宽；
      half_height：垂直半高；
      lines：平行四边形内文本行。
    """

    tensor: TensorNode
    cx: float
    cy: float
    half_width: float
    half_height: float
    lines: tuple[str, ...]


@dataclass(frozen=True)
class RunBox:
    """Layer 代表子图的背景框。

    属性：
      view：Layer 区间视图；
      x：左边界；
      y：上边界；
      width：宽度；
      height：高度。
    """

    view: LayerRunView
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class LayoutResult:
    """完整布局结果。

    属性：
      width：画布宽度；
      height：画布高度；
      computes：融合算子框；
      tensors：tensor 数据边框；
      run_boxes：Layer 区间背景框。
    """

    width: float
    height: float
    computes: dict[str, ComputeBox]
    tensors: dict[str, TensorBox]
    run_boxes: tuple[RunBox, ...]


def _scatter_categories(composition: tuple[str, ...]) -> list[str]:
    """提取散算子类别并去重。

    参数：
      composition：融合组成列表，例如 ``linear(gate)``。

    返回：
      去重后的基础类别列表。
    """
    result = []
    for item in composition:
        base = item.split("(", 1)[0].strip()
        if base and base not in result:
            result.append(base)
    return result


def _node_lines(node: Node) -> list[str]:
    """生成融合算子节点文本。

    参数：
      node：融合算子节点。

    返回：
      文本行列表；不包含属性。
    """
    lines = wrap_lines(node.name, 40)
    lines.append(
        f"{op_type_label(node.op_type)}  [{node.category} / "
        f"{precision_label(node.precision)}]")
    categories = _scatter_categories(node.composition)
    if categories:
        scatter = "散算子: " + ", ".join(categories)
    else:
        scatter = "散算子: 未登记"
    lines.extend(wrap_lines(scatter, 42))
    for weight in node.weights:
        lines.append(
            f"W: {precision_label(weight.precision)}, "
            f"{format_shape(weight.shape)}")
    return lines


def _node_size(lines: list[str]) -> tuple[float, float]:
    """根据文本行估算节点尺寸。

    参数：
      lines：文本行列表。

    返回：
      ``(width, height)``。
    """
    text_width = max((display_width(line) for line in lines), default=0)
    width = min(360.0, max(220.0, text_width * 7.0 + 28.0))
    height = 24.0 + len(lines) * 17.0
    return width, height


def _tensor_display(
    tensor: TensorNode,
) -> tuple[list[str], float, float]:
    """计算 tensor 数据边文本与半宽/半高。

    参数：
      tensor：tensor 节点。

    返回：
      ``(lines, half_width, half_height)``。
    """
    lines = [
        precision_label(tensor.precision),
        format_shape(tensor.shape),
    ]
    text_width = max(display_width(line) for line in lines) * 6.0
    half_width = max(54.0, text_width / 2.0 + 18.0)
    return lines, half_width, _TENSOR_HALF_HEIGHT


def _ranks(view: ViewModel) -> dict[str, int]:
    """计算主数据流节点的 rank。

    参数：
      view：视图模型。

    返回：
      融合算子与 tensor 平行四边形的 rank 映射。
    """
    ranks = {}
    for node in view.nodes:
        ranks[node.id] = 0
    for tensor in view.tensors:
        ranks[tensor.id] = 0
    links = [
        link for link in view.links
        if link.src in ranks and link.dst in ranks
    ]
    for _ in range(len(ranks) + 1):
        changed = False
        for link in links:
            candidate = ranks[link.src] + 1
            if candidate > ranks[link.dst]:
                ranks[link.dst] = candidate
                changed = True
        if not changed:
            break
    return ranks


def _order_map(view: ViewModel) -> dict[str, int]:
    """生成同级节点的稳定排序键。

    参数：
      view：视图模型。

    返回：
      节点/数据边 ID 到顺序号的映射。
    """
    order = {}
    sequence = 0
    for tensor in view.tensors:
        if tensor.kind == "input":
            order[tensor.id] = sequence
            sequence += 1
    for node in view.nodes:
        order[node.id] = sequence
        sequence += 1
    for tensor in view.tensors:
        if tensor.kind != "input":
            order[tensor.id] = sequence
            sequence += 1
    return order


def layout_view(view: ViewModel) -> LayoutResult:
    """计算视图的自上而下布局。

    参数：
      view：视图模型。

    返回：
      ``LayoutResult``；包含算子框、tensor 数据边和 Layer 背景框。
    """
    ranks = _ranks(view)
    order = _order_map(view)

    compute_lines = {node.id: _node_lines(node) for node in view.nodes}
    compute_sizes = {
        node.id: _node_size(compute_lines[node.id])
        for node in view.nodes
    }
    tensor_lines = {
        tensor.id: _tensor_display(tensor)
        for tensor in view.tensors
    }

    rows: dict[int, list[str]] = {}
    for item_id in ranks:
        rows.setdefault(ranks[item_id], []).append(item_id)
    for row in rows.values():
        row.sort(key=lambda item_id: order.get(item_id, 0))

    def item_size(item_id: str) -> tuple[float, float]:
        """返回主数据流节点的布局尺寸。"""
        if item_id in compute_sizes:
            return compute_sizes[item_id]
        _, half_width, half_height = tensor_lines[item_id]
        return half_width * 2.0, half_height * 2.0

    row_widths = {}
    row_heights = {}
    for rank, row in rows.items():
        widths = [item_size(item_id)[0] for item_id in row]
        heights = [item_size(item_id)[1] for item_id in row]
        row_widths[rank] = sum(widths) + _NODE_GAP * (len(widths) - 1)
        row_heights[rank] = max(heights)

    left_margin = 310.0
    right_margin = 80.0
    max_row_width = max(row_widths.values(), default=0.0)
    width = max(1100.0, max_row_width + left_margin + right_margin)

    computes: dict[str, ComputeBox] = {}
    tensors: dict[str, TensorBox] = {}
    y = 160.0
    for rank in sorted(rows):
        row = rows[rank]
        x = left_margin + (
            width - left_margin - right_margin - row_widths[rank]
        ) / 2.0
        for item_id in row:
            item_width, item_height = item_size(item_id)
            if item_id in compute_sizes:
                node = next(
                    node for node in view.nodes if node.id == item_id)
                computes[item_id] = ComputeBox(
                    node=node,
                    x=x,
                    y=y + (row_heights[rank] - item_height) / 2.0,
                    width=item_width,
                    height=item_height,
                    lines=tuple(compute_lines[item_id]),
                )
            else:
                tensor = next(
                    tensor for tensor in view.tensors
                    if tensor.id == item_id)
                lines, half_width, half_height = tensor_lines[item_id]
                tensors[item_id] = TensorBox(
                    tensor=tensor,
                    cx=x + item_width / 2.0,
                    cy=y + row_heights[rank] / 2.0,
                    half_width=half_width,
                    half_height=half_height,
                    lines=tuple(lines),
                )
            x += item_width + _NODE_GAP
        y += row_heights[rank] + 62.0

    height = max(y + 70.0, 700.0)

    run_boxes = []
    for run_view in view.runs:
        run_computes = [
            computes[node_id] for node_id in run_view.node_ids
            if node_id in computes
        ]
        if not run_computes:
            continue
        min_x = min(box.x for box in run_computes) - 30.0
        min_y = min(box.y for box in run_computes) - 40.0
        max_x = max(box.x + box.width for box in run_computes) + 30.0
        max_y = max(box.y + box.height for box in run_computes) + 30.0
        for tensor in tensors.values():
            if min_x <= tensor.cx <= max_x and min_y <= tensor.cy <= max_y:
                min_x = min(min_x, tensor.cx - tensor.half_width - 10.0)
                max_x = max(max_x, tensor.cx + tensor.half_width + 10.0)
                min_y = min(min_y, tensor.cy - tensor.half_height - 10.0)
                max_y = max(max_y, tensor.cy + tensor.half_height + 10.0)
        run_boxes.append(RunBox(
            view=run_view,
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
        ))

    return LayoutResult(
        width=width,
        height=height,
        computes=computes,
        tensors=tensors,
        run_boxes=tuple(run_boxes),
    )
