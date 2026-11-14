"""模块：视图模型的至上而下布局。

职责：
  计算节点文本、尺寸、rank、坐标，以及 Layer 代表子图的背景框。

边界：
  - 不做 SVG 绘制；
  - 参数边不参与 rank 计算；
  - 布局失败（如 DAG 异常）时回退到顺序布局。
"""

from __future__ import annotations

from dataclasses import dataclass

from graph_visualizer.formatting import (
    display_width,
    format_attrs,
    format_shape,
    precision_label,
    wrap_lines,
)
from graph_visualizer.composition import op_type_label
from graph_visualizer.view_model import LayerRunView, Node, ViewModel

_NODE_GAP = 180.0


@dataclass(frozen=True)
class NodeBox:
    """节点的布局结果。

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
      boxes：节点 ID 到布局框的映射；
      run_boxes：Layer 区间背景框列表。
    """

    width: float
    height: float
    boxes: dict[str, NodeBox]
    run_boxes: tuple[RunBox, ...]


def _node_lines(node: Node, max_label_items: int) -> list[str]:
    """生成节点显示文本行。

    参数：
      node：节点；
      max_label_items：属性列表内联上限。

    返回：
      文本行列表。
    """
    if node.node_type in ("input", "output"):
        shape = format_shape(node.shape)
        precision = precision_label(node.precision)
        return [node.name, f"[{precision}, {shape}]"]

    lines = wrap_lines(node.name, 40)
    lines.append(
        f"{op_type_label(node.op_type)}  [{node.category} / "
        f"{precision_label(node.precision)}]")
    composition = " → ".join(node.composition)
    if not composition:
        composition = "未登记融合组成"
    lines.extend(wrap_lines(f"组成: {composition}", 42))
    attrs = format_attrs(node.attrs, max_label_items)
    if attrs:
        lines.extend(wrap_lines(f"attrs: {attrs}", 42))
    return lines


def _node_size(lines: list[str]) -> tuple[float, float]:
    """根据文本行估算节点尺寸。

    参数：
      lines：文本行列表。

    返回：
      ``(width, height)``。
    """
    text_width = max((display_width(line) for line in lines), default=0)
    width = min(380.0, max(220.0, text_width * 7.0 + 28.0))
    height = 24.0 + len(lines) * 17.0
    return width, height


def _ranks(view: ViewModel) -> dict[str, int]:
    """计算节点的自上而下 rank。

    参数：
      view：视图模型。

    返回：
      节点 ID 到 rank 的映射。
    """
    node_ids = {node.id for node in view.nodes}
    ranks = {node_id: 0 for node_id in node_ids}
    edges = [
        edge for edge in view.edges
        if edge.kind != "parameter"
        and edge.src is not None
        and edge.src in node_ids
        and edge.dst in node_ids
    ]
    for _ in range(len(node_ids) + 1):
        changed = False
        for edge in edges:
            candidate = ranks[edge.src] + 1
            if candidate > ranks[edge.dst]:
                ranks[edge.dst] = candidate
                changed = True
        if not changed:
            break
    return ranks


def layout_view(
    view: ViewModel,
    max_label_items: int = 4,
) -> LayoutResult:
    """计算视图的自上而下布局。

    参数：
      view：视图模型；
      max_label_items：属性列表内联上限。

    返回：
      ``LayoutResult``；包含节点框和 Layer 背景框。
    """
    ranks = _ranks(view)
    order = {node.id: index for index, node in enumerate(view.nodes)}
    rows: dict[int, list[Node]] = {}
    for node in view.nodes:
        rows.setdefault(ranks[node.id], []).append(node)
    for row_nodes in rows.values():
        row_nodes.sort(key=lambda node: order[node.id])

    prepared: dict[str, tuple[list[str], float, float]] = {}
    for node in view.nodes:
        lines = _node_lines(node, max_label_items)
        width, height = _node_size(lines)
        prepared[node.id] = (lines, width, height)

    row_widths = {}
    row_heights = {}
    for rank, row_nodes in rows.items():
        widths = [prepared[node.id][1] for node in row_nodes]
        heights = [prepared[node.id][2] for node in row_nodes]
        row_widths[rank] = sum(widths) + _NODE_GAP * (len(widths) - 1)
        row_heights[rank] = max(heights)

    left_margin = 300.0
    right_margin = 70.0
    max_row_width = max(row_widths.values(), default=0.0)
    width = max(1100.0, max_row_width + left_margin + right_margin)

    boxes: dict[str, NodeBox] = {}
    y = 150.0
    for rank in sorted(rows):
        row_nodes = rows[rank]
        x = left_margin + (width - left_margin - right_margin
                           - row_widths[rank]) / 2.0
        for node in row_nodes:
            lines, node_width, node_height = prepared[node.id]
            boxes[node.id] = NodeBox(
                node=node,
                x=x,
                y=y + (row_heights[rank] - node_height) / 2.0,
                width=node_width,
                height=node_height,
                lines=tuple(lines),
            )
            x += node_width + _NODE_GAP
        y += row_heights[rank] + 58.0

    height = max(y + 70.0, 700.0)

    run_boxes = []
    for run_view in view.runs:
        run_node_boxes = [
            boxes[node_id] for node_id in run_view.node_ids
            if node_id in boxes
        ]
        if not run_node_boxes:
            continue
        min_x = min(box.x for box in run_node_boxes) - 20.0
        min_y = min(box.y for box in run_node_boxes) - 34.0
        max_x = max(box.x + box.width for box in run_node_boxes) + 20.0
        max_y = max(box.y + box.height for box in run_node_boxes) + 20.0
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
        boxes=boxes,
        run_boxes=tuple(run_boxes),
    )
