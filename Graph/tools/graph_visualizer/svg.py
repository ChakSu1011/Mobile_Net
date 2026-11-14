"""模块：自包含 SVG 渲染。

职责：
  把 ViewModel 与 LayoutResult 渲染为离线可打开的 SVG，包含节点、边、
  参数边、Layer 背景框、图例和哈希信息。

边界：
  - 不计算布局、不读取模型包；
  - 不引用外部 JS/CSS/字体；
  - 相同输入生成字节一致的 SVG。
"""

from __future__ import annotations

from pathlib import Path

from graph_visualizer.composition import category_color
from graph_visualizer.formatting import (
    clip_text,
    format_shape,
    precision_label,
    xml_escape,
)
from graph_visualizer.layout import LayoutResult, NodeBox, RunBox
from graph_visualizer.view_model import Edge, Node, ViewModel


def _fmt(value: float) -> str:
    """格式化 SVG 坐标。

    参数：
      value：浮点坐标。

    返回：
      一位小数文本。
    """
    return f"{value:.1f}"


def _node_fill(node: Node) -> str:
    """返回节点填充色。

    参数：
      node：节点。

    返回：
      十六进制颜色。
    """
    if node.node_type == "input":
        return "#eff6ff"
    if node.node_type == "output":
        return "#ecfdf5"
    return "#ffffff"


def _edge_color(edge: Edge) -> str:
    """返回边颜色。

    参数：
      edge：边。

    返回：
      十六进制颜色。
    """
    if edge.kind == "parameter":
        return "#d97706"
    if edge.kind == "input":
        return "#2563eb"
    if edge.kind == "output":
        return "#16a34a"
    return "#334155"


def _marker_id(edge: Edge) -> str:
    """返回边使用的箭头 marker ID。

    参数：
      edge：边。

    返回：
      marker ID。
    """
    if edge.kind == "parameter":
        return "arrow-param"
    if edge.kind == "input":
        return "arrow-input"
    if edge.kind == "output":
        return "arrow-output"
    return "arrow-activation"


def _edge_text(edge: Edge) -> str:
    """返回边的可见标签。

    参数：
      edge：边。

    返回：
      参数边只显示精度与 shape；其他边显示 tensor 名、精度和 shape。
    """
    precision = precision_label(edge.precision)
    shape = format_shape(edge.shape)
    if edge.kind == "parameter":
        return f"[{precision}, {shape}]"
    return f"{edge.tensor} [{precision}, {shape}]"


def _text_width(text: str, size: float) -> float:
    """估算 SVG 文本宽度。

    参数：
      text：文本；
      size：字号。

    返回：
      估算像素宽度。
    """
    width = 0.0
    for char in text:
        width += size * (1.8 if ord(char) > 0x2E80 else 0.62)
    return width


def _label(
    x: float,
    y: float,
    text: str,
    size: float,
    color: str,
    *,
    anchor: str = "start",
    bold: bool = False,
) -> str:
    """生成 SVG 文本元素。

    参数：
      x：横坐标；
      y：纵坐标；
      text：文本；
      size：字号；
      color：颜色；
      anchor：text-anchor；
      bold：是否加粗。

    返回：
      ``<text>`` 字符串。
    """
    weight = " font-weight=\"600\"" if bold else ""
    return (
        f'<text x="{_fmt(x)}" y="{_fmt(y)}" font-size="{size}" '
        f'fill="{color}" text-anchor="{anchor}"{weight}>'
        f"{xml_escape(text)}</text>"
    )


def _edge_label(x: float, y: float, text: str) -> str:
    """生成带白色背景的边标签。

    参数：
      x：中心横坐标；
      y：中心纵坐标；
      text：标签文本。

    返回：
      SVG 片段。
    """
    clipped = clip_text(text, 52)
    width = _text_width(clipped, 10.0) + 10.0
    height = 16.0
    return (
        f'<rect x="{_fmt(x - width / 2.0)}" '
        f'y="{_fmt(y - height / 2.0)}" width="{_fmt(width)}" '
        f'height="{_fmt(height)}" rx="3" fill="#ffffff" '
        f'fill-opacity="0.92"/>\n'
        + _label(x, y + 3.5, clipped, 10, "#475569", anchor="middle")
    )


def _render_legend(width: float) -> str:
    """生成图例。

    参数：
      width：画布宽度。

    返回：
      SVG 片段。
    """
    x = width - 250.0
    y = 34.0
    parts = [
        f'<rect x="{_fmt(x - 14)}" y="{_fmt(y - 20)}" '
        f'width="236" height="76" rx="8" fill="#ffffff" '
        f'stroke="#cbd5e1"/>',
        _label(x, y, "图例", 12, "#0f172a", bold=True),
    ]
    rows = (
        ("#334155", "", "输入 / 激活 / 输出"),
        ("#d97706", "5 4", "参数（仅精度与 shape）"),
        ("#dc2626", "4 3", "差异 Layer"),
    )
    for index, (color, dash, text) in enumerate(rows):
        row_y = y + 20.0 + index * 18.0
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{_fmt(x)}" y1="{_fmt(row_y - 4)}" '
            f'x2="{_fmt(x + 38)}" y2="{_fmt(row_y - 4)}" '
            f'stroke="{color}" stroke-width="2"{dash_attr}/>'
        )
        parts.append(_label(x + 48, row_y, text, 10, "#334155"))
    return "\n".join(parts)


def _render_run_box(run_box: RunBox) -> str:
    """生成 Layer 代表子图背景框。

    参数：
      run_box：Layer 背景框布局。

    返回：
      SVG 片段。
    """
    run = run_box.view.run
    color = "#dc2626" if run.is_variant else "#94a3b8"
    dash = "6 4"
    if run.repeat_count > 1:
        label = (
            f"DecoderLayer ×{run.repeat_count} "
            f"(layer.{run.start}–layer.{run.end})")
        if run.is_variant:
            label = "变体 " + label
    else:
        label = f"Layer {run.start}"
        if run.is_variant:
            label += " （差异）"
    parts = [
        f'<rect x="{_fmt(run_box.x)}" y="{_fmt(run_box.y)}" '
        f'width="{_fmt(run_box.width)}" height="{_fmt(run_box.height)}" '
        f'rx="14" fill="#f8fafc" fill-opacity="0.72" '
        f'stroke="{color}" stroke-width="2" '
        f'stroke-dasharray="{dash}"/>',
        _label(
            run_box.x + 14,
            run_box.y + 22,
            label,
            13,
            color,
            bold=True,
        ),
    ]
    return "\n".join(parts)


def _render_node(box: NodeBox) -> str:
    """生成节点框与文本。

    参数：
      box：节点布局框。

    返回：
      SVG 片段。
    """
    node = box.node
    stroke = category_color(node.category)
    parts = [
        f'<rect x="{_fmt(box.x)}" y="{_fmt(box.y)}" '
        f'width="{_fmt(box.width)}" height="{_fmt(box.height)}" '
        f'rx="7" fill="{_node_fill(node)}" stroke="{stroke}" '
        f'stroke-width="1.5"/>',
    ]
    text_x = box.x + 12.0
    text_y = box.y + 20.0
    for index, line in enumerate(box.lines):
        parts.append(_label(
            text_x,
            text_y + index * 17.0,
            line,
            13 if index == 0 else 10.5,
            "#0f172a",
            bold=index == 0,
        ))
    return "\n".join(parts)


def _render_parameter_edges(
    view: ViewModel,
    layout: LayoutResult,
) -> str:
    """生成参数边。

    参数：
      view：视图模型；
      layout：布局结果。

    返回：
      SVG 片段。
    """
    grouped: dict[str, list[Edge]] = {}
    for edge in view.edges:
        if edge.kind == "parameter":
            grouped.setdefault(edge.dst, []).append(edge)
    parts = []
    for dst, edges in grouped.items():
        box = layout.boxes.get(dst)
        if box is None:
            continue
        center_y = box.y + box.height / 2.0
        count = len(edges)
        for index, edge in enumerate(edges):
            y = center_y + (index - (count - 1) / 2.0) * 18.0
            parts.append(
                f'<circle cx="{_fmt(box.x - 48)}" cy="{_fmt(y)}" '
                f'r="9" fill="#fff7ed" stroke="#d97706"/>'
            )
            parts.append(_label(
                box.x - 48, y + 3.5, "W", 9, "#b45309",
                anchor="middle", bold=True))
            parts.append(
                f'<line x1="{_fmt(box.x - 38)}" y1="{_fmt(y)}" '
                f'x2="{_fmt(box.x - 2)}" y2="{_fmt(y)}" '
                f'stroke="#d97706" stroke-width="1.4" '
                f'stroke-dasharray="5 4" '
                f'marker-end="url(#arrow-param)"/>'
            )
            parts.append(_label(
                box.x - 62, y + 3.5, _edge_text(edge), 9.5,
                "#92400e", anchor="end"))
    return "\n".join(parts)


def _render_flow_edges(
    view: ViewModel,
    layout: LayoutResult,
) -> str:
    """生成激活/输入/输出边。

    参数：
      view：视图模型；
      layout：布局结果。

    返回：
      SVG 片段。

    主要逻辑：
      边标签放在目标节点上方，同一目标的多条入边向上错开，避免多条边在
      中点附近互相覆盖。
    """
    incoming: dict[str, int] = {}
    totals: dict[str, int] = {}
    for edge in view.edges:
        if edge.kind != "parameter" and edge.src is not None:
            totals[edge.dst] = totals.get(edge.dst, 0) + 1

    parts = []
    for edge in view.edges:
        if edge.kind == "parameter" or edge.src is None:
            continue
        src = layout.boxes.get(edge.src)
        dst = layout.boxes.get(edge.dst)
        if src is None or dst is None:
            continue
        x1 = src.x + src.width / 2.0
        y1 = src.y + src.height
        x2 = dst.x + dst.width / 2.0
        y2 = dst.y
        control = max(40.0, abs(y2 - y1) * 0.45)
        path = (
            f"M {_fmt(x1)} {_fmt(y1)} C {_fmt(x1)} "
            f"{_fmt(y1 + control)}, {_fmt(x2)} "
            f"{_fmt(y2 - control)}, {_fmt(x2)} {_fmt(y2)}"
        )
        color = _edge_color(edge)
        parts.append(
            f'<path d="{path}" fill="none" stroke="{color}" '
            f'stroke-width="1.6" marker-end="url(#{_marker_id(edge)})"/>'
        )
        index = incoming.get(edge.dst, 0)
        incoming[edge.dst] = index + 1
        total = totals[edge.dst]
        label_y = dst.y - 12.0 - (total - 1 - index) * 14.0
        parts.append(_edge_label(x2, label_y, _edge_text(edge)))
    return "\n".join(parts)


def render_svg(
    view: ViewModel,
    layout: LayoutResult,
    output_path: Path,
) -> Path:
    """渲染并写出 SVG 文件。

    参数：
      view：视图模型；
      layout：布局结果；
      output_path：输出 SVG 路径。

    返回：
      实际写出的路径。

    异常：
      OSError：目录或文件不可写。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_fmt(layout.width)}" height="{_fmt(layout.height)}" '
        f'viewBox="0 0 {_fmt(layout.width)} {_fmt(layout.height)}">',
        "<defs>",
        '<marker id="arrow-activation" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#334155"/></marker>',
        '<marker id="arrow-input" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#2563eb"/></marker>',
        '<marker id="arrow-output" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#16a34a"/></marker>',
        '<marker id="arrow-param" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#d97706"/></marker>',
        "</defs>",
        f'<rect width="{_fmt(layout.width)}" '
        f'height="{_fmt(layout.height)}" fill="#ffffff"/>',
        _label(28, 38, view.title, 22, "#0f172a", bold=True),
        _label(
            28,
            62,
            f"model: {view.model_name}  |  "
            f"manifest: {view.manifest_sha256[:16]}...  |  "
            f"ir: {view.ir_sha256[:16]}...",
            11,
            "#475569",
        ),
        _render_legend(layout.width),
    ]
    parts.extend(_render_run_box(box) for box in layout.run_boxes)
    parts.append(_render_flow_edges(view, layout))
    parts.append(_render_parameter_edges(view, layout))
    parts.extend(_render_node(box) for box in layout.boxes.values())
    parts.append(_label(
        28,
        layout.height - 34,
        f"manifest SHA-256: {view.manifest_sha256}",
        9.5,
        "#64748b",
    ))
    parts.append(_label(
        28,
        layout.height - 18,
        f"IR SHA-256: {view.ir_sha256}",
        9.5,
        "#64748b",
    ))
    parts.append("</svg>")
    text = "\n".join(parts) + "\n"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return output_path
