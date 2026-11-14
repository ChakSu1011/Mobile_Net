"""模块：自包含 SVG 渲染。

职责：
  把 ViewModel 与 LayoutResult 渲染为离线可打开的 SVG。融合算子为矩形节点，
  输入/输出/激活/参数为串在数据流中的圆角菱形，菱形只显示精度与 shape。

边界：
  - 不计算布局、不读取模型包；
  - 不引用外部 JS/CSS/字体；
  - 相同输入生成字节一致的 SVG。
"""

from __future__ import annotations

import math
from pathlib import Path

from graph_visualizer.composition import category_color
from graph_visualizer.formatting import xml_escape
from graph_visualizer.layout import (
    ComputeBox,
    LayoutResult,
    RunBox,
    TensorBox,
)
from graph_visualizer.view_model import ViewModel


def _fmt(value: float) -> str:
    """格式化 SVG 坐标。

    参数：
      value：浮点坐标。

    返回：
      一位小数文本。
    """
    return f"{value:.1f}"


def _tensor_color(tensor_kind: str) -> str:
    """返回 tensor 菱形和连线的颜色。

    参数：
      tensor_kind：input/activation/output/parameter。

    返回：
      十六进制颜色。
    """
    if tensor_kind == "input":
        return "#2563eb"
    if tensor_kind == "output":
        return "#16a34a"
    if tensor_kind == "parameter":
        return "#d97706"
    return "#334155"


def _tensor_marker(tensor_kind: str) -> str:
    """返回 tensor 连线箭头 marker ID。

    参数：
      tensor_kind：tensor 类型。

    返回：
      marker ID。
    """
    if tensor_kind == "input":
        return "arrow-input"
    if tensor_kind == "output":
        return "arrow-output"
    if tensor_kind == "parameter":
        return "arrow-param"
    return "arrow-activation"


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
    weight = ' font-weight="600"' if bold else ""
    return (
        f'<text x="{_fmt(x)}" y="{_fmt(y)}" font-size="{size}" '
        f'fill="{color}" text-anchor="{anchor}"{weight}>'
        f"{xml_escape(text)}</text>"
    )


def _diamond_path(
    cx: float,
    cy: float,
    half_width: float,
    half_height: float,
    radius: float = 10.0,
) -> str:
    """生成圆角菱形路径。

    参数：
      cx：中心横坐标；
      cy：中心纵坐标；
      half_width：水平半宽；
      half_height：垂直半高；
      radius：圆角距离。

    返回：
      SVG path 的 ``d`` 属性值。
    """
    top = (cx, cy - half_height)
    right = (cx + half_width, cy)
    bottom = (cx, cy + half_height)
    left = (cx - half_width, cy)

    def near(origin: tuple[float, float],
             target: tuple[float, float]) -> tuple[float, float]:
        """返回 origin 朝 target 方向 radius 处的点。"""
        dx = target[0] - origin[0]
        dy = target[1] - origin[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return origin
        scale = min(radius, length / 2.0) / length
        return (origin[0] + dx * scale, origin[1] + dy * scale)

    top_left = near(top, left)
    top_right = near(top, right)
    right_top = near(right, top)
    right_bottom = near(right, bottom)
    bottom_right = near(bottom, right)
    bottom_left = near(bottom, left)
    left_bottom = near(left, bottom)
    left_top = near(left, top)
    return (
        f"M {_fmt(top_left[0])} {_fmt(top_left[1])} "
        f"Q {_fmt(top[0])} {_fmt(top[1])} "
        f"{_fmt(top_right[0])} {_fmt(top_right[1])} "
        f"L {_fmt(right_top[0])} {_fmt(right_top[1])} "
        f"Q {_fmt(right[0])} {_fmt(right[1])} "
        f"{_fmt(right_bottom[0])} {_fmt(right_bottom[1])} "
        f"L {_fmt(bottom_right[0])} {_fmt(bottom_right[1])} "
        f"Q {_fmt(bottom[0])} {_fmt(bottom[1])} "
        f"{_fmt(bottom_left[0])} {_fmt(bottom_left[1])} "
        f"L {_fmt(left_bottom[0])} {_fmt(left_bottom[1])} "
        f"Q {_fmt(left[0])} {_fmt(left[1])} "
        f"{_fmt(left_top[0])} {_fmt(left_top[1])} Z"
    )


def _render_legend(width: float) -> str:
    """生成图例。

    参数：
      width：画布宽度。

    返回：
      SVG 片段。
    """
    x = width - 290.0
    y = 36.0
    parts = [
        f'<rect x="{_fmt(x - 14)}" y="{_fmt(y - 22)}" '
        f'width="276" height="96" rx="8" fill="#ffffff" '
        f'stroke="#cbd5e1"/>',
        _label(x, y, "图例", 12, "#0f172a", bold=True),
    ]
    rows = (
        ("rect", "#475569", "融合算子"),
        ("diamond", "#334155", "输入/激活/输出边（精度+shape）"),
        ("diamond-param", "#d97706", "参数边（精度+shape）"),
        ("box", "#dc2626", "差异 Layer"),
    )
    for index, (shape, color, text) in enumerate(rows):
        row_y = y + 20.0 + index * 18.0
        if shape in ("rect", "box"):
            dash = ' stroke-dasharray="4 3"' if shape == "box" else ""
            parts.append(
                f'<rect x="{_fmt(x)}" y="{_fmt(row_y - 10)}" '
                f'width="28" height="12" rx="3" fill="#ffffff" '
                f'stroke="{color}"{dash}/>')
        else:
            dash = (
                ' stroke-dasharray="5 3"'
                if shape == "diamond-param" else ""
            )
            parts.append(
                f'<path d="{_diamond_path(x + 14, row_y - 4, 15, 9, 4)}" '
                f'fill="#ffffff" stroke="{color}"{dash}/>')
        parts.append(_label(x + 40, row_y, text, 10, "#334155"))
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
    return "\n".join([
        f'<rect x="{_fmt(run_box.x)}" y="{_fmt(run_box.y)}" '
        f'width="{_fmt(run_box.width)}" height="{_fmt(run_box.height)}" '
        f'rx="14" fill="#f8fafc" fill-opacity="0.72" '
        f'stroke="{color}" stroke-width="2" stroke-dasharray="6 4"/>',
        _label(
            run_box.x + 14,
            run_box.y + 22,
            label,
            13,
            color,
            bold=True,
        ),
    ])


def _render_compute(box: ComputeBox) -> str:
    """生成融合算子矩形节点。

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
        f'rx="7" fill="#ffffff" stroke="{stroke}" stroke-width="1.5"/>',
    ]
    for index, line in enumerate(box.lines):
        parts.append(_label(
            box.x + 12.0,
            box.y + 20.0 + index * 17.0,
            line,
            13 if index == 0 else 10.5,
            "#0f172a",
            bold=index == 0,
        ))
    return "\n".join(parts)


def _render_tensor(box: TensorBox) -> str:
    """生成圆角菱形 tensor 边。

    参数：
      box：tensor 菱形布局。

    返回：
      SVG 片段。
    """
    tensor = box.tensor
    color = _tensor_color(tensor.kind)
    dash = (
        ' stroke-dasharray="6 4"'
        if tensor.kind == "parameter" else ""
    )
    path = _diamond_path(
        box.cx, box.cy, box.half_width, box.half_height)
    parts = [
        f'<path d="{path}" fill="#ffffff" stroke="{color}" '
        f'stroke-width="1.6"{dash}/>',
        _label(
            box.cx,
            box.cy - 3.0,
            box.lines[0],
            10,
            color,
            anchor="middle",
            bold=True,
        ),
        _label(
            box.cx,
            box.cy + 11.0,
            box.lines[1],
            9.5,
            "#334155",
            anchor="middle",
        ),
    ]
    if tensor.caption:
        if tensor.kind == "input":
            caption_y = box.cy - box.half_height - 12.0
        else:
            caption_y = box.cy + box.half_height + 14.0
        parts.append(_label(
            box.cx,
            caption_y,
            tensor.caption,
            11,
            color,
            anchor="middle",
            bold=True,
        ))
    return "\n".join(parts)


def _render_links(view: ViewModel, layout: LayoutResult) -> str:
    """生成算子与菱形之间的连线。

    参数：
      view：视图模型；
      layout：布局结果。

    返回：
      SVG 片段。
    """
    parts = []
    for link in view.links:
        src_compute = layout.computes.get(link.src)
        dst_compute = layout.computes.get(link.dst)
        src_tensor = layout.tensors.get(link.src)
        dst_tensor = layout.tensors.get(link.dst)
        color = "#334155"
        marker = "arrow-activation"
        x1 = y1 = x2 = y2 = 0.0
        if src_compute is not None and dst_tensor is not None:
            x1 = src_compute.x + src_compute.width / 2.0
            y1 = src_compute.y + src_compute.height
            x2 = dst_tensor.cx
            y2 = dst_tensor.cy - dst_tensor.half_height
            color = _tensor_color(dst_tensor.tensor.kind)
            marker = _tensor_marker(dst_tensor.tensor.kind)
        elif src_tensor is not None and dst_compute is not None:
            x1 = src_tensor.cx
            y1 = src_tensor.cy + src_tensor.half_height
            x2 = dst_compute.x + dst_compute.width / 2.0
            y2 = dst_compute.y
            color = _tensor_color(src_tensor.tensor.kind)
            marker = _tensor_marker(src_tensor.tensor.kind)
            if src_tensor.tensor.kind == "parameter":
                x1 = src_tensor.cx + src_tensor.half_width
                y1 = src_tensor.cy
                x2 = dst_compute.x
                y2 = dst_compute.y + dst_compute.height / 2.0
        else:
            continue
        parts.append(
            f'<line x1="{_fmt(x1)}" y1="{_fmt(y1)}" '
            f'x2="{_fmt(x2)}" y2="{_fmt(y2)}" stroke="{color}" '
            f'stroke-width="1.6" marker-end="url(#{marker})"/>'
        )
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
    parts.append(_render_links(view, layout))
    parts.extend(_render_tensor(box) for box in layout.tensors.values())
    parts.extend(_render_compute(box) for box in layout.computes.values())
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
