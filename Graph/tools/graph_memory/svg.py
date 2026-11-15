"""模块：显存排布自包含 SVG 渲染。

职责：
  绘制 Activation 面板与权重面板，包括地址轴、复用条、明细条和统计信息。

边界：
  - 不计算布局、不读取模型包；
  - 无外部 JS/CSS/字体；
  - 相同输入生成字节一致 SVG。
"""

from __future__ import annotations

from pathlib import Path

from graph_memory.formatting import (
    format_bytes,
    xml_escape,
)
from graph_memory.layout import (
    DetailRow,
    GroupRow,
    MemoryLayout,
    StripRow,
)
from graph_memory.view_model import MemoryViewModel

_BAR_X = 550.0
_BAR_WIDTH = 1000.0
_LEFT = 40.0
_TEXT_COLOR = "#0f172a"
_MUTED = "#475569"


def _fmt(value: float) -> str:
    """格式化坐标。

    参数：
      value：浮点坐标。

    返回：
      一位小数文本。
    """
    return f"{value:.1f}"


def _label(
    x: float,
    y: float,
    text: str,
    size: float,
    color: str = _TEXT_COLOR,
    *,
    anchor: str = "start",
    bold: bool = False,
) -> str:
    """生成 SVG 文本。

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


def _clip(text: str, limit: int = 72) -> str:
    """按字符数截断文本。

    参数：
      text：原始文本；
      limit：最大字符数。

    返回：
      截断文本。
    """
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."


def _bar_x(offset: int, size: int, total: int, y: float,
           color: str, *, reuse: bool = False,
           variant: bool = False) -> str:
    """生成地址轴上的区块。

    参数：
      offset：字节偏移；
      size：字节数；
      total：arena 总字节数；
      y：纵坐标；
      color：颜色；
      reuse：是否使用复用纹理；
      variant：是否差异标记。

    返回：
      SVG 片段。
    """
    if total <= 0:
        return ""
    x = _BAR_X + _BAR_WIDTH * offset / total
    width = max(2.0, _BAR_WIDTH * size / total)
    fill = "url(#reuse-hatch)" if reuse else color
    stroke = "#dc2626" if variant else color
    return (
        f'<rect x="{_fmt(x)}" y="{_fmt(y)}" width="{_fmt(width)}" '
        f'height="10" rx="2" fill="{fill}" fill-opacity="0.72" '
        f'stroke="{stroke}" stroke-width="1"/>'
    )


def _axis(y: float, total: int) -> str:
    """生成地址轴和刻度。

    参数：
      y：轴纵坐标；
      total：总字节数。

    返回：
      SVG 片段。
    """
    parts = [
        f'<line x1="{_fmt(_BAR_X)}" y1="{_fmt(y)}" '
        f'x2="{_fmt(_BAR_X + _BAR_WIDTH)}" y2="{_fmt(y)}" '
        f'stroke="#94a3b8" stroke-width="1"/>'
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = _BAR_X + _BAR_WIDTH * fraction
        parts.append(
            f'<line x1="{_fmt(x)}" y1="{_fmt(y - 5)}" '
            f'x2="{_fmt(x)}" y2="{_fmt(y + 5)}" '
            f'stroke="#94a3b8"/>')
        parts.append(_label(
            x, y + 17, format_bytes(int(total * fraction)),
            9, _MUTED, anchor="middle"))
        parts.append(
            f'<line x1="{_fmt(x)}" y1="{_fmt(y)}" '
            f'x2="{_fmt(x)}" y2="{_fmt(y + 2000)}" '
            f'stroke="#e2e8f0" stroke-width="0.5"/>')
    return "\n".join(parts)


def _render_strip(row: StripRow, total: int) -> str:
    """生成一层复用条。

    参数：
      row：复用条布局；
      total：arena 字节数。

    返回：
      SVG 片段。
    """
    parts = [
        f'<rect x="{_fmt(_BAR_X)}" y="{_fmt(row.y)}" '
        f'width="{_fmt(_BAR_WIDTH)}" height="8" fill="#f1f5f9" '
        f'rx="2"/>',
        _label(
            _LEFT + 10, row.y + 8, f"L{row.layer}", 8, _MUTED),
    ]
    for offset, size, color in row.blocks:
        parts.append(_bar_x(
            offset, size, total, row.y, color,
            variant=row.variant))
    if row.representative:
        parts.append(
            f'<rect x="{_fmt(_BAR_X - 4)}" y="{_fmt(row.y - 1)}" '
            f'width="{_fmt(_BAR_WIDTH + 8)}" height="10" rx="3" '
            f'fill="none" stroke="#0f172a" stroke-width="1.5"/>')
    return "\n".join(parts)


def _render_detail(
    row: DetailRow,
    total: int,
    *,
    reuse: bool,
) -> str:
    """生成一行明细条。

    参数：
      row：明细布局；
      total：地址轴总字节数；
      reuse：是否复用纹理。

    返回：
      SVG 片段。
    """
    parts = [
        f'<rect x="{_fmt(_LEFT + 4)}" y="{_fmt(row.y - 2)}" '
        f'width="{_fmt(_BAR_X - _LEFT - 12)}" height="60" '
        f'rx="5" fill="#f8fafc"/>',
        f'<rect x="{_fmt(_BAR_X)}" y="{_fmt(row.y + 8)}" '
        f'width="{_fmt(_BAR_WIDTH)}" height="10" rx="2" '
        f'fill="#e2e8f0"/>',
    ]
    for index, line in enumerate(row.lines[:4]):
        parts.append(_label(
            _LEFT + 14,
            row.y + 14 + index * 13,
            _clip(line),
            10.5 if index == 0 else 9.5,
            _TEXT_COLOR,
            bold=index == 0,
        ))
    parts.append(_bar_x(
        row.offset, row.size, total, row.y + 8, row.color,
        reuse=reuse, variant=row.variant))
    if reuse:
        parts.append(_label(
            _BAR_X + 8, row.y + 30, "REUSE", 8.5,
            "#b45309", bold=True))
    if row.variant:
        parts.append(_label(
            _BAR_X + 8, row.y + 44, "差异", 8.5,
            "#dc2626", bold=True))
    return "\n".join(parts)


def _render_group(row: GroupRow, total: int) -> str:
    """生成权重总览分组条。

    参数：
      row：分组布局；
      total：权重区字节数。

    返回：
      SVG 片段。
    """
    parts = [
        _label(_LEFT + 14, row.y + 14, _clip(row.lines[0]), 10.5,
               _TEXT_COLOR, bold=True),
        _label(_LEFT + 14, row.y + 29, _clip(row.lines[1]), 9.5,
               _MUTED),
        f'<rect x="{_fmt(_BAR_X)}" y="{_fmt(row.y + 4)}" '
        f'width="{_fmt(_BAR_WIDTH)}" height="14" rx="3" '
        f'fill="#e2e8f0"/>',
    ]
    parts.append(_bar_x(
        row.offset, row.size, total, row.y + 6, row.color))
    return "\n".join(parts)


def _legend(width: float) -> str:
    """生成图例。

    参数：
      width：画布宽度。

    返回：
      SVG 片段。
    """
    x = width - 360.0
    y = 34.0
    parts = [
        f'<rect x="{_fmt(x - 14)}" y="{_fmt(y - 22)}" '
        f'width="340" height="86" rx="8" fill="#ffffff" '
        f'stroke="#cbd5e1"/>',
        _label(x, y, "图例", 12, _TEXT_COLOR, bold=True),
    ]
    rows = (
        ("#2563eb", "Activation / Weight"),
        ("url(#reuse-hatch)", "复用区块（共享 offset）"),
        ("#7c3aed", "LayerWeights ×N"),
        ("#dc2626", "差异 Layer"),
    )
    for index, (color, text) in enumerate(rows):
        row_y = y + 20.0 + index * 16.0
        parts.append(
            f'<rect x="{_fmt(x)}" y="{_fmt(row_y - 9)}" '
            f'width="24" height="10" rx="2" fill="{color}" '
            f'fill-opacity="0.72" stroke="{color}"/>')
        parts.append(_label(x + 34, row_y, text, 9.5, _MUTED))
    return "\n".join(parts)


def render_memory_svg(
    view: MemoryViewModel,
    layout: MemoryLayout,
    output_path: Path,
) -> Path:
    """渲染并写出显存排布 SVG。

    参数：
      view：显存视图模型；
      layout：布局结果；
      output_path：输出路径。

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
        '<pattern id="reuse-hatch" width="7" height="7" '
        'patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<rect width="7" height="7" fill="#fef3c7"/>'
        '<line x1="0" y1="0" x2="0" y2="7" stroke="#d97706"/>'
        "</pattern>",
        "</defs>",
        f'<rect width="{_fmt(layout.width)}" '
        f'height="{_fmt(layout.height)}" fill="#ffffff"/>',
        _label(28, 38, view.title, 22, _TEXT_COLOR, bold=True),
        _label(
            28, 62,
            f"model: {view.model_name} | manifest: "
            f"{view.manifest_sha256[:16]}... | ir: "
            f"{view.ir_sha256[:16]}...",
            10.5, _MUTED),
        _legend(layout.width),
    ]

    ax, ay, aw, ah = layout.activation_panel
    parts.append(
        f'<rect x="{_fmt(ax)}" y="{_fmt(ay)}" width="{_fmt(aw)}" '
        f'height="{_fmt(ah)}" rx="10" fill="#ffffff" '
        f'stroke="#cbd5e1"/>')
    parts.append(_label(
        ax + 18, ay + 28, "Activation Memory Arena", 17,
        _TEXT_COLOR, bold=True))
    parts.append(_label(
        ax + 18, ay + 52,
        f"arena {format_bytes(view.arena_bytes)} | "
        f"peak {format_bytes(view.peak_bytes)} | "
        f"alignment {view.alignment} B | "
        f"entries {len(view.activation_blocks)}",
        11, _MUTED))
    parts.append(_axis(layout.activation_axis_y, view.arena_bytes))
    parts.extend(_render_strip(row, view.arena_bytes)
                 for row in layout.strips)
    parts.extend(_render_detail(
        row, view.arena_bytes, reuse=row.reuse)
        for row in layout.activation_rows)

    wx, wy, ww, wh = layout.weight_panel
    parts.append(
        f'<rect x="{_fmt(wx)}" y="{_fmt(wy)}" width="{_fmt(ww)}" '
        f'height="{_fmt(wh)}" rx="10" fill="#ffffff" '
        f'stroke="#cbd5e1"/>')
    parts.append(_label(
        wx + 18, wy + 28, "Parameter / Weight Memory", 17,
        _TEXT_COLOR, bold=True))
    planned = " | planned layout" if view.planned_weights else ""
    parts.append(_label(
        wx + 18, wy + 52,
        f"constants {format_bytes(view.constants_bytes)} | "
        f"weight arena {format_bytes(view.weight_arena_bytes)} | "
        f"weights {len(view.weight_blocks)}{planned}",
        11, _MUTED))
    parts.append(_axis(layout.weight_axis_y, view.weight_arena_bytes))
    parts.extend(_render_group(row, view.weight_arena_bytes)
                 for row in layout.weight_groups)
    parts.extend(_render_detail(
        row, view.weight_arena_bytes, reuse=False)
        for row in layout.weight_rows)

    parts.append(_label(
        28, layout.height - 34,
        f"manifest SHA-256: {view.manifest_sha256}",
        9.5, "#64748b"))
    parts.append(_label(
        28, layout.height - 18,
        f"IR SHA-256: {view.ir_sha256}",
        9.5, "#64748b"))
    parts.append("</svg>")
    text = "\n".join(parts) + "\n"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return output_path
