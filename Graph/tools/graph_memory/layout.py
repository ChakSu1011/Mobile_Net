"""模块：显存排布两面板布局。

职责：
  计算 Activation/权重面板、地址轴、Layer 复用条和明细行的坐标。

边界：
  不做 SVG 绘制，不读取模型包。
"""

from __future__ import annotations

from dataclasses import dataclass

from graph_memory.formatting import (
    format_bytes,
    format_dims,
    format_offset,
    precision_label,
)
from graph_memory.view_model import (
    ActivationBlock,
    LayerRun,
    MemoryViewModel,
    WeightBlock,
)

_LAYER_COLORS = (
    "#2563eb", "#0891b2", "#0f766e", "#7c3aed",
    "#b45309", "#be123c", "#4d7c0f", "#0369a1",
)
_NON_LAYER_COLORS = {
    "embedding": "#2563eb",
    "final_norm": "#0f766e",
    "lm_head": "#7c3aed",
}
_WIDTH = 1600.0
_LEFT = 40.0
_ROW_HEIGHT = 66.0


@dataclass(frozen=True)
class StripRow:
    """Layer 复用条。

    属性：
      layer：层号；
      y：纵坐标；
      blocks：``(offset, size, color)`` 列表；
      representative：是否为代表层；
      variant：是否为差异层。
    """

    layer: int
    y: float
    blocks: tuple[tuple[int, int, str], ...]
    representative: bool
    variant: bool


@dataclass(frozen=True)
class DetailRow:
    """Activation/权重明细行。

    属性：
      y：纵坐标；
      lines：左侧文本行；
      offset：字节偏移；
      size：字节数；
      color：条形颜色；
      reuse：是否共享 offset；
      repeated：重复标记；
      variant：差异标记。
    """

    y: float
    lines: tuple[str, ...]
    offset: int
    size: int
    color: str
    reuse: bool
    repeated: str | None
    variant: bool


@dataclass(frozen=True)
class GroupRow:
    """权重总览分组行。

    属性：
      y：纵坐标；
      lines：文本行；
      offset：起始偏移；
      size：总大小；
      color：颜色。
    """

    y: float
    lines: tuple[str, ...]
    offset: int
    size: int
    color: str


@dataclass(frozen=True)
class MemoryLayout:
    """完整布局结果。

    属性：
      width：画布宽度；
      height：画布高度；
      activation_panel：Activation 面板矩形；
      weight_panel：权重面板矩形；
      activation_axis_y：Activation 地址轴纵坐标；
      weight_axis_y：权重地址轴纵坐标；
      strips：Layer 复用条；
      activation_rows：Activation 明细；
      weight_groups：权重总览；
      weight_rows：权重明细。
    """

    width: float
    height: float
    activation_panel: tuple[float, float, float, float]
    weight_panel: tuple[float, float, float, float]
    activation_axis_y: float
    weight_axis_y: float
    strips: tuple[StripRow, ...]
    activation_rows: tuple[DetailRow, ...]
    weight_groups: tuple[GroupRow, ...]
    weight_rows: tuple[DetailRow, ...]


def _color_for_layer(layer: int | None, name: str) -> str:
    """返回区块颜色。

    参数：
      layer：层号；None 表示非层；
      name：tensor/权重名。

    返回：
      十六进制颜色。
    """
    if layer is not None:
        return _LAYER_COLORS[layer % len(_LAYER_COLORS)]
    for prefix, color in _NON_LAYER_COLORS.items():
        if name.startswith(prefix):
            return color
    return "#475569"


def _run_for_layer(
    layer: int,
    runs: tuple[LayerRun, ...],
) -> LayerRun | None:
    """返回层所属的 LayerRun。

    参数：
      layer：层号；
      runs：层区间。

    返回：
      所属区间；不存在时返回 None。
    """
    for run in runs:
        if run.start <= layer <= run.end:
            return run
    return None


def _activation_lines(
    block: ActivationBlock,
    repeated: str | None,
    reuse: bool,
) -> tuple[str, ...]:
    """生成 Activation 明细文本行。

    参数：
      block：Activation 区块；
      repeated：重复标记；
      reuse：是否共享 offset。

    返回：
      文本行。
    """
    pitch = format_dims(block.pitch)
    suffix = " (contiguous)" if block.pitch_derived else ""
    lines = [
        block.tensor,
        f"owner {block.owner_op} | shape {format_dims(block.shape)}",
        f"pitch {pitch}{suffix} | {precision_label(block.precision)}"
        f" | {format_bytes(block.size)}",
        f"offset {format_offset(block.offset)} | lifespan "
        f"{block.birth}..{block.last_use}"
        f"{' | reusable' if block.reusable else ''}",
    ]
    if reuse:
        lines[-1] += " | REUSE"
    if repeated:
        lines[-1] += f" | {repeated}"
    return tuple(lines)


def _weight_lines(
    block: WeightBlock,
    repeated: str | None,
    planned: bool,
) -> tuple[str, ...]:
    """生成权重明细文本行。

    参数：
      block：权重区块；
      repeated：重复标记；
      planned：是否 planned layout。

    返回：
      文本行。
    """
    owners = ", ".join(block.owner_ops)
    lines = [
        block.name,
        f"owner {owners} | shape {format_dims(block.shape)}",
        f"pitch {format_dims(block.pitch)} | "
        f"{precision_label(block.precision)} | "
        f"{format_bytes(block.size)}",
        f"offset {format_offset(block.offset)}"
        f"{' | planned' if planned else ''}",
    ]
    if repeated:
        lines[-1] += f" | {repeated}"
    return tuple(lines)


def layout_memory(view: MemoryViewModel) -> MemoryLayout:
    """计算两面板布局。

    参数：
      view：显存视图模型。

    返回：
      ``MemoryLayout``。
    """
    activation_runs = {
        run.start: run for run in view.activation_runs
    }
    representative_layers = set(activation_runs)
    activation_by_layer: dict[int, list[ActivationBlock]] = {}
    for block in view.activation_blocks:
        if block.layer_index is not None:
            activation_by_layer.setdefault(
                block.layer_index, []).append(block)

    activation_panel_top = 90.0
    activation_title_y = 118.0
    activation_summary_y = 142.0
    activation_axis_y = 172.0
    strip_start = 200.0
    strip_gap = 3.0
    strip_height = 8.0
    layers = sorted(activation_by_layer)
    strips = []
    for index, layer in enumerate(layers):
        run = _run_for_layer(layer, view.activation_runs)
        representative = (
            run is not None and layer == run.start)
        blocks = tuple(
            (block.offset, block.size,
             _color_for_layer(layer, block.tensor))
            for block in activation_by_layer[layer]
        )
        strips.append(StripRow(
            layer=layer,
            y=strip_start + index * (strip_height + strip_gap),
            blocks=blocks,
            representative=representative,
            variant=bool(run and run.is_variant),
        ))
    strip_end = (
        strip_start + len(layers) * (strip_height + strip_gap))
    activation_detail_top = strip_end + 26.0

    offset_counts: dict[tuple[int, int], int] = {}
    for block in view.activation_blocks:
        key = (block.offset, block.size)
        offset_counts[key] = offset_counts.get(key, 0) + 1

    detail_blocks = [
        block for block in view.activation_blocks
        if block.layer_index is None
        or block.layer_index in representative_layers
    ]
    detail_blocks.sort(key=lambda block: (
        block.offset, block.tensor))
    activation_rows = []
    y = activation_detail_top
    for block in detail_blocks:
        run = (
            _run_for_layer(block.layer_index, view.activation_runs)
            if block.layer_index is not None else None
        )
        repeated = (
            f"×{run.count}" if run and block.layer_index == run.start
            and run.count > 1 else None
        )
        reuse = offset_counts[(block.offset, block.size)] > 1
        activation_rows.append(DetailRow(
            y=y,
            lines=_activation_lines(block, repeated, reuse),
            offset=block.offset,
            size=block.size,
            color=_color_for_layer(block.layer_index, block.tensor),
            reuse=reuse,
            repeated=repeated,
            variant=bool(run and run.is_variant),
        ))
        y += _ROW_HEIGHT
    activation_panel_bottom = y + 26.0

    weight_panel_top = activation_panel_bottom + 30.0
    weight_title_y = weight_panel_top + 28.0
    weight_summary_y = weight_panel_top + 52.0
    weight_axis_y = weight_panel_top + 82.0
    group_y = weight_panel_top + 112.0
    weight_groups = []
    layer_weights: dict[int, list[WeightBlock]] = {}
    for block in view.weight_blocks:
        if block.layer_index is not None:
            layer_weights.setdefault(block.layer_index, []).append(block)
    for run in view.weight_runs:
        blocks = [
            block for layer, values in layer_weights.items()
            if run.start <= layer <= run.end for block in values
        ]
        if not blocks:
            continue
        offset = min(block.offset for block in blocks)
        end = max(block.offset + block.size for block in blocks)
        weight_groups.append(GroupRow(
            y=group_y,
            lines=(
                f"LayerWeights ×{run.count} "
                f"(layer.{run.start}–layer.{run.end})",
                f"offset {format_offset(offset)} | "
                f"size {format_bytes(end - offset)}",
            ),
            offset=offset,
            size=end - offset,
            color="#7c3aed" if not run.is_variant else "#dc2626",
        ))
        group_y += 46.0
    non_layer_weights = [
        block for block in view.weight_blocks
        if block.layer_index is None
    ]
    non_layer_weights.sort(key=lambda block: block.offset)
    for block in non_layer_weights:
        weight_groups.append(GroupRow(
            y=group_y,
            lines=(
                block.name,
                f"owner {', '.join(block.owner_ops)} | "
                f"offset {format_offset(block.offset)} | "
                f"size {format_bytes(block.size)}",
            ),
            offset=block.offset,
            size=block.size,
            color=_color_for_layer(None, block.name),
        ))
        group_y += 46.0
    weight_groups.sort(key=lambda row: row.offset)
    group_y_start = weight_panel_top + 112.0
    weight_groups = [
        GroupRow(
            y=group_y_start + index * 46.0,
            lines=row.lines,
            offset=row.offset,
            size=row.size,
            color=row.color,
        )
        for index, row in enumerate(weight_groups)
    ]
    group_y = group_y_start + len(weight_groups) * 46.0
    weight_detail_top = group_y + 20.0
    weight_detail_blocks = [
        block for block in view.weight_blocks
        if block.layer_index is None
        or block.layer_index in representative_layers
    ]
    weight_detail_blocks.sort(key=lambda block: (
        block.offset, block.name))
    weight_rows = []
    y = weight_detail_top
    for block in weight_detail_blocks:
        run = (
            _run_for_layer(block.layer_index, view.weight_runs)
            if block.layer_index is not None else None
        )
        repeated = (
            f"×{run.count}" if run and block.layer_index == run.start
            and run.count > 1 else None
        )
        weight_rows.append(DetailRow(
            y=y,
            lines=_weight_lines(block, repeated, view.planned_weights),
            offset=block.offset,
            size=block.size,
            color=_color_for_layer(block.layer_index, block.name),
            reuse=False,
            repeated=repeated,
            variant=bool(run and run.is_variant),
        ))
        y += _ROW_HEIGHT
    weight_panel_bottom = y + 28.0
    height = weight_panel_bottom + 60.0
    return MemoryLayout(
        width=_WIDTH,
        height=height,
        activation_panel=(
            _LEFT,
            activation_panel_top,
            _WIDTH - 2 * _LEFT,
            activation_panel_bottom - activation_panel_top,
        ),
        weight_panel=(
            _LEFT,
            weight_panel_top,
            _WIDTH - 2 * _LEFT,
            weight_panel_bottom - weight_panel_top,
        ),
        activation_axis_y=activation_axis_y,
        weight_axis_y=weight_axis_y,
        strips=tuple(strips),
        activation_rows=tuple(activation_rows),
        weight_groups=tuple(weight_groups),
        weight_rows=tuple(weight_rows),
    )
