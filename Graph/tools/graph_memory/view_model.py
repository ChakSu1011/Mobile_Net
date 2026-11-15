"""模块：显存排布视图模型构建。

职责：
  把 ``ModelGraph``、memory_plan 与权重条目转换为 Activation/权重区块，
  并识别重复 Layer。

边界：
  - 不读取权重内容、不修改模型包；
  - Activation Pitch 在缺失时按 row-major contiguous 推导；
  - 权重无实际 offset 时使用 planned layout。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ToolChain.common.model_ir import ModelGraph, TensorMeta

from graph_memory.package import LoadedPackage
from graph_memory.repeats import LayerRun, analyze_layer_runs

_LAYER_REF = re.compile(r"layer\.(?P<index>\d+)\.")


@dataclass(frozen=True)
class ActivationBlock:
    """Activation 显存区块。

    属性：
      tensor：tensor 名；
      owner_op：生产该 tensor 的算子；
      precision：精度；
      shape：逻辑形状；
      pitch：物理 pitch；
      pitch_derived：pitch 是否由 shape 推导；
      offset：arena 内偏移；
      size：字节数；
      birth：出生 op 下标；
      last_use：末次使用 op 下标；
      reusable：是否允许复用；
      layer_index：所在层号；非层为 None。
    """

    tensor: str
    owner_op: str
    precision: str
    shape: tuple[int | str, ...]
    pitch: tuple[int, ...]
    pitch_derived: bool
    offset: int
    size: int
    birth: int
    last_use: int
    reusable: bool
    layer_index: int | None


@dataclass(frozen=True)
class WeightBlock:
    """权重区块。

    属性：
      name：常量名；
      owner_ops：消费该权重的算子；
      precision：精度；
      shape：逻辑形状；
      pitch：物理 pitch；
      offset：权重区偏移；
      size：字节数；
      source：源权重名；
      available：是否有真实权重；
      layer_index：所在层号；非层为 None。
    """

    name: str
    owner_ops: tuple[str, ...]
    precision: str
    shape: tuple[int | str, ...]
    pitch: tuple[int, ...]
    offset: int
    size: int
    source: str | None
    available: bool
    layer_index: int | None


@dataclass(frozen=True)
class MemoryViewModel:
    """显存排布视图模型。

    属性：
      title：标题；
      model_name：图名；
      manifest_sha256：manifest 哈希；
      ir_sha256：IR 哈希；
      alignment：对齐字节数；
      arena_bytes：Activation arena 大小；
      peak_bytes：峰值；
      constants_bytes：常量逻辑字节数；
      weight_arena_bytes：权重布局总大小；
      activation_blocks：全部 Activation 区块；
      weight_blocks：全部权重区块；
      activation_runs：Activation 层区间；
      weight_runs：权重层区间；
      planned_weights：是否使用 planned 权重布局；
      warnings：构建警告。
    """

    title: str
    model_name: str
    manifest_sha256: str
    ir_sha256: str
    alignment: int
    arena_bytes: int
    peak_bytes: int
    constants_bytes: int
    weight_arena_bytes: int
    activation_blocks: tuple[ActivationBlock, ...]
    weight_blocks: tuple[WeightBlock, ...]
    activation_runs: tuple[LayerRun, ...]
    weight_runs: tuple[LayerRun, ...]
    planned_weights: bool
    warnings: tuple[str, ...]


def _shape_of(meta: TensorMeta) -> tuple[int | str, ...]:
    """返回张量形状。

    参数：
      meta：张量元数据。

    返回：
      维度值元组。
    """
    return tuple(dim.value for dim in meta.shape)


def _pitch_of(
    meta: TensorMeta,
    shape: tuple[int | str, ...],
) -> tuple[tuple[int, ...], bool]:
    """返回 pitch 以及是否为推导值。

    参数：
      meta：张量元数据；
      shape：逻辑形状。

    返回：
      ``(pitch, derived)``；pitch 为空时按 row-major contiguous 推导。
    """
    if meta.pitch:
        return tuple(int(value) for value in meta.pitch), False
    try:
        return tuple(int(value) for value in shape), True
    except (TypeError, ValueError):
        return (), True


def _layer_index(name: str) -> int | None:
    """解析名称中的层号。

    参数：
      name：tensor/权重名。

    返回：
      层号；非层名称返回 None。
    """
    match = _LAYER_REF.match(name)
    if match is None:
        return None
    return int(match.group("index"))


def _normalize(name: str) -> str:
    """把名称中的层号替换为 ``layer.*.``。

    参数：
      name：原始名称。

    返回：
      归一化名称。
    """
    return _LAYER_REF.sub("layer.*.", name)


def _align_up(value: int, alignment: int) -> int:
    """向上对齐。

    参数：
      value：待对齐值；
      alignment：对齐字节数。

    返回：
      对齐后的值。
    """
    return (value + alignment - 1) // alignment * alignment


def _activation_block(
    graph: ModelGraph,
    entry: dict[str, Any],
    producer: dict[str, str],
) -> ActivationBlock | None:
    """构造一个 Activation 区块。

    参数：
      graph：模型图；
      entry：memory_plan 条目；
      producer：tensor 到生产算子的映射。

    返回：
      ``ActivationBlock``；tensor 缺失时返回 None。
    """
    name = str(entry["name"])
    try:
        meta = graph.tensor(name)
    except KeyError:
        return None
    shape = _shape_of(meta)
    pitch, derived = _pitch_of(meta, shape)
    return ActivationBlock(
        tensor=name,
        owner_op=producer.get(name, "unknown"),
        precision=meta.precision,
        shape=shape,
        pitch=pitch,
        pitch_derived=derived,
        offset=int(entry["offset"]),
        size=int(entry["size"]),
        birth=int(entry["birth"]),
        last_use=int(entry["last_use"]),
        reusable=bool(entry["reusable"]),
        layer_index=_layer_index(name),
    )


def _owner_map(graph: ModelGraph) -> dict[str, tuple[str, ...]]:
    """建立常量到消费算子的映射。

    参数：
      graph：模型图。

    返回：
      常量名到消费算子元组的映射。
    """
    owners: dict[str, list[str]] = {}
    for op in graph.ops:
        for name in op.inputs:
            if name not in graph.constants:
                continue
            owners.setdefault(name, []).append(op.name)
    return {
        name: tuple(values) for name, values in owners.items()
    }


def _weight_blocks(
    package: LoadedPackage,
    warnings: list[str],
) -> tuple[tuple[WeightBlock, ...], int, bool]:
    """构造权重区块与权重区大小。

    参数：
      package：模型包；
      warnings：警告收集列表。

    返回：
      ``(blocks, weight_arena_bytes, planned)``。
    """
    graph = package.model_graph
    plan = package.memory_plan
    alignment = int(plan.get("alignment", 256))
    owners = _owner_map(graph)
    entries = {
        str(item.get("name")): item
        for item in package.weight_entries
    }
    actual_offsets = [
        entries.get(name, {}).get("offset")
        for name in graph.constants
    ]
    planned = any(offset is None for offset in actual_offsets)
    if planned:
        warnings.append(
            "weights have no physical offsets; using planned layout")
    offset = 0
    blocks = []
    for name in graph.constants:
        meta = graph.tensor(name)
        shape = _shape_of(meta)
        pitch, _ = _pitch_of(meta, shape)
        entry = entries.get(name, {})
        size = int(entry.get("byte_size", meta.byte_size))
        if planned:
            offset = _align_up(offset, alignment)
            block_offset = offset
            offset += size
        else:
            block_offset = int(entry["offset"])
            offset = max(offset, block_offset + size)
        blocks.append(WeightBlock(
            name=name,
            owner_ops=owners.get(name, ("unused",)),
            precision=meta.precision,
            shape=shape,
            pitch=pitch,
            offset=block_offset,
            size=size,
            source=(
                None if entry.get("source") is None
                else str(entry["source"])
            ),
            available=bool(entry.get("available", False)),
            layer_index=_layer_index(name),
        ))
    weight_arena_bytes = max(
        offset, int(plan.get("constants_bytes", 0)))
    return tuple(blocks), weight_arena_bytes, planned


def _signatures(
    activation_blocks: tuple[ActivationBlock, ...],
    weight_blocks: tuple[WeightBlock, ...],
) -> dict[int, tuple]:
    """按层构造内存布局签名。

    参数：
      activation_blocks：Activation 区块；
      weight_blocks：权重区块。

    返回：
      层号到归一化签名的映射。
    """
    activation_by_layer: dict[int, list[tuple]] = {}
    weight_by_layer: dict[int, list[tuple]] = {}
    for block in activation_blocks:
        if block.layer_index is None:
            continue
        activation_by_layer.setdefault(
            block.layer_index, []).append((
                _normalize(block.tensor),
                _normalize(block.owner_op),
                block.shape,
                block.pitch,
                block.size,
            ))
    for block in weight_blocks:
        if block.layer_index is None:
            continue
        weight_by_layer.setdefault(block.layer_index, []).append((
            _normalize(block.name),
            tuple(_normalize(owner) for owner in block.owner_ops),
            block.shape,
            block.pitch,
            block.size,
        ))
    indices = set(activation_by_layer) | set(weight_by_layer)
    return {
        index: (
            tuple(activation_by_layer.get(index, ())),
            tuple(weight_by_layer.get(index, ())),
        )
        for index in indices
    }


def build_memory_view(package: LoadedPackage) -> MemoryViewModel:
    """构建显存排布视图模型。

    参数：
      package：已校验的模型包。

    返回：
      ``MemoryViewModel``。

    异常：
      KeyError/ValueError：memory_plan 字段缺失或非法。
    """
    graph = package.model_graph
    plan = package.memory_plan
    warnings: list[str] = []
    producer = {
        output: op.name
        for op in graph.ops for output in op.outputs
    }
    activation_blocks = []
    for entry in plan.get("entries", []):
        block = _activation_block(graph, entry, producer)
        if block is None:
            warnings.append(
                f"activation tensor missing: {entry.get('name')}")
            continue
        meta = graph.tensor(block.tensor)
        if block.size != meta.byte_size:
            warnings.append(
                "activation size differs from tensor byte_size: "
                f"{block.tensor}")
        activation_blocks.append(block)
    weight_blocks, weight_arena_bytes, planned = _weight_blocks(
        package, warnings)
    signatures = _signatures(
        tuple(activation_blocks), weight_blocks)
    runs = analyze_layer_runs(signatures)
    for run in runs:
        if run.is_variant:
            warnings.append(
                f"layer {run.start}-{run.end} has a different "
                "memory layout")
    return MemoryViewModel(
        title=f"{package.model_name} Memory Layout",
        model_name=package.model_name,
        manifest_sha256=package.manifest_sha256,
        ir_sha256=package.ir_sha256,
        alignment=int(plan.get("alignment", 256)),
        arena_bytes=int(plan.get("arena_bytes", 0)),
        peak_bytes=int(plan.get("peak_bytes", 0)),
        constants_bytes=int(plan.get("constants_bytes", 0)),
        weight_arena_bytes=weight_arena_bytes,
        activation_blocks=tuple(activation_blocks),
        weight_blocks=weight_blocks,
        activation_runs=runs,
        weight_runs=runs,
        planned_weights=planned,
        warnings=tuple(warnings),
    )
