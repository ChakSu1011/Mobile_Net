"""模块：Memory Manage Pass。

职责：
  基于拓扑序计算激活张量的生命周期，按对齐约束做 first-fit 复用排版，
  输出 ``MemoryPlan``，并校验生命周期与字节区间不变量。

边界：
  - 只生成逻辑偏移，不分配真实显存；
  - 常量与图输入不进入 Activation arena；
  - 不修改图结构与张量元数据（输出复用标记除外）。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ToolChain.common.errors import BuildError
from ToolChain.common.model_ir import (
    MemoryEntry,
    MemoryPlan,
    ModelGraph,
)
from ToolChain.optimizer.config import OptimizeConfig


def _align_up(value: int, alignment: int) -> int:
    """向上对齐到 alignment 的整数倍。

    参数：
      value：非负整数；
      alignment：2 的幂。

    返回：
      对齐后的值。
    """
    return (value + alignment - 1) // alignment * alignment


class MemoryPass:
    """激活显存生命周期分析与复用排版。"""

    def __init__(self, config: OptimizeConfig) -> None:
        """保存配置。

        参数：
          config：优化配置；提供 alignment 与 enable_reuse。
        """
        self._config = config

    def run(self, model_graph: ModelGraph) -> MemoryPlan:
        """执行显存规划。

        参数：
          model_graph：已完成精度与形状推导的模型图；调用方应保证是深拷贝。

        返回：
          显存计划；同时把计划摘要写入 ``metadata["memory"]``。

        异常：
          BuildError：生命周期缺失、偏移未对齐或复用区间重叠。
        """
        for tensor in model_graph.activation_tensors:
            if tensor.static_shape is None or not tensor.static_shape:
                raise BuildError(
                    "optimizer",
                    "activation shape is not resolved; "
                    "run shape inference first",
                    target=tensor.name,
                )
        topological = model_graph.topological_order()
        op_index = {
            op.name: index for index, op in enumerate(topological)}
        output_index = len(topological)

        births, last_uses = self._lifetimes(
            model_graph, topological, op_index, output_index)
        entries = self._allocate(model_graph, births, last_uses)
        constants_bytes = model_graph.constant_bytes
        peak_bytes = self._peak_bytes(
            model_graph, births, last_uses, output_index)
        arena_bytes = max(
            (entry.end for entry in entries), default=0)

        plan = MemoryPlan(
            alignment=self._config.alignment,
            arena_bytes=arena_bytes,
            peak_bytes=peak_bytes,
            constants_bytes=constants_bytes,
            entries=tuple(entries),
        )
        self._verify(model_graph, plan)
        model_graph.metadata["memory"] = {
            "alignment": plan.alignment,
            "arena_bytes": plan.arena_bytes,
            "peak_bytes": plan.peak_bytes,
            "constants_bytes": plan.constants_bytes,
            "entries": len(plan.entries),
        }
        return plan

    def _lifetimes(
        self,
        model_graph: ModelGraph,
        topological: list[Any],
        op_index: dict[str, int],
        output_index: int,
    ) -> tuple[dict[str, int], dict[str, int]]:
        """计算每个 activation 的出生与末次使用下标。

        参数：
          model_graph：模型图；
          topological：拓扑序 op 列表；
          op_index：op 名到拓扑下标的映射；
          output_index：输出绑定虚拟消费者下标。

        返回：
          ``(births, last_uses)``；均为 activation 张量名到下标。

        异常：
          BuildError：activation 没有生产 op。
        """
        producers = model_graph.producer_map()
        births: dict[str, int] = {}
        last_uses: dict[str, int] = {}
        for op in topological:
            for output in op.outputs:
                births[output] = op_index[op.name]
                last_uses[output] = op_index[op.name]

        consumers: dict[str, list[int]] = defaultdict(list)
        for op in topological:
            for tensor_name in op.inputs:
                tensor = model_graph.tensor(tensor_name)
                if tensor.kind == "activation":
                    consumers[tensor_name].append(op_index[op.name])
        for tensor_name, indices in consumers.items():
            if tensor_name not in last_uses:
                raise BuildError(
                    "optimizer",
                    "activation tensor has no producer",
                    target=tensor_name,
                )
            last_uses[tensor_name] = max(
                last_uses[tensor_name], max(indices))

        for binding in model_graph.outputs:
            tensor_name = binding.tensor
            if tensor_name not in producers:
                raise BuildError(
                    "optimizer",
                    "output tensor has no producer",
                    target=tensor_name,
                )
            last_uses[tensor_name] = max(
                last_uses[tensor_name], output_index)
            model_graph.tensor(tensor_name).reusable = False
        return births, last_uses

    def _allocate(
        self,
        model_graph: ModelGraph,
        births: dict[str, int],
        last_uses: dict[str, int],
    ) -> list[MemoryEntry]:
        """按 first-fit 规则复用激活显存。

        参数：
          model_graph：模型图；
          births：出生下标；
          last_uses：末次使用下标。

        返回：
          显存条目列表，按分配顺序排列。

        异常：
          BuildError：activation 缺少生命周期信息。
        """
        alignment = self._config.alignment
        activations = [
            tensor
            for tensor in model_graph.tensors.values()
            if tensor.kind == "activation"
        ]
        activations.sort(key=lambda tensor: (
            births[tensor.name],
            -tensor.byte_size,
            tensor.name,
        ))

        blocks: list[tuple[int, int, int, str]] = []
        entries: list[MemoryEntry] = []
        arena_end = 0
        for tensor in activations:
            if tensor.name not in births or tensor.name not in last_uses:
                raise BuildError(
                    "optimizer",
                    "activation tensor lacks lifetime information",
                    target=tensor.name,
                )
            birth = births[tensor.name]
            last_use = last_uses[tensor.name]
            size = tensor.byte_size
            reusable = (
                tensor.reusable and self._config.enable_reuse)
            if size == 0:
                offset = 0
            elif reusable:
                offset = self._first_fit(
                    blocks, size, birth, alignment)
            else:
                offset = _align_up(arena_end, alignment)
            entry = MemoryEntry(
                name=tensor.name,
                offset=offset,
                size=size,
                birth=birth,
                last_use=last_use,
                reusable=tensor.reusable,
            )
            entries.append(entry)
            if size > 0:
                blocks.append(
                    (offset, offset + size, last_use, tensor.name))
                arena_end = max(arena_end, offset + size)
        return entries

    @staticmethod
    def _first_fit(
        blocks: list[tuple[int, int, int, str]],
        size: int,
        birth: int,
        alignment: int,
    ) -> int:
        """在活跃区间不重叠的前提下寻找最低可用偏移。

        参数：
          blocks：已分配区间 ``(offset, end, last_use, name)``；
          size：本次分配字节数；
          birth：本次分配的出生下标；
          alignment：对齐字节数。

        返回：
          对齐后的偏移。
        """
        offset = 0
        while True:
            offset = _align_up(offset, alignment)
            conflict_end = -1
            for block_offset, block_end, block_last_use, _ in blocks:
                if block_last_use < birth:
                    continue
                if offset < block_end and block_offset < offset + size:
                    conflict_end = max(conflict_end, block_end)
            if conflict_end < 0:
                return offset
            offset = conflict_end

    @staticmethod
    def _peak_bytes(
        model_graph: ModelGraph,
        births: dict[str, int],
        last_uses: dict[str, int],
        output_index: int,
    ) -> int:
        """计算任一时刻活跃 activation 字节数之和的最大值。

        参数：
          model_graph：模型图；
          births：出生下标；
          last_uses：末次使用下标；
          output_index：输出绑定下标。

        返回：
          理论峰值字节数。

        异常：
          BuildError：生命周期下标非法。
        """
        delta = [0] * (output_index + 2)
        for tensor in model_graph.tensors.values():
            if tensor.kind != "activation":
                continue
            birth = births[tensor.name]
            last_use = last_uses[tensor.name]
            if birth < 0 or last_use < birth or last_use > output_index:
                raise BuildError(
                    "optimizer",
                    "invalid activation lifetime",
                    target=tensor.name,
                )
            delta[birth] += tensor.byte_size
            delta[last_use + 1] -= tensor.byte_size
        peak = 0
        running = 0
        for value in delta:
            running += value
            peak = max(peak, running)
        return peak

    def _verify(
        self, model_graph: ModelGraph, plan: MemoryPlan) -> None:
        """校验显存计划不变量。

        参数：
          model_graph：模型图；
          plan：待校验的显存计划。

        异常：
          BuildError：偏移未对齐、区间重叠或统计不一致。
        """
        names: set[str] = set()
        for entry in plan.entries:
            if entry.name in names:
                raise BuildError(
                    "optimizer",
                    "duplicate memory entry",
                    target=entry.name,
                )
            names.add(entry.name)
            if entry.offset % plan.alignment != 0:
                raise BuildError(
                    "optimizer",
                    f"memory offset {entry.offset} is not aligned to "
                    f"{plan.alignment}",
                    target=entry.name,
                )
        for index, first in enumerate(plan.entries):
            for second in plan.entries[index + 1:]:
                if not first.overlaps(second):
                    continue
                lifetime_overlap = (
                    first.birth <= second.last_use
                    and second.birth <= first.last_use
                )
                if lifetime_overlap:
                    raise BuildError(
                        "optimizer",
                        "live memory ranges overlap",
                        target=f"{first.name}/{second.name}",
                    )
        activation_bytes = sum(
            tensor.byte_size
            for tensor in model_graph.tensors.values()
            if tensor.kind == "activation"
        )
        if sum(entry.size for entry in plan.entries) != activation_bytes:
            raise BuildError(
                "optimizer",
                "memory entries do not cover all activation bytes",
            )
        if plan.arena_bytes < plan.peak_bytes:
            raise BuildError(
                "optimizer",
                f"arena {plan.arena_bytes} is smaller than peak "
                f"{plan.peak_bytes}",
            )
        if plan.constants_bytes != model_graph.constant_bytes:
            raise BuildError(
                "optimizer",
                "constant bytes do not match tensor table",
            )
