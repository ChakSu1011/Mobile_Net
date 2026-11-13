"""模块：Optimizer 报告与不变量检查。

职责：
  汇总精度、形状与显存阶段的结构化结果，逐项验证优化不变量，
  生成可 JSON 序列化、无时间戳的确定性报告。

边界：
  - 不执行 Pass，不修改模型图；
  - 只报告事实与判定，失败时抛出 ``BuildError``。
"""

from __future__ import annotations

import math
from typing import Any

from Build.common.dtypes import INTEGER_PRECISIONS, element_bytes
from Build.common.errors import BuildError
from Build.common.model_ir import MemoryPlan, ModelGraph
from Build.optimizer.config import OptimizeConfig


def build_report(
    model_graph: ModelGraph,
    config: OptimizeConfig,
    precision_report: dict[str, Any],
    shape_report: dict[str, Any],
    memory_plan: MemoryPlan,
) -> dict[str, Any]:
    """生成优化报告并校验不变量。

    参数：
      model_graph：优化后的模型图；
      config：优化配置；
      precision_report：Precision Pass 返回值；
      shape_report：Shape Pass 返回值；
      memory_plan：Memory Pass 返回值。

    返回：
      报告字典；字段可 JSON 序列化。

    异常：
      BuildError：任一不变量失败。
    """
    invariants = _check_invariants(model_graph, config, memory_plan)
    failed = [item for item in invariants if not item["passed"]]
    if failed:
        detail = "; ".join(
            f"{item['name']}: {item['detail']}" for item in failed)
        raise BuildError(
            "optimizer", f"invariant check failed: {detail}")
    return {
        "model": {
            "name": model_graph.name,
            "adapter": model_graph.adapter,
        },
        "bounds": {
            "batch": config.batch_size,
            "sequence": config.max_num_tokens,
            "context_length": config.context_length,
        },
        "precision": precision_report,
        "shape": shape_report,
        "memory": {
            "alignment": memory_plan.alignment,
            "arena_bytes": memory_plan.arena_bytes,
            "peak_bytes": memory_plan.peak_bytes,
            "constants_bytes": memory_plan.constants_bytes,
            "entries": len(memory_plan.entries),
        },
        "invariants": invariants,
    }


def _check_invariants(
    model_graph: ModelGraph,
    config: OptimizeConfig,
    memory_plan: MemoryPlan,
) -> list[dict[str, Any]]:
    """逐项检查优化不变量。

    参数：
      model_graph：优化后的模型图；
      config：优化配置；
      memory_plan：显存计划。

    返回：
      ``{"name", "passed", "detail"}`` 列表，顺序固定。
    """
    results: list[dict[str, Any]] = []
    results.append(_precision_complete(model_graph))
    results.append(_precision_matches_target(model_graph, config))
    results.append(_shape_complete(model_graph))
    results.append(_memory_ranges_disjoint(memory_plan))
    results.append(_memory_aligned(memory_plan))
    results.append(_memory_covers_peak(memory_plan))
    results.append(_activation_bytes_covered(model_graph, memory_plan))
    results.append(_constants_match(model_graph, memory_plan))
    results.append(_io_not_reusable(model_graph))
    return results


def _result(name: str, failures: list[str]) -> dict[str, Any]:
    """构造单条不变量结果。

    参数：
      name：不变量名；
      failures：失败描述列表；为空表示通过。

    返回：
      不变量结果字典。
    """
    return {
        "name": name,
        "passed": not failures,
        "detail": "ok" if not failures else "; ".join(failures[:5]),
    }


def _precision_complete(model_graph: ModelGraph) -> dict[str, Any]:
    """检查所有张量精度已确定。"""
    failures = [
        f"{name}={tensor.precision}"
        for name, tensor in model_graph.tensors.items()
        if tensor.precision == "unknown"
    ]
    return _result("precision_complete", failures)


def _precision_matches_target(
    model_graph: ModelGraph,
    config: OptimizeConfig,
) -> dict[str, Any]:
    """检查浮点张量精度与配置目标一致。"""
    if config.target_precision == "keep":
        return _result("precision_matches_target", [])
    failures = []
    for name, tensor in model_graph.tensors.items():
        if tensor.precision in INTEGER_PRECISIONS:
            continue
        if tensor.precision != config.target_precision:
            failures.append(
                f"{name}={tensor.precision} "
                f"!= {config.target_precision}")
    return _result("precision_matches_target", failures)


def _shape_complete(model_graph: ModelGraph) -> dict[str, Any]:
    """检查所有 activation 具有静态形状与正确字节数。"""
    failures = []
    for name, tensor in model_graph.tensors.items():
        if tensor.kind != "activation":
            continue
        static_shape = tensor.static_shape
        if static_shape is None or not static_shape:
            failures.append(f"{name}: shape={tensor.shape}")
            continue
        expected = (
            math.prod(static_shape) * element_bytes(tensor.precision))
        if tensor.byte_size != expected:
            failures.append(
                f"{name}: byte_size={tensor.byte_size} "
                f"!= {expected}")
    return _result("shape_complete", failures)


def _memory_ranges_disjoint(memory_plan: MemoryPlan) -> dict[str, Any]:
    """检查生命周期重叠的张量不共享字节。"""
    failures = []
    entries = memory_plan.entries
    for index, first in enumerate(entries):
        for second in entries[index + 1:]:
            lifetime_overlap = (
                first.birth <= second.last_use
                and second.birth <= first.last_use
            )
            byte_overlap = (
                first.offset < second.end
                and second.offset < first.end
            )
            if lifetime_overlap and byte_overlap:
                failures.append(f"{first.name}/{second.name}")
    return _result("memory_ranges_disjoint", failures)


def _memory_aligned(memory_plan: MemoryPlan) -> dict[str, Any]:
    """检查所有显存偏移按 alignment 对齐。"""
    failures = [
        f"{entry.name}@{entry.offset}"
        for entry in memory_plan.entries
        if entry.offset % memory_plan.alignment != 0
    ]
    return _result("memory_aligned", failures)


def _memory_covers_peak(memory_plan: MemoryPlan) -> dict[str, Any]:
    """检查 arena 高水位不小于理论峰值。"""
    if memory_plan.arena_bytes < memory_plan.peak_bytes:
        return _result(
            "memory_covers_peak",
            [f"arena={memory_plan.arena_bytes} "
             f"peak={memory_plan.peak_bytes}"],
        )
    return _result("memory_covers_peak", [])


def _activation_bytes_covered(
    model_graph: ModelGraph,
    memory_plan: MemoryPlan,
) -> dict[str, Any]:
    """检查显存条目完整覆盖 activation 字节。"""
    activation_bytes = sum(
        tensor.byte_size
        for tensor in model_graph.tensors.values()
        if tensor.kind == "activation"
    )
    entries_bytes = sum(entry.size for entry in memory_plan.entries)
    if activation_bytes != entries_bytes:
        return _result(
            "activation_bytes_covered",
            [f"activations={activation_bytes} "
             f"entries={entries_bytes}"],
        )
    return _result("activation_bytes_covered", [])


def _constants_match(
    model_graph: ModelGraph,
    memory_plan: MemoryPlan,
) -> dict[str, Any]:
    """检查常量字节统计与张量表一致。"""
    if model_graph.constant_bytes != memory_plan.constants_bytes:
        return _result(
            "constants_match",
            [f"tensors={model_graph.constant_bytes} "
             f"plan={memory_plan.constants_bytes}"],
        )
    return _result("constants_match", [])


def _io_not_reusable(model_graph: ModelGraph) -> dict[str, Any]:
    """检查图输入与图输出张量不参与复用。"""
    failures = []
    for name in model_graph.inputs:
        if model_graph.tensor(name).reusable:
            failures.append(f"input {name}")
    for binding in model_graph.outputs:
        if model_graph.tensor(binding.tensor).reusable:
            failures.append(f"output {binding.tensor}")
    return _result("io_not_reusable", failures)
