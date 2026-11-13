"""模块：Precision Infer Pass。

职责：
  把浮点常量与激活统一到目标精度，为数值敏感算子写入 FP32 累加属性，
  保持整数输入不变，并输出精度统计。

边界：
  - 不做量化、不重排权重、不改变张量字节数；
  - 不推导 shape，不分配显存；
  - 不修改 Graph_Fixed 图对象，只修改 ``ModelGraph``。
"""

from __future__ import annotations

from typing import Any

from Build.common.dtypes import (
    INTEGER_PRECISIONS,
    element_bytes,
)
from Build.common.errors import BuildError
from Build.common.model_ir import ModelGraph
from Build.optimizer.config import OptimizeConfig

_SENSITIVE_OPS = frozenset({
    "rms_norm",
    "linear",
    "linear_swi_glu",
    "mega_attention",
    "rope",
})


class PrecisionPass:
    """精度推导与转换。"""

    def __init__(self, config: OptimizeConfig) -> None:
        """保存配置。

        参数：
          config：优化配置；构造时已校验。
        """
        self._config = config

    def run(self, model_graph: ModelGraph) -> dict[str, Any]:
        """执行精度转换。

        参数：
          model_graph：待修改的模型图；调用方应保证是深拷贝。

        返回：
          精度报告字典。

        异常：
          BuildError：常量精度未知、字节数会变化或属性精度不一致。
        """
        target = self._config.target_precision
        accumulate = self._config.accumulate_precision
        converted = 0
        kept_integer = 0

        for tensor in model_graph.tensors.values():
            if tensor.precision in INTEGER_PRECISIONS:
                kept_integer += 1
                continue
            if target == "keep":
                continue
            if tensor.kind == "constant":
                if tensor.precision == "unknown":
                    raise BuildError(
                        "optimizer",
                        "constant precision is unknown",
                        target=tensor.name,
                    )
                source_bytes = element_bytes(tensor.precision)
                target_bytes = element_bytes(target)
                if source_bytes != target_bytes:
                    raise BuildError(
                        "optimizer",
                        "precision conversion would change byte size: "
                        f"{tensor.precision}->{target}",
                        target=tensor.name,
                    )
            if tensor.precision != target:
                converted += 1
            tensor.precision = target

        if target != "keep":
            for op in model_graph.ops:
                if "precision" not in op.attrs:
                    continue
                if op.attrs["precision"] != target:
                    op.attrs["precision"] = target
        for op in model_graph.ops:
            if op.op_type in _SENSITIVE_OPS:
                op.attrs["accumulate_precision"] = accumulate

        self._verify(model_graph, target)

        histogram: dict[str, int] = {}
        for tensor in model_graph.tensors.values():
            histogram[tensor.precision] = (
                histogram.get(tensor.precision, 0) + 1)
        return {
            "target": target,
            "accumulate": accumulate,
            "converted_tensors": converted,
            "kept_integer_tensors": kept_integer,
            "precision_histogram": dict(sorted(histogram.items())),
        }

    def _verify(self, model_graph: ModelGraph, target: str) -> None:
        """校验精度转换不变量。

        参数：
          model_graph：已转换的模型图；
          target：目标精度或 ``keep``。

        异常：
          BuildError：精度仍为 unknown，或属性与张量精度不一致。
        """
        for tensor in model_graph.tensors.values():
            if tensor.precision == "unknown":
                raise BuildError(
                    "optimizer",
                    "tensor precision is unknown after precision pass",
                    target=tensor.name,
                )
        if target == "keep":
            return
        for op in model_graph.ops:
            attribute = op.attrs.get("precision")
            if attribute is None:
                continue
            if attribute != target:
                raise BuildError(
                    "optimizer",
                    "op precision attribute does not match target",
                    target=op.name,
                )
            for output in op.outputs:
                tensor = model_graph.tensor(output)
                if (tensor.precision not in INTEGER_PRECISIONS
                        and tensor.precision != target):
                    raise BuildError(
                        "optimizer",
                        "output tensor precision does not match target",
                        target=output,
                    )
