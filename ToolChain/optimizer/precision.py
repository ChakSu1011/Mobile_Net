"""模块：Precision Infer Pass。

职责：
  把浮点常量与激活统一到目标精度，为数值敏感算子写入 FP32 累加属性，
  保持整数输入不变，并输出精度统计；提供权重路径时执行 BF16→FP16
  权重字节转换。

边界：
  - 不做 INT8/INT4 量化；量化是后续扩展点；
  - 不推导 shape，不分配显存；
  - 不修改 Graph对象，只修改 ``ModelGraph`` 与转换权重文件。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ToolChain.common.dtypes import (
    INTEGER_PRECISIONS,
    element_bytes,
)
from ToolChain.common.errors import BuildError
from ToolChain.common.model_ir import ModelGraph
from ToolChain.optimizer.config import OptimizeConfig
from ToolChain.optimizer.weights import (
    WeightConversionResult,
    convert_weights_to_fp16,
)

_SENSITIVE_OPS = frozenset({
    "rms_norm",
    "linear",
    "linear_swi_glu",
    "linear_per_head_rms_norm_rope",
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
        self._weight_conversion: WeightConversionResult | None = None

    @property
    def weight_conversion(self) -> WeightConversionResult | None:
        """返回本次权重转换结果；未转换时为 None。"""
        return self._weight_conversion

    def run(
        self,
        model_graph: ModelGraph,
        *,
        weights_path: Path | None = None,
        weights_output: Path | None = None,
    ) -> dict[str, Any]:
        """执行精度转换与可选权重字节转换。

        参数：
          model_graph：待修改的模型图；调用方应保证是深拷贝；
          weights_path：可选源 safetensors 文件或目录；
          weights_output：转换后的 safetensors 输出路径。

        返回：
          精度报告字典，包含 ``weight_conversion`` 字段。

        异常：
          BuildError：常量精度未知、字节数会变化、属性精度不一致或
            权重溢出/写出失败。
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
                if self._all_outputs_integer(model_graph, op):
                    continue
                if op.attrs["precision"] != target:
                    op.attrs["precision"] = target
        for op in model_graph.ops:
            if op.op_type in _SENSITIVE_OPS:
                op.attrs["accumulate_precision"] = accumulate

        self._verify(model_graph, target)
        self._weight_conversion = None
        if weights_path is not None and weights_output is not None:
            self._weight_conversion = convert_weights_to_fp16(
                model_graph,
                weights_path,
                weights_output,
                target,
            )

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
            "weight_conversion": (
                None if self._weight_conversion is None
                else self._weight_conversion.to_dict()
            ),
        }

    @staticmethod
    def _all_outputs_integer(model_graph: ModelGraph, op: Any) -> bool:
        """判断 op 的所有输出是否都是整数张量。

        参数：
          model_graph：模型图；
          op：算子元数据。

        返回：
          True 表示所有输出均为整数精度；无输出时返回 False。

        异常：
          BuildError：输出张量未登记。
        """
        if not op.outputs:
            return False
        for name in op.outputs:
            tensor = model_graph.tensor(name)
            if tensor.precision not in INTEGER_PRECISIONS:
                return False
        return True

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
            if self._all_outputs_integer(model_graph, op):
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
