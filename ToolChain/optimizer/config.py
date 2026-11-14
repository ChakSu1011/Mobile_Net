"""模块：Optimizer 配置。

职责：
  定义精度策略、形状规划边界与显存对齐参数，并在构造时校验。

边界：
  不读取模型、不执行优化、不保存状态。
"""

from __future__ import annotations

from dataclasses import dataclass

from ToolChain.common.errors import BuildError

_TARGET_PRECISIONS = ("float16", "bfloat16", "float32", "keep")
_ACCUMULATE_PRECISIONS = ("float32",)


@dataclass(frozen=True)
class OptimizeConfig:
    """优化配置。

    属性：
      target_precision：目标精度；``keep`` 表示关闭精度转换；
      accumulate_precision：数值敏感算子的累加精度；
      batch_size：规划 batch 上界，首期固定 1；
      max_num_tokens：单 Step Token 规划上界，绑定符号维 ``S``；
      context_length：单请求上下文上限，必须不小于 max_num_tokens；
      alignment：显存偏移对齐字节数；
      enable_reuse：是否启用 Activation 复用。
    """

    target_precision: str = "float16"
    accumulate_precision: str = "float32"
    batch_size: int = 1
    max_num_tokens: int = 256
    context_length: int = 40960
    alignment: int = 256
    enable_reuse: bool = True

    def __post_init__(self) -> None:
        """校验配置取值范围。

        异常：
          BuildError：字段非法。
        """
        if self.target_precision not in _TARGET_PRECISIONS:
            raise BuildError(
                "optimizer",
                f"unsupported target_precision: "
                f"{self.target_precision}",
            )
        if self.accumulate_precision not in _ACCUMULATE_PRECISIONS:
            raise BuildError(
                "optimizer",
                "unsupported accumulate_precision: "
                f"{self.accumulate_precision}",
            )
        if self.batch_size <= 0:
            raise BuildError(
                "optimizer", "batch_size must be positive")
        if self.max_num_tokens <= 0:
            raise BuildError(
                "optimizer", "max_num_tokens must be positive")
        if self.context_length <= 0:
            raise BuildError(
                "optimizer", "context_length must be positive")
        if self.max_num_tokens > self.context_length:
            raise BuildError(
                "optimizer",
                "max_num_tokens must not exceed context_length",
            )
        if self.alignment <= 0:
            raise BuildError(
                "optimizer", "alignment must be positive")
        if self.alignment & (self.alignment - 1):
            raise BuildError(
                "optimizer", "alignment must be a power of two")

    def to_dict(self) -> dict[str, object]:
        """返回 JSON 原生表示。"""
        return {
            "target_precision": self.target_precision,
            "accumulate_precision": self.accumulate_precision,
            "batch_size": self.batch_size,
            "max_num_tokens": self.max_num_tokens,
            "context_length": self.context_length,
            "alignment": self.alignment,
            "enable_reuse": self.enable_reuse,
        }
