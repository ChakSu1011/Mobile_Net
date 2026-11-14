"""模块：Optimizer 公共接口。

职责：
  导出优化配置、优化器、优化结果与输出写出函数。

边界：
  不包含 Pass 实现；实现分文件放置于本包内。
"""

from __future__ import annotations

from ToolChain.optimizer.config import OptimizeConfig
from ToolChain.optimizer.memory import MemoryPass
from ToolChain.optimizer.optimizer import (
    OptimizeResult,
    Optimizer,
    write_optimization_outputs,
)
from ToolChain.optimizer.precision import PrecisionPass
from ToolChain.optimizer.report import build_report
from ToolChain.optimizer.shape import ShapePass

__all__ = [
    "MemoryPass",
    "OptimizeConfig",
    "OptimizeResult",
    "Optimizer",
    "PrecisionPass",
    "ShapePass",
    "build_report",
    "write_optimization_outputs",
]
