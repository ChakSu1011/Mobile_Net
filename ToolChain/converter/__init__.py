"""模块：Converter（模型转换为 Graph）。

职责：
  导出转换编排、Qwen3 Adapter、记录代理与结果类型。

边界：
  只负责离线模型转换；精度/形状/显存优化与序列化属于下游模块。
"""

from __future__ import annotations

from ToolChain.converter.converter import (
    conversion_report,
    convert,
    convert_qwen3,
    write_conversion_outputs,
)
from ToolChain.converter.qwen3_adapter import Qwen3Adapter
from ToolChain.converter.recording_graph import RecordingGraph
from ToolChain.converter.result import ConversionResult

__all__ = [
    "ConversionResult",
    "Qwen3Adapter",
    "RecordingGraph",
    "conversion_report",
    "convert",
    "convert_qwen3",
    "write_conversion_outputs",
]
