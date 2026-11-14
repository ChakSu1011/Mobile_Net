"""模块：转换结果数据结构。

职责：
  定义 Converter 的统一返回值 ``ConversionResult``，同时携带可执行的
  Graph与旁路记录的 ``ModelGraph``。

边界：
  只承载结果，不执行构图、校验或持久化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ToolChain.common.model_ir import ModelGraph


@dataclass(frozen=True)
class ConversionResult:
    """一次模型转换的产物。

    属性：
      graph：已调用 ``sort_op()`` 的 Graph；
      model_graph：旁路记录的模型元数据；
      adapter：Adapter 名称。
    """

    graph: Any
    model_graph: ModelGraph
    adapter: str

    @property
    def ir_text(self) -> str:
        """返回图的 IR 文本。"""
        return self.graph.dump_ir()
