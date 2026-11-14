"""模块：Serializer 公共接口。

职责：
  导出模型包序列化、加载、图重建与格式数据结构。

边界：
  不包含 CLI 与文件格式实现细节；这些位于同包其他文件。
"""

from __future__ import annotations

from ToolChain.serializer.format import (
    FORMAT_NAME,
    FORMAT_VERSION,
    ModelPackage,
    PackageHeader,
    WeightEntry,
)
from ToolChain.serializer.loader import (
    load_package,
    package_summary,
    rebuild_graph,
    rebuild_graph_from_model,
)
from ToolChain.serializer.serializer import (
    SerializationResult,
    serialize_package,
)

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "ModelPackage",
    "PackageHeader",
    "SerializationResult",
    "WeightEntry",
    "load_package",
    "package_summary",
    "rebuild_graph",
    "rebuild_graph_from_model",
    "serialize_package",
]
