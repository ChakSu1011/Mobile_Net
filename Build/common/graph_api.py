"""模块：Graph Python 模块加载与枚举名称转换。

职责：
  定位并加载 ``mobilenet_runtime_graph``，提供 Precision/Layout/OpType
  在 Python 枚举与稳定字符串之间的双向映射。

边界：
  不实现 Graph/Tensor，不修改 Graph；本模块只做导入与名称转换。
"""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

_CACHED_API: Any | None = None

_PRECISION_NAMES = {
    "kUnknown": "unknown",
    "kFloat64": "float64",
    "kFloat32": "float32",
    "kFloat16": "float16",
    "kBFloat16": "bfloat16",
    "kFloat8E4M3": "float8_e4m3",
    "kFloat8E5M2": "float8_e5m2",
    "kInt64": "int64",
    "kInt32": "int32",
    "kInt16": "int16",
    "kInt8": "int8",
    "kUInt8": "uint8",
    "kBool": "bool",
}
_LAYOUT_NAMES = {
    "kRowMajor": "row_major",
    "kColumnMajor": "column_major",
}
_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def load_graph_api() -> Any:
    """加载 Graph 的 Python 公共模块。

    优先使用当前 ``sys.path``；失败时把仓库的 ``Graph/python`` 与
    原生扩展目录 ``Build/graph/python`` 加入 ``sys.path`` 后重试。
    结果会被缓存，重复调用不会重复导入。

    返回：
      ``mobilenet_runtime_graph`` 模块对象。

    异常：
      ModuleNotFoundError：公共模块或 ``_graph`` 扩展均不可用时抛出。
    """
    global _CACHED_API
    if _CACHED_API is not None:
        return _CACHED_API
    try:
        _CACHED_API = importlib.import_module(
            "mobilenet_runtime_graph")
        return _CACHED_API
    except ModuleNotFoundError:
        pass

    repository_root = Path(__file__).resolve().parents[2]
    for candidate in (
        repository_root / "Graph" / "python",
        repository_root / "Build" / "graph" / "python",
    ):
        if not candidate.is_dir():
            continue
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
    _CACHED_API = importlib.import_module("mobilenet_runtime_graph")
    return _CACHED_API


def precision_name(precision: Any) -> str:
    """把 Precision 枚举转换为稳定字符串。

    参数：
      precision：``GraphApi.Precision`` 成员。

    返回：
      小写 snake_case 名称。

    异常：
      BuildError：枚举值不在映射表中时由调用方处理；
      KeyError：未知枚举名称。
    """
    return _PRECISION_NAMES[precision.name]


def precision_from_name(api: Any, name: str) -> Any:
    """把稳定字符串转换为 Precision 枚举。

    参数：
      api：``mobilenet_runtime_graph`` 模块；
      name：``precision_name`` 支持的小写名称。

    返回：
      ``api.Precision`` 成员。

    异常：
      ValueError：名称不受支持。
    """
    for member_name, text in _PRECISION_NAMES.items():
        if text == name:
            return getattr(api.Precision, member_name)
    raise ValueError(f"unsupported precision: {name}")


def layout_name(layout: Any) -> str:
    """把 Layout 枚举转换为稳定字符串。"""
    return _LAYOUT_NAMES[layout.name]


def layout_from_name(api: Any, name: str) -> Any:
    """把稳定字符串转换为 Layout 枚举。

    异常：
      ValueError：名称不受支持。
    """
    for member_name, text in _LAYOUT_NAMES.items():
        if text == name:
            return getattr(api.Layout, member_name)
    raise ValueError(f"unsupported layout: {name}")


def op_type_name(op_type: Any) -> str:
    """把 OpType 枚举转换为稳定字符串。

    参数：
      op_type：``GraphApi.OpType`` 成员。

    返回：
      去掉 ``k`` 前缀并转为 snake_case 的名称，例如
      ``kLinearSwiGlu`` → ``linear_swi_glu``。
    """
    name = op_type.name
    if name.startswith("k"):
        name = name[1:]
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def op_type_from_name(api: Any, name: str) -> Any:
    """把稳定字符串转换为 OpType 枚举。

    参数：
      api：``mobilenet_runtime_graph`` 模块；
      name：``op_type_name`` 输出的名称。

    返回：
      ``api.OpType`` 成员。

    异常：
      ValueError：名称不受支持。
    """
    members = getattr(api.OpType, "__members__")
    for member in members.values():
        if op_type_name(member) == name:
            return member
    raise ValueError(f"unsupported op type: {name}")


def op_type_names(api: Any) -> dict[str, Any]:
    """返回全部 OpType 的稳定字符串到枚举映射。"""
    members = getattr(api.OpType, "__members__")
    return {
        op_type_name(member): member for member in members.values()
    }
