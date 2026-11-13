"""模块：Runtime Graph Python 公共包。

职责：
  统一导出 ``_graph`` 扩展中的 Graph、Tensor 与枚举；所有图对象仍由
  C++ 扩展创建，Python 侧不定义第二套 Graph/Tensor 数据结构。

边界：
  - 不实现纯 Python 图结构，不执行 Forward；
  - 不暴露 C++ Op、Storage、Shape 对象；
  - Tensor 句柄的生命周期由 Graph 持有，调用方必须保持 Graph 存活。

依赖：
  ``_graph`` 扩展、Python 3.10+ 标准库。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _load_native_module():
    """加载并返回 ``_graph`` 扩展模块。

    优先使用当前 ``sys.path``；找不到时依次尝试环境变量
    ``GRAPH_PYTHON_DIR`` 与仓库默认构建目录
    ``Build/graph/python``。

    返回：
      ``_graph`` 模块对象。
    异常：
      ModuleNotFoundError：所有候选目录均未找到扩展时抛出。
    """
    try:
        return importlib.import_module("_graph")
    except ModuleNotFoundError:
        pass

    candidates = []
    environment_dir = os.environ.get("GRAPH_PYTHON_DIR")
    if environment_dir:
        candidates.append(Path(environment_dir))
    repository_root = Path(__file__).resolve().parents[3]
    candidates.append(repository_root / "Build" / "graph" / "python")
    for candidate in candidates:
        extension_names = (
            "_graph.pyd",
            "_graph.so",
            "_graph.dylib",
        )
        extension_exists = any(
            (candidate / name).exists() for name in extension_names
        )
        if not extension_exists:
            continue
        sys.path.insert(0, str(candidate))
        return importlib.import_module("_graph")

    raise ModuleNotFoundError(
        "cannot find the _graph extension; set GRAPH_PYTHON_DIR"
    )


_native = _load_native_module()

# 图与张量类型。
Graph = _native.Graph
Tensor = _native.Tensor

# 枚举。
DeviceType = _native.DeviceType
Layout = _native.Layout
OpType = _native.OpType
Precision = _native.Precision

__all__ = [
    "DeviceType",
    "Graph",
    "Layout",
    "OpType",
    "Precision",
    "Tensor",
]
