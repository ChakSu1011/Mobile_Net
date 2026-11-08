"""模块：Graph 模块 Python 端公共包。

职责：
  统一导出 Graph IR 的 Python 类与枚举，并提供 Python 侧构图辅助与模型
  构建器。所有图对象仍由 C++ 扩展 ``_graph`` 创建，属性在 Python 侧呈现为
  ``torch.Tensor`` / ``torch.dtype``，不存在第二套图定义。

边界：
  - 不实现纯 Python 的 Graph/Node/Edge 数据结构；
  - 不执行 Forward；RuntimeGraph 不暴露给 Python；
  - 不读取或修改真实权重。

依赖：
  ``_graph`` 扩展、``torch``、``graph_tools``、``builders.qwen3``。
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def _load_native_module():
    """加载并返回 ``_graph`` 扩展模块。

    优先使用当前 ``sys.path``；找不到时依次尝试环境变量
    ``MOBILENET_GRAPH_PYTHON_DIR`` 与仓库默认构建目录
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
    environment_dir = os.environ.get("MOBILENET_GRAPH_PYTHON_DIR")
    if environment_dir:
        candidates.append(Path(environment_dir))
    repository_root = Path(__file__).resolve().parents[3]
    candidates.append(repository_root / "Build" / "graph" / "python")
    for candidate in candidates:
        extension_exists = any(
            (candidate / name).exists()
            for name in ("_graph.pyd", "_graph.so", "_graph.dylib")
        )
        if not extension_exists:
            continue
        sys.path.insert(0, str(candidate))
        return importlib.import_module("_graph")

    raise ModuleNotFoundError(
        "cannot find the _graph extension; set MOBILENET_GRAPH_PYTHON_DIR"
    )


_native = _load_native_module()

# 错误与枚举。
ErrorCode = _native.ErrorCode
OpType = _native.OpType
CopyKind = _native.CopyKind
BlockStage = _native.BlockStage
RopeLayout = _native.RopeLayout
QuantGranularity = _native.QuantGranularity

# 基础值类型。
Dim = _native.Dim
SymbolicShape = _native.SymbolicShape
QuantizationDesc = _native.QuantizationDesc
RopeConfig = _native.RopeConfig
ModelConfig = _native.ModelConfig
FusedOp = _native.FusedOp

# 图类型。
Edge = _native.Edge
WeightDesc = _native.WeightDesc
Node = _native.Node
Graph = _native.Graph
GraphBuilder = _native.GraphBuilder
GraphError = _native.GraphError

# 校验与融合规则。
ValidationIssue = _native.ValidationIssue
ValidationReport = _native.ValidationReport
FusionRule = _native.FusionRule
FusionRuleTable = _native.FusionRuleTable

# 测试/转换辅助函数。
to_tensor_desc = _native.to_tensor_desc
convert_graph_to_runtime_json = _native.convert_graph_to_runtime_json
build_qwen3_reference_graph = _native.build_qwen3_reference_graph

from . import graph_tools  # noqa: E402
from .builders import qwen3  # noqa: E402

build_qwen3_python_graph = qwen3.build_qwen3_python_graph

__all__ = [
    "BlockStage",
    "CopyKind",
    "Dim",
    "Edge",
    "ErrorCode",
    "FusedOp",
    "FusionRule",
    "FusionRuleTable",
    "Graph",
    "GraphBuilder",
    "GraphError",
    "ModelConfig",
    "Node",
    "OpType",
    "QuantGranularity",
    "QuantizationDesc",
    "RopeConfig",
    "RopeLayout",
    "SymbolicShape",
    "ValidationIssue",
    "ValidationReport",
    "WeightDesc",
    "build_qwen3_python_graph",
    "build_qwen3_reference_graph",
    "convert_graph_to_runtime_json",
    "graph_tools",
    "qwen3",
    "to_tensor_desc",
]
