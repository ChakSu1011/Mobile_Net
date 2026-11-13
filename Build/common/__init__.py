"""模块：Build 阶段公共工具。

职责：
  提供模型 IR 数据类、Graph 动态加载、规范化 JSON、哈希与错误类型，
  供 converter、optimizer、serializer 共享。

边界：
  不实现具体模型 Adapter、优化 Pass 或文件格式；这些逻辑留在各模块内。
"""

from __future__ import annotations

from Build.common.dtypes import element_bytes
from Build.common.errors import BuildError
from Build.common.graph_api import load_graph_api
from Build.common.hashing import sha256_bytes, sha256_file
from Build.common.jsonio import canonical_json, read_json, write_json
from Build.common.model_ir import (
    Dim,
    IOBinding,
    MemoryEntry,
    MemoryPlan,
    ModelGraph,
    OpMeta,
    TensorMeta,
    normalize_json_value,
)
from Build.common.safetensors import (
    SafetensorsEntry,
    iter_tensor_chunks,
    read_index,
    read_tensor_bytes,
)

__all__ = [
    "BuildError",
    "Dim",
    "IOBinding",
    "MemoryEntry",
    "MemoryPlan",
    "ModelGraph",
    "OpMeta",
    "SafetensorsEntry",
    "TensorMeta",
    "canonical_json",
    "element_bytes",
    "iter_tensor_chunks",
    "load_graph_api",
    "normalize_json_value",
    "read_index",
    "read_json",
    "read_tensor_bytes",
    "sha256_bytes",
    "sha256_file",
    "write_json",
]
