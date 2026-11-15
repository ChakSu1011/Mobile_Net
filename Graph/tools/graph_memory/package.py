"""模块：读取并校验 model.mgf 模型包。

职责：
  调用 Serializer 的加载接口校验文件头、manifest 哈希，并暴露 graph、
  memory_plan 与权重元数据。

边界：
  不读取 weights.bin 内容，不修改模型包。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ToolChain.common.errors import BuildError
from ToolChain.common.model_ir import ModelGraph
from ToolChain.serializer import load_package


@dataclass(frozen=True)
class LoadedPackage:
    """显存可视化使用的模型包描述。

    属性：
      package_dir：包含 model.mgf 的目录；
      model_graph：模型图元数据；
      memory_plan：manifest 中的 memory_plan 字典；
      weight_entries：manifest 中的权重条目列表；
      manifest_sha256：manifest SHA-256；
      ir_sha256：IR SHA-256；
      model_name：图名称。
    """

    package_dir: Path
    model_graph: ModelGraph
    memory_plan: dict[str, Any]
    weight_entries: tuple[dict[str, Any], ...]
    manifest_sha256: str
    ir_sha256: str
    model_name: str


def resolve_package_dir(path: Path) -> Path:
    """把文件或目录路径统一解析为包目录。

    参数：
      path：model.mgf 文件路径或包目录。

    返回：
      包目录。
    """
    path = Path(path)
    return path.parent if path.is_file() else path


def load_model_package(path: Path) -> LoadedPackage:
    """读取并校验模型包。

    参数：
      path：model.mgf 文件路径或包目录。

    返回：
      ``LoadedPackage``。

    异常：
      BuildError：包缺失、头部非法、哈希不符或缺少 memory_plan。
    """
    package_dir = resolve_package_dir(path)
    package = load_package(package_dir, verify=True)
    graph_dict = package.manifest.get("graph")
    if not isinstance(graph_dict, dict):
        raise BuildError(
            "graph_memory",
            "manifest is missing graph object",
            target=str(package_dir),
        )
    ir_sha256 = graph_dict.get("ir_sha256")
    if not isinstance(ir_sha256, str):
        raise BuildError(
            "graph_memory",
            "manifest graph is missing ir_sha256",
            target=str(package_dir),
        )
    optimization = package.manifest.get("optimization", {})
    memory_plan = (
        optimization.get("memory_plan")
        if isinstance(optimization, dict) else None
    )
    if not isinstance(memory_plan, dict):
        raise BuildError(
            "graph_memory",
            "manifest is missing optimization.memory_plan",
            target=str(package_dir),
        )
    weights = package.manifest.get("weights", {})
    entries = weights.get("entries", []) if isinstance(weights, dict) else []
    model_graph = ModelGraph.from_dict(graph_dict)
    return LoadedPackage(
        package_dir=package_dir,
        model_graph=model_graph,
        memory_plan=memory_plan,
        weight_entries=tuple(entries),
        manifest_sha256=package.manifest_sha256,
        ir_sha256=ir_sha256,
        model_name=model_graph.name,
    )
