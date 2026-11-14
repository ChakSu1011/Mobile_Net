"""模块：读取并校验 model.mgf 模型包。

职责：
  调用 Serializer 的加载接口校验文件头、manifest 哈希与格式版本，
  并暴露可视化所需的 ``ModelGraph``、模型名与哈希信息。

边界：
  - 不读取 weights.bin 内容；
  - 不重建 Graph、不依赖 _graph 扩展；
  - 不修改模型包或 manifest。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ToolChain.common.errors import BuildError
from ToolChain.common.model_ir import ModelGraph
from ToolChain.serializer import load_package


@dataclass(frozen=True)
class LoadedPackage:
    """可视化使用的模型包描述。

    属性：
      package_dir：包含 model.mgf 的目录；
      model_graph：manifest 中记录的模型图元数据；
      manifest_sha256：manifest SHA-256（十六进制）；
      ir_sha256：重建图 IR SHA-256（十六进制）；
      format_version：容器格式版本；
      model_name：图名称。
    """

    package_dir: Path
    model_graph: ModelGraph
    manifest_sha256: str
    ir_sha256: str
    format_version: int
    model_name: str


def resolve_package_dir(path: Path) -> Path:
    """把文件路径或目录路径统一解析为包目录。

    参数：
      path：model.mgf 文件路径或包含它的目录。

    返回：
      包目录；如果传入的是普通文件，则返回其父目录。
    """
    path = Path(path)
    return path.parent if path.is_file() else path


def load_model_package(path: Path) -> LoadedPackage:
    """读取并校验模型包。

    参数：
      path：model.mgf 文件路径或包目录。

    返回：
      ``LoadedPackage``；manifest 已完成 SHA-256 校验。

    异常：
      BuildError：包缺失、头部非法、哈希不符、manifest 缺字段或 JSON 非法。
    """
    package_dir = resolve_package_dir(path)
    package = load_package(package_dir, verify=True)
    graph_dict = package.manifest.get("graph")
    if not isinstance(graph_dict, dict):
        raise BuildError(
            "visualizer",
            "manifest is missing graph object",
            target=str(package_dir),
        )
    ir_sha256 = graph_dict.get("ir_sha256")
    if not isinstance(ir_sha256, str):
        raise BuildError(
            "visualizer",
            "manifest graph is missing ir_sha256",
            target=str(package_dir),
        )
    model_graph = ModelGraph.from_dict(graph_dict)
    format_version = package.manifest.get("format_version", 1)
    return LoadedPackage(
        package_dir=package_dir,
        model_graph=model_graph,
        manifest_sha256=package.manifest_sha256,
        ir_sha256=ir_sha256,
        format_version=int(format_version),
        model_name=model_graph.name,
    )
