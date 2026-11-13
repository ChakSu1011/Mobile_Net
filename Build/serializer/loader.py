"""模块：模型包加载与图重建。

职责：
  读取并校验 ``model.mgf`` 固定头、manifest 哈希与 ``weights.bin`` 哈希；
  按 manifest 重放 Graph_Fixed 构图 API，重建可执行图并校验 IR 哈希。

边界：
  - 不修改包文件；
  - 不加载真实设备显存、不创建 Kernel；
  - 不解析权重数值内容，只做边界与哈希校验。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Build.common.errors import BuildError
from Build.common.graph_api import (
    layout_from_name,
    load_graph_api,
    op_type_from_name,
    precision_from_name,
)
from Build.common.hashing import sha256_bytes, sha256_file
from Build.common.model_ir import ModelGraph
from Build.serializer.format import (
    ModelPackage,
    decode_manifest,
    unpack_header,
)

MODEL_FILE_NAME = "model.mgf"
WEIGHTS_FILE_NAME = "weights.bin"
REPORT_FILE_NAME = "compile_report.json"


def load_package(
    package_dir: Path,
    *,
    verify: bool = True,
) -> ModelPackage:
    """读取并校验模型包。

    参数：
      package_dir：包含 ``model.mgf`` 的目录；
      verify：是否校验 manifest 与权重哈希。

    返回：
      ``ModelPackage``；已解析 manifest。

    异常：
      BuildError：文件缺失、头部非法、越界、哈希不符或 JSON 非法。
    """
    package_dir = Path(package_dir)
    model_file = package_dir / MODEL_FILE_NAME
    if not model_file.is_file():
        raise BuildError(
            "serializer",
            f"model file not found: {model_file}",
        )
    data = model_file.read_bytes()
    header = unpack_header(data)
    manifest_end = header.manifest_offset + header.manifest_length
    if manifest_end > len(data):
        raise BuildError(
            "serializer",
            "manifest extends beyond file: "
            f"end={manifest_end}, size={len(data)}",
        )
    manifest_bytes = data[
        header.manifest_offset:manifest_end]
    manifest_sha256 = sha256_bytes(manifest_bytes)
    if verify and manifest_sha256 != header.manifest_sha256.hex():
        raise BuildError(
            "serializer",
            "manifest SHA-256 mismatch: "
            f"expected {header.manifest_sha256.hex()}, "
            f"got {manifest_sha256}",
        )
    manifest = decode_manifest(manifest_bytes)

    weights_path: Path | None = None
    weights_sha256: str | None = None
    if header.has_weights:
        weights_path = package_dir / WEIGHTS_FILE_NAME
        if not weights_path.is_file():
            raise BuildError(
                "serializer",
                f"weights file not found: {weights_path}",
            )
        actual_size = weights_path.stat().st_size
        if actual_size != header.weights_length:
            raise BuildError(
                "serializer",
                "weights file size mismatch: "
                f"expected {header.weights_length}, "
                f"got {actual_size}",
            )
        actual_hash = sha256_file(weights_path)
        if verify and actual_hash != header.weights_sha256.hex():
            raise BuildError(
                "serializer",
                "weights SHA-256 mismatch: "
                f"expected {header.weights_sha256.hex()}, "
                f"got {actual_hash}",
            )
        weights_sha256 = actual_hash
        _validate_weight_entries(manifest, actual_size)
    return ModelPackage(
        path=package_dir,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
    )


def _validate_weight_entries(
    manifest: dict[str, Any],
    weights_size: int,
) -> None:
    """校验 manifest 权重条目的偏移与大小边界。

    参数：
      manifest：已解析 manifest；
      weights_size：weights.bin 实际字节数。

    异常：
      BuildError：偏移未对齐、越界或可用性矛盾。
    """
    weights = manifest.get("weights", {})
    alignment = int(weights.get("alignment", 0))
    if alignment <= 0:
        raise BuildError(
            "serializer", "invalid weight alignment")
    total_bytes = int(weights.get("total_bytes", -1))
    if total_bytes != weights_size:
        raise BuildError(
            "serializer",
            "weights total_bytes mismatch: "
            f"manifest={total_bytes}, file={weights_size}",
        )
    for item in weights.get("entries", []):
        available = bool(item.get("available", False))
        if not available:
            continue
        offset = int(item.get("offset", -1))
        size = int(item.get("byte_size", -1))
        if offset < 0 or size < 0:
            raise BuildError(
                "serializer",
                f"invalid weight offset/size: {item.get('name')}",
            )
        if offset % alignment != 0:
            raise BuildError(
                "serializer",
                f"weight offset {offset} is not aligned to {alignment}",
            )
        if offset + size > weights_size:
            raise BuildError(
                "serializer",
                f"weight {item.get('name')} exceeds weights file",
            )


def rebuild_graph(
    package: ModelPackage,
    *,
    verify_ir: bool = True,
) -> Any:
    """按 manifest 重建 Graph_Fixed 图。

    参数：
      package：已加载的模型包；
      verify_ir：是否校验重建图 IR 哈希。

    返回：
      ``mobilenet_runtime_graph.Graph`` 实例；已调用 ``sort_op()``。

    异常：
      BuildError：manifest 图非法、重建失败或 IR 哈希不符。
    """
    graph_dict = package.manifest["graph"]
    model_graph = ModelGraph.from_dict(graph_dict)
    graph = rebuild_graph_from_model(model_graph)
    if verify_ir:
        expected = graph_dict.get("ir_sha256")
        if not isinstance(expected, str):
            raise BuildError(
                "serializer",
                "manifest is missing graph.ir_sha256",
            )
        actual = sha256_bytes(
            graph.dump_ir().encode("utf-8"))
        if actual != expected:
            raise BuildError(
                "serializer",
                "rebuilt graph IR SHA-256 mismatch: "
                f"expected {expected}, got {actual}",
            )
    return graph


def rebuild_graph_from_model(model_graph: ModelGraph) -> Any:
    """按 ``ModelGraph`` 记录顺序重放构图 API。

    参数：
      model_graph：模型图元数据。

    返回：
      ``mobilenet_runtime_graph.Graph`` 实例；已排序。

    异常：
      BuildError：精度/布局/op 类型未知、输出名不符或构图失败。
    """
    api = load_graph_api()
    graph = api.Graph(model_graph.name)
    tensors: dict[str, Any] = {}
    try:
        for name in model_graph.constants:
            meta = model_graph.tensor(name)
            static_shape = meta.static_shape
            if static_shape is None:
                raise BuildError(
                    "serializer",
                    "constant shape has symbolic dimensions",
                    target=name,
                )
            tensors[name] = graph.create_constant(
                name,
                dims=list(static_shape),
                pitch=list(meta.pitch),
                precision=precision_from_name(api, meta.precision),
                layout=layout_from_name(api, meta.layout),
                byte_size=meta.byte_size,
                data_ptr=0,
            )
        for name in model_graph.inputs:
            meta = model_graph.tensor(name)
            tensors[name] = graph.create_input(
                name,
                precision_from_name(api, meta.precision),
                layout_from_name(api, meta.layout),
            )
        for op in model_graph.ops:
            sources = [tensors[name] for name in op.inputs]
            outputs = graph.create_op(
                op.name,
                sources,
                len(op.outputs),
                op_type_from_name(api, op.op_type),
                dict(op.attrs),
            )
            actual_names = [tensor.name for tensor in outputs]
            if actual_names != list(op.outputs):
                raise BuildError(
                    "serializer",
                    "rebuilt op output names differ: "
                    f"expected {list(op.outputs)}, "
                    f"got {actual_names}",
                    target=op.name,
                )
            for tensor_name, tensor in zip(actual_names, outputs):
                tensors[tensor_name] = tensor
        for binding in model_graph.outputs:
            graph.create_output(
                binding.name, tensors[binding.tensor])
        graph.sort_op()
    except BuildError:
        raise
    except (KeyError, RuntimeError, ValueError) as error:
        raise BuildError(
            "serializer",
            f"graph rebuild failed: {error}",
            target=model_graph.name,
        ) from error
    return graph


def package_summary(package: ModelPackage) -> dict[str, Any]:
    """返回模型包摘要。

    参数：
      package：已加载的模型包。

    返回：
      JSON 可序列化摘要。
    """
    manifest = package.manifest
    weights = manifest.get("weights", {})
    entries = weights.get("entries", [])
    available = sum(
        1 for item in entries if item.get("available"))
    return {
        "package": str(package.path),
        "model": manifest["model"]["name"],
        "adapter": manifest["model"]["adapter"],
        "tensors": manifest["statistics"]["tensors"],
        "ops": manifest["statistics"]["ops"],
        "constants": manifest["statistics"]["constants"],
        "weights_total_bytes": weights.get("total_bytes", 0),
        "weights_available": available,
        "manifest_sha256": package.manifest_sha256,
        "weights_sha256": package.weights_sha256,
        "ir_sha256": manifest["graph"].get("ir_sha256"),
    }


def dump_manifest(package_dir: Path) -> str:
    """返回包 manifest 的缩进 JSON 文本，供 CLI 检查。

    参数：
      package_dir：包目录。

    返回：
      UTF-8 JSON 文本。

    异常：
      BuildError：包非法。
    """
    package = load_package(package_dir, verify=True)
    return json.dumps(
        package.manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
