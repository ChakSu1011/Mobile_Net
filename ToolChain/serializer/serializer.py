"""模块：模型包序列化。

职责：
  校验优化结果，打包可用权重，计算重建图 IR 哈希，组装 manifest，
  写出 ``model.mgf``、可选 ``weights.bin`` 与 ``compile_report.json``，
  并在写出后重新加载自检。

边界：
  - 不执行图优化，不修改 ``ModelGraph``；
  - 不转换权重 dtype，BF16 等字节原样打包；
  - 不生成运行时 Kernel 或设备资源。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ToolChain.common.errors import BuildError
from ToolChain.common.hashing import sha256_bytes
from ToolChain.common.jsonio import write_json
from ToolChain.common.model_ir import ModelGraph
from ToolChain.common.safetensors import (
    SafetensorsEntry,
    iter_tensor_chunks,
    read_index,
)
from ToolChain.optimizer.optimizer import OptimizeResult
from ToolChain.serializer.format import (
    FLAG_WEIGHTS,
    FORMAT_NAME,
    FORMAT_VERSION,
    HEADER_SIZE,
    MANIFEST_OFFSET,
    WEIGHT_ALIGNMENT,
    ModelPackage,
    PackageHeader,
    WeightEntry,
    build_manifest,
    encode_manifest,
)
from ToolChain.serializer.loader import (
    MODEL_FILE_NAME,
    REPORT_FILE_NAME,
    WEIGHTS_FILE_NAME,
    load_package,
    rebuild_graph_from_model,
)


@dataclass(frozen=True)
class SerializationResult:
    """序列化结果。

    属性：
      package_dir：包目录；
      model_file：model.mgf 路径；
      weights_file：weights.bin 路径；无权重时为 None；
      report_file：compile_report.json 路径；
      manifest_sha256：manifest 哈希（十六进制）；
      weights_sha256：weights.bin 哈希（十六进制）；
      graph_ir_sha256：重建图 IR 哈希。
    """

    package_dir: Path
    model_file: Path
    weights_file: Path | None
    report_file: Path
    manifest_sha256: str
    weights_sha256: str | None
    graph_ir_sha256: str


def serialize_package(
    result: OptimizeResult,
    output_dir: Path,
    *,
    weights_path: Path | None = None,
    overwrite: bool = False,
) -> SerializationResult:
    """把优化结果写成模型包。

    参数：
      result：Optimizer 输出；必须已完成精度与形状推导；
      output_dir：包目录；非空时必须显式 ``overwrite=True``；
      weights_path：可选 safetensors 文件或目录；
      overwrite：是否允许覆盖已有模型包文件。

    返回：
      ``SerializationResult``；包已重新加载自检。

    异常：
      BuildError：结果未优化、目录非空、权重损坏或写出不一致；
      OSError：文件读写失败。
    """
    _validate_result(result)
    output_dir = Path(output_dir)
    _prepare_output_dir(output_dir, overwrite)
    model_graph = result.model_graph

    entries, ready = _collect_weight_entries(
        model_graph, weights_path)
    entries, weights_total, weights_sha256 = _pack_weights(
        entries, ready, output_dir)

    graph = rebuild_graph_from_model(model_graph)
    ir_text = graph.dump_ir()
    ir_sha256 = sha256_bytes(ir_text.encode("utf-8"))
    ir_line_count = len(ir_text.splitlines())
    manifest = build_manifest(
        model_graph,
        result.config,
        result.memory_plan,
        ir_sha256=ir_sha256,
        ir_line_count=ir_line_count,
        weight_entries=entries,
        weights_total_bytes=weights_total,
    )
    manifest_bytes = encode_manifest(manifest)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    weights_hash_bytes = (
        bytes.fromhex(weights_sha256)
        if weights_sha256 is not None else b"\0" * 32)
    header = PackageHeader(
        version=FORMAT_VERSION,
        flags=FLAG_WEIGHTS if weights_sha256 is not None else 0,
        manifest_offset=MANIFEST_OFFSET,
        manifest_length=len(manifest_bytes),
        manifest_sha256=bytes.fromhex(manifest_sha256),
        weights_offset=(
            HEADER_SIZE + len(manifest_bytes)
            if weights_sha256 is not None else 0
        ),
        weights_length=weights_total,
        weights_sha256=weights_hash_bytes,
    )

    model_file = output_dir / MODEL_FILE_NAME
    _write_atomic(model_file, header.to_bytes() + manifest_bytes)
    report_file = output_dir / REPORT_FILE_NAME
    write_json(
        report_file,
        _compile_report(
            result,
            manifest,
            model_file,
            weights_total,
            weights_sha256,
            manifest_sha256,
            ir_line_count,
        ),
    )
    package = load_package(output_dir, verify=True)
    _self_check(package, manifest_sha256, weights_sha256)
    return SerializationResult(
        package_dir=output_dir,
        model_file=model_file,
        weights_file=(
            output_dir / WEIGHTS_FILE_NAME
            if weights_sha256 is not None else None
        ),
        report_file=report_file,
        manifest_sha256=manifest_sha256,
        weights_sha256=weights_sha256,
        graph_ir_sha256=ir_sha256,
    )


def _validate_result(result: OptimizeResult) -> None:
    """校验优化结果满足序列化前提。

    参数：
      result：优化结果。

    异常：
      BuildError：精度未知、形状未解析或显存计划缺失。
    """
    model_graph = result.model_graph
    for name, tensor in model_graph.tensors.items():
        if tensor.precision == "unknown":
            raise BuildError(
                "serializer",
                "tensor precision is unknown; run optimizer first",
                target=name,
            )
        if (tensor.kind == "activation"
                and (tensor.static_shape is None
                     or not tensor.static_shape)):
            raise BuildError(
                "serializer",
                "activation shape is not resolved; run optimizer first",
                target=name,
            )
    entry_names = {
        entry.name for entry in result.memory_plan.entries}
    for tensor in model_graph.activation_tensors:
        if tensor.name not in entry_names:
            raise BuildError(
                "serializer",
                "activation is missing from memory plan",
                target=tensor.name,
            )


def _prepare_output_dir(
    output_dir: Path, overwrite: bool) -> None:
    """创建或校验输出目录。

    参数：
      output_dir：目标目录；
      overwrite：是否允许目录内已有文件。

    异常：
      BuildError：路径是文件，或目录非空且未授权覆盖。
    """
    if output_dir.exists() and not output_dir.is_dir():
        raise BuildError(
            "serializer",
            f"output path is not a directory: {output_dir}",
        )
    if output_dir.is_dir():
        existing = list(output_dir.iterdir())
        if existing and not overwrite:
            raise BuildError(
                "serializer",
                f"output directory is not empty: {output_dir}",
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def _collect_weight_entries(
    model_graph: ModelGraph,
    weights_path: Path | None,
) -> tuple[list[WeightEntry], list[tuple[str, SafetensorsEntry]]]:
    """扫描常量与可选权重文件，生成权重条目。

    参数：
      model_graph：优化后的模型图；
      weights_path：可选 safetensors 文件或目录。

    返回：
      ``(entries, ready)``；``ready`` 为可打包的
      ``(常量名, safetensors 条目)`` 列表。

    异常：
      BuildError：权重路径不存在或文件损坏。
    """
    index: dict[str, SafetensorsEntry] | None = None
    if weights_path is not None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise BuildError(
                "serializer",
                f"weights path not found: {weights_path}",
            )
        index = read_index(weights_path)
    entries: list[WeightEntry] = []
    ready: list[tuple[str, SafetensorsEntry]] = []
    for name in model_graph.constants:
        tensor = model_graph.tensor(name)
        static_shape = tensor.static_shape or ()
        source = tensor.weight_source
        reason: str | None = None
        found: SafetensorsEntry | None = None
        if source is None:
            reason = "no source mapping"
        elif index is None:
            reason = "weights not supplied"
        elif source not in index:
            reason = "source tensor not found"
        else:
            candidate = index[source]
            if candidate.shape != static_shape:
                reason = (
                    f"shape mismatch: weights {list(candidate.shape)}, "
                    f"graph {list(static_shape)}")
            elif candidate.precision != tensor.precision:
                reason = (
                    f"dtype mismatch: weights {candidate.precision}, "
                    f"graph {tensor.precision}")
            elif candidate.byte_size != tensor.byte_size:
                reason = (
                    f"byte_size mismatch: weights {candidate.byte_size}, "
                    f"graph {tensor.byte_size}")
            else:
                found = candidate
        entries.append(
            WeightEntry(
                name=name,
                source=source,
                shape=static_shape,
                dtype=tensor.precision,
                byte_size=tensor.byte_size,
                offset=None,
                sha256=None,
                available=False,
                reason=reason,
            )
        )
        if found is not None:
            ready.append((name, found))
    return entries, ready


def _pack_weights(
    entries: list[WeightEntry],
    ready: list[tuple[str, SafetensorsEntry]],
    output_dir: Path,
) -> tuple[list[WeightEntry], int, str | None]:
    """把可用权重按对齐要求写入 weights.bin。

    参数：
      entries：待更新的权重条目；
      ready：可打包权重列表；
      output_dir：包目录。

    返回：
      ``(updated_entries, total_bytes, sha256)``；无可用权重时
      ``total_bytes=0`` 且 ``sha256=None``。

    异常：
      BuildError/OSError：读取或写入失败。
    """
    weights_file = output_dir / WEIGHTS_FILE_NAME
    if not ready:
        if weights_file.exists():
            weights_file.unlink()
        return entries, 0, None
    temporary = output_dir / (WEIGHTS_FILE_NAME + ".tmp")
    index_by_name = {
        entry.name: index for index, entry in enumerate(entries)}
    offset = 0
    file_hasher = hashlib.sha256()
    try:
        with temporary.open("wb") as stream:
            for name, source in ready:
                padding = (
                    -offset % WEIGHT_ALIGNMENT)
                if padding:
                    zeros = b"\0" * padding
                    stream.write(zeros)
                    file_hasher.update(zeros)
                    offset += padding
                entry_hasher = hashlib.sha256()
                for chunk in iter_tensor_chunks(source):
                    stream.write(chunk)
                    entry_hasher.update(chunk)
                    file_hasher.update(chunk)
                index = index_by_name[name]
                entries[index] = replace(
                    entries[index],
                    offset=offset,
                    sha256=entry_hasher.hexdigest(),
                    available=True,
                    reason=None,
                )
                offset += source.byte_size
        os.replace(temporary, weights_file)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return entries, offset, file_hasher.hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    """原子写入文件。

    参数：
      path：目标路径；
      data：完整字节内容。

    异常：
      OSError：写入失败。
    """
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _self_check(
    package: ModelPackage,
    manifest_sha256: str,
    weights_sha256: str | None,
) -> None:
    """校验重新加载的包与写出内容一致。

    参数：
      package：重新加载的包；
      manifest_sha256：期望 manifest 哈希；
      weights_sha256：期望权重哈希。

    异常：
      BuildError：哈希不一致。
    """
    if package.manifest_sha256 != manifest_sha256:
        raise BuildError(
            "serializer",
            "self-check manifest SHA-256 mismatch",
        )
    if package.weights_sha256 != weights_sha256:
        raise BuildError(
            "serializer",
            "self-check weights SHA-256 mismatch",
        )


def _compile_report(
    result: OptimizeResult,
    manifest: dict[str, Any],
    model_file: Path,
    weights_total: int,
    weights_sha256: str | None,
    manifest_sha256: str,
    ir_line_count: int,
) -> dict[str, Any]:
    """生成编译报告。

    参数：
      result：优化结果；
      manifest：已写出的 manifest；
      model_file：model.mgf 路径；
      weights_total：weights.bin 字节数；
      weights_sha256：weights.bin 哈希；
      manifest_sha256：manifest 哈希；
      ir_line_count：重建图 IR 行数。

    返回：
      含生成时间的 JSON 可序列化报告。
    """
    entries = manifest["weights"]["entries"]
    available = sum(
        1 for entry in entries if entry["available"])
    files: dict[str, Any] = {
        MODEL_FILE_NAME: {
            "size": model_file.stat().st_size,
            "sha256": sha256_bytes(model_file.read_bytes()),
        }
    }
    weights_file = Path(model_file).parent / WEIGHTS_FILE_NAME
    if weights_sha256 is not None:
        files[WEIGHTS_FILE_NAME] = {
            "size": weights_total,
            "sha256": weights_sha256,
        }
    elif weights_file.exists():
        weights_file.unlink()
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": result.model_graph.name,
            "adapter": result.model_graph.adapter,
        },
        "graph": {
            "ir_sha256": manifest["graph"]["ir_sha256"],
            "ir_line_count": ir_line_count,
            "tensors": manifest["statistics"]["tensors"],
            "ops": manifest["statistics"]["ops"],
            "constants": manifest["statistics"]["constants"],
            "activation_tensors": manifest["statistics"][
                "activation_tensors"],
        },
        "memory": {
            "alignment": result.memory_plan.alignment,
            "arena_bytes": result.memory_plan.arena_bytes,
            "peak_bytes": result.memory_plan.peak_bytes,
            "constants_bytes": result.memory_plan.constants_bytes,
            "entries": len(result.memory_plan.entries),
        },
        "weights": {
            "total_bytes": weights_total,
            "available": available,
            "unavailable": len(entries) - available,
            "sha256": weights_sha256,
        },
        "verification": {
            "manifest_sha256": manifest_sha256,
            "package_reloaded": True,
        },
        "files": files,
    }
