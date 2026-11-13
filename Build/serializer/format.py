"""模块：模型包格式定义。

职责：
  定义 ``model.mgf`` 固定头、manifest 数据结构、权重表条目与模型包
  数据结构，并提供头部与 manifest 的编码/解析函数。

边界：
  - 不读写实际文件、不打包权重、不重建图；
  - 只定义格式与纯函数转换，文件 I/O 位于 serializer/loader。
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Build.common.errors import BuildError
from Build.common.jsonio import canonical_json
from Build.common.model_ir import MemoryPlan, ModelGraph
from Build.optimizer.config import OptimizeConfig

MAGIC = b"MNETMGF\0"
FORMAT_NAME = "MobileNet.ModelPackage"
FORMAT_VERSION = 1
MANIFEST_OFFSET = 112
HEADER_SIZE = MANIFEST_OFFSET
FLAG_WEIGHTS = 0x1
WEIGHT_ALIGNMENT = 256
_HEADER = struct.Struct("<8sIIQQ32sQQ32s")


@dataclass(frozen=True)
class PackageHeader:
    """``model.mgf`` 固定头。

    属性：
      version：容器版本；
      flags：位标记，bit0 表示存在 weights.bin；
      manifest_offset：manifest 起始偏移；
      manifest_length：manifest 字节数；
      manifest_sha256：manifest SHA-256（32 字节）；
      weights_offset：权重区逻辑偏移；无权重为 0；
      weights_length：weights.bin 字节数；
      weights_sha256：weights.bin SHA-256（32 字节）。
    """

    version: int
    flags: int
    manifest_offset: int
    manifest_length: int
    manifest_sha256: bytes
    weights_offset: int
    weights_length: int
    weights_sha256: bytes

    @property
    def has_weights(self) -> bool:
        """返回是否存在权重文件。"""
        return bool(self.flags & FLAG_WEIGHTS)

    def to_bytes(self) -> bytes:
        """编码为 112 字节头部。

        返回：
          小端头部字节。

        异常：
          BuildError：字段长度或组合非法。
        """
        if self.version != FORMAT_VERSION:
            raise BuildError(
                "serializer",
                f"unsupported container version: {self.version}",
            )
        if len(self.manifest_sha256) != 32:
            raise BuildError(
                "serializer", "manifest SHA-256 must be 32 bytes")
        if len(self.weights_sha256) != 32:
            raise BuildError(
                "serializer", "weights SHA-256 must be 32 bytes")
        if self.has_weights != (self.weights_length > 0):
            raise BuildError(
                "serializer",
                "weights flag and length are inconsistent",
            )
        return _HEADER.pack(
            MAGIC,
            self.version,
            self.flags,
            self.manifest_offset,
            self.manifest_length,
            self.manifest_sha256,
            self.weights_offset,
            self.weights_length,
            self.weights_sha256,
        )


def unpack_header(data: bytes) -> PackageHeader:
    """解析并校验固定头。

    参数：
      data：至少 112 字节的文件前缀。

    返回：
      ``PackageHeader``；magic 与版本已校验。

    异常：
      BuildError：长度不足、magic 错误、版本不支持或字段矛盾。
    """
    if len(data) < HEADER_SIZE:
        raise BuildError(
            "serializer",
            f"file is too short for header: {len(data)} bytes",
        )
    (
        magic,
        version,
        flags,
        manifest_offset,
        manifest_length,
        manifest_sha256,
        weights_offset,
        weights_length,
        weights_sha256,
    ) = _HEADER.unpack(data[:HEADER_SIZE])
    if magic != MAGIC:
        raise BuildError(
            "serializer",
            f"bad magic: {magic!r}",
        )
    if version != FORMAT_VERSION:
        raise BuildError(
            "serializer",
            f"unsupported container version: {version}",
        )
    if flags & ~FLAG_WEIGHTS:
        raise BuildError(
            "serializer", f"unsupported flags: {flags}")
    if manifest_offset < HEADER_SIZE or manifest_length == 0:
        raise BuildError(
            "serializer",
            "invalid manifest offset/length: "
            f"{manifest_offset}/{manifest_length}",
        )
    has_weights = bool(flags & FLAG_WEIGHTS)
    if has_weights != (weights_length > 0):
        raise BuildError(
            "serializer",
            "weights flag and length are inconsistent",
        )
    if has_weights and weights_offset < HEADER_SIZE:
        raise BuildError(
            "serializer",
            f"invalid weights offset: {weights_offset}")
    return PackageHeader(
        version=version,
        flags=flags,
        manifest_offset=manifest_offset,
        manifest_length=manifest_length,
        manifest_sha256=manifest_sha256,
        weights_offset=weights_offset,
        weights_length=weights_length,
        weights_sha256=weights_sha256,
    )


@dataclass(frozen=True)
class WeightEntry:
    """manifest 中的单个常量权重条目。

    属性：
      name：图内常量名；
      source：源 safetensors 张量名；无映射时为 None；
      shape：静态形状；
      dtype：源权重精度；
      byte_size：字节数；
      offset：weights.bin 内偏移；不可用时为 None；
      sha256：权重字节哈希；不可用时为 None；
      available：是否已打包；
      reason：不可用原因。
    """

    name: str
    source: str | None
    shape: tuple[int, ...]
    dtype: str
    byte_size: int
    offset: int | None
    sha256: str | None
    available: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """返回 manifest 中的 JSON 表示。"""
        result: dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "byte_size": self.byte_size,
            "offset": self.offset,
            "sha256": self.sha256,
            "available": self.available,
        }
        if self.reason is not None:
            result["reason"] = self.reason
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WeightEntry":
        """从 manifest 字典构造条目。

        异常：
          KeyError/ValueError：必填字段缺失或非法。
        """
        return cls(
            name=str(data["name"]),
            source=(
                None if data.get("source") is None
                else str(data["source"])
            ),
            shape=tuple(int(value) for value in data["shape"]),
            dtype=str(data["dtype"]),
            byte_size=int(data["byte_size"]),
            offset=(
                None if data.get("offset") is None
                else int(data["offset"])
            ),
            sha256=(
                None if data.get("sha256") is None
                else str(data["sha256"])
            ),
            available=bool(data["available"]),
            reason=(
                None if data.get("reason") is None
                else str(data["reason"])
            ),
        )


@dataclass(frozen=True)
class ModelPackage:
    """加载后的模型包描述。

    属性：
      path：包目录；
      manifest：解析后的 manifest；
      manifest_sha256：manifest 哈希（十六进制）；
      weights_path：weights.bin 路径；无权重时为 None；
      weights_sha256：weights.bin 哈希（十六进制）。
    """

    path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    weights_path: Path | None
    weights_sha256: str | None


def encode_manifest(manifest: dict[str, Any]) -> bytes:
    """把 manifest 编码为 canonical JSON UTF-8 字节。

    参数：
      manifest：JSON 原生字典。

    返回：
      紧凑、键排序、UTF-8 字节。
    """
    return canonical_json(manifest).encode("utf-8")


def decode_manifest(data: bytes) -> dict[str, Any]:
    """解析并校验 manifest 顶层字段。

    参数：
      data：UTF-8 JSON 字节。

    返回：
      manifest 字典。

    异常：
      BuildError：JSON 非法、顶层字段缺失或版本不支持。
    """
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(
            "serializer", f"invalid manifest JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise BuildError(
            "serializer", "manifest must be a JSON object")
    if manifest.get("format") != FORMAT_NAME:
        raise BuildError(
            "serializer",
            f"unsupported manifest format: {manifest.get('format')}",
        )
    if int(manifest.get("format_version", 0)) != FORMAT_VERSION:
        raise BuildError(
            "serializer",
            "unsupported manifest format_version: "
            f"{manifest.get('format_version')}",
        )
    for key in ("model", "graph", "optimization", "weights",
                "statistics"):
        if key not in manifest:
            raise BuildError(
                "serializer", f"manifest is missing field: {key}")
    return manifest


def build_manifest(
    model_graph: ModelGraph,
    config: OptimizeConfig,
    memory_plan: MemoryPlan,
    *,
    ir_sha256: str,
    ir_line_count: int,
    weight_entries: list[WeightEntry],
    weights_total_bytes: int,
) -> dict[str, Any]:
    """组装模型包 manifest。

    参数：
      model_graph：优化后的模型图；
      config：优化配置；
      memory_plan：显存计划；
      ir_sha256：重建图 ``dump_ir()`` 的 SHA-256；
      ir_line_count：重建图 IR 行数；
      weight_entries：逐常量权重条目；
      weights_total_bytes：weights.bin 字节数。

    返回：
      JSON 原生 manifest 字典。
    """
    graph = model_graph.to_dict()
    graph["ir_sha256"] = ir_sha256
    graph["ir_line_count"] = ir_line_count
    statistics = {
        "tensors": len(model_graph.tensors),
        "ops": len(model_graph.ops),
        "constants": len(model_graph.constants),
        "outputs": len(model_graph.outputs),
        "activation_tensors": len(model_graph.activation_tensors),
        "constant_bytes": model_graph.constant_bytes,
    }
    return {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "model": {
            "name": model_graph.name,
            "adapter": model_graph.adapter,
            "config": model_graph.metadata.get("model_config", {}),
        },
        "graph": graph,
        "optimization": {
            "config": config.to_dict(),
            "memory_plan": memory_plan.to_dict(),
        },
        "weights": {
            "file": "weights.bin",
            "alignment": WEIGHT_ALIGNMENT,
            "total_bytes": weights_total_bytes,
            "entries": [entry.to_dict() for entry in weight_entries],
        },
        "statistics": statistics,
    }
