"""模块：Precision Pass 的 BF16→FP16 权重转换。

职责：
  读取源 safetensors 索引，按 ``ModelGraph`` 常量映射找到权重，流式把
  BF16 字节转换为 FP16，并写出供 Serializer 使用的转换文件。

边界：
  - 只做 BF16→FP16 数值转换；其他 dtype 原样复制；
  - 不修改 ModelGraph、不执行推理；
  - 不一次性加载完整权重，按块处理；
  - numpy 是可选加速路径，缺失时使用 struct 逐元素转换。
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ToolChain.common.errors import BuildError
from ToolChain.common.model_ir import ModelGraph
from ToolChain.common.safetensors import (
    SafetensorsEntry,
    iter_tensor_chunks,
    read_index,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover - 无 numpy 环境走纯 Python
    np = None

_CHUNK_BYTES = 4 * 1024 * 1024
_MAX_FP16 = 65504.0


@dataclass(frozen=True)
class WeightConversionResult:
    """一次权重转换的结果。

    属性：
      output_path：转换后的 safetensors 路径；
      source_count：写入转换文件的源张量数；
      converted_count：实际 BF16→FP16 的张量数；
      copied_count：原样复制的张量数；
      target_precision：目标精度。
    """

    output_path: Path
    source_count: int
    converted_count: int
    copied_count: int
    target_precision: str

    def to_dict(self) -> dict[str, Any]:
        """返回确定性报告字段（不包含绝对路径）。"""
        return {
            "source_count": self.source_count,
            "converted_count": self.converted_count,
            "copied_count": self.copied_count,
            "target_precision": self.target_precision,
            "output_name": self.output_path.name,
        }


def _convert_bf16_chunk(chunk: bytes, tensor_name: str) -> bytes:
    """把一块 BF16 字节转换为 FP16 字节。

    参数：
      chunk：BF16 原始字节，长度必须为偶数；
      tensor_name：用于错误定位的张量名。

    返回：
      FP16 字节，长度与输入相同。

    异常：
      BuildError：长度非法或有限值超出 FP16 范围。
    """
    if len(chunk) % 2 != 0:
        raise BuildError(
            "optimizer",
            "BF16 tensor byte size must be even",
            target=tensor_name,
        )
    if np is not None:
        bits = np.frombuffer(chunk, dtype="<u2")
        values = (bits.astype(np.uint32) << 16).view("<f4")
        overflow = np.isfinite(values) & (np.abs(values) > _MAX_FP16)
        if bool(overflow.any()):
            raise BuildError(
                "optimizer",
                "BF16 value exceeds FP16 range: "
                f"{int(overflow.sum())} elements",
                target=tensor_name,
            )
        return values.astype("<f2").tobytes()

    output = bytearray()
    for (bits,) in struct.iter_unpack("<H", chunk):
        value = struct.unpack(
            "<f", struct.pack("<I", bits << 16))[0]
        if math.isfinite(value) and abs(value) > _MAX_FP16:
            raise BuildError(
                "optimizer",
                "BF16 value exceeds FP16 range",
                target=tensor_name,
            )
        output.extend(struct.pack("<e", value))
    return bytes(output)


def _source_entries(
    model_graph: ModelGraph,
    weights_path: Path,
) -> list[SafetensorsEntry]:
    """返回图中常量引用的源 safetensors 条目。

    参数：
      model_graph：模型图；
      weights_path：safetensors 文件或目录。

    返回：
      按常量登记顺序去重后的源张量条目。
    """
    index = read_index(weights_path)
    entries = []
    seen: set[str] = set()
    for name in model_graph.constants:
        source = model_graph.tensor(name).weight_source
        if source is None or source in seen:
            continue
        entry = index.get(source)
        if entry is None:
            continue
        seen.add(source)
        entries.append(entry)
    return entries


def convert_weights_to_fp16(
    model_graph: ModelGraph,
    weights_path: Path,
    output_path: Path,
    target_precision: str,
) -> WeightConversionResult | None:
    """把图中的 BF16 权重转换为 FP16 并写出 safetensors。

    参数：
      model_graph：模型图，常量必须带 ``weight_source``；
      weights_path：源 safetensors 文件或目录；
      output_path：转换文件输出路径；
      target_precision：目标精度，只有 ``float16`` 才触发转换。

    返回：
      ``WeightConversionResult``；无需转换时返回 None。

    异常：
      BuildError：源文件损坏、权重溢出或写出失败。
    """
    if target_precision != "float16":
        return None
    entries = _source_entries(model_graph, weights_path)
    if not entries:
        return None
    if not any(entry.precision == "bfloat16" for entry in entries):
        return None

    header: dict[str, Any] = {}
    records: list[tuple[SafetensorsEntry, str]] = []
    offset = 0
    converted = 0
    copied = 0
    for entry in entries:
        if entry.precision == "bfloat16":
            dtype = "F16"
            converted += 1
        else:
            dtype = entry.dtype
            copied += 1
        size = entry.byte_size
        header[entry.name] = {
            "dtype": dtype,
            "shape": list(entry.shape),
            "data_offsets": [offset, offset + size],
        }
        records.append((entry, dtype))
        offset += size

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_bytes = json.dumps(
        header,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(header_bytes)) % 8
    header_bytes += b" " * padding
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(struct.pack("<Q", len(header_bytes)))
            stream.write(header_bytes)
            for entry, dtype in records:
                if entry.precision == "bfloat16":
                    for chunk in iter_tensor_chunks(
                            entry, chunk_size=_CHUNK_BYTES):
                        stream.write(_convert_bf16_chunk(
                            chunk, entry.name))
                else:
                    for chunk in iter_tensor_chunks(
                            entry, chunk_size=_CHUNK_BYTES):
                        stream.write(chunk)
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    return WeightConversionResult(
        output_path=output_path,
        source_count=len(entries),
        converted_count=converted,
        copied_count=copied,
        target_precision=target_precision,
    )
