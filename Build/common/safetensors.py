"""模块：safetensors 头信息与原始字节读取。

职责：
  直接解析 safetensors 文件的 8 字节头长度与 JSON 头，登记张量名称、
  dtype、shape 与数据区偏移；按偏移读取原始字节，不依赖 torch/numpy。

边界：
  - 不支持写入 safetensors；
  - 不解释张量数值、不做 dtype 转换、不做形状广播；
  - 只支持 safetensors 规范中的标准 dtype 名称。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from Build.common.errors import BuildError

_HEADER_PREFIX_BYTES = 8
_MAX_HEADER_BYTES = 100 * 1024 * 1024
_DTYPE_TO_PRECISION = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3",
    "F8_E5M2": "float8_e5m2",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


@dataclass(frozen=True)
class SafetensorsEntry:
    """safetensors 中的单个张量条目。

    属性：
      name：张量名；
      dtype：safetensors 原生 dtype，如 ``BF16``；
      shape：逻辑形状；
      begin/end：相对数据区起点的字节偏移（左闭右开）；
      data_offset：数据区在文件中的绝对起点；
      shard：所属 safetensors 文件。
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int
    data_offset: int
    shard: Path

    @property
    def byte_size(self) -> int:
        """返回张量字节数。"""
        return self.end - self.begin

    @property
    def precision(self) -> str:
        """返回 manifest 使用的精度名。

        异常：
          BuildError：dtype 不受支持。
        """
        precision = _DTYPE_TO_PRECISION.get(self.dtype)
        if precision is None:
            raise BuildError(
                "safetensors",
                f"unsupported dtype: {self.dtype}",
                target=self.name,
            )
        return precision


def _safetensors_files(path: Path) -> list[Path]:
    """把文件或目录展开为 safetensors 文件列表。

    参数：
      path：safetensors 文件或包含 ``*.safetensors`` 的目录。

    返回：
      按文件名排序的文件列表。

    异常：
      BuildError：路径不存在，或目录中没有任何 safetensors 文件。
    """
    if path.is_dir():
        files = sorted(path.glob("*.safetensors"))
        if not files:
            raise BuildError(
                "safetensors",
                "no *.safetensors file found in directory",
                target=str(path),
            )
        return files
    if path.is_file():
        return [path]
    raise BuildError(
        "safetensors", "weights path does not exist", target=str(path))


def _parse_shard(path: Path) -> dict[str, SafetensorsEntry]:
    """解析单个 safetensors 文件的头信息。

    参数：
      path：safetensors 文件。

    返回：
      张量名到条目的映射。

    异常：
      BuildError：文件截断、头长度非法、JSON 非法或偏移越界。
    """
    file_size = path.stat().st_size
    with open(path, "rb") as stream:
        prefix = stream.read(_HEADER_PREFIX_BYTES)
        if len(prefix) != _HEADER_PREFIX_BYTES:
            raise BuildError(
                "safetensors",
                "file is truncated before header length",
                target=str(path),
            )
        header_length = int.from_bytes(prefix, "little")
        if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
            raise BuildError(
                "safetensors",
                f"invalid header length: {header_length}",
                target=str(path),
            )
        if _HEADER_PREFIX_BYTES + header_length > file_size:
            raise BuildError(
                "safetensors",
                "header length exceeds file size",
                target=str(path),
            )
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise BuildError(
                "safetensors",
                "file is truncated inside header",
                target=str(path),
            )

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError(
            "safetensors",
            f"invalid JSON header: {error}",
            target=str(path),
        ) from error
    if not isinstance(header, dict):
        raise BuildError(
            "safetensors",
            "header must be a JSON object",
            target=str(path),
        )

    data_offset = _HEADER_PREFIX_BYTES + header_length
    data_size = file_size - data_offset
    entries: dict[str, SafetensorsEntry] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise BuildError(
                "safetensors",
                "tensor entry must be an object",
                target=f"{path}:{name}",
            )
        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if not isinstance(dtype, str):
            raise BuildError(
                "safetensors",
                "tensor dtype missing or invalid",
                target=f"{path}:{name}",
            )
        if not isinstance(shape, list) or any(
                not isinstance(item, int) or item < 0
                for item in shape):
            raise BuildError(
                "safetensors",
                "tensor shape missing or invalid",
                target=f"{path}:{name}",
            )
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(not isinstance(item, int) for item in offsets)):
            raise BuildError(
                "safetensors",
                "tensor data_offsets missing or invalid",
                target=f"{path}:{name}",
            )
        begin, end = offsets
        if begin < 0 or end < begin or end > data_size:
            raise BuildError(
                "safetensors",
                f"tensor offsets out of range: [{begin},{end}]",
                target=f"{path}:{name}",
            )
        entries[name] = SafetensorsEntry(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            begin=begin,
            end=end,
            data_offset=data_offset,
            shard=path,
        )
    return entries


def read_index(paths: Path | Sequence[Path]) -> dict[str, SafetensorsEntry]:
    """读取一个或多个 safetensors 文件的头索引。

    参数：
      paths：文件、目录或二者的序列；目录会展开为 ``*.safetensors``。

    返回：
      张量名到条目的映射。

    异常：
      BuildError：路径非法、文件损坏或不同文件出现重名张量。
    """
    if isinstance(paths, (str, Path)):
        raw_paths: Sequence[Path] = (Path(paths),)
    else:
        raw_paths = tuple(Path(item) for item in paths)

    files: list[Path] = []
    for path in raw_paths:
        files.extend(_safetensors_files(path))

    index: dict[str, SafetensorsEntry] = {}
    for file_path in files:
        for name, entry in _parse_shard(file_path).items():
            if name in index:
                raise BuildError(
                    "safetensors",
                    "duplicate tensor name across shards",
                    target=name,
                )
            index[name] = entry
    return index


def iter_tensor_chunks(
    entry: SafetensorsEntry,
    chunk_size: int = 1024 * 1024,
) -> Iterator[bytes]:
    """按块读取张量原始字节。

    参数：
      entry：safetensors 张量条目；
      chunk_size：单次读取字节数，必须为正数。

    返回：
      字节块迭代器，总长度等于 ``entry.byte_size``。

    异常：
      ValueError：chunk_size 非正；
      OSError：文件读取失败。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    with open(entry.shard, "rb") as stream:
        stream.seek(entry.data_offset + entry.begin)
        remaining = entry.byte_size
        while remaining > 0:
            chunk = stream.read(min(chunk_size, remaining))
            if not chunk:
                raise BuildError(
                    "safetensors",
                    "unexpected end of file while reading tensor",
                    target=str(entry.shard),
                )
            remaining -= len(chunk)
            yield chunk


def read_tensor_bytes(entry: SafetensorsEntry) -> bytes:
    """读取张量的全部原始字节。

    参数：
      entry：safetensors 张量条目。

    返回：
      原始字节，长度等于 ``entry.byte_size``。
    """
    return b"".join(iter_tensor_chunks(entry))
