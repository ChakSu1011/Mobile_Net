"""模块：SHA-256 哈希工具。

职责：
  为模型包、manifest 与权重文件提供统一的 SHA-256 计算函数。

边界：
  只计算哈希，不做签名、加密、校验策略选择或缓存。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_DEFAULT_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    """计算字节串的 SHA-256。

    参数：
      data：待计算字节；允许为空字节串。

    返回：
      64 个小写十六进制字符。
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> str:
    """流式计算文件的 SHA-256。

    参数：
      path：文件路径；必须存在且可读；
      chunk_size：单次读取字节数，必须为正数。

    返回：
      64 个小写十六进制字符。

    异常：
      ValueError：chunk_size 小于等于 0；
      OSError：文件不存在或读取失败。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
