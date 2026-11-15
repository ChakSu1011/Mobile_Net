"""模块：显存图文本格式化。

职责：
  提供精度、shape、pitch、字节数文本与 XML 转义。

边界：
  不做布局、绘图或模型解析。
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable
from html import escape

_PRECISION_LABELS = {
    "float64": "fp64",
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float8_e4m3": "fp8e4m3",
    "float8_e5m2": "fp8e5m2",
    "int64": "i64",
    "int32": "i32",
    "int16": "i16",
    "int8": "i8",
    "uint8": "u8",
    "bool": "bool",
}


def precision_label(precision: str) -> str:
    """返回精度紧凑名。

    参数：
      precision：稳定精度名。

    返回：
      紧凑名称。
    """
    return _PRECISION_LABELS.get(precision, precision)


def format_dims(values: Iterable[object]) -> str:
    """把维度序列格式化为 ``1×256×1024``。

    参数：
      values：维度值序列。

    返回：
      使用 ``×`` 连接的文本；空序列返回 ``[]``。
    """
    items = [str(getattr(item, "value", item)) for item in values]
    if not items:
        return "[]"
    return "×".join(items)


def format_bytes(value: int) -> str:
    """返回带千分位的字节数。

    参数：
      value：字节数。

    返回：
      例如 ``1,048,576 B``。
    """
    return f"{value:,} B"


def format_offset(value: int) -> str:
    """返回带千分位的偏移。

    参数：
      value：字节偏移。

    返回：
      例如 ``524,288``。
    """
    return f"{value:,}"


def wrap_lines(text: str, width: int = 64) -> list[str]:
    """按字符宽度换行。

    参数：
      text：原始文本；
      width：每行字符数。

    返回：
      行列表。
    """
    if not text:
        return []
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


def xml_escape(text: str) -> str:
    """转义 XML 特殊字符。

    参数：
      text：原始文本。

    返回：
      可放入 SVG 的文本。
    """
    return escape(text, quote=True)
