"""模块：可视化标签和形状格式化。

职责：
  提供精度缩写、shape 文本、文本换行与 XML 转义，保证 SVG 输出稳定且可读。

边界：
  - 不读取模型包、不构建视图模型；
  - 不负责布局和绘图，只做纯文本转换。
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable
from html import escape
_PRECISION_LABELS = {
    "unknown": "unknown",
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
    """返回精度的紧凑显示名称。

    参数：
      precision：稳定精度名，如 ``float16``。

    返回：
      紧凑名称，如 ``fp16``；未知名称原样返回。
    """
    return _PRECISION_LABELS.get(precision, precision)


def format_shape(shape: Iterable[object]) -> str:
    """把 shape 序列格式化为 ``1×256×1024``。

    参数：
      shape：int、str 或带 ``value`` 属性的维度对象序列。

    返回：
      使用 ``×`` 连接的文本；空 shape 返回 ``[]``。
    """
    values: list[str] = []
    for dim in shape:
        values.append(str(getattr(dim, "value", dim)))
    if not values:
        return "[]"
    return "×".join(values)


def display_width(text: str) -> int:
    """按东亚宽字符双宽度估算显示宽度。

    参数：
      text：待估算文本。

    返回：
      显示列数。
    """
    width = 0
    for char in text:
        code = ord(char)
        if (
            0x1100 <= code <= 0x115F
            or 0x2E80 <= code <= 0xA4CF
            or 0xAC00 <= code <= 0xD7A3
            or 0xF900 <= code <= 0xFAFF
            or 0xFE30 <= code <= 0xFE4F
            or 0xFF00 <= code <= 0xFF60
            or 0xFFE0 <= code <= 0xFFE6
        ):
            width += 2
        else:
            width += 1
    return width


def wrap_lines(text: str, width: int = 44) -> list[str]:
    """按字符宽度换行。

    参数：
      text：原始文本；
      width：每行最大字符数（东亚字符按单宽近似）。

    返回：
      换行后的行列表；空文本返回空列表。
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
      可安全放入 SVG 文本节点的字符串。
    """
    return escape(text, quote=True)
