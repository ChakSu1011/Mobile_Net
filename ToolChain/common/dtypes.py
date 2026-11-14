"""模块：精度与元素字节数映射。

职责：
  为 Build 阶段提供稳定的精度名称到单元素字节数的映射。

边界：
  只描述存储宽度，不描述数值范围、舍入或累加语义。
"""

from __future__ import annotations

ELEMENT_BYTES = {
    "unknown": 0,
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3": 1,
    "float8_e5m2": 1,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}

INTEGER_PRECISIONS = frozenset({
    "int64",
    "int32",
    "int16",
    "int8",
    "uint8",
    "bool",
})

FLOAT_PRECISIONS = frozenset({
    "float64",
    "float32",
    "float16",
    "bfloat16",
    "float8_e4m3",
    "float8_e5m2",
})


def element_bytes(precision: str) -> int:
    """返回单元素字节数。

    参数：
      precision：稳定精度名，如 ``bfloat16``。

    返回：
      字节数；``unknown`` 返回 0。

    异常：
      ValueError：精度名不受支持。
    """
    try:
        return ELEMENT_BYTES[precision]
    except KeyError as error:
        raise ValueError(
            f"unsupported precision: {precision}") from error
