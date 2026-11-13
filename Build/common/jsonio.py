"""模块：JSON 读写与规范化。

职责：
  提供确定性 JSON 编码（key 排序、紧凑分隔符、UTF-8）以及人类可读的
  JSON 文件读写，用于 `ModelGraph`、manifest 与报告。

边界：
  不做 JSON Schema 校验，不处理非标准类型；调用方必须先把值转换为
  bool/int/float/str/list/dict 等 JSON 原生类型。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def canonical_json(data: Any) -> str:
    """把对象编码为规范化 JSON 文本。

    功能：
      使用 key 排序、紧凑分隔符、UTF-8 非 ASCII 原样输出，保证同一对象
      在不同运行与不同进程中产生完全一致的字节。

    参数：
      data：可被 ``json.dumps`` 序列化的对象；不得包含 NaN/Infinity。

    返回：
      不含末尾换行的 JSON 文本。

    异常：
      TypeError：包含不可序列化对象；
      ValueError：包含 NaN 或 Infinity。
    """
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """把对象写入 JSON 文件。

    参数：
      path：输出路径；父目录必须已存在；
      data：可序列化对象；
      indent：缩进空格数；0 表示紧凑输出。

    异常：
      OSError：目录不存在或写入失败；
      TypeError/ValueError：对象不可序列化。
    """
    if indent <= 0:
        text = canonical_json(data)
    else:
        text = json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )
    path.write_text(text + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """读取并解析 JSON 文件。

    参数：
      path：输入路径。

    返回：
      解析后的 Python 对象。

    异常：
      OSError：文件不存在或读取失败；
      json.JSONDecodeError：内容不是合法 JSON。
    """
    return json.loads(path.read_text(encoding="utf-8"))
