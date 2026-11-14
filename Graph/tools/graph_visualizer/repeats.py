"""模块：重复 Layer 识别与差异检测。

职责：
  按 ``layer.<n>.<op>`` 命名识别连续 Decoder Layer，比较归一化结构签名，
  将连续相同层合并为 ``LayerRun`` 并标记基准/差异层。

边界：
  - 只分析 ModelGraph 的 op/tensor 名称与属性，不修改图；
  - 只识别 ``layer.<n>.`` 命名模式；
  - 不负责布局与 SVG 渲染。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Sequence

from ToolChain.common.model_ir import OpMeta

_LAYER_OP = re.compile(r"^layer\.(?P<index>\d+)\.")
_LAYER_REF = re.compile(r"layer\.\d+\.")


@dataclass(frozen=True)
class LayerRun:
    """一组连续且结构相同的 Layer。

    属性：
      start：起始层号；
      end：结束层号；
      signature：归一化结构签名哈希；
      repeat_count：层数；
      is_variant：是否与基准签名不同。
    """

    start: int
    end: int
    signature: str
    repeat_count: int
    is_variant: bool


@dataclass(frozen=True)
class _RawRun:
    """内部使用的连续层区间。

    属性：
      start：起始层号；
      end：结束层号；
      signature：完整签名元组。
    """

    start: int
    end: int
    signature: tuple


def _normalize_ref(name: str) -> str:
    """把 tensor 名称中的层号替换为 ``layer.*.``。

    参数：
      name：op 或 tensor 名称。

    返回：
      归一化名称。
    """
    return _LAYER_REF.sub("layer.*.", name)


def _layer_signature(
    ops: Sequence[OpMeta],
    constants: set[str],
) -> tuple:
    """计算一层算子的结构签名。

    参数：
      ops：该层全部 op，按 ModelGraph 顺序排列；
      constants：模型图的常量张量名集合。

    返回：
      可比较的签名元组，包含归一化名称、类型、属性和连接。

    主要逻辑：
      层内产生的 activation 和本层常量保留归一化名称；来自其他层或图输入的
      外部 activation 统一记为 ``external:<consumer>:<index>``，避免第一层
      的 embedding 输入与后续层的前一层输出被误判为结构差异。
    """
    internal = {
        output for op in ops for output in op.outputs
    }
    items = []
    for op in ops:
        attrs = json.dumps(
            op.attrs,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        normalized_inputs = []
        suffix = _normalize_ref(op.name)
        for index, name in enumerate(op.inputs):
            if name in internal or name in constants:
                normalized_inputs.append(_normalize_ref(name))
            else:
                normalized_inputs.append(
                    f"external:{suffix.split('.', 2)[-1]}:{index}")
        items.append((
            _normalize_ref(op.name),
            op.op_type,
            attrs,
            tuple(normalized_inputs),
            tuple(_normalize_ref(name) for name in op.outputs),
        ))
    return tuple(items)


def _collect_layers(
    ops: Sequence[OpMeta],
) -> dict[int, list[OpMeta]]:
    """按层号收集 op。

    参数：
      ops：ModelGraph 的 op 序列。

    返回：
      层号到 op 列表的映射；无层名的 op 被忽略。
    """
    layers: dict[int, list[OpMeta]] = {}
    for op in ops:
        match = _LAYER_OP.match(op.name)
        if match is None:
            continue
        index = int(match.group("index"))
        layers.setdefault(index, []).append(op)
    return layers


def _raw_runs(
    layers: dict[int, list[OpMeta]],
    constants: set[str],
) -> list[_RawRun]:
    """把连续且签名相同的层合并为区间。

    参数：
      layers：层号到 op 列表的映射；
      constants：模型图的常量张量名集合。

    返回：
      连续区间列表；层号必须连续且签名一致才合并。
    """
    indices = sorted(layers)
    if not indices:
        return []
    runs: list[_RawRun] = []
    start = indices[0]
    previous = indices[0]
    signature = _layer_signature(layers[start], constants)
    for index in indices[1:]:
        current = _layer_signature(layers[index], constants)
        if index == previous + 1 and current == signature:
            previous = index
            continue
        runs.append(_RawRun(start, previous, signature))
        start = index
        previous = index
        signature = current
    runs.append(_RawRun(start, previous, signature))
    return runs


def _signature_hash(signature: tuple) -> str:
    """返回签名元组的稳定哈希。

    参数：
      signature：结构签名元组。

    返回：
      SHA-256 十六进制字符串。
    """
    text = repr(signature).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def analyze_layer_runs(
    ops: Sequence[OpMeta],
    constants: set[str] | None = None,
) -> list[LayerRun]:
    """识别连续相同层并标记差异层。

    参数：
      ops：ModelGraph 的 op 序列；
      constants：模型图常量名集合；省略时按空集处理。

    返回：
      ``LayerRun`` 列表；基准为重复次数最多的区间，重复次数相同时取最早。
    """
    raw = _raw_runs(
        _collect_layers(ops),
        set(constants or ()),
    )
    if not raw:
        return []
    counts = [run.end - run.start + 1 for run in raw]
    base_index = max(
        range(len(raw)),
        key=lambda index: (counts[index], -raw[index].start),
    )
    base_signature = raw[base_index].signature
    result = []
    for run, count in zip(raw, counts):
        result.append(LayerRun(
            start=run.start,
            end=run.end,
            signature=_signature_hash(run.signature),
            repeat_count=count,
            is_variant=run.signature != base_signature,
        ))
    return result
